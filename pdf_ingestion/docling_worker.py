import os
import tempfile
import fitz
from pathlib import Path
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions,
    RapidOcrOptions, EasyOcrOptions, TableFormerMode
)
from docling.datamodel.base_models import InputFormat

try:
    from docling_core.types.doc import ImageRefMode
except ImportError:
    ImageRefMode = None

try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

# 🛠️ FIX: Global variable to hold the converter per worker process
_worker_converter = None
_worker_log = None

def init_worker(config_dict: dict):
    """
    Runs ONCE per worker process to load heavy models into memory.
    Prevents reloading the model for every single batch.
    """
    global _worker_converter, _worker_log
    
    user_id = config_dict.get("user_id", "unknown")
    _worker_log = setup_logger(f"docling-worker-{os.getpid()}", user_id)
    _worker_log.info(f"Worker {os.getpid()} initializing models...")

    # 1. Hardware Acceleration
    if config_dict["use_gpu"]:
        if torch.cuda.is_available():
            device = AcceleratorDevice.CUDA
            _worker_log.info("Hardware detected: NVIDIA GPU (CUDA)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = AcceleratorDevice.MPS
            _worker_log.info("Hardware detected: Apple Silicon (MPS)")
        else:
            device = AcceleratorDevice.CPU
            _worker_log.info("Hardware detected: CPU (Fallback)")
    else:
        device = AcceleratorDevice.CPU
        _worker_log.info("Hardware forced: CPU via config")

    accel_opts = AcceleratorOptions(num_threads=4, device=device)

    # 2. Pipeline Options
    pipeline_opts = PdfPipelineOptions(
        accelerator_options=accel_opts,
        do_table_structure=config_dict["do_table_structure"],
        images_scale=config_dict["image_scale"]
    )

    # 3. Table & Picture Optimization
    if config_dict["use_fast_tables"]:
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST
    else:
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE

    if config_dict.get("extract_pictures", True):
        pipeline_opts.generate_picture_images = True

    # 4. OCR Optimization
    pipeline_opts.do_ocr = True
    if config_dict["use_rapid_ocr"]:
        pipeline_opts.ocr_options = RapidOcrOptions()
    else:
        pipeline_opts.ocr_options = EasyOcrOptions(
            lang=config_dict["ocr_lang"],
            use_gpu=config_dict["use_gpu"]
        )

    pipeline_opts.ocr_options.force_full_page_ocr = config_dict.get("force_full_page_ocr", True)
    pipeline_opts.ocr_options.bitmap_area_threshold = config_dict.get("bitmap_area_threshold", 0.01)

    # 5. Initialize Converter ONCE
    _worker_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    _worker_log.info(f"Worker {os.getpid()} initialized successfully.")

def process_batch(batch_info: dict, output_dir: str, config_dict: dict):
    """
    Processes an OCR batch using the pre-loaded Docling converter.
    """
    global _worker_converter, _worker_log
    
    batch_path = batch_info["path"]
    batch_name = Path(batch_path).stem
    user_id = config_dict.get("user_id", "unknown")
    
    # Fallback logger if init_worker somehow failed to set it
    if _worker_log is None:
        _worker_log = setup_logger(f"docling-{batch_name}", user_id)
    if _worker_converter is None:
        init_worker(config_dict)
        
    _worker_log.info(f"Processing OCR batch: {batch_path}")

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        full_markdown = []
        
        # 🛠️ FIX: Use context manager for the main batch document
        with fitz.open(batch_path) as batch_doc:
            for local_idx, original_page_num in enumerate(batch_info["pages"]):
                # 🛠️ FIX: Use context manager for single page doc
                with fitz.open() as single_page_doc:
                    single_page_doc.insert_pdf(batch_doc, from_page=local_idx, to_page=local_idx)
                    
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                    os.close(tmp_fd) 
                    
                    try:
                        single_page_doc.save(tmp_path)
                        result = _worker_converter.convert(tmp_path)
                        
                        if result.status == "success":
                            page_marker = f"\n\n<!-- page: {original_page_num + 1} -->\n\n"
                            
                            if config_dict.get("extract_pictures", True) and ImageRefMode is not None:
                                try:
                                    md_text = result.document.export_to_markdown(
                                        image_mode=ImageRefMode.REFERENCED,
                                        artifacts_dir=images_dir
                                    )
                                except TypeError:
                                    md_text = result.document.export_to_markdown(
                                        image_mode=ImageRefMode.REFERENCED
                                    )
                            else:
                                md_text = result.document.export_to_markdown()
                                
                            full_markdown.append(page_marker + md_text)
                        else:
                            _worker_log.warning(f"Docling failed on page {original_page_num + 1}: {result.status}")
                            
                    except Exception as page_e:
                        _worker_log.error(f"Error processing page {original_page_num + 1}: {page_e}")
                    finally:
                        if os.path.exists(tmp_path):
                            try: os.unlink(tmp_path)
                            except Exception: pass

        # Write combined markdown to disk
        out_path = Path(output_dir) / f"{batch_name}.md"
        out_path.write_text("\n\n".join(full_markdown), encoding="utf-8")
        
        _worker_log.info(f" {batch_name} (OCR) completed with page markers.")
        return {
            "batch": batch_name, "type": "ocr", "status": "success",
            "output": str(out_path), "pages": batch_info["pages"]
        }
        
    except Exception as e:
        _worker_log.exception(f" Docling crashed on {batch_name}: {e}")
        return {"batch": batch_name, "type": "ocr", "status": "error", "error": str(e)}