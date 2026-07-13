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
from logger import setup_logger

def process_batch(batch_info: dict, output_dir: str, config_dict: dict):
    """
    Processes an OCR batch using Docling.
    Refactored to process page-by-page to guarantee exact page numbers for RAG.
    """
    batch_path = batch_info["path"]
    batch_name = Path(batch_path).stem
    user_id = config_dict.get("user_id", "unknown")
    log = setup_logger(f"docling-{batch_name}", user_id)
    log.info(f"Processing OCR batch: {batch_path}")

    # 1. Hardware Acceleration (Dynamic Detection)
    if config_dict["use_gpu"]:
        if torch.cuda.is_available():
            device = AcceleratorDevice.CUDA
            log.info("Hardware detected: NVIDIA GPU (CUDA)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = AcceleratorDevice.MPS
            log.info("Hardware detected: Apple Silicon (MPS)")
        else:
            device = AcceleratorDevice.CPU
            log.info("Hardware detected: CPU (Fallback - No GPU available)")
    else:
        device = AcceleratorDevice.CPU
        log.info("Hardware forced: CPU via config")
        
    accel_opts = AcceleratorOptions(num_threads=4, device=device)
    
    # 2. Pipeline Options (Memory Safe)
    pipeline_opts = PdfPipelineOptions(
        accelerator_options=accel_opts,
        do_table_structure=config_dict["do_table_structure"],
        images_scale=config_dict["image_scale"]
    )
    
    # 3. Table Optimization
    if config_dict["use_fast_tables"]:
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST
        log.info("Table mode: FAST")
    else:
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE
        log.info("Table mode: ACCURATE")
        
    # 3b. Picture / chart / diagram extraction
    if config_dict.get("extract_pictures", True):
        pipeline_opts.generate_picture_images = True
        pipeline_opts.images_scale = config_dict["image_scale"]
        log.info("Picture/chart/diagram extraction: ON")
        
    # 4. OCR Optimization
    pipeline_opts.do_ocr = True
    if config_dict["use_rapid_ocr"]:
        pipeline_opts.ocr_options = RapidOcrOptions()
        log.info("Using RapidOCR (ONNX) - High Speed")
    else:
        pipeline_opts.ocr_options = EasyOcrOptions(
            lang=config_dict["ocr_lang"],
            use_gpu=config_dict["use_gpu"]
        )
        log.info("Using EasyOCR (PyTorch) - Fallback")

    # 4b. Handwriting / small-region OCR safety net.
    # By default Docling only OCRs bitmap regions covering >5% of the page,
    # and skips re-OCRing anything already covered by a native text layer.
    # That silently drops small handwritten signatures/margin notes embedded
    # in an otherwise-digital page. Force a full re-OCR pass and lower the
    # area threshold so those regions actually get read.
    force_full_page_ocr = config_dict.get("force_full_page_ocr", True)
    bitmap_area_threshold = config_dict.get("bitmap_area_threshold", 0.01)
    pipeline_opts.ocr_options.force_full_page_ocr = force_full_page_ocr
    pipeline_opts.ocr_options.bitmap_area_threshold = bitmap_area_threshold
    log.info(
        f"OCR coverage: force_full_page_ocr={force_full_page_ocr}, "
        f"bitmap_area_threshold={bitmap_area_threshold} (catches small handwritten regions)"
    )

    # 5. Initialize Converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    
    # Images directory for charts/diagrams
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        batch_doc = fitz.open(batch_path)
        full_markdown = []
        
        # Process each page individually to preserve exact page mapping
        for local_idx, original_page_num in enumerate(batch_info["pages"]):
            # Create a single-page PDF in memory
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(batch_doc, from_page=local_idx, to_page=local_idx)
            
            # 🛠️ FIX: Generate a unique temporary file path.
            # On Windows, tempfile.NamedTemporaryFile keeps the file open/locked.
            # Using mkstemp and immediately closing the descriptor fixes the "Permission denied" error.
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(tmp_fd)  # Release the lock immediately so PyMuPDF can write to it
            
            try:
                single_page_doc.save(tmp_path)
                
                # Convert the single page
                result = converter.convert(tmp_path)
                
                if result.status == "success":
                    # INJECT INVISIBLE PAGE MARKER (1-based index for humans)
                    page_marker = f"\n\n<!-- page: {original_page_num + 1} -->\n\n"
                    
                    # Export to markdown
                    if config_dict.get("extract_pictures", True) and ImageRefMode is not None:
                        try:
                            # Try the newer API with artifacts_dir
                            md_text = result.document.export_to_markdown(
                                image_mode=ImageRefMode.REFERENCED,
                                artifacts_dir=images_dir
                            )
                        except TypeError:
                            # Fallback for older Docling versions that don't support artifacts_dir
                            md_text = result.document.export_to_markdown(
                                image_mode=ImageRefMode.REFERENCED
                            )
                            log.warning(f"Docling version lacks 'artifacts_dir'. Images saved to default location.")
                    else:
                        md_text = result.document.export_to_markdown()
                        
                    full_markdown.append(page_marker + md_text)
                else:
                    log.warning(f"Docling failed on page {original_page_num + 1}: {result.status}")
            except Exception as page_e:
                log.error(f"Error processing page {original_page_num + 1}: {page_e}")
            finally:
                # Ensure memory is freed and temp file is cleaned up, even if it crashes
                single_page_doc.close()
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    
        batch_doc.close()
        
        # Write combined markdown to disk
        out_path = Path(output_dir) / f"{batch_name}.md"
        out_path.write_text("\n\n".join(full_markdown), encoding="utf-8")
        log.info(f" {batch_name} (OCR) completed with page markers.")
        
        return {
            "batch": batch_name, "type": "ocr", "status": "success",
            "output": str(out_path), "pages": batch_info["pages"]
        }
    except Exception as e:
        log.exception(f" Docling crashed on {batch_name}: {e}")
        return {"batch": batch_name, "type": "ocr", "status": "error", "error": str(e)}