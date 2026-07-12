from pathlib import Path
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, AcceleratorDevice, AcceleratorOptions, 
    RapidOcrOptions, EasyOcrOptions, TableFormerMode
)
from docling.datamodel.base_models import InputFormat
from logger import setup_logger

def process_batch(batch_info: dict, output_dir: str, config_dict: dict):
    """
    Processes an OCR batch using Docling.
    Function name MUST be 'process_batch' to match the import in extractor.py.
    """
    batch_path = batch_info["path"]
    batch_name = Path(batch_path).stem
    
    # Extract user_id for logging
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

    # 5. Initialize Converter (Strictly typed PdfFormatOption)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    
    try:
        result = converter.convert(batch_path)
        
        if result.status == "success":
            out_path = Path(output_dir) / f"{batch_name}.md"
            out_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
            log.info(f"{batch_name} (OCR) completed.")
            return {
                "batch": batch_name, "type": "ocr", "status": "success",
                "output": str(out_path), "pages": batch_info["pages"]
            }
        else:
            return {"batch": batch_name, "type": "ocr", "status": "failed", "error": result.status}
    except Exception as e:
        log.exception(f"Docling crashed on {batch_name}: {e}")
        return {"batch": batch_name, "type": "ocr", "status": "error", "error": str(e)}