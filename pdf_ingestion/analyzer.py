import fitz  # PyMuPDF
from pathlib import Path
from logger import setup_logger
from config import PipelineConfig

log = setup_logger("analyzer")

def analyze_pdf(pdf_path: str, config: PipelineConfig) -> dict:
    """
    We scan every page and classify it as "text" or "ocr_needed".
    This allows us to batch similar pages together for optimal performance.
    """
    log = setup_logger("analyzer", config.user_id)
    log.info(f" Analyzing {Path(pdf_path).name}...")
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    text_pages = []
    ocr_pages = []
    
    # Classify every page
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Decision logic: If page has enough text, it's a text page
        if len(text) >= config.min_text_length:
            text_pages.append(page_num)
        else:
            ocr_pages.append(page_num)
    
    doc.close()
    
    analysis = {
        "total_pages": total_pages,
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "text_count": len(text_pages),
        "ocr_count": len(ocr_pages),
        "ocr_percentage": (len(ocr_pages) / total_pages * 100) if total_pages > 0 else 0
    }
    
    log.info(f"   Analysis complete:")
    log.info(f"   Total pages: {total_pages}")
    log.info(f"   Text pages: {len(text_pages)} ({100 - analysis['ocr_percentage']:.1f}%)")
    log.info(f"   OCR needed: {len(ocr_pages)} ({analysis['ocr_percentage']:.1f}%)")
    
    return analysis


def create_smart_batches(pdf_path: str, analysis: dict, config: PipelineConfig) -> dict:
    """
    Top 1% Move: Create separate batches for text and OCR pages.
    Text pages can be processed in larger batches (faster).
    OCR pages need smaller batches (memory-intensive).
    """
    from utils import ensure_dir
    ensure_dir(config.temp_dir)
    
    doc = fitz.open(pdf_path)
    batches = {
        "text_batches": [],
        "ocr_batches": []
    }
    
    # Create text page batches (larger batches, faster processing)
    text_pages = analysis["text_pages"]
    for i in range(0, len(text_pages), config.text_pages_per_batch):
        batch_pages = text_pages[i:i + config.text_pages_per_batch]
        batch_doc = fitz.open()
        
        for page_num in batch_pages:
            batch_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        batch_path = Path(config.temp_dir) / f"text_batch_{i // config.text_pages_per_batch:03d}.pdf"
        batch_doc.save(str(batch_path))
        batch_doc.close()
        
        batches["text_batches"].append({
            "path": str(batch_path),
            "pages": batch_pages,
            "type": "text"
        })
    
    # Create OCR page batches (smaller batches, memory-safe)
    ocr_pages = analysis["ocr_pages"]
    for i in range(0, len(ocr_pages), config.ocr_pages_per_batch):
        batch_pages = ocr_pages[i:i + config.ocr_pages_per_batch]
        batch_doc = fitz.open()
        
        for page_num in batch_pages:
            batch_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        batch_path = Path(config.temp_dir) / f"ocr_batch_{i // config.ocr_pages_per_batch:03d}.pdf"
        batch_doc.save(str(batch_path))
        batch_doc.close()
        
        batches["ocr_batches"].append({
            "path": str(batch_path),
            "pages": batch_pages,
            "type": "ocr"
        })
    
    doc.close()
    
    log.info(f" Created {len(batches['text_batches'])} text batches, "
             f"{len(batches['ocr_batches'])} OCR batches")
    
    return batches