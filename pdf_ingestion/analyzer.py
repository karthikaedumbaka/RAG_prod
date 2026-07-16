import fitz  # PyMuPDF
from pathlib import Path

try:
    from .logger import setup_logger
    from .config import PipelineConfig
except ImportError:
    from logger import setup_logger
    from config import PipelineConfig

log = setup_logger("analyzer")

def _find_template_image_xrefs(doc, config: PipelineConfig) -> set:
    """
    Reports often repeat the same banner/logo/background images on every
    single page (e.g. an SDG icon strip, a page border). Those inflate the
    per-page image count without indicating real content, so we identify
    them up front and exclude them from the per-page "is this a visual page"
    check. An image is treated as "template" if it appears on more than
    60% of pages.
    """
    total_pages = len(doc)
    if total_pages == 0:
        return set()

    xref_page_counts = {}
    for page_num in range(total_pages):
        xrefs_on_page = {img[0] for img in doc[page_num].get_images()}
        for xref in xrefs_on_page:
            xref_page_counts[xref] = xref_page_counts.get(xref, 0) + 1

    threshold = max(1, int(total_pages * 0.6))
    return {xref for xref, count in xref_page_counts.items() if count >= threshold}


def _page_has_ink_annotations(page) -> bool:
    """
    Hand-drawn "ink" annotations (scribbled notes, handwritten signatures
    added in a PDF viewer) have zero presence in get_text() - they're just
    stroke geometry, not characters. The ONLY way to recover them is to
    rasterize the page and run OCR over the pixels, which only the
    Docling path does. So their mere presence must force OCR routing,
    regardless of how much clean text is otherwise on the page.
    """
    try:
        for annot in page.annots() or []:
            # PDF annotation type 15 == "Ink" in the PDF spec / PyMuPDF
            if annot.type[0] == 15 or annot.type[1] == "Ink":
                return True
    except Exception:
        pass
    return False


def _page_has_visual_content(page, template_xrefs: set, config: PipelineConfig) -> tuple[bool, dict]:
    """
    A page can be full of paragraph text AND still contain pie charts, bar
    charts, tables, or diagrams drawn as vector graphics (paths/rects/curves),
    genuinely unique embedded images, or hand-drawn ink annotations. Plain
    get_text() extraction captures none of that structure - it only grabs
    stray text labels. So we detect visual density here and use it to force
    those pages down the rich OCR/Docling path (table structure + layout +
    image extraction + full-page OCR) instead of the fast-but-blind PyMuPDF
    text-only path.
    """
    drawings = page.get_drawings()
    images = page.get_images()

    num_drawings = len(drawings)
    novel_images = [img for img in images if img[0] not in template_xrefs]
    num_novel_images = len(novel_images)
    has_ink = config.route_ink_annotations_to_ocr and _page_has_ink_annotations(page)

    is_visual = (
        num_drawings >= config.max_drawings_for_text_path
        or num_novel_images >= config.min_images_for_visual_page
        or has_ink
    )

    return is_visual, {
        "drawings": num_drawings,
        "novel_images": num_novel_images,
        "has_ink_annotations": has_ink,
    }


def analyze_pdf(pdf_path: str, config: PipelineConfig) -> dict:
    """
    We scan every page and classify it as "text" or "ocr_needed".
    This allows us to batch similar pages together for optimal performance.

    Classification now considers BOTH text length AND visual complexity
    (vector drawings / embedded images), since a page with plenty of text
    can still be dominated by a pie chart, bar chart, or table that plain
    text extraction would silently drop.
    """
    log = setup_logger("analyzer", config.user_id)
    log.info(f" Analyzing {Path(pdf_path).name}...")
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    text_pages = []
    ocr_pages = []
    visual_page_count = 0

    template_xrefs = _find_template_image_xrefs(doc, config)
    if template_xrefs:
        log.info(f"   Detected {len(template_xrefs)} repeating template image(s) (logos/banners) - excluded from visual detection")
    
    # Classify every page
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        has_enough_text = len(text) >= config.min_text_length

        is_visual, visual_stats = _page_has_visual_content(page, template_xrefs, config)
        if is_visual:
            visual_page_count += 1

        # Decision logic:
        #   - No visual content AND enough text  -> fast text path
        #   - Visual content (charts/tables/diagrams) OR too little text
        #     -> rich OCR/Docling path, regardless of how much text is present
        if has_enough_text and not is_visual:
            text_pages.append(page_num)
        else:
            ocr_pages.append(page_num)

        if is_visual:
            reason = "ink/handwritten annotation" if visual_stats["has_ink_annotations"] else "chart/table/diagram"
            log.debug(
                f"   Page {page_num}: routed to OCR (drawings={visual_stats['drawings']}, "
                f"novel_images={visual_stats['novel_images']}, "
                f"ink={visual_stats['has_ink_annotations']}) - likely {reason}"
            )
    
    doc.close()
    
    analysis = {
        "total_pages": total_pages,
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "text_count": len(text_pages),
        "ocr_count": len(ocr_pages),
        "visual_page_count": visual_page_count,
        "ocr_percentage": (len(ocr_pages) / total_pages * 100) if total_pages > 0 else 0
    }
    
    log.info(f"   Analysis complete:")
    log.info(f"   Total pages: {total_pages}")
    log.info(f"   Text pages: {len(text_pages)} ({100 - analysis['ocr_percentage']:.1f}%)")
    log.info(f"   OCR needed: {len(ocr_pages)} ({analysis['ocr_percentage']:.1f}%)")
    log.info(f"   Pages with charts/tables/diagrams detected: {visual_page_count}")
    
    return analysis


def create_smart_batches(pdf_path: str, analysis: dict, config: PipelineConfig) -> dict:
    """
    Top 1% Move: Create separate batches for text and OCR pages.
    Text pages can be processed in larger batches (faster).
    OCR pages need smaller batches (memory-intensive).
    """
    try:
        from .utils import ensure_dir
    except ImportError:
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
