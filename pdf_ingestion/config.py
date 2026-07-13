import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent 

@dataclass
class PipelineConfig:
    user_id: str = "unknown"
    
    # Auto-discovery
    data_dir: str = str(PROJECT_ROOT / "data")
    output_dir: str = str(PROJECT_ROOT / "output")
    temp_dir: str = str(PROJECT_ROOT / "temp_batches")
    
    # Hardware (MEMORY SAFE SETTINGS)
    use_gpu: bool = True
    num_workers: int = 2       
    num_text_threads: int = 16 
    
    # Intelligent batching
    text_pages_per_batch: int = 100 
    ocr_pages_per_batch: int = 5  
    
    # Docling pipeline
    do_table_structure: bool = True          # was False - tables were being dropped entirely
    use_fast_tables: bool = False             # ACCURATE mode: FAST misreads merged/colored report tables
    extract_pictures: bool = True             # pull embedded charts / pie diagrams / figures out as images
    ocr_lang: list = field(default_factory=lambda: ["en"])
    
    # Checkpoint & recovery
    checkpoint_file: str = "pipeline_state.json"
    
    # PERFORMANCE OVERRIDES
    image_scale: float = 2.0                  # was 1.25 - too low-res to read small chart labels/legends
    use_rapid_ocr: bool = True      
    min_text_length: int = 50       
    
    # --- Visual-content routing (NEW) ---
    # A page can have lots of *text* (paragraphs, headers) while ALSO containing
    # pie charts / bar charts / tables drawn as vector graphics. Those pages must
    # go through the Docling (OCR + layout + table) pipeline, not the plain
    # PyMuPDF fast-text path, or the chart/table data is silently lost.
    max_drawings_for_text_path: int = 130     # vector paths above this = charts/tables/diagrams present
    min_images_for_visual_page: int = 1       # any NON-template embedded image = real diagram/photo content
    route_ink_annotations_to_ocr: bool = True # hand-drawn "ink" annotations (scribbled notes/signatures)
                                               # have no text layer at all - force these pages to OCR
    # Note: thresholds are intentionally biased low. Sending a plain text page
    # through the OCR/Docling path costs a little extra time; sending a
    # chart/table page through the fast PyMuPDF-only path silently loses data.
    # When in doubt, route to OCR.

    # --- Handwriting / small-region OCR (NEW) ---
    # Docling, by default, only OCRs bitmap regions that cover >5% of the page
    # AND skips regions already covered by a native text layer unless forced.
    # That means a small handwritten signature or margin note embedded in an
    # otherwise-digital page gets silently skipped even on OCR-routed pages.
    force_full_page_ocr: bool = True          # re-OCR the whole rendered page, ignore the native text layer gap
    bitmap_area_threshold: float = 0.01       # OCR any bitmap region >=1% of page (was Docling default 0.05)
    
    def __post_init__(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)