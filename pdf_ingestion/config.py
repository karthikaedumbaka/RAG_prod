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
    do_table_structure: bool = False 
    ocr_lang: list = field(default_factory=lambda: ["en"])
    
    # Checkpoint & recovery
    checkpoint_file: str = "pipeline_state.json"
    
    # PERFORMANCE OVERRIDES
    image_scale: float = 1.25     
    use_rapid_ocr: bool = True      
    use_fast_tables: bool = True    
    min_text_length: int = 50       
    
    def __post_init__(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)