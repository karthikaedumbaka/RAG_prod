from analyzer import create_smart_batches
from config import PipelineConfig
from logger import setup_logger

log = setup_logger("batcher")

def create_batches(pdf_path: str, analysis: dict, config: PipelineConfig) -> dict:
    """Wrapper for smart batching"""
    return create_smart_batches(pdf_path, analysis, config)