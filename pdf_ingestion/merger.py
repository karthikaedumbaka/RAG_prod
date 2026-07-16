from pathlib import Path

try:
    from .config import PipelineConfig
    from .logger import setup_logger
except ImportError:
    from config import PipelineConfig
    from logger import setup_logger



def merge_outputs(pdf_name: str, batches: dict, config: PipelineConfig):
    log = setup_logger("merger", config.user_id)
    log.info(f" Merging dual-path outputs for {pdf_name}...")
    final_md = Path(config.output_dir) / f"{pdf_name}.md"
    
    # Combine and map all batches
    all_batch_info = []
    for batch in batches["text_batches"] + batches["ocr_batches"]:
        all_batch_info.append({
            "path": Path(config.output_dir) / f"{Path(batch['path']).stem}.md",
            "pages": batch["pages"]
        })
    
    # CRITICAL: Sort by the original page index to maintain document flow
    all_batch_info.sort(key=lambda x: min(x["pages"]) if x["pages"] else 0)
    
    # Stream to disk
    with open(final_md, "w", encoding="utf-8") as out_f:
        for info in all_batch_info:
            if info["path"].exists():
                out_f.write(info["path"].read_text(encoding="utf-8"))
                out_f.write("\n\n---\n\n")
            else:
                log.warning(f" Missing: {info['path']}")
                
    log.info(f" Merge complete: {final_md}")
    return final_md
