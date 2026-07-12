import shutil
from pathlib import Path
from logger import setup_logger



def cleanup_artifacts(config):
    """
    Safely deletes intermediate batches and checkpoints to save disk space.
    """
    log = setup_logger("cleaner", config.user_id)
    log.info(" Cleaning up intermediate artifacts...")
    
    # 1. Clear temp batches directory entirely
    temp_path = Path(config.temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
        log.info(f"   ️ Deleted temp directory: {temp_path.name}")
        
    # 2. Delete intermediate batch markdowns in output dir
    output_path = Path(config.output_dir)
    if output_path.exists():
        # SAFETY: Only delete files with "batch" in the name. 
        # This guarantees we NEVER delete the final merged document.
        batch_files = list(output_path.glob("*batch*.md"))
        if batch_files:
            for file in batch_files:
                file.unlink()
            log.info(f"    Deleted {len(batch_files)} intermediate batch files.")
            
    # 3. Delete checkpoint file to ensure a fresh state for the next run
    checkpoint = output_path / config.checkpoint_file
    if checkpoint.exists():
        checkpoint.unlink()
        log.info("    Deleted checkpoint file.")
        
    log.info(" Cleanup complete. Disk space reclaimed.")