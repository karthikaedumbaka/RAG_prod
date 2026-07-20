import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .config import PipelineConfig
    from .checkpoint import CheckpointManager
    from .docling_worker import process_batch as process_ocr_batch, init_worker
    from .pymupdf_worker import process_text_batch
    from .logger import setup_logger
except ImportError:
    from config import PipelineConfig
    from checkpoint import CheckpointManager
    from docling_worker import process_batch as process_ocr_batch, init_worker
    from pymupdf_worker import process_text_batch
    from logger import setup_logger

def extract_batches(batches: dict, config: PipelineConfig):
    log = setup_logger("extractor", config.user_id)
    ctx = mp.get_context('spawn')
    checkpoint = CheckpointManager(config)
    
    config_dict = {
        "user_id": config.user_id,
        "use_gpu": config.use_gpu,
        "do_table_structure": config.do_table_structure,
        "ocr_lang": config.ocr_lang,
        "image_scale": config.image_scale,
        "use_rapid_ocr": config.use_rapid_ocr,
        "use_fast_tables": config.use_fast_tables,
        "extract_pictures": config.extract_pictures,
        "force_full_page_ocr": config.force_full_page_ocr,
        "bitmap_area_threshold": config.bitmap_area_threshold
    }

    pending_text = [b for b in batches["text_batches"] if not checkpoint.is_completed(Path(b["path"]).stem)]
    pending_ocr = [b for b in batches["ocr_batches"] if not checkpoint.is_completed(Path(b["path"]).stem)]
    
    total_pending = len(pending_text) + len(pending_ocr)
    if total_pending == 0:
        log.info("All batches already completed.")
        return

    log.info(f"Dispatching {len(pending_text)} Text batches (Threads) & {len(pending_ocr)} OCR batches (Processes)")
    
    futures = {}
    
    # 1. Process Text Batches (I/O Bound -> Threads)
    with ThreadPoolExecutor(max_workers=config.num_text_threads) as text_exec:
        for batch in pending_text:
            f = text_exec.submit(process_text_batch, batch, config.output_dir, config.user_id)
            futures[f] = batch
            
        for future in as_completed(futures):
            batch = futures[future]
            try:
                res = future.result()
                if res["status"] == "success":
                    checkpoint.mark_completed(res["batch"])
            except Exception as e:
                log.error(f"Text batch failed: {e}")

    futures.clear()

    # 2. Process OCR Batches (CPU/GPU Bound -> Processes)
    if pending_ocr:
        # 🛠️ FIX: Pass initializer and initargs to load models ONCE per worker
        with ProcessPoolExecutor(
            max_workers=config.num_workers, 
            mp_context=ctx,
            initializer=init_worker,      # <--- Load models here
            initargs=(config_dict,)       # <--- Pass config to the initializer
        ) as ocr_exec:
            for batch in pending_ocr:
                f = ocr_exec.submit(process_ocr_batch, batch, config.output_dir, config_dict)
                futures[f] = batch
                
            completed = 0
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    res = future.result()
                    completed += 1
                    if res["status"] == "success":
                        checkpoint.mark_completed(res["batch"])
                        log.info(f"OCR Progress: {completed}/{len(pending_ocr)}")
                    else:
                        log.error(f"OCR batch failed: {res.get('error')}")
                except Exception as e:
                    log.exception(f"OCR worker crashed: {e}")