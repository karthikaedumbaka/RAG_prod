import time
import uuid
import getpass # 🛠️ FIX: Replaced msvcrt with cross-platform getpass
from pathlib import Path

try:
    from .config import PipelineConfig
    from .auth import authenticate_or_register
    from .analyzer import analyze_pdf
    from .batcher import create_batches
    from .extractor import extract_batches
    from .merger import merge_outputs
    from .cleaner import cleanup_artifacts
    from .utils import ensure_dir, find_pdfs
    from .logger import setup_logger
except ImportError:
    from config import PipelineConfig
    from auth import authenticate_or_register
    from analyzer import analyze_pdf
    from batcher import create_batches
    from extractor import extract_batches
    from merger import merge_outputs
    from cleaner import cleanup_artifacts
    from utils import ensure_dir, find_pdfs
    from logger import setup_logger

def get_masked_password(prompt="Enter Password: "):
    """
    Cross-platform masked password input.
    """
    return getpass.getpass(prompt)

def authenticate_user() -> str:
    """
    Dynamic CLI Authentication.
    Auto-creates DB, logs in existing users, or registers new users.
    """
    print("=" * 50)
    print("USER AUTHENTICATION")
    print("=" * 50)
    try:
        user_id = input("Enter User ID: ").strip()
        password = get_masked_password("Enter Password: ")
    except EOFError as exc:
        raise RuntimeError(
            "CLI authentication requires an interactive terminal. "
            "Run `python -m pdf_ingestion.main` interactively or launch the UI instead."
        ) from exc

    if not user_id or not password:
        raise RuntimeError("User ID and password are required.")

    # Call the dynamic auth function
    auth_status = authenticate_or_register(user_id, password)

    if auth_status == "authenticated":
        print(f"Authentication successful. Welcome back, {user_id}!")
        return user_id
    elif auth_status == "registered":
        print(f"New user detected. Account created successfully. Welcome, {user_id}!")
        return user_id
    else:
        # Password mismatch for an existing user
        print(f"Authentication failed for user '{user_id}'. Incorrect password.")
        unique_id = f"guest_{uuid.uuid4().hex[:8]}"
        print(f"Assigned unique guest ID: {unique_id}")
        return unique_id

def process_single_pdf(pdf_path: Path, config: PipelineConfig) -> dict:
    """Process a single PDF through the entire pipeline"""
    log = setup_logger("main", config.user_id)
    pdf_name = pdf_path.stem
    log.info("=" * 60)
    log.info(f"Processing: {pdf_path.name}")
    log.info("=" * 60)
    
    start_time = time.time()
    try:
        analysis = analyze_pdf(str(pdf_path), config)
        batches = create_batches(str(pdf_path), analysis, config)
        extract_batches(batches, config)
        final_md = merge_outputs(pdf_name, batches, config)
        
        if final_md.exists():
            cleanup_artifacts(config)
        else:
            log.warning("Final merge failed or file missing. Keeping artifacts for debugging.")
            
        elapsed = time.time() - start_time
        pages_per_sec = analysis["total_pages"] / elapsed if elapsed > 0 else 0
        
        log.info(f"Completed {pdf_path.name}")
        log.info(f"Time: {elapsed:.2f}s | Speed: {pages_per_sec:.2f} pages/sec")
        log.info(f"Output: {final_md}")
        
        return {
            "pdf": str(pdf_path),
            "pages": analysis["total_pages"],
            "time": elapsed,
            "speed": pages_per_sec,
            "output": str(final_md),
            "status": "success"
        }
    except Exception as e:
        log.exception(f"Critical failure processing {pdf_path.name}: {e}")
        return {
            "pdf": str(pdf_path),
            "status": "failed",
            "error": str(e)
        }

def run_pipeline():
    # 1. AUTHENTICATE BEFORE ANYTHING ELSE
    user_id = authenticate_user()
    start_time = time.time()
    
    config = PipelineConfig()
    config.user_id = user_id  # Pass authenticated user_id to config
    ensure_dir(config.output_dir)
    ensure_dir(config.temp_dir)
    
    log = setup_logger("main", config.user_id)
    log.info("=" * 60)
    log.info("PDF INGESTION PIPELINE STARTING")
    log.info("=" * 60)
    log.info(f"Data directory: {config.data_dir}")
    log.info(f"Output directory: {config.output_dir}")
    log.info(f"Workers: {config.num_workers} | GPU: {config.use_gpu}")
    
    pdf_files = find_pdfs(config.data_dir)
    if not pdf_files:
        log.error(f"No PDFs found in {config.data_dir}")
        log.info("Tip: Place your PDFs in the 'data/' folder and run again.")
        return
        
    log.info(f"Found {len(pdf_files)} PDF(s) to process")
    
    results = []
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path, config)
        results.append(result)
        
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    total_pages = sum(r.get("pages", 0) for r in results if r.get("status") == "success")
    overall_speed = total_pages / total_time if total_time > 0 else 0
    
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"PDFs processed: {successful}/{len(pdf_files)}")
    log.info(f"Failed: {failed}")
    log.info(f"Total pages: {total_pages}")
    log.info(f"Total time: {total_time:.2f}s")
    log.info(f"Overall speed: {overall_speed:.2f} pages/sec")
    log.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()