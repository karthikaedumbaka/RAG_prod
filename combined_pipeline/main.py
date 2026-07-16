
import time
import uuid
import getpass
from pathlib import Path

# Import from pdf_ingestion
try:
    from pdf_ingestion.config import PipelineConfig as PDFIngestionConfig
    from pdf_ingestion.auth import authenticate_or_register
    from pdf_ingestion.analyzer import analyze_pdf
    from pdf_ingestion.batcher import create_batches
    from pdf_ingestion.extractor import extract_batches
    from pdf_ingestion.merger import merge_outputs
    from pdf_ingestion.cleaner import cleanup_artifacts
    from pdf_ingestion.utils import ensure_dir, find_pdfs
    from pdf_ingestion.logger import setup_logger
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pdf_ingestion.config import PipelineConfig as PDFIngestionConfig
    from pdf_ingestion.auth import authenticate_or_register
    from pdf_ingestion.analyzer import analyze_pdf
    from pdf_ingestion.batcher import create_batches
    from pdf_ingestion.extractor import extract_batches
    from pdf_ingestion.merger import merge_outputs
    from pdf_ingestion.cleaner import cleanup_artifacts
    from pdf_ingestion.utils import ensure_dir, find_pdfs
    from pdf_ingestion.logger import setup_logger

# Import from chunking_and_embedding
try:
    from chunking_and_embedding.config import ChunkingEmbeddingConfig
    from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
    from chunking_and_embedding.embedder import create_embedder
    from chunking_and_embedding.vector_store import init_pinecone_index, store_in_pinecone
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chunking_and_embedding.config import ChunkingEmbeddingConfig
    from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
    from chunking_and_embedding.embedder import create_embedder
    from chunking_and_embedding.vector_store import init_pinecone_index, store_in_pinecone


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
            "Run `python -m combined_pipeline.main` interactively or launch the UI instead."
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


def process_single_pdf(pdf_path: Path, config: PDFIngestionConfig) -> dict:
    """Process a single PDF through the entire PDF ingestion pipeline"""
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


def run_pdf_ingestion_pipeline(user_id: str):
    """Run the PDF ingestion pipeline"""
    start_time = time.time()
    
    config = PDFIngestionConfig()
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
        return []
        
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
    log.info("PDF INGESTION PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"PDFs processed: {successful}/{len(pdf_files)}")
    log.info(f"Failed: {failed}")
    log.info(f"Total pages: {total_pages}")
    log.info(f"Total time: {total_time:.2f}s")
    log.info(f"Overall speed: {overall_speed:.2f} pages/sec")
    log.info("=" * 60)
    
    return results


def run_chunking_embedding_pipeline():
    """Run the chunking and embedding pipeline"""
    print("=" * 60)
    print("CHUNKING & EMBEDDING PIPELINE STARTING")
    print("=" * 60)

    config = ChunkingEmbeddingConfig()
    start_time = time.time()

    # Step 1: Load markdown files
    print("Loading processed markdown files...")
    docs = load_markdown_files(config.input_dir)
    if not docs:
        print(f"No markdown files found in {config.input_dir}")
        return
    print(f"Loaded {len(docs)} document(s)")

    # Step 2: Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(
        docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators
    )
    print(f"Created {len(chunks)} chunks")

    # Step 3: Create embedder
    print("Creating embedder...")
    embedder = create_embedder(config.google_api_key, model=config.embedding_model)

    # Step 4: Initialize vector DB index
    print("Initializing vector database...")
    init_pinecone_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region
    )

    # Step 5: Store chunks in vector DB
    print("Storing chunks in vector database...")
    store_in_pinecone(
        chunks=chunks,
        embedder=embedder,
        index_name=config.pinecone_index_name,
        api_key=config.pinecone_api_key
    )

    elapsed = time.time() - start_time
    print("=" * 60)
    print("CHUNKING & EMBEDDING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)


def run_combined_pipeline():
    """Run the entire combined pipeline: PDF Ingestion -> Chunking & Embedding"""
    print("=" * 60)
    print("COMBINED RAG PIPELINE STARTING")
    print("=" * 60)
    
    # Step 1: Authenticate user
    user_id = authenticate_user()
    
    # Step 2: Run PDF ingestion
    pdf_results = run_pdf_ingestion_pipeline(user_id)
    
    # Check if any PDFs were processed successfully
    successful_pdfs = [r for r in pdf_results if r.get("status") == "success"]
    if not successful_pdfs:
        print("No PDFs processed successfully, skipping chunking and embedding.")
        return
    
    # Step 3: Run chunking and embedding
    run_chunking_embedding_pipeline()
    
    print("=" * 60)
    print("COMBINED RAG PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_combined_pipeline()
