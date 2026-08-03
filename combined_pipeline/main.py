import time
import uuid
import getpass
import sys
from pathlib import Path

# ==============================================================================
# 1. IMPORTS (With fallback for running as script vs module)
# ==============================================================================
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
    from chunking_and_embedding.eval_dimensions import find_optimal_dimension
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chunking_and_embedding.config import ChunkingEmbeddingConfig
    from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
    from chunking_and_embedding.embedder import create_embedder
    from chunking_and_embedding.vector_store import init_pinecone_index, store_in_pinecone
    from chunking_and_embedding.eval_dimensions import find_optimal_dimension

#  Import the Query Pipeline (Chat Interface)
try:
    from query_pipeline.main import run_chat_interface
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from query_pipeline.main import run_chat_interface

#  NEW: Import the Evaluation Script
try:
    from evaluation.evaluate_rag import main as run_evaluation
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.evaluate_rag import main as run_evaluation

# ==============================================================================
# 2. AUTHENTICATION
# ==============================================================================
def get_masked_password(prompt="Enter Password: "):
    """Cross-platform masked password input."""
    return getpass.getpass(prompt)

def authenticate_user() -> str:
    """Dynamic CLI Authentication."""
    print("=" * 50)
    print(" USER AUTHENTICATION")
    print("=" * 50)
    try:
        user_id = input("Enter User ID: ").strip()
        password = get_masked_password("Enter Password: ")
    except EOFError as exc:
        raise RuntimeError("CLI authentication requires an interactive terminal.") from exc

    if not user_id or not password:
        raise RuntimeError("User ID and password are required.")

    auth_status = authenticate_or_register(user_id, password)
    if auth_status == "authenticated":
        print(f" Authentication successful. Welcome back, {user_id}!")
        return user_id
    elif auth_status == "registered":
        print(f" New user detected. Account created successfully. Welcome, {user_id}!")
        return user_id
    else:
        print(f" Authentication failed for user '{user_id}'. Incorrect password.")
        unique_id = f"guest_{uuid.uuid4().hex[:8]}"
        print(f" Assigned unique guest ID: {unique_id}")
        return unique_id

# ==============================================================================
# 3. PDF INGESTION PIPELINE
# ==============================================================================
def process_single_pdf(pdf_path: Path, config: PDFIngestionConfig) -> dict:
    """Process a single PDF through the entire PDF ingestion pipeline."""
    log = setup_logger("main", config.user_id)
    pdf_name = pdf_path.stem
    log.info("=" * 60)
    log.info(f" Processing: {pdf_path.name}")
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
        log.info(f" Completed {pdf_path.name}")
        log.info(f"️ Time: {elapsed:.2f}s |  Speed: {pages_per_sec:.2f} pages/sec")
        log.info(f" Output: {final_md}")
        return {"pdf": str(pdf_path), "pages": analysis["total_pages"], "time": elapsed, "speed": pages_per_sec, "output": str(final_md), "status": "success"}
    except Exception as e:
        log.exception(f" Critical failure processing {pdf_path.name}: {e}")
        return {"pdf": str(pdf_path), "status": "failed", "error": str(e)}

def run_pdf_ingestion_pipeline(user_id: str):
    """Run the PDF ingestion pipeline for all PDFs in the data directory."""
    start_time = time.time()
    config = PDFIngestionConfig()
    config.user_id = user_id
    
    ensure_dir(config.output_dir)
    ensure_dir(config.temp_dir)
    
    log = setup_logger("main", config.user_id)
    log.info("=" * 60)
    log.info(" PDF INGESTION PIPELINE STARTING")
    log.info("=" * 60)
    
    pdf_files = find_pdfs(config.data_dir)
    if not pdf_files:
        log.error(f" No PDFs found in {config.data_dir}")
        return []
        
    log.info(f" Found {len(pdf_files)} PDF(s) to process")
    results = []
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path, config)
        results.append(result)
        
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get("status") == "success")
    log.info("=" * 60)
    log.info(" PDF INGESTION PIPELINE COMPLETE")
    log.info(f" PDFs processed: {successful}/{len(pdf_files)} | ️ Total time: {total_time:.2f}s")
    log.info("=" * 60)
    return results

# ==============================================================================
# 4. CHUNKING & EMBEDDING PIPELINE (WITH AUTO DIMENSION EVAL)
# ==============================================================================
def run_chunking_embedding_pipeline(user_id: str = "unknown"):
    """Run the chunking and embedding pipeline with automatic dimension tuning."""
    log = setup_logger("combined_pipeline", user_id)
    log.info("=" * 60)
    log.info(" CHUNKING & EMBEDDING PIPELINE STARTING")
    log.info("=" * 60)
    config = ChunkingEmbeddingConfig()
    start_time = time.time()

    docs = load_markdown_files(config.input_dir)
    if not docs:
        log.error(f" No markdown files found in {config.input_dir}")
        return
    log.info(f" Loaded {len(docs)} document(s)")

    chunks = chunk_documents(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separators=config.separators)
    log.info(f" Created {len(chunks)} chunks")

    if getattr(config, 'auto_evaluate_dimensions', True):
        log.info(" Determining optimal embedding dimension...")
        eval_chunks = chunks[:500] if len(chunks) > 500 else chunks
        optimal_dim = find_optimal_dimension(config, eval_chunks)
        config.embedding_dimension = optimal_dim
        log.info(f" Final Vector DB will use dimension: {optimal_dim}")

    log.info(f" Creating embedder (Target Dim={config.embedding_dimension})...")
    embedder = create_embedder(model=config.embedding_model, output_dimensionality=config.embedding_dimension)

    try:
        embedding_dimension = len(embedder.embed_query("dimension probe"))
    except Exception:
        embedding_dimension = config.embedding_dimension

    log.info(" Initializing final Pinecone vector database...")
    init_pinecone_index(
        api_key=config.pinecone_api_key, index_name=config.pinecone_index_name,
        cloud=config.pinecone_cloud, region=config.pinecone_region, dimension=embedding_dimension
    )

    log.info(" Storing chunks in final vector database...")
    store_in_pinecone(chunks=chunks, embedder=embedder, index_name=config.pinecone_index_name, api_key=config.pinecone_api_key)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info(" CHUNKING & EMBEDDING PIPELINE COMPLETE")
    log.info(f" Optimal Dimension Used: {embedding_dimension} | ️ Total time: {elapsed:.2f}s")
    log.info("=" * 60)

# ==============================================================================
# 5. MASTER ORCHESTRATOR
# ==============================================================================
def run_combined_pipeline():
    """Run the entire combined pipeline: Auth -> Ingestion -> Embedding -> Eval -> Chat."""
    print("=" * 60)
    print(" COMBINED RAG PIPELINE STARTING")
    print("=" * 60)

    # Step 1: Authenticate user
    user_id = authenticate_user()

    # Step 2: Run PDF ingestion
    pdf_results = run_pdf_ingestion_pipeline(user_id)
    successful_pdfs = [r for r in pdf_results if r.get("status") == "success"]
    if not successful_pdfs:
        print("️ No PDFs processed successfully, skipping downstream steps.")
        return

    # Step 3: Run chunking and embedding
    run_chunking_embedding_pipeline(user_id)

    print("\n" + "=" * 60)
    print(" DATA INGESTION & EMBEDDING COMPLETE")
    print("=" * 60)

    #  Step 4: Run End-to-End Evaluation
    print("\n" + "=" * 60)
    print(" RUNNING END-TO-END RAG EVALUATION...")
    print("=" * 60 + "\n")
    try:
        run_evaluation()
    except Exception as e:
        print(f"️ Evaluation failed or was skipped: {e}")

    # Step 5: Automatically launch the chat interface
    print("\n" + "=" * 60)
    print(" AUTOMATICALLY LAUNCHING CHAT INTERFACE...")
    print("=" * 60 + "\n")
    
    run_chat_interface()

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_combined_pipeline()