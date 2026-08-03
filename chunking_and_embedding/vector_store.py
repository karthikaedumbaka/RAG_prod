import json
import re
import time
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from tenacity import (
    retry, retry_if_exception, stop_after_attempt,
    wait_exponential, before_sleep_log,
)
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

# Initialize the centralized logger
logger = setup_logger("vector_store")

def init_pinecone_index(api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1", dimension: int = 768):
    """
    Initializes or verifies a Pinecone index.
    
    Args:
        api_key: Pinecone API key.
        index_name: Name of the index.
        cloud: Cloud provider (aws/gcp/azure).
        region: Cloud region.
        dimension: Embedding dimension.
        
    Returns:
        The Pinecone Index object.
    """
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        logger.info(f"️ Creating new Pinecone index '{index_name}' (dim={dimension})...")
        pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))
    else:
        try:
            index_desc = pc.describe_index(index_name)
            existing_dimension = getattr(index_desc, "dimension", None)
            if existing_dimension is not None and int(existing_dimension) != int(dimension):
                raise RuntimeError(
                    f"Pinecone index '{index_name}' exists with dim={existing_dimension}, "
                    f"but model outputs dim={dimension}. Use a new index name or delete the existing one."
                )
        except RuntimeError:
            raise
        except Exception:
            pass
    return pc.Index(index_name)

def store_in_pinecone(chunks: List[Document], embedder, index_name: str, api_key: str):
    """Basic wrapper to store documents in Pinecone."""
    return PineconeVectorStore.from_documents(
        documents=chunks, embedding=embedder, index_name=index_name, pinecone_api_key=api_key
    )

def load_pinecone_vector_store(embedder, index_name: str, api_key: str):
    """Loads an existing Pinecone vector store."""
    return PineconeVectorStore(embedding=embedder, index_name=index_name, pinecone_api_key=api_key)

_RETRY_DELAY_RE = re.compile(r"""retryDelay['"]?\s*:\s*['"]?(\d+(?:\.\d+)?)s""")

def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, PineconeApiException) and getattr(exc, "status", None) == 429

def _extract_retry_delay(exc: BaseException, default: float = 30.0) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) + 2.0 if match else default

def _load_checkpoint(checkpoint_path: Path) -> set:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_batches", []))
    return set()

def _save_checkpoint(checkpoint_path: Path, completed_batches: set):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"completed_batches": sorted(completed_batches)}, f)

def store_in_pinecone_resumable(
    chunks: List[Document], embedder, index_name: str, api_key: str,
    batch_size: int = 100, delay_between_batches: float = 0.5, checkpoint_path: Optional[str] = None,
) -> "PineconeVectorStore":
    """
    Stores chunks in Pinecone with resumability, rate-limit handling, and throughput tracking.
    
    Args:
        chunks: List of Document objects.
        embedder: Embedding model instance.
        index_name: Pinecone index name.
        api_key: Pinecone API key.
        batch_size: Number of chunks per upsert batch.
        delay_between_batches: Base delay between batches.
        checkpoint_path: Path to save resume state.
        
    Returns:
        The initialized PineconeVectorStore.
    """
    logger.info(f" Starting resumable Pinecone upsert for {len(chunks)} chunks...")
    start_time = time.time()
    
    vector_store = PineconeVectorStore(embedding=embedder, index_name=index_name, pinecone_api_key=api_key)
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    total = len(batches)
    completed = _load_checkpoint(Path(checkpoint_path)) if checkpoint_path else set()
    
    if completed:
        logger.info(f" Resuming from checkpoint: {len(completed)}/{total} batches already stored")

    @retry(
        retry=retry_if_exception(_is_rate_limit_error), stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=5, max=120), before_sleep=before_sleep_log(logger, logging.WARNING), reraise=True,
    )
    def _add_batch_with_retry(batch, state):
        try:
            vector_store.add_documents(batch)
        except PineconeApiException as e:
            if _is_rate_limit_error(e):
                wait_s = _extract_retry_delay(e)
                logger.warning(f"⏳ Rate limited (429, Pinecone). Waiting {wait_s:.0f}s before retry...")
                time.sleep(wait_s)
                state["delay"] = min(state["delay"] * 1.6, 90.0)
                state["hit_limit"] = True
            raise

    pacing_state = {"delay": delay_between_batches, "hit_limit": False}
    
    for i, batch in enumerate(batches):
        if i in completed:
            continue
            
        logger.info(f" Batch {i + 1}/{total} ({len(batch)} chunks)...")
        _add_batch_with_retry(batch, pacing_state)
        completed.add(i)
        
        if checkpoint_path:
            _save_checkpoint(Path(checkpoint_path), completed)
            
        if i < total - 1:
            if pacing_state["hit_limit"]:
                logger.info(f" Pacing up: waiting {pacing_state['delay']:.0f}s before next batch (adaptive backoff)...")
            time.sleep(pacing_state["delay"])

    if checkpoint_path:
        Path(checkpoint_path).unlink(missing_ok=True)
        
    elapsed = time.time() - start_time
    chunks_per_sec = len(chunks) / elapsed if elapsed > 0 else 0
    logger.info(f" Upsert complete in {elapsed:.2f}s ({chunks_per_sec:.1f} chunks/sec)")
    
    return vector_store