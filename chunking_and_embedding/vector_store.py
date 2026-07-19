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
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)

def init_pinecone_index(api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1", dimension: int = 768):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    else:
        try:
            index_desc = pc.describe_index(index_name)
            existing_dimension = getattr(index_desc, "dimension", None)
            if existing_dimension is not None and int(existing_dimension) != int(dimension):
                raise RuntimeError(
                    f"Pinecone index '{index_name}' already exists with dimension={existing_dimension}, "
                    f"but your embedding model outputs dimension={dimension}. "
                    "Use a new index name or delete/recreate the existing index."
                )
        except RuntimeError:
            raise
        except Exception:
            return pc.Index(index_name)
    return pc.Index(index_name)

def store_in_pinecone(chunks: List[Document], embedder, index_name: str, api_key: str):
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedder,
        index_name=index_name,
        pinecone_api_key=api_key
    )
    return vector_store

def load_pinecone_vector_store(embedder, index_name: str, api_key: str):
    return PineconeVectorStore(embedding=embedder, index_name=index_name, pinecone_api_key=api_key)

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")

def _is_rate_limit_error(exc: BaseException) -> bool:
    # Only check for Pinecone 429s now, as local embeddings don't have API quotas
    if isinstance(exc, PineconeApiException) and getattr(exc, "status", None) == 429:
        return True
    return False

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
    chunks: List[Document],
    embedder,
    index_name: str,
    api_key: str,
    batch_size: int = 100,
    delay_between_batches: float = 0.5,
    checkpoint_path: Optional[str] = None,
) -> "PineconeVectorStore":
    vector_store = PineconeVectorStore(embedding=embedder, index_name=index_name, pinecone_api_key=api_key)
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    total = len(batches)
    completed = _load_checkpoint(Path(checkpoint_path)) if checkpoint_path else set()
    
    if completed:
        print(f"Resuming from checkpoint: {len(completed)}/{total} batches already stored")

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=5, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _add_batch_with_retry(batch, state):
        try:
            vector_store.add_documents(batch)
        except PineconeApiException as e:
            if _is_rate_limit_error(e):
                wait_s = _extract_retry_delay(e)
                print(f"  Rate limited (429, Pinecone). Waiting {wait_s:.0f}s before retry...")
                time.sleep(wait_s)
                state["delay"] = min(state["delay"] * 1.6, 90.0)
                state["hit_limit"] = True
            raise

    pacing_state = {"delay": delay_between_batches, "hit_limit": False}
    for i, batch in enumerate(batches):
        if i in completed:
            continue
        print(f"  Batch {i + 1}/{total} ({len(batch)} chunks)...")
        _add_batch_with_retry(batch, pacing_state)
        completed.add(i)
        if checkpoint_path:
            _save_checkpoint(Path(checkpoint_path), completed)
        if i < total - 1:
            if pacing_state["hit_limit"]:
                print(f"  Pacing up: waiting {pacing_state['delay']:.0f}s before next batch (adaptive backoff)...")
            time.sleep(pacing_state["delay"])
            
    if checkpoint_path:
        Path(checkpoint_path).unlink(missing_ok=True)
    return vector_store