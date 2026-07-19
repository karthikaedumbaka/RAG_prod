"""
Chunking and Embedding Module
Connects: config -> chunker -> embedder -> vector_store -> main
"""
from .config import ChunkingEmbeddingConfig
from .chunker import load_markdown_files, chunk_documents
from .embedder import (
    create_embedder,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSION,
)
from .vector_store import (
    init_pinecone_index,
    store_in_pinecone,
    store_in_pinecone_resumable,
    load_pinecone_vector_store,
)
from .main import run_chunking_embedding_pipeline

__all__ = [
    "ChunkingEmbeddingConfig",
    "load_markdown_files",
    "chunk_documents",
    "create_embedder",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSION",
    "init_pinecone_index",
    "store_in_pinecone",
    "store_in_pinecone_resumable",
    "load_pinecone_vector_store",
    "run_chunking_embedding_pipeline",
]