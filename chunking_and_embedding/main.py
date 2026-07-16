
import time
from pathlib import Path

try:
    from .config import ChunkingEmbeddingConfig
    from .chunker import load_markdown_files, chunk_documents
    from .embedder import create_embedder
    from .vector_store import init_pinecone_index, store_in_pinecone
except ImportError:
    from config import ChunkingEmbeddingConfig
    from chunker import load_markdown_files, chunk_documents
    from embedder import create_embedder
    from vector_store import init_pinecone_index, store_in_pinecone


def run_chunking_embedding_pipeline():
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
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    run_chunking_embedding_pipeline()
