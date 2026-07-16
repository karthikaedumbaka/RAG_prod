
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
from chunking_and_embedding.config import ChunkingEmbeddingConfig

def test_chunking():
    log_file = Path("test_chunking.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Testing chunking module...\n")
        config = ChunkingEmbeddingConfig()
        f.write(f"Loading markdown files from {config.input_dir}\n")
        docs = load_markdown_files(config.input_dir)
        f.write(f"Loaded {len(docs)} documents\n")
        
        if docs:
            f.write(f"First document metadata: {docs[0].metadata}\n")
            f.write(f"First document content preview: {docs[0].page_content[:200]}\n")
        
        f.write("Chunking documents...\n")
        chunks = chunk_documents(
            docs,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
        f.write(f"Created {len(chunks)} chunks\n")
        
        if chunks:
            f.write(f"First chunk metadata: {chunks[0].metadata}\n")
            f.write(f"First chunk content preview: {chunks[0].page_content[:200]}\n")
        f.write("Chunking test complete!\n")
    print(f"Test results written to {log_file}")

if __name__ == "__main__":
    test_chunking()
