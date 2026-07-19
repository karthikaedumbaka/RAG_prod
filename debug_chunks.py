"""
Debug script to check if markdown files are being loaded and chunked correctly.
"""
from pathlib import Path
from chunking_and_embedding.config import ChunkingEmbeddingConfig
from chunking_and_embedding.chunker import load_markdown_files, chunk_documents

def main():
    config = ChunkingEmbeddingConfig()
    print(f"📂 Looking for markdown files in: {config.input_dir}")
    
    # 1. Load documents
    docs = load_markdown_files(config.input_dir)
    print(f"\n✅ Loaded {len(docs)} document(s) after cleaning.")
    
    if not docs:
        print("❌ ERROR: No documents were loaded! Check if the 'output' folder has .md files.")
        return

    # 2. Inspect the loaded documents
    for doc in docs:
        print(f"\n📄 File: {doc.metadata['source']}")
        print(f"📏 Cleaned Text Length: {len(doc.page_content)} characters")
        print(f"👀 Preview (first 300 chars):\n{doc.page_content[:300]}...\n")

    # 3. Chunk the documents
    print("✂️ Chunking documents...")
    chunks = chunk_documents(
        docs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators
    )
    
    print(f"\n🧩 Total Chunks Created: {len(chunks)}")
    
    if chunks:
        print(f"👀 Chunk 1 Preview:\n{chunks[0].page_content[:400]}")
        print(f"🏷️ Chunk 1 Metadata: {chunks[0].metadata}")
    else:
        print("❌ ERROR: 0 chunks were created! The text might be getting completely stripped by the cleaner.")

if __name__ == "__main__":
    main()