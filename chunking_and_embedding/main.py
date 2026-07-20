import sys
import time
from pathlib import Path
import json

# FIX: Inject project root into sys.path so direct execution works
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try relative imports first (for -m execution), fallback to absolute (for direct execution)
try:
    from .config import ChunkingEmbeddingConfig
    from .chunker import load_markdown_files, chunk_documents
    from .embedder import create_embedder
    from .vector_store import init_pinecone_index, store_in_pinecone_resumable
    from .generate_eval_questions import generate_questions_with_kimi
    from .eval_dimensions import evaluate_dimensions_recall_mmr  # <-- Updated import
except ImportError:
    from chunking_and_embedding.config import ChunkingEmbeddingConfig
    from chunking_and_embedding.chunker import load_markdown_files, chunk_documents
    from chunking_and_embedding.embedder import create_embedder
    from chunking_and_embedding.vector_store import init_pinecone_index, store_in_pinecone_resumable
    from chunking_and_embedding.generate_eval_questions import generate_questions_with_kimi
    from chunking_and_embedding.eval_dimensions import evaluate_dimensions_recall_mmr  # <-- Updated import

def run_chunking_embedding_pipeline():
    print("=" * 60)
    print("🚀 CHUNKING & EMBEDDING PIPELINE STARTING")
    print("=" * 60)
    config = ChunkingEmbeddingConfig()
    start_time = time.time()

    # ---------------------------------------------------------
    # STEP 1: Handle Evaluation Questions (Strict Interactive Prompt)
    # ---------------------------------------------------------
    eval_path = Path(__file__).parent / "EvalQuestions.json"
    
    try:
        if eval_path.exists():
            print(f"\n✅ Found existing {eval_path.name}")
            overwrite = input("🔄 Do you want to overwrite and regenerate it using the LLM API? [y/N]: ").strip().lower()
            
            if overwrite == 'y':
                print("🚀 Triggering API to regenerate benchmark questions...")
                generate_questions_with_kimi(config.input_dir, eval_path)
            else:
                print("🛑 User selected 'N'. Using existing EvalQuestions.json only. Skipping API generation.")
        else:
            print(f"\n⚠️ {eval_path.name} not found locally.")
            generate = input("🔍 Do you want to generate it now using the LLM API? [Y/n]: ").strip().lower()
            
            if generate != 'n':
                print("🚀 Triggering API to generate benchmark questions...")
                generate_questions_with_kimi(config.input_dir, eval_path)
            else:
                raise RuntimeError(f"❌ Cannot proceed without {eval_path.name}. Please create it manually or allow the script to generate it.")
    except EOFError:
        raise RuntimeError("CLI requires an interactive terminal to answer prompts. Run directly via `uv run ...` instead of piping input.")

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        raise ValueError("❌ EvalQuestions.json is empty or contains no valid questions.")
    print(f"📝 Loaded {len(questions)} benchmark questions for evaluation.")

    # ---------------------------------------------------------
    # STEP 2: Load Data & Initial Chunking (For Evaluation)
    # ---------------------------------------------------------
    print("\n📂 Loading processed markdown files...")
    docs = load_markdown_files(config.input_dir)
    if not docs:
        raise FileNotFoundError(f"❌ No markdown files found in {config.input_dir}")
    print(f"   Loaded {len(docs)} document(s)")

    print("🔪 Chunking documents for evaluation...")
    chunks = chunk_documents(
        docs, 
        chunk_size=config.chunk_size, 
        chunk_overlap=config.chunk_overlap, 
        separators=config.separators
    )
    print(f"   Created {len(chunks)} chunks")

    # ---------------------------------------------------------
    # STEP 3: Find Best Dimension using Recall & MMR
    # ---------------------------------------------------------
    print("\n📏 Evaluating different vector dimensions using Recall@K and MMR...")
    
    # Define the dimensions you want to test
    dimensions_to_test = [256, 384, 512, 768] 
    
    best_result = evaluate_dimensions_recall_mmr(
        config=config,
        chunks=chunks,
        questions=questions,
        dimensions_to_test=dimensions_to_test
    )
    
    best_dim = best_result['dimension']
    
    # Dynamically update config in memory
    config.embedding_dimension = best_dim
    config.pinecone_index_name = f"rag-index-nomic-{best_dim}"
    
    print(f"\n🏆 Best Dimension Selected: {best_dim}")
    print(f"   -> Avg Recall: {best_result['avg_recall']:.3f}")
    print(f"   -> Avg MMR:    {best_result['avg_mmr']:.3f}")
    print(f"   -> Final Score:{best_result['final_score']:.3f}")
    print(f"   -> Updated Index Name: {config.pinecone_index_name}")

    # ---------------------------------------------------------
    # STEP 4: Final Embedding & Vector DB Upsert
    # ---------------------------------------------------------
    print(f"\n🧠 Creating final local embedder (model={config.embedding_model}, dimension={config.embedding_dimension})...")
    embedder = create_embedder(
        model=config.embedding_model,
        output_dimensionality=config.embedding_dimension,
    )

    print("🌲 Initializing final Pinecone vector database...")
    init_pinecone_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
        dimension=config.embedding_dimension
    )

    print("💾 Storing chunks in vector database...")
    store_in_pinecone_resumable(
        chunks=chunks,
        embedder=embedder,
        index_name=config.pinecone_index_name,
        api_key=config.pinecone_api_key,
        batch_size=config.embedding_batch_size,
        delay_between_batches=config.embedding_batch_delay_seconds,
        checkpoint_path=config.embedding_checkpoint_path,
    )

    # ---------------------------------------------------------
    # COMPLETION
    # ---------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ CHUNKING & EMBEDDING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Final Index: {config.pinecone_index_name} (Dim: {config.embedding_dimension})")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_chunking_embedding_pipeline()