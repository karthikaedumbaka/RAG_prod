import json
import time
import numpy as np
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from .embedder import create_embedder
from .logger import setup_logger

log = setup_logger("eval_dimensions")

DIMENSIONS_TO_TEST = [256, 512, 768]
EVAL_QUESTIONS_PATH = Path(__file__).parent / "EvalQuestions.json"
OPTIMAL_DIM_CACHE = Path(__file__).parent.parent / "optimal_dimension.json"

def _load_eval_questions():
    """Loads the ground-truth Q&A dataset for evaluation."""
    if not EVAL_QUESTIONS_PATH.exists():
        log.warning(f"⚠️ EvalQuestions.json not found at {EVAL_QUESTIONS_PATH}.")
        return None
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _evaluate_single_dimension(dim: int, chunks, config, questions, pc: Pinecone) -> dict:
    """Tests a single dimension by creating a temp index, evaluating, and guaranteeing cleanup."""
    # Unique index name to prevent collisions
    index_name = f"rag-eval-{dim}-{int(time.time())}"
    
    try:
        log.info(f"  🏗️ Creating temp Pinecone index '{index_name}' (dim={dim})...")
        pc.create_index(
            name=index_name, dimension=dim, metric="cosine", 
            spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region)
        )
        log.info(f"  ⏳ Waiting 15s for index initialization...")
        time.sleep(15) 
        
        # Initialize embedder and vector store for this dimension
        embedder = create_embedder(model=config.embedding_model, output_dimensionality=dim)
        vs = PineconeVectorStore(
            embedding=embedder, index_name=index_name, pinecone_api_key=config.pinecone_api_key
        )
        
        # Upsert chunks in batches
        log.info(f"  📤 Upserting {len(chunks)} chunks for evaluation...")
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            vs.add_documents(chunks[i:i+batch_size])
        time.sleep(2) # Allow Pinecone to finish indexing
        
        # Evaluate Recall@5 and MRR
        hits_at_5 = 0
        reciprocal_ranks = []
        
        for item in questions:
            results = vs.similarity_search(item["question"], k=5)
            rank_of_hit = None
            
            for rank, doc in enumerate(results, start=1):
                # Check if the expected snippet is in the retrieved chunk
                snippet = item.get("expected_content_snippet", "")
                if snippet and snippet.lower() in doc.page_content.lower():
                    rank_of_hit = rank
                    break
            
            if rank_of_hit and rank_of_hit <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank_of_hit if rank_of_hit else 0.0)
            
        recall_5 = hits_at_5 / len(questions)
        mrr = sum(reciprocal_ranks) / len(questions)
        
        # Composite score (heavily weight Recall, use MRR as tiebreaker)
        score = recall_5 + (mrr * 0.1) 
        
        return {"dimension": dim, "recall@5": recall_5, "mrr": mrr, "score": score}
        
    finally:
        # 🧹 GUARANTEED CLEANUP (Crucial to avoid Pinecone index limits/costs)
        if index_name in pc.list_indexes().names():
            log.info(f"  🧹 Cleaning up temp index '{index_name}'...")
            pc.delete_index(index_name)

def find_optimal_dimension(config, chunks) -> int:
    """
    Main entry point. Checks cache first. If no cache, runs the evaluation 
    across all dimensions and returns the mathematically best one.
    """
    # 1. CHECK CACHE (Saves ~4 minutes on subsequent runs)
    if OPTIMAL_DIM_CACHE.exists():
        try:
            with open(OPTIMAL_DIM_CACHE, "r") as f:
                cached = json.load(f)
                log.info(f"✅ Using cached optimal dimension: {cached['dimension']} (Score: {cached['score']:.3f})")
                return cached['dimension']
        except Exception:
            pass # Cache corrupted, re-evaluate

    # 2. LOAD QUESTIONS
    questions = _load_eval_questions()
    if not questions:
        log.warning("⚠️ No evaluation questions found. Falling back to default dimension (768).")
        return 768

    log.info("=" * 60)
    log.info("🔬 EVALUATING EMBEDDING DIMENSIONS (This may take 3-4 minutes...)")
    log.info("=" * 60)
    
    pc = Pinecone(api_key=config.pinecone_api_key)
    best_dim = 768
    best_score = -1.0
    
    # 3. TEST EACH DIMENSION
    for dim in DIMENSIONS_TO_TEST:
        try:
            metrics = _evaluate_single_dimension(dim, chunks, config, questions, pc)
            log.info(
                f"  📊 Dim {dim} -> Recall@5: {metrics['recall@5']:.2%} | "
                f"MRR: {metrics['mrr']:.3f} | Final Score: {metrics['score']:.3f}"
            )
            if metrics['score'] > best_score:
                best_score = metrics['score']
                best_dim = dim
        except Exception as e:
            log.exception(f"❌ Error evaluating dimension {dim}: {e}")
            
    log.info(f"🏆 BEST DIMENSION SELECTED: {best_dim} (Score: {best_score:.3f})")
    log.info("=" * 60)
    
    # 4. SAVE TO CACHE
    with open(OPTIMAL_DIM_CACHE, "w") as f:
        json.dump({"dimension": best_dim, "score": best_score}, f)
        
    return best_dim