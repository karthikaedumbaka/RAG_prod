"""
Evaluates embedding dimensions against a benchmark dataset to find the optimal dimension.
"""
import time
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from .embedder import create_embedder

# Matryoshka dimensions supported by nomic-ai/nomic-embed-text-v1.5
DIMENSIONS_TO_TEST = [256, 512, 768]

def evaluate_dimension(dimension: int, chunks, config, questions) -> dict:
    """
    Tests a specific dimension by creating a temp Pinecone index, 
    evaluating Recall/MRR, and guaranteeing cleanup.
    """
    index_name = f"rag-eval-{dimension}"
    pc = Pinecone(api_key=config.pinecone_api_key)
    
    try:
        # 1. Create Temporary Index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name, 
                dimension=dimension, 
                metric="cosine", 
                spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region)
            )
            print(f"    Waiting for index '{index_name}' to initialize...")
            time.sleep(15) # Pinecone needs a moment to become searchable
            
        # 2. Embed and Store
        embedder = create_embedder(model=config.embedding_model, output_dimensionality=dimension)
        vector_store = PineconeVectorStore(embedding=embedder, index_name=index_name, pinecone_api_key=config.pinecone_api_key)
        
        # Batch upload to prevent timeouts
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            vector_store.add_documents(chunks[i:i+batch_size])
            
        # 3. Evaluate Metrics
        hits_at = {1: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        
        for item in questions:
            results = vector_store.similarity_search(item["question"], k=10)
            rank_of_first_hit = None
            
            for rank, doc in enumerate(results, start=1):
                source_ok = doc.metadata.get("source") == item["expected_source"]
                snippet_ok = not item.get("expected_content_snippet") or \
                             item["expected_content_snippet"].lower() in doc.page_content.lower()
                             
                if source_ok and snippet_ok:
                    rank_of_first_hit = rank
                    break
                    
            for k in hits_at:
                if rank_of_first_hit and rank_of_first_hit <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / rank_of_first_hit if rank_of_first_hit else 0.0)
            
        n = len(questions)
        return {
            "dimension": dimension,
            "recall@5": hits_at[5] / n,
            "mrr": sum(reciprocal_ranks) / n
        }
        
    finally:
        # 4. GUARANTEED CLEANUP (Crucial for Prod)
        if index_name in pc.list_indexes().names():
            print(f"   🧹 Cleaning up temporary index '{index_name}'...")
            pc.delete_index(index_name)

def find_best_dimension(config, chunks, questions) -> int:
    """Runs evaluation across all dimensions and returns the best one."""
    print("\n" + "="*60)
    print(" EVALUATING EMBEDDING DIMENSIONS FOR YOUR DATA")
    print("="*60)
    
    best_dim = config.embedding_dimension
    best_score = -1.0
    
    for dim in DIMENSIONS_TO_TEST:
        print(f"\n Testing Dimension: {dim}")
        try:
            metrics = evaluate_dimension(dim, chunks, config, questions)
            # Composite score: heavily weight Recall@5, with MRR as tiebreaker
            score = metrics["recall@5"] + (metrics["mrr"] * 0.1) 
            
            print(f"    Recall@5: {metrics['recall@5']:.2%} | MRR: {metrics['mrr']:.3f}")
            
            if score > best_score:
                best_score = score
                best_dim = dim
        except Exception as e:
            print(f"    Error evaluating dimension {dim}: {e}")
            
    print(f"\n BEST DIMENSION SELECTED: {best_dim}")
    return best_dim