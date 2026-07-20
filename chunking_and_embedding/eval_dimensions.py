import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Adjust import based on your project structure
try:
    from .embedder import create_embedder
except ImportError:
    from chunking_and_embedding.embedder import create_embedder

def calculate_hit(retrieved_docs, expected_source, expected_snippet):
    """
    Checks if the retrieved documents contain the expected source or snippet.
    Returns 1.0 (Hit) or 0.0 (Miss).
    """
    for doc in retrieved_docs:
        doc_source = doc.metadata.get('source', '')
        
        # Check Source Match
        if expected_source and expected_source.lower() in doc_source.lower():
            return 1.0
            
        # Check Snippet Match
        if expected_snippet and expected_snippet.lower() in doc.page_content.lower():
            return 1.0
            
    return 0.0

def evaluate_dimensions_recall_mmr(config, chunks, questions, dimensions_to_test, k=5):
    """
    Tests multiple dimensions using Recall@K and MMR, returning the best dimension.
    """
    pc = Pinecone(api_key=config.pinecone_api_key)
    results = []

    for dim in dimensions_to_test:
        print(f"\n--- Testing Dimension: {dim} ---")
        index_name = f"eval-temp-{dim}"
        
        # 1. Create temporary index
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            time.sleep(5)
            
        print(f"  🏗️ Creating temporary Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name, 
            dimension=dim, 
            metric="cosine", 
            spec=ServerlessSpec(cloud=config.pinecone_cloud, region=config.pinecone_region)
        )
        time.sleep(10) # Wait for Pinecone index initialization

        try:
            # 2. Initialize Embedder and Vector Store
            embedder = create_embedder(
                model=config.embedding_model, 
                output_dimensionality=dim
            )
            vector_store = PineconeVectorStore(
                embedding=embedder, 
                index_name=index_name, 
                pinecone_api_key=config.pinecone_api_key
            )

            # 3. Upsert chunks for evaluation
            print(f"  📤 Upserting {len(chunks)} chunks for evaluation...")
            vector_store.add_documents(chunks)
            time.sleep(2) # Allow Pinecone to finish indexing

            # 4. Evaluate Recall and MMR
            recall_scores = []
            mmr_scores = []

            for q_data in questions:
                query = q_data['question']
                exp_source = q_data.get('expected_source', '')
                exp_snippet = q_data.get('expected_content_snippet', '')

                # A) Standard Similarity Search (Calculates Recall@K)
                docs_sim = vector_store.similarity_search(query, k=k)
                recall_scores.append(calculate_hit(docs_sim, exp_source, exp_snippet))

                # B) Maximal Marginal Relevance Search (Calculates MMR Hit Rate)
                # fetch_k=20 retrieves more candidates to ensure diversity before picking top k
                docs_mmr = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=20)
                mmr_scores.append(calculate_hit(docs_mmr, exp_source, exp_snippet))

            avg_recall = np.mean(recall_scores)
            avg_mmr = np.mean(mmr_scores)
            
            # Final Score: Weighted average (60% weight on pure Recall, 40% on MMR diversity)
            final_score = (0.6 * avg_recall) + (0.4 * avg_mmr)

            results.append({
                "dimension": dim,
                "avg_recall": avg_recall,
                "avg_mmr": avg_mmr,
                "final_score": final_score
            })
            print(f"  ✅ Dim {dim} -> Recall: {avg_recall:.3f}, MMR: {avg_mmr:.3f}, Final Score: {final_score:.3f}")

        finally:
            # 5. Cleanup temporary index to save costs/space
            print(f"  🧹 Cleaning up temporary index '{index_name}'...")
            pc.delete_index(index_name)

    # Find and return the dimension with the highest final score
    best_result = max(results, key=lambda x: x['final_score'])
    return best_result