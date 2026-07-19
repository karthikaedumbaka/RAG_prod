import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 768

class MatryoshkaEmbeddings:
    """
    Wrapper to manually truncate and L2-normalize embeddings 
    for Nomic's Matryoshka Representation Learning (MRL).
    """
    def __init__(self, base_embedder, target_dim: int):
        self.base_embedder = base_embedder
        self.target_dim = target_dim

    def _process_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        processed = []
        for emb in embeddings:
            # 1. Truncate to the target dimension
            vec = np.array(emb[:self.target_dim], dtype=np.float32)
            # 2. L2 Normalize (crucial for maintaining cosine similarity)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            processed.append(vec.tolist())
        return processed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw_embeddings = self.base_embedder.embed_documents(texts)
        return self._process_embeddings(raw_embeddings)

    def embed_query(self, text: str) -> List[float]:
        raw_embedding = self.base_embedder.embed_query(text)
        return self._process_embeddings([raw_embedding])[0]

def create_embedder(
    model: str = DEFAULT_EMBEDDING_MODEL,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSION,
):
    """Create a local Hugging Face embeddings instance."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local embedding model '{model}' on {device.upper()} (Target Dim={output_dimensionality})...")
    
    model_kwargs = {
        "device": device,
        "trust_remote_code": True,
    }
    
    # Initialize the base embedder WITHOUT encode_kwargs
    base_embedder = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs
    )
    
    # If a smaller dimension is requested, wrap it to manually truncate & normalize
    if output_dimensionality < 768:
        print(f"  -> Applying Matryoshka truncation to {output_dimensionality} dimensions...")
        return MatryoshkaEmbeddings(base_embedder, output_dimensionality)
    
    return base_embedder

def embed_chunks(chunks: List[Document], embedder) -> List[Document]:
    """Embed chunks using the provided embedder (optional, vector stores usually do this)"""
    return chunks