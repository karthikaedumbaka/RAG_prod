import time
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

log = setup_logger("embedder")

DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 768

class MatryoshkaEmbeddings:
    """
    Wrapper to manually truncate and L2-normalize embeddings for Nomic's MRL.
    This allows using smaller dimensions (e.g., 256, 512) to save Pinecone costs.
    """
    def __init__(self, base_embedder, target_dim: int):
        self.base_embedder = base_embedder
        self.target_dim = target_dim

    def _process_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Truncates and L2-normalizes the embeddings."""
        processed = []
        for emb in embeddings:
            vec = np.array(emb[:self.target_dim], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            processed.append(vec.tolist())
        return processed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents."""
        raw_embeddings = self.base_embedder.embed_documents(texts)
        return self._process_embeddings(raw_embeddings)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        raw_embedding = self.base_embedder.embed_query(text)
        return self._process_embeddings([raw_embedding])[0]

def create_embedder(
    model: str = DEFAULT_EMBEDDING_MODEL,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSION,
):
    """
    Creates a local Hugging Face embeddings instance, optionally applying 
    Matryoshka truncation if the target dimension is less than 768.
    
    Args:
        model: HuggingFace model name.
        output_dimensionality: Target embedding dimension.
        
    Returns:
        An embedder instance (either base or Matryoshka wrapped).
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log.info(f" Loading local embedding model '{model}' on {device.upper()} (Target Dim={output_dimensionality})...")
    start_time = time.time()
    
    model_kwargs = {
        "device": device,
        "trust_remote_code": True,
    }
    
    base_embedder = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs
    )
    
    elapsed = time.time() - start_time
    log.info(f" Model loaded in {elapsed:.2f}s")
    
    if output_dimensionality < 768:
        log.info(f"   -> Applying Matryoshka truncation to {output_dimensionality} dimensions...")
        return MatryoshkaEmbeddings(base_embedder, output_dimensionality)
        
    return base_embedder

def embed_chunks(chunks: List[Document], embedder) -> List[Document]:
    """
    Placeholder for embedding chunks directly if needed. 
    (Usually handled by PineconeVectorStore directly).
    """
    return chunks