import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Ensure project root is in sys.path for absolute imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
#  AUTO-SYNC: Import settings directly from the ingestion pipeline
# ==============================================================================
from chunking_and_embedding.config import ChunkingEmbeddingConfig
_ingestion_config = ChunkingEmbeddingConfig()

# Check if the auto-evaluator saved an optimal dimension during ingestion
OPTIMAL_DIM_CACHE = PROJECT_ROOT / "optimal_dimension.json"
if OPTIMAL_DIM_CACHE.exists():
    try:
        with open(OPTIMAL_DIM_CACHE, "r") as f:
            cached_data = json.load(f)
            OPTIMAL_DIM = cached_data.get("dimension", _ingestion_config.embedding_dimension)
    except Exception:
        OPTIMAL_DIM = _ingestion_config.embedding_dimension
else:
    OPTIMAL_DIM = _ingestion_config.embedding_dimension

# ==============================================================================
# QUERY CONFIGURATION
# ==============================================================================
class QueryConfig:
    #  Pinecone Settings (Automatically synced with the ingestion pipeline!)
    PINECONE_API_KEY: str = _ingestion_config.pinecone_api_key
    PINECONE_INDEX_NAME: str = _ingestion_config.pinecone_index_name
    
    #  LLM Settings (Using Groq for ultra-fast, free inference)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama-3.3-70b-versatile" 
    
    # Retrieval Settings
    RETRIEVAL_K: int = 8          # Increased from 5 to 8
    USE_MMR: bool = True          
    MMR_FETCH_K: int = 30         # Increased from 20 to 30 (fetch more candidates to filter for diversity)
    
    #  Embedding Settings (Synced with ingestion + optimal dimension cache)
    EMBEDDING_MODEL: str = _ingestion_config.embedding_model
    EMBEDDING_DIMENSION: int = OPTIMAL_DIM
    
    #  Local paths for BM25 Hybrid Search
    INPUT_DIR: str = _ingestion_config.input_dir
    CHUNK_SIZE: int = _ingestion_config.chunk_size
    CHUNK_OVERLAP: int = _ingestion_config.chunk_overlap