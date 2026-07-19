import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

@dataclass
class ChunkingEmbeddingConfig:
    # File paths
    input_dir: str = str(PROJECT_ROOT / "output")
    temp_dir: str = str(PROJECT_ROOT / "temp_chunks")
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = None
    
    # Embedding settings (100% open, no Hugging Face login required)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dimension: int = 768 
    
    # Vector database settings
    vector_db_provider: str = "pinecone"
    pinecone_api_key: str = None
    # Vector database settings
    pinecone_index_name: str = "rag-index-nomic-768" # Changed to avoid conflict with old 3072 
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # Rate-limiting / resumability 
    # Local models don't have API rate limits, but we keep batching for 
    # memory efficiency and to respect Pinecone's upsert limits.
    embedding_batch_size: int = 100
    embedding_batch_delay_seconds: float = 0.5  # Much faster locally

    def __post_init__(self):
        self.pinecone_api_key = self.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", self.pinecone_index_name)
        self.separators = self.separators or ["\n\n", "\n", " ", ""]
        Path(self.input_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    @property
    def embedding_checkpoint_path(self) -> str:
        return str(Path(self.temp_dir) / f"{self.pinecone_index_name}_checkpoint.json")