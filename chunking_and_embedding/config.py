
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ChunkingEmbeddingConfig:
    # File paths
    input_dir: str = str(PROJECT_ROOT / "output")  # Where processed PDFs are stored (markdown files)
    temp_dir: str = str(PROJECT_ROOT / "temp_chunks")

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = None

    # Embedding settings
    google_api_key: str = None
    embedding_model: str = "models/text-embedding-004"

    # Vector database settings
    vector_db_provider: str = "pinecone"  # or "chroma" (local)
    pinecone_api_key: str = None
    pinecone_index_name: str = "rag-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    def __post_init__(self):
        self.google_api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = self.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.separators = self.separators or ["\n\n", "\n", " ", ""]

        Path(self.input_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
