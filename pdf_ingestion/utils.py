import hashlib
from pathlib import Path

def get_file_hash(file_path: str) -> str:
    """Fast hash for checkpointing"""
    return hashlib.md5(file_path.encode()).hexdigest()[:8]

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def find_pdfs(directory: str) -> list[Path]:
    """Find all PDF files in directory"""
    return list(Path(directory).glob("*.pdf"))