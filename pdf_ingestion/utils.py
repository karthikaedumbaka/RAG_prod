import hashlib
from pathlib import Path

def get_file_hash(file_path: str) -> str:
    """
    Fast hash based on actual file content to detect real changes.
    Reads in chunks to avoid loading massive PDFs into RAM.
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read in 8KB chunks
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    except FileNotFoundError:
        return "file_not_found"

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def find_pdfs(directory: str) -> list[Path]:
    """Find all PDF files in directory"""
    return list(Path(directory).glob("*.pdf"))