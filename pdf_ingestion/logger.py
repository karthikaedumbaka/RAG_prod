import logging
import sys
from pathlib import Path

def setup_logger(name: str, user_id: str = "unknown") -> logging.Logger:
    """
    Process-safe logger that routes all logs to a single file per user_id.
    Uses append mode to allow multiple concurrent workers to write safely.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding duplicate handlers if this function is called multiple times
    # for the same logger name (e.g., in a loop or across imports).
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent logs from bubbling up to the root logger
        
        # 1. Console handler (Standard output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 2. File handler (ONE FILE PER USER)
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # The filename is strictly based on the user_id
        log_file = log_dir / f"user_{user_id}.log"
        
        file_handler = logging.FileHandler(
            log_file,
            mode='a',  # Append mode: crucial for multiple workers writing to the same file
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        formatter = logging.Formatter(
            f'%(asctime)s | [USER: {user_id}] | %(name)-25s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger