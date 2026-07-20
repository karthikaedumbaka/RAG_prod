import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, user_id: str = "unknown") -> logging.Logger:
    """
    Process-safe logger that routes all logs to a single file per user_id.
    Uses RotatingFileHandler to prevent infinite log growth.
    """
    logger_name = f"{name}.{user_id}"
    logger = logging.getLogger(logger_name)
    
    # Prevent adding duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False  
        
        # 1. Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 2. File handler (ONE FILE PER USER, ROTATING)
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"user_{user_id}.log"
        
        # Use RotatingFileHandler (10MB max, 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,              # Keep 5 historical files
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