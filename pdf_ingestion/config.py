# -----------------------------------------------------
'''
for thsi config.py we are doing the input/output path checking or creation
OCR Batch Settings
Page Analysis like TEXT_THRESHOLD_RATIO etc and \
logs

'''
# -----------------------------------------------------
from pathlib import Path
import os

# -----------------------------------------------------
# Project Directories
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "output"

TEMP_DIR = BASE_DIR / "temp"

LOG_DIR = BASE_DIR / "logs"

# -----------------------------------------------------
# Auto create directories
# -----------------------------------------------------

for directory in [
    OUTPUT_DIR,
    TEMP_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------
# Input PDF
# -----------------------------------------------------

PDF_FILES = list(DATA_DIR.glob("*.pdf"))

if not PDF_FILES:
    raise FileNotFoundError(
        f"No PDF found inside {DATA_DIR}"
    )

PDF_PATH = PDF_FILES[0]

# -----------------------------------------------------
# OCR Batch Settings
# -----------------------------------------------------

OCR_BATCH_SIZE = 20

# Memory safe

MAX_WORKERS = min(4, os.cpu_count() or 2)

# -----------------------------------------------------
# Page Analysis
# -----------------------------------------------------

TEXT_THRESHOLD_RATIO = 0.70

MAX_DRAWINGS = 25

MAX_IMAGE_PERCENTAGE = 0.30

# -----------------------------------------------------
# Output
# -----------------------------------------------------

MARKDOWN_DIR = OUTPUT_DIR / "markdown"

MARKDOWN_DIR.mkdir(exist_ok=True)

FINAL_MARKDOWN = OUTPUT_DIR / "final_document.md"

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

PAGE_REPORT = OUTPUT_DIR / "page_report.json"

# -----------------------------------------------------
# Logging
# -----------------------------------------------------

LOG_FILE = LOG_DIR / "ingestion.log"

# -----------------------------------------------------
# Retry
# -----------------------------------------------------

MAX_RETRY = 2

# -----------------------------------------------------
# Temporary Batch PDFs
# -----------------------------------------------------

BATCH_DIR = TEMP_DIR / "batches"

BATCH_DIR.mkdir(exist_ok=True)