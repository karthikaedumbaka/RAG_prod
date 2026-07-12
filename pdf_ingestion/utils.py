"""
utils.py

Common utility functions used across the PDF ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import fitz

from logger import logger


# ==========================================================
# JSON
# ==========================================================

def read_json(path: Path, default=None):

    if default is None:
        default = {}

    if not path.exists():
        return default

    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def write_json(path: Path, data: Any):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf8") as f



#data_ingestion