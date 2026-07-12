"""
 Console logging
 File logging
 Colored logs
 Log rotation
 Execution timer
 Memory usage
 Progress statistics and 
 Error logging
"""


import logging
import time
import psutil

from functools import wraps
from logging.handlers import RotatingFileHandler

from config import LOG_FILE


LOGGER_NAME = "PDF_INGESTION"


def get_logger():

    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----------------------------
    # Console
    # ----------------------------

    console = logging.StreamHandler()

    console.setFormatter(formatter)

    # ----------------------------
    # File
    # ----------------------------

    file = RotatingFileHandler(
        LOG_FILE,
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf8",
    )

    file.setFormatter(formatter)

    logger.addHandler(console)

    logger.addHandler(file)

    logger.propagate = False

    return logger


logger = get_logger()


# ======================================================
# Performance
# ======================================================

class Performance:

    @staticmethod
    def cpu():

        return psutil.cpu_percent(interval=None)

    @staticmethod
    def process_memory():

        process = psutil.Process()

        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def system_memory():

        return psutil.virtual_memory().percent

    @staticmethod
    def disk():

        return psutil.disk_usage("/").percent


# ======================================================
# Decorator
# ======================================================

def log_time(task_name=None):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            name = task_name or func.__name__

            logger.info("=" * 70)

            logger.info(f"START : {name}")

            start = time.perf_counter()

            try:

                result = func(*args, **kwargs)

                elapsed = time.perf_counter() - start

                logger.info(f"END   : {name}")

                logger.info(f"TIME  : {elapsed:.2f} sec")

                logger.info("=" * 70)

                return result

            except Exception:

                logger.exception(f"FAILED : {name}")

                raise

        return wrapper

    return decorator


# ======================================================
# Helper Functions
# ======================================================

def system_stats():

    logger.info("-" * 70)

    logger.info(f"CPU Usage          : {Performance.cpu()} %")

    logger.info(
        f"Process Memory     : {Performance.process_memory():.2f} MB"
    )

    logger.info(
        f"System RAM Usage   : {Performance.system_memory()} %"
    )

    logger.info(
        f"Disk Usage         : {Performance.disk()} %"
    )

    logger.info("-" * 70)


def stage(name):

    logger.info("")
    logger.info("=" * 70)
    logger.info(name)
    logger.info("=" * 70)


def page(page_number):

    logger.info(f"Processing Page : {page_number}")


def batch(batch_number, pages):

    logger.info(
        f"Batch {batch_number} | Pages : {pages}"
    )


def success(msg):

    logger.info(f"SUCCESS : {msg}")


def warning(msg):

    logger.warning(msg)


def error(msg):

    logger.error(msg)


def exception(e):

    logger.exception(e)


def summary(
    total_pages,
    text_pages,
    ocr_pages,
    elapsed,
):

    logger.info("")
    logger.info("=" * 70)

    logger.info("INGESTION SUMMARY")

    logger.info("=" * 70)

    logger.info(f"Total Pages      : {total_pages}")

    logger.info(f"Text Pages       : {text_pages}")

    logger.info(f"OCR Pages        : {ocr_pages}")

    logger.info(
        f"Execution Time   : {elapsed:.2f} sec"
    )

    logger.info(
        f"Process Memory   : {Performance.process_memory():.2f} MB"
    )

    logger.info("=" * 70)