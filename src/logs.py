"""
Logging script.
"""

import logging
from datetime import datetime
import pandas as pd
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "..", "logs")
DATA_LOG_FILE = os.path.join(LOG_DIR, "log_data.csv")

os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"app_{timestamp}.log")

def setup_logger():
    """
    Set up a logger that writes messages to a log file and the console.
    """
    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def append_to_log_data(data):
    """
    Append structured log data to a CSV file using pandas.

    Args:
        data (dict): A dictionary where keys are column names, and values are data to append.
    """
    df = pd.DataFrame([data])

    if os.path.exists(DATA_LOG_FILE):
        existing_df = pd.read_csv(DATA_LOG_FILE)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(DATA_LOG_FILE, index=False)
    else:
        df.to_csv(DATA_LOG_FILE, index=False)

def log_event(event_message):
    """Log an informational event."""
    logger.info(event_message)

def log_error(error_message):
    """Log an error event."""
    logger.error(error_message)

def log_critical(critical_message):
    """Log a critical event."""
    logger.critical(critical_message)

def log_warning(warning_message):
    """Log a warning."""
    logger.warning(warning_message)