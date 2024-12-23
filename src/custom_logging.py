"""
Logging script with separate directories for application and test logs.
"""

import logging
from datetime import datetime
import pandas as pd
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "..", "logs")
APP_LOG_DIR = os.path.join(LOG_DIR, "application_logs")
TEST_LOG_DIR = os.path.join(LOG_DIR, "test_logs")
DATA_LOG_FILE = os.path.join(LOG_DIR, "log_data.csv")

os.makedirs(APP_LOG_DIR, exist_ok=True)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
APP_LOG_FILE = os.path.join(APP_LOG_DIR, f"app_{timestamp}.log")
TEST_LOG_FILE = os.path.join(TEST_LOG_DIR, f"test_{timestamp}.log")

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """
    Set up a logger that writes messages to a log file and the console.

    Args:
        name (str): Logger name.
        log_file (str): Path to the log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler for logging to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

app_logger = setup_logger("application", APP_LOG_FILE)
test_logger = setup_logger("tests", TEST_LOG_FILE)

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

def log_event(event_message, logger_type="application"):
    """
    Log an informational event.

    Args:
        event_message (str): Message to log.
        logger_type (str): The type of logger ("application" or "tests").
    """
    if logger_type == "application":
        app_logger.info(event_message)
    elif logger_type == "tests":
        test_logger.info(event_message)

def log_error(error_message, logger_type="application"):
    """
    Log an error event.

    Args:
        error_message (str): Error message to log.
        logger_type (str): The type of logger ("application" or "tests").
    """
    if logger_type == "application":
        app_logger.error(error_message)
    elif logger_type == "tests":
        test_logger.error(error_message)

def log_critical(critical_message, logger_type="application"):
    """
    Log a critical event.

    Args:
        critical_message (str): Critical message to log.
        logger_type (str): The type of logger ("application" or "tests").
    """
    if logger_type == "application":
        app_logger.critical(critical_message)
    elif logger_type == "tests":
        test_logger.critical(critical_message)

def log_warning(warning_message, logger_type="application"):
    """
    Log a warning.

    Args:
        warning_message (str): Warning message to log.
        logger_type (str): The type of logger ("application" or "tests").
    """
    if logger_type == "application":
        app_logger.warning(warning_message)
    elif logger_type == "tests":
        test_logger.warning(warning_message)
