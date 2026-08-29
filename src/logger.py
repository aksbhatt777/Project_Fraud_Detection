"""
Centralized logging setup.

Every module gets a logger via `get_logger(__name__)` so log lines are
attributable to the exact component that produced them, and every run also
writes to a timestamped file under `logs/` in addition to stdout.
"""

import logging
import os
import sys
from datetime import datetime

_LOG_DIR = "logs"
_LOG_FILE = None
_CONFIGURED = False


def _resolve_log_file() -> str:
    global _LOG_FILE
    if _LOG_FILE is None:
        os.makedirs(_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _LOG_FILE = os.path.join(_LOG_DIR, f"pipeline_{timestamp}.log")
    return _LOG_FILE


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_file = _resolve_log_file()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring handlers on first call."""
    _configure_root_logger()
    return logging.getLogger(name)
