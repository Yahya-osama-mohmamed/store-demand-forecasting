"""
Structured Logging Configuration.

Provides consistent, JSON-formatted logging across the entire application.
Supports both console output and file-based logging for production use.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"model": "xgboost", "n_samples": 5000})
"""

import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as structured JSON.
    
    This makes logs machine-parseable for monitoring systems while
    remaining human-readable in the console.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include extra fields if provided (e.g., model name, metrics)
        if hasattr(record, "model"):
            log_data["model"] = record.model
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable colored console formatter.
    
    Uses ANSI escape codes for color-coding log levels
    to improve readability during development.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with color-coded level and timestamp."""
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{color}[{timestamp}] {record.levelname:8s}{self.RESET} "
            f"| {record.name} | {record.getMessage()}"
        )


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        log_to_file: Whether to also write logs to a file.
        log_file: Custom log file name. Defaults to 'app.log'.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # --- Console handler (human-readable) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # --- File handler (structured JSON for production) ---
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / (log_file or "app.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger
