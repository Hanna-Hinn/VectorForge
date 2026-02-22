"""Structured logging setup for VectorForge.

Supports JSON and text output formats with optional file rotation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from vectorforge.config.settings import MonitoringConfig


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge extra fields if present
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            log_entry.update(extra)

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def configure_logging(config: MonitoringConfig) -> None:
    """Configure the root logger based on MonitoringConfig.

    Sets up handlers (stdout and optional file) with the chosen format.
    Suppresses noisy third-party loggers.

    Args:
        config: Monitoring configuration with logging preferences.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level)

    # Remove existing handlers to avoid duplicates on reconfiguration
    root_logger.handlers.clear()

    # Select formatter
    if config.log_format == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Optional file handler with rotation
    if config.log_file:
        file_handler = RotatingFileHandler(
            filename=config.log_file,
            maxBytes=10_485_760,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "sqlalchemy.engine"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    root_logger.info(
        "Logging configured",
        extra={
            "extra": {
                "log_level": config.log_level,
                "log_format": config.log_format,
                "log_file": config.log_file,
            }
        },
    )
