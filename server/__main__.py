"""Entry point for ``python -m server``."""

from __future__ import annotations

import logging
import sys

import uvicorn

from server.app import create_app
from server.config import APIConfig


def _configure_logging() -> None:
    """Set up structured logging for the server process."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def main() -> None:
    """Start the VectorForge API server."""
    _configure_logging()
    config = APIConfig()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
