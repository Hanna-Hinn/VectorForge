"""Main CLI application for VectorForge.

Assembles the top-level typer app and registers sub-command groups.
"""

from __future__ import annotations

import logging
import sys

import typer

from vectorforge.cli.collections import collections_app
from vectorforge.cli.config import config_app
from vectorforge.cli.evaluate import evaluate_app
from vectorforge.cli.query import query_app

app = typer.Typer(
    name="vectorforge",
    help="VectorForge — high-performance standalone RAG engine CLI.",
    no_args_is_help=True,
)
app.add_typer(collections_app, name="collections", help="Manage document collections.")
app.add_typer(config_app, name="config", help="View and validate configuration.")
app.add_typer(evaluate_app, name="evaluate", help="Run and view RAG quality evaluations.")
app.add_typer(query_app, name="query", help="Execute RAG queries.")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Configure global CLI options."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.command()
def version() -> None:
    """Print the VectorForge version."""
    from vectorforge import __version__

    typer.echo(f"VectorForge v{__version__}")
