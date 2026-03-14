"""CLI commands for viewing and validating configuration."""

from __future__ import annotations

import json

import typer

config_app = typer.Typer(no_args_is_help=True)


@config_app.command("show")
def show_config() -> None:
    """Show the current configuration (from environment)."""
    from vectorforge.config.settings import load_config

    config = load_config()
    data = config.model_dump(mode="json")

    # Redact sensitive fields
    _redact_sensitive(data)

    typer.echo(json.dumps(data, indent=2, default=str))


@config_app.command("validate")
def validate_config() -> None:
    """Validate the configuration and report any errors."""
    from pydantic import ValidationError

    from vectorforge.config.settings import load_config

    try:
        config = load_config()
        typer.echo("Configuration is valid.")
        typer.echo(f"  Database: {config.database.host}:{config.database.port}")
        typer.echo(f"  Embedding: {config.embedding.default_provider}")
        typer.echo(f"  LLM: {config.llm.default_provider}")
    except ValidationError as exc:
        typer.echo("Configuration errors:", err=True)
        for error in exc.errors():
            loc = " → ".join(str(p) for p in error["loc"])
            typer.echo(f"  [{loc}] {error['msg']}", err=True)
        raise typer.Exit(code=1) from None


def _redact_sensitive(data: dict[str, object]) -> None:
    """Redact known sensitive keys in-place.

    Args:
        data: Config dictionary to sanitize.
    """
    sensitive_keys = {"password", "api_key", "secret", "token"}
    for key, value in data.items():
        if isinstance(value, dict):
            _redact_sensitive(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _redact_sensitive(item)
        elif isinstance(value, str) and any(s in key.lower() for s in sensitive_keys):
            data[key] = "***REDACTED***"
