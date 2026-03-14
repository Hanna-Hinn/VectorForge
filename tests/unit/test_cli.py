"""Unit tests for the VectorForge CLI."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import typer.testing

from vectorforge.cli.main import app

runner = typer.testing.CliRunner()


# ---------------------------------------------------------------------------
# Version command
# ---------------------------------------------------------------------------


class TestVersionCommand:
    """Tests for the 'version' command."""

    def test_version_output(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "VectorForge v" in result.output


# ---------------------------------------------------------------------------
# Config commands
# ---------------------------------------------------------------------------


class TestConfigCommands:
    """Tests for the 'config' sub-commands."""

    @patch("vectorforge.config.settings.load_config")
    def test_show_config(self, mock_load: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {
            "database": {"host": "localhost", "password": "secret123"},
            "llm": {"api_key": "sk-test"},
        }
        mock_load.return_value = mock_config

        result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["database"]["password"] == "***REDACTED***"
        assert output["llm"]["api_key"] == "***REDACTED***"

    @patch("vectorforge.config.settings.load_config")
    def test_validate_config_success(self, mock_load: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.database.host = "localhost"
        mock_config.database.port = 5432
        mock_config.embedding.default_provider = "voyage"
        mock_config.llm.default_provider = "openai"
        mock_load.return_value = mock_config

        result = runner.invoke(app, ["config", "validate"])

        assert result.exit_code == 0
        assert "Configuration is valid" in result.output


# ---------------------------------------------------------------------------
# Collections commands
# ---------------------------------------------------------------------------


class TestCollectionsCommands:
    """Tests for the 'collections' sub-commands."""

    @patch("vectorforge.cli.collections._list_collections")
    @patch("vectorforge.cli.collections.run_async")
    def test_list_calls_run_async(
        self, mock_run: MagicMock, mock_fn: MagicMock,
    ) -> None:
        mock_run.return_value = None
        result = runner.invoke(app, ["collections", "list"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("vectorforge.cli.collections._get_collection")
    @patch("vectorforge.cli.collections.run_async")
    def test_get_with_valid_uuid(
        self, mock_run: MagicMock, mock_fn: MagicMock,
    ) -> None:
        mock_run.return_value = None
        cid = str(uuid.uuid4())
        result = runner.invoke(app, ["collections", "get", cid])
        assert result.exit_code == 0

    def test_get_with_invalid_uuid(self) -> None:
        result = runner.invoke(app, ["collections", "get", "not-a-uuid"])
        assert result.exit_code == 1
        assert "Invalid UUID" in result.output

    @patch("vectorforge.cli.collections._create_collection")
    @patch("vectorforge.cli.collections.run_async")
    def test_create(self, mock_run: MagicMock, mock_fn: MagicMock) -> None:
        mock_run.return_value = None
        result = runner.invoke(app, ["collections", "create", "my-coll"])
        assert result.exit_code == 0

    def test_delete_invalid_uuid(self) -> None:
        result = runner.invoke(app, ["collections", "delete", "bad-uuid"])
        assert result.exit_code == 1

    @patch("vectorforge.cli.collections._delete_collection")
    @patch("vectorforge.cli.collections.run_async")
    def test_delete_with_force(
        self, mock_run: MagicMock, mock_fn: MagicMock,
    ) -> None:
        mock_run.return_value = None
        cid = str(uuid.uuid4())
        result = runner.invoke(app, ["collections", "delete", cid, "--force"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Query commands
# ---------------------------------------------------------------------------


class TestQueryCommands:
    """Tests for the 'query' sub-commands."""

    def test_run_invalid_uuid(self) -> None:
        result = runner.invoke(app, ["query", "run", "not-uuid", "hello"])
        assert result.exit_code == 1
        assert "Invalid UUID" in result.output

    @patch("vectorforge.cli.query._run_query")
    @patch("vectorforge.cli.query.run_async")
    def test_run_valid(self, mock_run: MagicMock, mock_fn: MagicMock) -> None:
        mock_run.return_value = None
        cid = str(uuid.uuid4())
        result = runner.invoke(app, ["query", "run", cid, "What is RAG?"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Verbose flag
# ---------------------------------------------------------------------------


class TestVerboseFlag:
    """Tests for the --verbose global option."""

    @patch("vectorforge.cli.main.logging")
    def test_verbose_sets_debug(self, mock_logging: MagicMock) -> None:
        result = runner.invoke(app, ["--verbose", "version"])
        assert result.exit_code == 0
        # The callback should have called basicConfig with DEBUG level
        mock_logging.basicConfig.assert_called_once()


# ---------------------------------------------------------------------------
# run_async helper
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for the run_async helper."""

    def test_run_async_simple_coro(self) -> None:
        from vectorforge.cli._helpers import run_async

        async def _add(a: int, b: int) -> int:
            return a + b

        assert run_async(_add(1, 2)) == 3
