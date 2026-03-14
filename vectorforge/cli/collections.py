"""CLI commands for managing document collections."""

from __future__ import annotations

import json
import uuid

import typer

from vectorforge.cli._helpers import managed_engine, run_async

collections_app = typer.Typer(no_args_is_help=True)


@collections_app.command("list")
def list_collections(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
    offset: int = typer.Option(0, "--offset", help="Pagination offset."),
) -> None:
    """List all collections."""
    run_async(_list_collections(limit, offset))


async def _list_collections(limit: int, offset: int) -> None:
    from vectorforge.db.repositories.collection_repo import CollectionRepository

    async with managed_engine() as db, db.get_session() as session:
        repo = CollectionRepository(session)
        collections = await repo.find_all(limit=limit, offset=offset)
        for coll in collections:
            typer.echo(f"  {coll.id}  {coll.name}  ({coll.description or '-'})")
        if not collections:
            typer.echo("No collections found.")


@collections_app.command("get")
def get_collection(
    collection_id: str = typer.Argument(help="Collection UUID."),
) -> None:
    """Get details of a single collection."""
    try:
        cid = uuid.UUID(collection_id)
    except ValueError:
        typer.echo(f"Invalid UUID: {collection_id}", err=True)
        raise typer.Exit(code=1) from None
    run_async(_get_collection(cid))


async def _get_collection(collection_id: uuid.UUID) -> None:
    from vectorforge.db.repositories.collection_repo import CollectionRepository

    async with managed_engine() as db, db.get_session() as session:
        repo = CollectionRepository(session)
        coll = await repo.find_by_id(collection_id)
        if coll is None:
            typer.echo(f"Collection {collection_id} not found.", err=True)
            raise typer.Exit(code=1)
        typer.echo(json.dumps(coll.model_dump(mode="json"), indent=2, default=str))


@collections_app.command("create")
def create_collection(
    name: str = typer.Argument(help="Collection name."),
    description: str = typer.Option("", "--description", "-d", help="Description."),
) -> None:
    """Create a new collection."""
    run_async(_create_collection(name, description))


async def _create_collection(name: str, description: str) -> None:
    from vectorforge.db.repositories.collection_repo import CollectionRepository
    from vectorforge.models.domain import CreateCollectionDTO

    async with managed_engine() as db, db.get_session() as session:
        repo = CollectionRepository(session)
        dto = CreateCollectionDTO(name=name, description=description)
        coll = await repo.create(dto)
        typer.echo(f"Created collection: {coll.id} ({coll.name})")


@collections_app.command("delete")
def delete_collection(
    collection_id: str = typer.Argument(help="Collection UUID."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete a collection."""
    try:
        cid = uuid.UUID(collection_id)
    except ValueError:
        typer.echo(f"Invalid UUID: {collection_id}", err=True)
        raise typer.Exit(code=1) from None

    if not force:
        confirmed = typer.confirm(f"Delete collection {cid}?")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    run_async(_delete_collection(cid))


async def _delete_collection(collection_id: uuid.UUID) -> None:
    from vectorforge.db.repositories.collection_repo import CollectionRepository

    async with managed_engine() as db, db.get_session() as session:
        repo = CollectionRepository(session)
        await repo.delete(collection_id)
        typer.echo(f"Deleted collection: {collection_id}")
