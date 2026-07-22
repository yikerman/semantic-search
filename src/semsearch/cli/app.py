import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any
from urllib.parse import urlsplit

import psycopg
import psycopg_pool
import typer
from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.daemon.run import (
    DAEMON_LOCK_ID,
    DaemonAlreadyRunningError,
    advisory_lock,
    run_daemon,
)
from semsearch.cli.ingest.chunk import (
    TokenizerError,
    load_tokenizer,
    token_chunks,
)
from semsearch.cli.ingest.feed import FeedError
from semsearch.cli.ingest.fetch import FetchError, Fetcher, create_fetcher
from semsearch.cli.models import Site
from semsearch.cli.sites import SiteError, add_site, list_sites, remove_site
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import (
    EmbeddingError,
    create_embeddings,
)
from semsearch.share.logging import configure_logging
from semsearch.share.status import fetch_index_stats, list_failed_jobs

app = typer.Typer(help="semsearch: indie blog search engine admin tool")
site_app = typer.Typer(help="Manage configured sites")
app.add_typer(site_app, name="site")


@dataclass(slots=True)
class Services:
    settings: Settings
    pool: AsyncConnectionPool
    fetcher: Fetcher


@asynccontextmanager
async def open_services() -> AsyncIterator[Services]:
    settings = get_settings()
    async with (
        create_pool(settings) as pool,
        create_fetcher(settings) as fetcher,
    ):
        yield Services(settings, pool, fetcher)


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except (
        FetchError,
        FeedError,
        EmbeddingError,
        TokenizerError,
        SiteError,
        DaemonAlreadyRunningError,
    ) as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc
    except psycopg.errors.UndefinedTable as exc:
        typer.secho(
            "error: database not initialized. Run: semsearch init-db",
            fg="red",
            err=True,
        )
        raise typer.Exit(1) from exc
    except (psycopg.OperationalError, psycopg_pool.PoolTimeout) as exc:
        typer.secho(
            f"database error: {exc}\n"
            "Is Postgres running (podman compose up -d db) "
            "and initialized (semsearch init-db)?",
            fg="red",
            err=True,
        )
        raise typer.Exit(1) from exc


@app.command("init-db")
def init_db() -> None:
    """Initialize an empty database."""

    async def _init() -> None:
        settings = get_settings()
        await db.init_schema(settings)
        typer.echo(
            f"Schema ready: {settings.embedding_model} "
            f"({settings.embedding_dim} dims) at {_redact_dsn(settings.database_url)}"
        )

    run(_init())


@site_app.command("add")
def site_add(
    url: str,
    sitemap: Annotated[
        str, typer.Option(help='"auto", "none", or a sitemap URL')
    ] = "auto",
    feed: Annotated[str, typer.Option(help='"auto" or a feed URL')] = "auto",
) -> None:
    """Add or update a feed-backed site."""

    async def _add() -> Site:
        async with open_services() as services:
            return await add_site(
                services.pool,
                services.fetcher,
                url,
                sitemap,
                feed,
                poll_interval_seconds=services.settings.site_poll_interval_seconds,
            )

    _echo_site(run(_add()))


@site_app.command("list")
def site_list() -> None:
    """List configured sites and synchronization state."""

    async def _list() -> list[Site]:
        async with open_services() as services:
            return await list_sites(services.pool)

    records = run(_list())
    if not records:
        typer.echo("No sites.")
        return
    for record in records:
        _echo_site(record)


@site_app.command("remove")
def site_remove(url: str) -> None:
    """Remove a configured site and all of its stored data."""

    async def _remove() -> str:
        settings = get_settings()
        async with create_pool(settings) as pool:
            async with advisory_lock(pool, DAEMON_LOCK_ID):
                return await remove_site(pool, url)

    typer.echo(f"Removed {run(_remove())}")


@app.command()
def daemon() -> None:
    """Continuously poll sites and ingest discovered posts."""

    async def _daemon() -> None:
        async with open_services() as services:
            settings = services.settings
            tokenizer = load_tokenizer(
                settings.embedding_tokenizer,
                settings.embedding_tokenizer_revision,
            )
            async with create_embeddings(settings) as embedder:
                await run_daemon(
                    services.pool,
                    embedder.embed_documents,
                    services.fetcher,
                    partial(
                        token_chunks,
                        tokenizer=tokenizer,
                        chunk_tokens=settings.chunk_tokens,
                        chunk_token_overlap=settings.chunk_token_overlap,
                    ),
                    settings,
                )

    run(_daemon())


@app.command()
def status() -> None:
    """Show index and crawl queue size."""

    async def _status() -> None:
        settings = get_settings()
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            try:
                stats = await fetch_index_stats(conn)
                failures = await list_failed_jobs(conn)
            except psycopg.errors.UndefinedTable:
                typer.echo("Database not initialized. Run: semsearch init-db")
                raise typer.Exit(1) from None
        typer.echo(f"sites:   {stats.site_count}")
        typer.echo(f"pages:   ~{stats.page_count}")
        typer.echo(f"chunks:  ~{stats.chunk_count}")
        typer.echo(f"queued:  {stats.queued_count}")
        typer.echo(f"retrying: {stats.retrying_count}")
        typer.echo(f"failed:  {stats.failed_count}")
        for failure in failures:
            typer.echo(
                f"  {failure.url} ({failure.attempt_count} attempts): "
                f"{failure.last_error}"
            )
        typer.echo(
            f"embedding config: {settings.embedding_model} "
            f"({settings.embedding_dim} dims)"
        )

    run(_status())


def _redact_dsn(url: str) -> str:
    parts = urlsplit(url)
    if not parts.hostname:
        return "the configured database"
    port = f":{parts.port}" if parts.port else ""
    name = parts.path.lstrip("/")
    return f"{parts.scheme}://{parts.hostname}{port}/{name}"


def _echo_site(site: Site) -> None:
    typer.echo(site.base_url)
    typer.echo(f"  sitemap: {site.sitemap_url or '-'}")
    typer.echo(f"  feed:    {site.feed_url}")
    typer.echo(f"  polled:  {site.last_polled_at or '-'}")
    typer.echo(f"  next:    {site.next_poll_at or '-'}")
    if site.history_pending:
        typer.echo("  history: pending")
    if site.history_error:
        typer.echo(f"  history error: {site.history_error}")
    if site.sync_error:
        typer.echo(f"  sync error: {site.sync_error}")


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
