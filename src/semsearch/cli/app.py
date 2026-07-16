import asyncio
from collections import Counter
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
from semsearch.cli.ingest.chunk import Chunker, char_chunks
from semsearch.cli.ingest.feed import FeedError
from semsearch.cli.ingest.fetch import FetchError, Fetcher, create_fetcher
from semsearch.cli.ingest.lease import LeaseLostError
from semsearch.cli.ingest.pipeline import IndexOutcome, IngestError
from semsearch.cli.ingest.worker import (
    WorkerAlreadyRunningError,
    drain_site_jobs,
    run_worker,
)
from semsearch.cli.models import Site
from semsearch.cli.sites import (
    PollOutcome,
    SiteError,
    add_site,
    list_sites,
    poll_site,
)
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import (
    EmbedDocuments,
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
    embed_documents: EmbedDocuments
    chunker: Chunker


@asynccontextmanager
async def open_services() -> AsyncIterator[Services]:
    settings = get_settings()
    async with (
        create_embeddings(settings) as embedder,
        create_pool(settings) as pool,
        create_fetcher(settings) as fetcher,
    ):
        yield Services(
            settings,
            pool,
            fetcher,
            embedder.embed_documents,
            partial(
                char_chunks,
                chunk_chars=settings.chunk_chars,
                chunk_overlap=settings.chunk_overlap,
            ),
        )


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except (
        IngestError,
        FetchError,
        FeedError,
        EmbeddingError,
        SiteError,
        WorkerAlreadyRunningError,
        LeaseLostError,
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
    """Apply the schema."""

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


@site_app.command("poll")
def site_poll_command(site: str) -> None:
    """Synchronize one configured site now."""

    async def _poll() -> tuple[PollOutcome, list[IndexOutcome]]:
        async with open_services() as services:
            outcome = await poll_site(
                services.pool,
                services.fetcher,
                services.settings,
                site,
            )
            indexed = await drain_site_jobs(
                services.pool,
                services.embed_documents,
                services.fetcher,
                services.chunker,
                outcome.site.id,
                _echo_outcome,
            )
            return outcome, indexed

    outcome, indexed = run(_poll())
    _echo_poll_summary(outcome, indexed)
    if outcome.error:
        raise typer.Exit(1)


@app.command()
def worker() -> None:
    """Continuously poll sites and ingest discovered posts."""

    async def _worker() -> None:
        async with open_services() as services:
            await run_worker(
                services.pool,
                services.embed_documents,
                services.fetcher,
                services.chunker,
                services.settings,
            )

    run(_worker())


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
        typer.echo(f"pages:   {stats.page_count}")
        typer.echo(f"chunks:  {stats.chunk_count}")
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


def _echo_outcome(outcome: IndexOutcome) -> None:
    detail = outcome.detail
    if outcome.status == "indexed":
        detail = f"{outcome.chunk_count} chunks"
    color = {"indexed": "green", "error": "red"}.get(outcome.status)
    typer.secho(
        f"[{outcome.status}] {outcome.url}" + (f" - {detail}" if detail else ""),
        fg=color,
    )


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


def _echo_poll_summary(poll: PollOutcome, outcomes: list[IndexOutcome]) -> None:
    if poll.not_modified:
        typer.echo(f"\n{poll.site.base_url}: feed unchanged")
        return
    counts = Counter(outcome.status for outcome in outcomes)
    summary = ", ".join(f"{key}: {value}" for key, value in counts.most_common())
    typer.echo(
        f"\n{poll.site.base_url}: queued {poll.discovered} URLs"
        + (f"; {summary}" if summary else "")
    )
    if poll.error:
        typer.secho(f"error: {poll.error}", fg="red", err=True)


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
