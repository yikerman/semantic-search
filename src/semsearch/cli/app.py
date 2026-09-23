import asyncio
import signal
from collections.abc import Coroutine
from typing import Annotated, Any

import psycopg
import psycopg_pool
import typer

from semsearch.cli import db
from semsearch.cli.crawl.run import run_crawl
from semsearch.cli.daemon import run_daemon
from semsearch.cli.index import run_index
from semsearch.cli.ingest.document import TokenizerError
from semsearch.cli.locks import (
    CRAWL_LOCK,
    INDEX_LOCK,
    AlreadyRunningError,
    command_lock,
)
from semsearch.cli.models import Site
from semsearch.cli.sites import SiteError, add_site, list_sites, remove_site
from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError
from semsearch.share.logging import configure_logging
from semsearch.share.status import fetch_index_stats, list_failed_articles

app = typer.Typer(help="semsearch: indie blog search engine admin tool")
site_app = typer.Typer(help="Manage configured sites")
app.add_typer(site_app, name="site")


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(_with_shutdown(coro))
    except asyncio.CancelledError as exc:
        raise typer.Exit(143) from exc
    except AlreadyRunningError as exc:
        typer.echo(f"Skipped: {exc}")
        return None
    except psycopg.errors.UndefinedTable as exc:
        typer.secho(
            "Database not initialized. Run: semsearch init-db", fg="red", err=True
        )
        raise typer.Exit(1) from exc
    except (
        SiteError,
        EmbeddingError,
        TokenizerError,
        psycopg.Error,
        psycopg_pool.PoolTimeout,
        RuntimeError,
    ) as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
        raise typer.Exit(1) from exc


async def _with_shutdown(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.get_running_loop()
    owner = asyncio.current_task()
    assert owner is not None
    loop.add_signal_handler(signal.SIGTERM, owner.cancel)
    try:
        return await coro
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


@app.command("init-db")
def init_db() -> None:
    """Initialize a fresh 1.0 database (no migration from older versions)."""
    run(db.init_schema(get_settings()))
    typer.echo("Schema ready")


@site_app.command("add")
def site_add(
    url: str,
    sitemap: Annotated[str, typer.Option(help="auto, none, or a sitemap URL")] = "auto",
    feed: Annotated[str, typer.Option(help="auto, none, or an RSS/Atom URL")] = "auto",
) -> None:
    """Record site configuration; source discovery happens during crawl."""

    async def operation() -> Site:
        async with create_pool(get_settings()) as pool:
            return await add_site(pool, url, sitemap, feed)

    _echo_site(run(operation()))


@site_app.command("list")
def site_list() -> None:
    async def operation() -> list[Site]:
        async with create_pool(get_settings()) as pool:
            return await list_sites(pool)

    for site in run(operation()):
        _echo_site(site)


@site_app.command("remove")
def site_remove(url: str) -> None:
    """Remove a site and its data when neither batch command is running."""

    async def operation() -> str:
        async with (
            create_pool(get_settings()) as pool,
            command_lock(pool, CRAWL_LOCK),
            command_lock(pool, INDEX_LOCK),
        ):
            return await remove_site(pool, url)

    removed = run(operation())
    if removed:
        typer.echo(f"Removed {removed}")


@app.command()
def crawl(retry_failed: bool = False) -> None:
    """Run one finite crawl batch; optionally retry rejected/failed articles."""

    async def operation() -> None:
        settings = get_settings()
        async with create_pool(settings) as pool:
            await run_crawl(pool, settings, retry_failed=retry_failed)

    run(operation())


@app.command("index")
def index_command() -> None:
    """Embed a snapshot of stored, unindexed articles."""

    async def operation() -> tuple[int, int]:
        settings = get_settings()
        async with create_pool(settings) as pool:
            return await run_index(pool, settings)

    result = run(operation())
    if result:
        succeeded, failed = result
        typer.echo(f"Indexed: {succeeded}; failed: {failed}")
        if failed:
            raise typer.Exit(1)


@app.command()
def daemon() -> None:
    """Run recurring crawl and index jobs until the container stops."""

    async def operation() -> None:
        settings = get_settings()
        async with create_pool(settings) as pool:
            await run_daemon(pool, settings)

    run(operation())


@app.command()
def status() -> None:
    """Show crawl outcomes and indexing backlog."""

    async def operation() -> None:
        async with create_pool(get_settings()) as pool, pool.connection() as conn:
            stats = await fetch_index_stats(conn)
            failures = await list_failed_articles(conn)
        for label, value in (
            ("sites", stats.site_count),
            ("pages", stats.page_count),
            ("indexed pages", stats.indexed_count),
            ("pending URLs", stats.queued_count),
            ("retrying URLs", stats.retrying_count),
            ("failed URLs", stats.failed_count),
            ("rejected URLs", stats.rejected_count),
            ("pending indexing", stats.pending_index_count),
            ("failed indexing", stats.failed_index_count),
            ("rejected indexing", stats.rejected_index_count),
        ):
            typer.echo(f"{label}: {value}")
        for failure in failures:
            typer.echo(f"  {failure.url}: {failure.last_error}")

    run(operation())


def _echo_site(site: Site) -> None:
    typer.echo(site.base_url)
    typer.echo(f"  feed: {site.feed_url}; sitemap: {site.sitemap_url}")
    typer.echo(
        f"  crawled: {site.last_crawled_at}; history complete: {site.history_complete}"
    )
    if site.last_error:
        typer.echo(f"  error: {site.last_error}")


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
