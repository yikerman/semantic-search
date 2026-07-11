import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any

import psycopg
import psycopg_pool
import typer

from semsearch.cli import db
from semsearch.cli.ingest.chunk import char_chunks
from semsearch.cli.ingest.fetch import FetchError, create_fetcher
from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.ingest.pipeline import IngestError, index_sitemap, index_url
from semsearch.cli.models import Site
from semsearch.cli.sites import (
    AddSite,
    IndexSite,
    ListSites,
    PollAll,
    PollOutcome,
    PollSite,
    SiteError,
    add_site,
    fetch_feed,
    get_site,
    index_site,
    list_sites,
    mark_polled,
    poll_all,
    poll_site,
)
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError, create_embeddings
from semsearch.share.logging import configure_logging

app = typer.Typer(help="semsearch: indie blog search engine admin tool")
site_app = typer.Typer(help="Manage configured sites")
app.add_typer(site_app, name="site")

ForceOption = Annotated[
    bool, typer.Option("--force", help="Re-index URLs that are already indexed")
]


@dataclass(slots=True)
class Services:
    settings: Settings
    add_site: AddSite
    list_sites: ListSites
    index_site: IndexSite
    poll_site: PollSite
    poll_all: PollAll


@asynccontextmanager
async def open_services() -> AsyncIterator[Services]:
    settings = get_settings()
    async with (
        create_embeddings(settings) as embedder,
        create_pool(settings) as pool,
        create_fetcher(settings) as fetcher,
    ):
        index_url_fn = partial(
            index_url,
            pool,
            embedder.embed_documents,
            fetcher.fetch_text,
            partial(
                char_chunks,
                chunk_chars=settings.chunk_chars,
                chunk_overlap=settings.chunk_overlap,
            ),
        )
        index_sitemap_fn = partial(index_sitemap, fetcher.fetch_text, index_url_fn)
        list_sites_fn = partial(list_sites, pool)
        poll_site_fn = partial(
            poll_site,
            partial(get_site, pool),
            partial(fetch_feed, fetcher),
            partial(mark_polled, pool),
            index_url_fn,
        )
        yield Services(
            settings,
            partial(add_site, pool, fetcher.fetch_text),
            list_sites_fn,
            partial(index_site, pool, index_sitemap_fn),
            poll_site_fn,
            partial(poll_all, list_sites_fn, poll_site_fn),
        )


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except (
        IngestError,
        FetchError,
        EmbeddingError,
        SiteError,
    ) as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
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
            f"({settings.embedding_dim} dims) at {settings.database_url}"
        )

    run(_init())


@site_app.command("add")
def site_add(
    url: str,
    sitemap: Annotated[
        str, typer.Option(help='"auto", "none", or a sitemap URL')
    ] = "auto",
    feed: Annotated[str, typer.Option(help='"auto", "none", or a feed URL')] = "auto",
    index: Annotated[bool, typer.Option("--index", help="Index after adding")] = False,
    force: ForceOption = False,
) -> None:
    """Add or update a site."""

    async def _add() -> tuple[Site, list[IndexOutcome] | None]:
        async with open_services() as services:
            site = await services.add_site(url, sitemap, feed)
            if not index:
                return site, None
            outcomes = await services.index_site(
                site.base_url,
                force,
                _echo_outcome,
            )
            return site, outcomes

    site, outcomes = run(_add())
    _echo_site(site)
    if outcomes is not None:
        _echo_index_summary(outcomes)


@site_app.command("list")
def site_list() -> None:
    """List configured sites."""

    async def _list() -> list[Site]:
        async with open_services() as services:
            return await services.list_sites()

    sites = run(_list())
    if not sites:
        typer.echo("No sites.")
        return
    for site in sites:
        _echo_site(site)


@site_app.command("index")
def site_index(site: str, force: ForceOption = False) -> None:
    """Index a configured site."""

    async def _index() -> list[IndexOutcome]:
        async with open_services() as services:
            return await services.index_site(
                site,
                force,
                _echo_outcome,
            )

    outcomes = run(_index())
    _echo_index_summary(outcomes)


@site_app.command("poll")
def site_poll(
    site: str | None = None,
    all_sites: Annotated[bool, typer.Option("--all", help="Poll every feed")] = False,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            min=1,
            help="Maximum number of configured sites to poll at once with --all",
        ),
    ] = None,
    force: ForceOption = False,
) -> None:
    """Poll RSS/Atom feeds."""

    async def _poll() -> list[PollOutcome]:
        if all_sites:
            if site is not None:
                raise SiteError("Use either a site origin or --all")
            selected_site = None
        else:
            if site is None:
                raise SiteError("Pass a site origin or --all")
            selected_site = site
        async with open_services() as services:
            if selected_site is None:
                return await services.poll_all(
                    force,
                    _echo_outcome,
                    concurrency or services.settings.site_poll_concurrency,
                )
            return [await services.poll_site(selected_site, force, _echo_outcome)]

    polls = run(_poll())
    for poll in polls:
        _echo_poll_summary(poll)


@app.command()
def status() -> None:
    """Show index size."""

    async def _status() -> None:
        settings = get_settings()
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            try:
                stats = await db.fetch_index_stats(conn)
            except psycopg.errors.UndefinedTable:
                typer.echo("Database not initialized. Run: semsearch init-db")
                raise typer.Exit(1) from None
        typer.echo(f"sites:  {stats.site_count}")
        typer.echo(f"pages:  {stats.page_count}")
        typer.echo(f"chunks: {stats.chunk_count}")
        typer.echo(
            f"embedding config: {settings.embedding_model} "
            f"({settings.embedding_dim} dims)"
        )

    run(_status())


def _echo_outcome(outcome: IndexOutcome) -> None:
    detail = outcome.detail
    if outcome.status == "indexed":
        detail = f"{outcome.chunk_count} chunks"
    color = {"indexed": "green", "error": "red"}.get(outcome.status)
    typer.secho(
        f"[{outcome.status}] {outcome.url}" + (f" — {detail}" if detail else ""),
        fg=color,
    )


def _echo_index_summary(outcomes: list[IndexOutcome]) -> None:
    counts = Counter(outcome.status for outcome in outcomes)
    summary = ", ".join(f"{status}: {count}" for status, count in counts.most_common())
    typer.echo(f"\n{len(outcomes)} pages — {summary}")


def _echo_site(site: Site) -> None:
    typer.echo(site.base_url)
    typer.echo(f"  sitemap: {site.sitemap_url or '-'}")
    typer.echo(f"  feed:    {site.feed_url or '-'}")
    if site.last_indexed_at is not None:
        typer.echo(f"  indexed: {site.last_indexed_at}")
    if site.last_polled_at is not None:
        typer.echo(f"  polled:  {site.last_polled_at}")


def _echo_poll_summary(poll: PollOutcome) -> None:
    if poll.not_modified:
        typer.echo(f"\n{poll.site.base_url}: feed unchanged")
        return
    typer.echo(f"\n{poll.site.base_url}")
    _echo_index_summary(poll.outcomes)


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
