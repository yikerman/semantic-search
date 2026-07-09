import asyncio
import logging
from collections import Counter
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import psycopg
import psycopg_pool
import typer
from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.config import Settings, get_settings
from semsearch.embeddings import get_embedding_provider
from semsearch.embeddings.openai_compat import EmbeddingError, OpenAICompatEmbeddings
from semsearch.ingest.fetch import FetchError
from semsearch.ingest.models import IndexOutcome
from semsearch.ingest.service import IngestError, IngestService
from semsearch.search.service import SearchService
from semsearch.sites import PollOutcome, Site, SiteError, SiteService

app = typer.Typer(help="semsearch: indie blog search engine admin tool")
site_app = typer.Typer(help="Manage configured sites")
app.add_typer(site_app, name="site")

ForceOption = Annotated[
    bool, typer.Option("--force", help="Re-index URLs that are already indexed")
]


@dataclass(slots=True)
class Services:
    settings: Settings
    pool: AsyncConnectionPool
    embedder: OpenAICompatEmbeddings


@asynccontextmanager
async def open_services() -> AsyncIterator[Services]:
    settings = get_settings()
    embedder = get_embedding_provider(settings)
    try:
        pool = db.create_pool(settings)
        await pool.open()
        try:
            yield Services(settings, pool, embedder)
        finally:
            await pool.close()
    finally:
        await embedder.aclose()


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        return asyncio.run(coro)
    except (
        db.IndexMetaError,
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
            sites = SiteService(services.pool, services.settings)
            try:
                site = await sites.add_site(url, sitemap_url=sitemap, feed_url=feed)
                if not index:
                    return site, None
                ingest = IngestService(
                    services.pool, services.embedder, services.settings
                )
                try:
                    outcomes = await sites.index_site(
                        site.base_url,
                        ingest,
                        force=force,
                        on_progress=_echo_outcome,
                    )
                finally:
                    await ingest.aclose()
                return site, outcomes
            finally:
                await sites.aclose()

    site, outcomes = run(_add())
    _echo_site(site)
    if outcomes is not None:
        _echo_index_summary(outcomes)


@site_app.command("list")
def site_list() -> None:
    """List configured sites."""

    async def _list() -> list[Site]:
        async with open_services() as services:
            sites = SiteService(services.pool, services.settings)
            try:
                return await sites.list_sites()
            finally:
                await sites.aclose()

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
            sites = SiteService(services.pool, services.settings)
            ingest = IngestService(services.pool, services.embedder, services.settings)
            try:
                return await sites.index_site(
                    site,
                    ingest,
                    force=force,
                    on_progress=_echo_outcome,
                )
            finally:
                await ingest.aclose()
                await sites.aclose()

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
        if site is not None and all_sites:
            raise SiteError("Use either a site origin or --all")
        if site is None and not all_sites:
            raise SiteError("Pass a site origin or --all")
        async with open_services() as services:
            sites = SiteService(services.pool, services.settings)
            ingest = IngestService(services.pool, services.embedder, services.settings)
            try:
                if all_sites:
                    return await sites.poll_all(
                        ingest,
                        force=force,
                        on_progress=_echo_outcome,
                        concurrency=(
                            concurrency or services.settings.site_poll_concurrency
                        ),
                    )
                assert site is not None
                return [
                    await sites.poll_site(
                        site,
                        ingest,
                        force=force,
                        on_progress=_echo_outcome,
                    )
                ]
            finally:
                await ingest.aclose()
                await sites.aclose()

    polls = run(_poll())
    for poll in polls:
        _echo_poll_summary(poll)


@app.command()
def search(
    query: str,
    limit: Annotated[int, typer.Option("-k", "--limit")] = 10,
) -> None:
    """Search from the terminal."""

    async def _search() -> list:
        async with open_services() as services:
            service = SearchService(services.pool, services.embedder, services.settings)
            return await service.search(query, limit=limit)

    results = run(_search())
    if not results:
        typer.echo("No results.")
        return
    for rank, result in enumerate(results, start=1):
        typer.secho(f"{rank}. {result.title or result.url}", bold=True)
        typer.echo(f"   {result.url}  (score {result.score:.3f})")
        snippet = " ".join(result.snippet.split())
        typer.echo(f"   {snippet[:200]}{'…' if len(snippet) > 200 else ''}\n")


@app.command()
def status() -> None:
    """Show index size."""

    async def _status() -> None:
        settings = get_settings()
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            try:
                cur = await conn.execute(
                    """
                    SELECT (SELECT count(*) FROM sites),
                           (SELECT count(*) FROM pages),
                           (SELECT count(*) FROM chunks),
                           (SELECT embedding_model FROM index_meta WHERE id = 1),
                           (SELECT embedding_dim FROM index_meta WHERE id = 1)
                    """
                )
            except psycopg.errors.UndefinedTable:
                typer.echo("Database not initialized. Run: semsearch init-db")
                raise typer.Exit(1) from None
            row = await cur.fetchone()
            assert row is not None
            sites, pages, chunks, model, dim = row
        typer.echo(f"sites:  {sites}")
        typer.echo(f"pages:  {pages}")
        typer.echo(f"chunks: {chunks}")
        typer.echo(f"embedding space: {model} ({dim} dims)")

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
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    app()


if __name__ == "__main__":
    main()
