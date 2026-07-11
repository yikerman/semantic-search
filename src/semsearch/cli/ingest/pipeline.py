import logging
from collections.abc import Awaitable, Callable

from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.ingest import sitemap
from semsearch.cli.ingest.chunk import Chunker
from semsearch.cli.ingest.extract import extract_page
from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.ingest.outcomes import collect_index_outcomes
from semsearch.cli.ingest.sitemap import FetchText
from semsearch.cli.url import normalize_origin
from semsearch.share.embeddings import EmbedDocuments

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


type ProgressCallback = Callable[[IndexOutcome], None]
type IndexUrl = Callable[[str, bool], Awaitable[IndexOutcome]]
type IndexSitemap = Callable[
    [str, bool, ProgressCallback | None], Awaitable[list[IndexOutcome]]
]


async def index_url(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetch_text: FetchText,
    chunker: Chunker,
    url: str,
    force: bool = False,
) -> IndexOutcome:
    async with pool.connection() as conn:
        if not force and await db.page_exists(conn, url=url):
            return IndexOutcome(url, "skipped", "already indexed")

    page = extract_page(await fetch_text(url), url)
    if page is None:
        return IndexOutcome(url, "no_content", "no extractable article text")

    chunks = chunker(page.text)
    embed_inputs = [
        f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
        for chunk in chunks
    ]
    vectors = await embed_documents(embed_inputs)

    async with pool.connection() as conn, conn.transaction():
        site_id = await db.ensure_site_origin(conn, base_url=normalize_origin(url))
        page_id = await db.upsert_page(
            conn,
            site_id=site_id,
            url=url,
            title=page.title,
            published_at=page.published_at,
        )
        await db.replace_page_chunks(
            conn,
            page_id=page_id,
            chunks=[
                db.ChunkInsert(
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_count=chunk.char_count,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return IndexOutcome(url, "indexed", chunk_count=len(chunks))


async def index_sitemap(
    fetch_text: FetchText,
    index_url: IndexUrl,
    url: str,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
    *,
    include: str | None = None,
    exclude: str | None = None,
) -> list[IndexOutcome]:
    sitemap_urls = (
        await sitemap.discover_sitemaps(fetch_text, url)
        if sitemap.is_site_root(url)
        else [url]
    )

    page_urls: dict[str, None] = {}
    for sitemap_url in sitemap_urls:
        for page_url in await sitemap.collect_page_urls(fetch_text, sitemap_url):
            page_urls.setdefault(page_url)
    filtered = sitemap.filter_urls(list(page_urls), include=include, exclude=exclude)
    if not filtered:
        raise IngestError(
            f"No page URLs found (sitemaps tried: {', '.join(sitemap_urls)})"
        )
    logger.info("Indexing %d pages from %s", len(filtered), url)

    async def index_one(page_url: str) -> IndexOutcome:
        return await index_url(page_url, force)

    def report_progress(outcome: IndexOutcome) -> None:
        logger.info(
            "[%s] %s %s",
            outcome.status,
            outcome.url,
            outcome.detail
            or (f"({outcome.chunk_count} chunks)" if outcome.chunk_count else ""),
        )
        if on_progress is not None:
            on_progress(outcome)

    return await collect_index_outcomes(
        filtered,
        index_one,
        on_progress=report_progress,
    )
