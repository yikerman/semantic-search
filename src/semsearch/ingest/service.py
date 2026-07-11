import logging
from collections.abc import Callable

from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.embeddings.base import EmbedDocuments
from semsearch.ingest import sitemap
from semsearch.ingest.chunk import Chunker
from semsearch.ingest.extract import extract_page
from semsearch.ingest.fetch import Fetcher
from semsearch.ingest.models import IndexOutcome
from semsearch.ingest.outcomes import collect_index_outcomes
from semsearch.url import normalize_origin

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


class IngestService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embed_documents: EmbedDocuments,
        fetcher: Fetcher,
        chunker: Chunker,
    ) -> None:
        self.pool = pool
        self.embed_documents = embed_documents
        self.fetcher = fetcher
        self.chunker = chunker

    async def index_url(self, url: str, force: bool = False) -> IndexOutcome:
        async with self.pool.connection() as conn:
            if not force:
                if await db.page_exists(conn, url=url):
                    return IndexOutcome(url, "skipped", "already indexed")

        html = await self.fetcher.fetch_text(url)
        page = extract_page(html, url)
        if page is None:
            return IndexOutcome(url, "no_content", "no extractable article text")

        chunks = self.chunker(page.text)
        if not chunks:
            return IndexOutcome(url, "no_content", "text produced no chunks")
        embed_inputs = [
            f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
            for chunk in chunks
        ]
        vectors = await self.embed_documents(embed_inputs)

        async with self.pool.connection() as conn, conn.transaction():
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
        self,
        url: str,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
        *,
        include: str | None = None,
        exclude: str | None = None,
    ) -> list[IndexOutcome]:
        if sitemap.is_site_root(url):
            sitemap_urls = await sitemap.discover_sitemaps(self.fetcher.fetch_text, url)
        else:
            sitemap_urls = [url]

        page_urls: dict[str, None] = {}
        for sitemap_url in sitemap_urls:
            for page_url in await sitemap.collect_page_urls(
                self.fetcher.fetch_text, sitemap_url
            ):
                page_urls.setdefault(page_url)
        filtered = sitemap.filter_urls(
            list(page_urls), include=include, exclude=exclude
        )
        if not filtered:
            raise IngestError(
                f"No page URLs found (sitemaps tried: {', '.join(sitemap_urls)})"
            )
        logger.info("Indexing %d pages from %s", len(filtered), url)

        async def index_one(page_url: str) -> IndexOutcome:
            return await self.index_url(page_url, force=force)

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

        outcomes = await collect_index_outcomes(
            filtered,
            index_one,
            on_progress=report_progress,
        )
        return outcomes
