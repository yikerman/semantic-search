import logging
from collections.abc import Callable

import psycopg
from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.config import Settings
from semsearch.embeddings.base import EmbeddingProvider
from semsearch.ingest import sitemap
from semsearch.ingest.chunk import CharChunker
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
        embedder: EmbeddingProvider,
        settings: Settings,
        *,
        fetcher: Fetcher | None = None,
        chunker: CharChunker | None = None,
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.settings = settings
        self.fetcher = fetcher or Fetcher(
            user_agent=settings.user_agent,
            timeout=settings.fetch_timeout_seconds,
            delay_seconds=settings.fetch_delay_seconds,
            impersonate=settings.fetch_impersonate,
        )
        self.chunker = chunker or CharChunker(
            chunk_chars=settings.chunk_chars,
            chunk_overlap=settings.chunk_overlap,
        )
        self._meta_checked = False

    async def index_url(self, url: str, *, force: bool = False) -> IndexOutcome:
        async with self.pool.connection() as conn:
            await self._ensure_meta(conn)
            if not force:
                if await db.page_exists(conn, url=url):
                    return IndexOutcome(url, "skipped", "already indexed")

        html = await self.fetcher.fetch_text(url)
        page = extract_page(html, url)
        if page is None:
            return IndexOutcome(url, "no_content", "no extractable article text")

        chunks = self.chunker.chunk(page.text)
        if not chunks:
            return IndexOutcome(url, "no_content", "text produced no chunks")
        embed_inputs = [
            f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
            for chunk in chunks
        ]
        vectors = await self.embedder.embed_documents(embed_inputs)

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
                    (
                        chunk.chunk_index,
                        chunk.content,
                        chunk.char_count,
                        vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        return IndexOutcome(url, "indexed", chunk_count=len(chunks))

    async def index_sitemap(
        self,
        url: str,
        *,
        include: str | None = None,
        exclude: str | None = None,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
    ) -> list[IndexOutcome]:
        if sitemap.is_site_root(url):
            sitemap_urls = await sitemap.discover_sitemaps(self.fetcher, url)
        else:
            sitemap_urls = [url]

        page_urls: dict[str, None] = {}
        for sitemap_url in sitemap_urls:
            for page_url in await sitemap.collect_page_urls(self.fetcher, sitemap_url):
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

    async def aclose(self) -> None:
        await self.fetcher.aclose()

    async def _ensure_meta(self, conn: psycopg.AsyncConnection) -> None:
        if not self._meta_checked:
            await db.check_index_meta(conn, self.settings)
            self._meta_checked = True
