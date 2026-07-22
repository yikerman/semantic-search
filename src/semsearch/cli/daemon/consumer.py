import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import psycopg
from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.daemon import queue
from semsearch.cli.daemon.lease import LeaseLostError, run_with_lease
from semsearch.cli.ingest.chunk import Chunker
from semsearch.cli.ingest.extract import extract_page
from semsearch.cli.ingest.fetch import FetchError, FetchResponse
from semsearch.cli.models import CrawlAttempt
from semsearch.cli.url import same_site
from semsearch.share.embeddings import EmbedDocuments, EmbeddingError

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (300, 1800, 7200, 21600, 86400)
_FLAKY_FETCH_ATTEMPTS = 3
_TRANSIENT_ATTEMPTS = 10
_IDLE_POLL_SECONDS = 1
_ERROR_BACKOFF_SECONDS = 5

type CrawlAttemptStatus = Literal[
    "indexed", "skipped", "retrying", "failed", "lease_lost"
]
type FetchPage = Callable[[str], Awaitable[FetchResponse]]


@dataclass(frozen=True, slots=True)
class CrawlAttemptOutcome:
    url: str
    status: CrawlAttemptStatus
    detail: str = ""
    chunk_count: int = 0


type ProcessNextCrawlJob = Callable[[], Awaitable[CrawlAttemptOutcome | None]]


class _ContentError(RuntimeError):
    pass


class _BusySites:
    def __init__(self) -> None:
        self.claim_lock = asyncio.Lock()
        self._counts: Counter[int] = Counter()

    def site_ids(self) -> tuple[int, ...]:
        return tuple(self._counts)

    def add(self, site_id: int) -> None:
        self._counts[site_id] += 1

    def remove(self, site_id: int) -> None:
        self._counts[site_id] -= 1
        if not self._counts[site_id]:
            del self._counts[site_id]


def create_crawl_job_processor(
    *,
    pool: AsyncConnectionPool,
    fetch_page: FetchPage,
    embed_documents: EmbedDocuments,
    chunker: Chunker,
) -> ProcessNextCrawlJob:
    busy_sites = _BusySites()
    return partial(
        _process_next_crawl_job,
        pool,
        fetch_page,
        embed_documents,
        chunker,
        busy_sites,
    )


async def _process_next_crawl_job(
    pool: AsyncConnectionPool,
    fetch_page: FetchPage,
    embed_documents: EmbedDocuments,
    chunker: Chunker,
    busy_sites: _BusySites,
) -> CrawlAttemptOutcome | None:
    async with busy_sites.claim_lock:
        excluded_sites = busy_sites.site_ids()
        async with pool.connection() as conn, conn.transaction():
            attempt = await queue.claim_crawl_job(conn, exclude_site_ids=excluded_sites)
            if attempt is None and excluded_sites:
                attempt = await queue.claim_crawl_job(conn)
        if attempt is None:
            return None
        busy_sites.add(attempt.site_id)

    try:
        return await _run_crawl_attempt(
            pool,
            fetch_page,
            embed_documents,
            chunker,
            attempt,
        )
    finally:
        busy_sites.remove(attempt.site_id)


async def _run_crawl_attempt(
    pool: AsyncConnectionPool,
    fetch_page: FetchPage,
    embed_documents: EmbedDocuments,
    chunker: Chunker,
    attempt: CrawlAttempt,
) -> CrawlAttemptOutcome:
    try:
        return await run_with_lease(
            partial(
                _ingest_crawl_attempt,
                pool,
                fetch_page,
                embed_documents,
                chunker,
                attempt,
            ),
            partial(_renew_crawl_lease, pool, attempt),
        )
    except LeaseLostError:
        return _lease_lost_outcome(attempt)
    except Exception as exc:  # noqa: BLE001
        attempt_number = attempt.attempt_count + 1
        failed = attempt_number >= _attempt_budget(exc)
        if isinstance(exc, (FetchError, _ContentError, EmbeddingError)):
            logger.warning(
                "%s %s (attempt %d): %s",
                "Failing" if failed else "Will retry",
                attempt.url,
                attempt_number,
                exc,
            )
        else:
            logger.exception(
                "Failed to ingest %s (attempt %d)", attempt.url, attempt_number
            )

        async with pool.connection() as conn, conn.transaction():
            if failed:
                transitioned = await queue.fail_crawl_job(
                    conn,
                    job_id=attempt.id,
                    lease_token=attempt.lease_token,
                    error=str(exc),
                )
                if not transitioned:
                    return _lease_lost_outcome(attempt)
                return CrawlAttemptOutcome(attempt.url, "failed", str(exc))

            delay = _RETRY_DELAYS[min(attempt.attempt_count, len(_RETRY_DELAYS) - 1)]
            transitioned = await queue.retry_crawl_job(
                conn,
                job_id=attempt.id,
                lease_token=attempt.lease_token,
                error=str(exc),
                delay_seconds=delay,
            )
            if not transitioned:
                return _lease_lost_outcome(attempt)
        return CrawlAttemptOutcome(attempt.url, "retrying", str(exc))


async def _ingest_crawl_attempt(
    pool: AsyncConnectionPool,
    fetch_page: FetchPage,
    embed_documents: EmbedDocuments,
    chunker: Chunker,
    attempt: CrawlAttempt,
) -> CrawlAttemptOutcome:
    async with pool.connection() as conn, conn.transaction():
        if await db.page_exists(conn, url=attempt.url):
            await _complete_owned_attempt(conn, attempt)
            return CrawlAttemptOutcome(attempt.url, "skipped", "already indexed")

    response = await fetch_page(attempt.url)
    if not same_site(response.url, attempt.url):
        raise _ContentError("page redirected to a different origin")
    page = await asyncio.to_thread(extract_page, response.text, response.url)
    if page is None:
        raise _ContentError("no extractable article text")

    chunks = await asyncio.to_thread(chunker, page.text)
    if not chunks:
        raise _ContentError("article produced no chunks")
    vectors = await embed_documents(
        [
            f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
            for chunk in chunks
        ]
    )

    async with pool.connection() as conn, conn.transaction():
        await _complete_owned_attempt(conn, attempt)
        if await db.page_exists(conn, url=attempt.url):
            return CrawlAttemptOutcome(attempt.url, "skipped", "already indexed")
        page_id = await db.insert_page(
            conn,
            site_id=attempt.site_id,
            url=attempt.url,
            title=page.title,
            content=page.text,
            published_at=page.published_at,
            language=page.language,
        )
        if page_id is None:
            return CrawlAttemptOutcome(attempt.url, "skipped", "already indexed")
        await db.insert_page_chunks(
            conn,
            page_id=page_id,
            chunks=[
                db.ChunkInsert(
                    start_offset=chunk.start_offset,
                    content=chunk.content,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    return CrawlAttemptOutcome(attempt.url, "indexed", chunk_count=len(chunks))


async def _complete_owned_attempt(
    conn: psycopg.AsyncConnection, attempt: CrawlAttempt
) -> None:
    completed = await queue.complete_crawl_job(
        conn,
        job_id=attempt.id,
        lease_token=attempt.lease_token,
    )
    if not completed:
        raise LeaseLostError("database lease was lost")


def _attempt_budget(exc: Exception) -> int:
    if isinstance(exc, _ContentError):
        return 1
    if isinstance(exc, FetchError) and exc.permanent:
        return _FLAKY_FETCH_ATTEMPTS
    return _TRANSIENT_ATTEMPTS


def _lease_lost_outcome(attempt: CrawlAttempt) -> CrawlAttemptOutcome:
    logger.warning("Lease no longer owned for %s; leaving job untouched", attempt.url)
    return CrawlAttemptOutcome(attempt.url, "lease_lost", "lease lost")


async def _renew_crawl_lease(pool: AsyncConnectionPool, attempt: CrawlAttempt) -> bool:
    async with pool.connection() as conn, conn.transaction():
        return await queue.renew_crawl_lease(
            conn, job_id=attempt.id, lease_token=attempt.lease_token
        )


async def crawl_loop(process_next: ProcessNextCrawlJob) -> None:
    while True:
        try:
            outcome = await process_next()
        except Exception:  # noqa: BLE001
            logger.exception("Crawl loop error; backing off")
            await asyncio.sleep(_ERROR_BACKOFF_SECONDS)
            continue
        if outcome is None:
            await asyncio.sleep(_IDLE_POLL_SECONDS)
        elif outcome.status == "indexed":
            logger.info("Indexed %s (%d chunks)", outcome.url, outcome.chunk_count)
        elif outcome.status == "skipped":
            logger.info("Skipped %s: %s", outcome.url, outcome.detail)
