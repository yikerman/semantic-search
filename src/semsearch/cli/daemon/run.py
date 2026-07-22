import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from psycopg_pool import AsyncConnectionPool

from semsearch.cli.daemon import schedule
from semsearch.cli.daemon.consumer import crawl_loop, create_crawl_job_processor
from semsearch.cli.daemon.producer import poll_loop
from semsearch.cli.ingest.chunk import Chunker
from semsearch.cli.ingest.fetch import Fetcher
from semsearch.share.config import Settings
from semsearch.share.embeddings import EmbedDocuments

logger = logging.getLogger(__name__)

DAEMON_LOCK_ID = 7_332_347_011


class DaemonAlreadyRunningError(RuntimeError):
    pass


type AdvisoryLock = Callable[
    [AsyncConnectionPool, int], AbstractAsyncContextManager[None]
]


@asynccontextmanager
async def advisory_lock(pool: AsyncConnectionPool, lock_id: int) -> AsyncIterator[None]:
    # Advisory locks are session-scoped, so the acquiring connection stays
    # checked out for the whole duration.
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
        row = await cur.fetchone()
        if row is None or not row[0]:
            raise DaemonAlreadyRunningError(
                "another semsearch daemon is already running"
            )
        await conn.commit()
        try:
            yield
        finally:
            try:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                await conn.commit()
            except Exception:
                logger.warning("Failed to release daemon advisory lock", exc_info=True)


async def run_daemon(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
    settings: Settings,
    *,
    lock: AdvisoryLock = advisory_lock,
) -> None:
    async with lock(pool, DAEMON_LOCK_ID):
        async with pool.connection() as conn, conn.transaction():
            await schedule.scatter_poll_schedule(
                conn,
                interval_seconds=settings.site_poll_interval_seconds,
            )
        logger.info("Daemon started")
        process_next = create_crawl_job_processor(
            pool=pool,
            fetch_page=fetcher.fetch_response,
            embed_documents=embed_documents,
            chunker=chunker,
        )
        async with asyncio.TaskGroup() as tasks:
            for _ in range(settings.site_poll_concurrency):
                tasks.create_task(poll_loop(pool, fetcher, settings))
            for _ in range(settings.ingest_concurrency):
                tasks.create_task(crawl_loop(process_next))
