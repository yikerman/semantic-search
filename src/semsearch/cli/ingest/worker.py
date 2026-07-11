import asyncio
import logging
from collections.abc import Callable
from functools import partial

from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.ingest.chunk import Chunker
from semsearch.cli.ingest.fetch import Fetcher
from semsearch.cli.ingest.fetch import FetchError
from semsearch.cli.ingest.lease import run_with_lease
from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.ingest.pipeline import IngestError, ingest_job
from semsearch.cli.models import CrawlJob
from semsearch.cli.sites import poll_site_record
from semsearch.share.config import Settings
from semsearch.share.embeddings import EmbedDocuments

logger = logging.getLogger(__name__)

WORKER_LOCK_ID = 7_332_347_011
_RETRY_DELAYS = (300, 1800, 7200, 21600, 86400)
_PERMANENT_ATTEMPTS = 3
_IDLE_POLL_SECONDS = 1
_ERROR_BACKOFF_SECONDS = 5


class WorkerAlreadyRunningError(RuntimeError):
    pass


async def run_worker(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
    settings: Settings,
) -> None:
    async with pool.connection() as lock_conn:
        cur = await lock_conn.execute(
            "SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_ID,)
        )
        row = await cur.fetchone()
        if row is None or not row[0]:
            raise WorkerAlreadyRunningError(
                "another semsearch worker is already running"
            )
        await lock_conn.commit()
        try:
            async with pool.connection() as conn, conn.transaction():
                await db.scatter_poll_schedule(
                    conn,
                    interval_seconds=settings.site_poll_interval_seconds,
                )
            logger.info("Worker started")
            async with asyncio.TaskGroup() as tasks:
                for _ in range(settings.site_poll_concurrency):
                    tasks.create_task(_poll_loop(pool, fetcher, settings))
                for _ in range(settings.ingest_concurrency):
                    tasks.create_task(
                        _ingest_loop(pool, embed_documents, fetcher, chunker)
                    )
        finally:
            try:
                await lock_conn.execute(
                    "SELECT pg_advisory_unlock(%s)", (WORKER_LOCK_ID,)
                )
                await lock_conn.commit()
            except Exception:
                logger.warning("Failed to release worker advisory lock", exc_info=True)


async def drain_site_jobs(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
    site_id: int,
    on_progress: Callable[[IndexOutcome], None] | None = None,
) -> list[IndexOutcome]:
    outcomes: list[IndexOutcome] = []
    while True:
        outcome = await process_one_job(
            pool,
            embed_documents,
            fetcher,
            chunker,
            site_id=site_id,
        )
        if outcome is None:
            return outcomes
        outcomes.append(outcome)
        if on_progress is not None:
            on_progress(outcome)


async def process_one_job(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
    *,
    site_id: int | None = None,
) -> IndexOutcome | None:
    async with pool.connection() as conn, conn.transaction():
        job = await db.claim_crawl_job(conn, site_id=site_id)
    if job is None:
        return None
    try:
        return await run_with_lease(
            partial(ingest_job, pool, embed_documents, fetcher, chunker, job),
            partial(_renew_crawl_lease, pool, job),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to ingest %s", job.url)
        async with pool.connection() as conn, conn.transaction():
            if _is_permanent(exc) and job.attempt_count + 1 >= _PERMANENT_ATTEMPTS:
                await db.fail_crawl_job(
                    conn,
                    job_id=job.id,
                    lease_token=job.lease_token,
                    error=str(exc),
                )
                return IndexOutcome(job.url, "error", f"permanent: {exc}")
            delay = _RETRY_DELAYS[min(job.attempt_count, len(_RETRY_DELAYS) - 1)]
            await db.retry_crawl_job(
                conn,
                job_id=job.id,
                lease_token=job.lease_token,
                error=str(exc),
                delay_seconds=delay,
            )
        return IndexOutcome(job.url, "error", f"will retry: {exc}")


async def _poll_loop(
    pool: AsyncConnectionPool, fetcher: Fetcher, settings: Settings
) -> None:
    # A transient database error must not tear down the whole worker (the
    # TaskGroup would cancel every sibling loop), so each iteration absorbs its
    # own failures and backs off instead of propagating.
    while True:
        try:
            async with pool.connection() as conn, conn.transaction():
                claimed = await db.claim_due_site(conn)
            if claimed is None:
                await asyncio.sleep(_IDLE_POLL_SECONDS)
                continue
            site, lease_token = claimed
            try:
                outcome = await poll_site_record(
                    pool, fetcher, settings, site, lease_token
                )
                logger.info(
                    "Polled %s: %d URLs queued%s",
                    site.base_url,
                    outcome.discovered,
                    f"; {outcome.error}" if outcome.error else "",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to poll %s", site.base_url)
                async with pool.connection() as conn, conn.transaction():
                    await db.mark_poll_failed(
                        conn,
                        site_id=site.id,
                        lease_token=lease_token,
                        error=str(exc),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Poll loop error; backing off")
            await asyncio.sleep(_ERROR_BACKOFF_SECONDS)


async def _ingest_loop(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
) -> None:
    while True:
        try:
            outcome = await process_one_job(pool, embed_documents, fetcher, chunker)
        except Exception:  # noqa: BLE001
            logger.exception("Ingest loop error; backing off")
            await asyncio.sleep(_ERROR_BACKOFF_SECONDS)
            continue
        if outcome is None:
            await asyncio.sleep(_IDLE_POLL_SECONDS)
        elif outcome.status == "indexed":
            logger.info("Indexed %s (%d chunks)", outcome.url, outcome.chunk_count)


def _is_permanent(exc: Exception) -> bool:
    return isinstance(exc, IngestError) or (
        isinstance(exc, FetchError) and exc.permanent
    )


async def _renew_crawl_lease(pool: AsyncConnectionPool, job: CrawlJob) -> bool:
    async with pool.connection() as conn, conn.transaction():
        return await db.renew_crawl_lease(
            conn, job_id=job.id, lease_token=job.lease_token
        )
