import asyncio
import logging

from psycopg_pool import AsyncConnectionPool

from semsearch.cli.daemon import schedule
from semsearch.cli.daemon.lease import LeaseLostError
from semsearch.cli.ingest.fetch import Fetcher
from semsearch.cli.sites import poll_site_record
from semsearch.share.config import Settings

logger = logging.getLogger(__name__)

_IDLE_POLL_SECONDS = 1
_ERROR_BACKOFF_SECONDS = 5


async def poll_loop(
    pool: AsyncConnectionPool, fetcher: Fetcher, settings: Settings
) -> None:
    # A transient database error must not tear down the daemon because the
    # TaskGroup would cancel every sibling loop.
    while True:
        try:
            async with pool.connection() as conn, conn.transaction():
                claimed = await schedule.claim_due_site(conn)
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
            except LeaseLostError:
                logger.warning(
                    "Poll lease lost for %s; leaving it to the new owner",
                    site.base_url,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to poll %s", site.base_url)
                async with pool.connection() as conn, conn.transaction():
                    await schedule.mark_poll_failed(
                        conn,
                        site_id=site.id,
                        lease_token=lease_token,
                        error=str(exc),
                        interval_seconds=settings.site_poll_interval_seconds,
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Poll loop error; backing off")
            await asyncio.sleep(_ERROR_BACKOFF_SECONDS)
