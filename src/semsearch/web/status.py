import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from psycopg_pool import AsyncConnectionPool

from semsearch.share.status import IndexStats, fetch_index_stats
from semsearch.web.db import (
    IndexingIssue,
    RecentActivity,
    list_indexing_issues,
    list_recent_activity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatusSnapshot:
    stats: IndexStats
    activity: tuple[RecentActivity, ...]
    issues: tuple[IndexingIssue, ...]
    updated_at: datetime
    details_failed: bool = False


@dataclass
class StatusState:
    snapshot: StatusSnapshot | None = None
    refresh_failed: bool = False


async def collect_status(pool: AsyncConnectionPool) -> StatusSnapshot:
    async with asyncio.timeout(5), pool.connection() as conn:
        await conn.execute("SET LOCAL statement_timeout = '5s'")
        stats = await fetch_index_stats(conn)
    updated_at = datetime.now(UTC)
    # Detail queries must not prevent publication of fresh totals.
    try:
        async with asyncio.timeout(5), pool.connection() as conn:
            await conn.execute("SET LOCAL statement_timeout = '5s'")
            activity = tuple(await list_recent_activity(conn))
            issues = tuple(await list_indexing_issues(conn))
    except Exception:
        logger.exception("Status details refresh failed")
        return StatusSnapshot(stats, (), (), updated_at, details_failed=True)
    return StatusSnapshot(stats, activity, issues, updated_at)


async def refresh_status(
    pool: AsyncConnectionPool, state: StatusState, *, interval: float = 30
) -> None:
    """Publish cached counters and bounded details without blocking HTTP requests."""
    while True:
        try:
            async with asyncio.timeout(15):
                snapshot = await collect_status(pool)
        except Exception:
            state.refresh_failed = True
            logger.exception("Status refresh failed; retaining previous snapshot")
        else:
            state.snapshot = snapshot
            state.refresh_failed = False
        await asyncio.sleep(interval)
