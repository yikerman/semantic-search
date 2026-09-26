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


@dataclass
class StatusState:
    snapshot: StatusSnapshot | None = None
    refresh_failed: bool = False


async def collect_status(pool: AsyncConnectionPool) -> StatusSnapshot:
    async with pool.connection() as conn:
        stats = await fetch_index_stats(conn)
        activity = tuple(await list_recent_activity(conn))
        issues = tuple(await list_indexing_issues(conn))
    return StatusSnapshot(stats, activity, issues, datetime.now(UTC))


async def refresh_status(
    pool: AsyncConnectionPool, state: StatusState, *, interval: float = 300
) -> None:
    """Publish complete snapshots without making HTTP requests wait for scans."""
    while True:
        try:
            async with asyncio.timeout(120):
                snapshot = await collect_status(pool)
        except Exception:
            state.refresh_failed = True
            logger.exception("Status refresh failed; retaining previous snapshot")
        else:
            state.snapshot = snapshot
            state.refresh_failed = False
        await asyncio.sleep(interval)
