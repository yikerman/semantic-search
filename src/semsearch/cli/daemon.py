import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial

from psycopg_pool import AsyncConnectionPool

from semsearch.cli.crawl.run import run_crawl
from semsearch.cli.index import run_index
from semsearch.cli.locks import AlreadyRunningError
from semsearch.share.config import Settings

logger = logging.getLogger(__name__)


async def repeat(
    name: str, operation: Callable[[], Awaitable[object]], interval: int
) -> None:
    """Run immediately, then wait between completed invocations; never overlap."""
    while True:
        try:
            await operation()
        except AlreadyRunningError:
            logger.info("Skipping %s: another invocation is running", name)
        # Unexpected failures propagate: cancel sibling jobs and let the
        # container restart the process instead of leaving a partial daemon.
        await asyncio.sleep(interval)


async def run_daemon(pool: AsyncConnectionPool, settings: Settings) -> None:
    """Own recurring jobs; each operation owns its resources and advisory lock."""
    async with asyncio.TaskGroup() as jobs:
        jobs.create_task(
            repeat(
                "crawl",
                partial(run_crawl, pool, settings),
                settings.crawl_interval_seconds,
            ),
            name="crawl",
        )
        jobs.create_task(
            repeat(
                "index",
                partial(run_index, pool, settings),
                settings.index_interval_seconds,
            ),
            name="index",
        )
