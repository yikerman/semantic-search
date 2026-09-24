import asyncio
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from functools import partial

from psycopg_pool import AsyncConnectionPool

from semsearch.cli.index import run_index
from semsearch.cli.locks import AlreadyRunningError
from semsearch.share.config import Settings

logger = logging.getLogger(__name__)


async def run_crawl(settings: Settings) -> None:
    """Keep Scrapy's synchronous scheduling work off the indexer's event loop."""
    crawl_env: dict[str, str] = os.environ.copy()
    crawl_env.update(
        {key.upper(): str(value) for key, value in settings.model_dump().items()}
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "semsearch.cli.app",
        "crawl",
        env=crawl_env,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = await process.wait()
        if code:
            raise RuntimeError(f"crawl process exited with status {code}")
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=60)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


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
                partial(run_crawl, settings),
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
