import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg_pool import AsyncConnectionPool

CRAWL_LOCK = 7_332_347_011
INDEX_LOCK = 7_332_347_012


class AlreadyRunningError(RuntimeError):
    pass


@asynccontextmanager
async def command_lock(pool: AsyncConnectionPool, lock_id: int) -> AsyncIterator[None]:
    """Hold a session lock; loss of the session cancels the owning operation."""
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
        row = await cur.fetchone()
        await conn.commit()
        if row is None or not row[0]:
            raise AlreadyRunningError("another invocation is already running")
        owner = asyncio.current_task()
        assert owner is not None
        connection_error: Exception | None = None

        async def monitor() -> None:
            nonlocal connection_error
            while True:
                await asyncio.sleep(5)
                try:
                    await conn.execute("SELECT 1")
                    await conn.commit()
                except psycopg.Error as exc:
                    connection_error = exc
                    owner.cancel()
                    return

        watcher = asyncio.create_task(monitor())
        try:
            yield
        except asyncio.CancelledError:
            if connection_error is not None:
                raise RuntimeError("command lock connection lost") from connection_error
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            if not conn.closed:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                await conn.commit()
