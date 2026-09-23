import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest

from semsearch.cli import locks


class LockConnection:
    def __init__(self, available=True):
        self.available = available
        self.closed = False
        self.unlocked = False
        self.fail_monitor = False

    async def execute(self, query, params=()):
        if query == "SELECT 1" and self.fail_monitor:
            self.closed = True
            raise psycopg.OperationalError("session lost")
        if "pg_advisory_unlock" in query:
            self.unlocked = True
        return self

    async def fetchone(self):
        return (self.available,)

    async def commit(self):
        pass

    @asynccontextmanager
    async def connection(self):
        yield self


async def test_overlap_skips_work():
    conn = LockConnection(available=False)
    with pytest.raises(locks.AlreadyRunningError):
        async with locks.command_lock(cast(Any, conn), locks.CRAWL_LOCK):
            pytest.fail("overlapping command started")


async def test_cancellation_releases_lock():
    conn = LockConnection()
    with pytest.raises(asyncio.CancelledError):
        async with locks.command_lock(cast(Any, conn), locks.CRAWL_LOCK):
            raise asyncio.CancelledError
    assert conn.unlocked


async def test_lock_session_loss_cancels_owner(monkeypatch):
    conn = LockConnection()
    conn.fail_monitor = True

    async def tick(delay):
        pass

    monkeypatch.setattr(
        locks,
        "asyncio",
        SimpleNamespace(
            sleep=tick,
            current_task=asyncio.current_task,
            create_task=asyncio.create_task,
            gather=asyncio.gather,
            CancelledError=asyncio.CancelledError,
        ),
    )

    async def command():
        async with locks.command_lock(cast(Any, conn), locks.CRAWL_LOCK):
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="lock connection lost"):
        await asyncio.wait_for(command(), timeout=1)
