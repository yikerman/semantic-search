import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

import semsearch.cli.daemon as daemon_module
from semsearch.cli.daemon import (
    DAEMON_LOCK_ID,
    DaemonAlreadyRunningError,
    advisory_lock,
    run_daemon,
)


class FakeConnection(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def transaction(self):
        return self


class FakePool:
    def connection(self):
        return FakeConnection()


class FakeFetcher:
    async def fetch_response(self, url: str):
        raise AssertionError("unexpected fetch")


class LockCursor:
    def __init__(self, acquired: bool) -> None:
        self._acquired = acquired

    async def fetchone(self):
        return (self._acquired,)


class LockConnection(AbstractAsyncContextManager):
    def __init__(
        self, *, acquired: bool = True, unlock_error: Exception | None = None
    ) -> None:
        self.acquired = acquired
        self.unlock_error = unlock_error
        self.statements: list[str] = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def execute(self, query, params=()):
        self.statements.append(query)
        if "pg_advisory_unlock" in query and self.unlock_error is not None:
            raise self.unlock_error
        return LockCursor(self.acquired)

    async def commit(self):
        self.commits += 1


class LockPool:
    def __init__(self, conn: LockConnection) -> None:
        self._conn = conn

    def connection(self):
        return self._conn


async def test_advisory_lock_acquires_then_releases():
    conn = LockConnection()

    async with advisory_lock(cast(Any, LockPool(conn)), 42):
        assert any("pg_try_advisory_lock" in s for s in conn.statements)
        assert not any("pg_advisory_unlock" in s for s in conn.statements)

    assert any("pg_advisory_unlock" in s for s in conn.statements)
    assert conn.commits == 2


async def test_advisory_lock_raises_when_already_held():
    conn = LockConnection(acquired=False)

    with pytest.raises(DaemonAlreadyRunningError):
        async with advisory_lock(cast(Any, LockPool(conn)), 42):
            raise AssertionError("body must not run")

    assert not any("pg_advisory_unlock" in s for s in conn.statements)


async def test_advisory_lock_releases_when_body_raises():
    conn = LockConnection()

    with pytest.raises(ValueError, match="boom"):
        async with advisory_lock(cast(Any, LockPool(conn)), 42):
            raise ValueError("boom")

    assert any("pg_advisory_unlock" in s for s in conn.statements)


async def test_advisory_lock_swallows_unlock_failure():
    conn = LockConnection(unlock_error=RuntimeError("connection lost"))

    async with advisory_lock(cast(Any, LockPool(conn)), 42):
        pass


def daemon_settings() -> Any:
    return cast(
        Any,
        SimpleNamespace(
            site_poll_interval_seconds=3600,
            site_poll_concurrency=1,
            ingest_concurrency=1,
        ),
    )


async def test_run_daemon_holds_lock_around_supervision(monkeypatch):
    events: list[tuple[str, int]] = []
    loops_started = asyncio.Event()

    @asynccontextmanager
    async def lock(pool, lock_id):
        events.append(("acquired", lock_id))
        try:
            yield
        finally:
            events.append(("released", lock_id))

    async def scatter(conn, *, interval_seconds):
        events.append(("scattered", interval_seconds))

    async def claim_due_site(conn):
        loops_started.set()
        return None

    async def process_next():
        return None

    monkeypatch.setattr("semsearch.cli.daemon.db.scatter_poll_schedule", scatter)
    monkeypatch.setattr("semsearch.cli.daemon.db.claim_due_site", claim_due_site)
    monkeypatch.setattr(
        "semsearch.cli.daemon.create_crawl_job_processor",
        lambda **kwargs: process_next,
    )

    task = asyncio.create_task(
        run_daemon(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, FakeFetcher()),
            cast(Any, None),
            daemon_settings(),
            lock=lock,
        )
    )
    await loops_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [
        ("acquired", DAEMON_LOCK_ID),
        ("scattered", 3600),
        ("released", DAEMON_LOCK_ID),
    ]


async def test_run_daemon_starts_nothing_when_lock_is_unavailable(monkeypatch):
    scattered: list[int] = []

    @asynccontextmanager
    async def lock(pool, lock_id):
        raise DaemonAlreadyRunningError("another semsearch daemon is already running")
        yield

    async def scatter(conn, *, interval_seconds):
        scattered.append(interval_seconds)

    monkeypatch.setattr("semsearch.cli.daemon.db.scatter_poll_schedule", scatter)

    with pytest.raises(DaemonAlreadyRunningError):
        await run_daemon(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, FakeFetcher()),
            cast(Any, None),
            daemon_settings(),
            lock=lock,
        )

    assert scattered == []


@pytest.mark.parametrize(
    ("result", "expected_delay"), [(None, 1), (RuntimeError("db"), 5)]
)
async def test_crawl_loop_uses_idle_and_error_backoff(
    monkeypatch, result, expected_delay
):
    delays: list[int] = []

    async def process_next():
        if isinstance(result, Exception):
            raise result
        return result

    async def sleep(delay):
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(daemon_module.asyncio, "sleep", sleep)

    with pytest.raises(asyncio.CancelledError):
        await daemon_module._crawl_loop(process_next)

    assert delays == [expected_delay]
