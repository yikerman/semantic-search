import asyncio
import signal
import sys
from typing import Any, cast

import pytest

from semsearch.cli import daemon
from semsearch.cli.locks import AlreadyRunningError
from semsearch.share.config import Settings


class ChildProcess:
    pid = 123

    def __init__(self, code=None, *, stuck=False):
        self.returncode = code
        self.stuck = stuck
        self.finished = asyncio.Event()
        self.terminated = False
        if code is not None:
            self.finished.set()

    async def wait(self):
        await self.finished.wait()
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stuck:
            self.returncode = 143
            self.finished.set()


@pytest.mark.parametrize("code", [0, 1])
async def test_crawl_child_receives_settings_and_propagates_failure(monkeypatch, code):
    child = ChildProcess(code)

    async def spawn(*args, **kwargs):
        assert args == (sys.executable, "-m", "semsearch.cli.app", "crawl")
        assert kwargs["env"]["CRAWL_ARTICLE_LIMIT"] == "123"
        assert kwargs["env"]["DATABASE_URL"] == "postgresql://test/test"
        assert kwargs["start_new_session"]
        return child

    monkeypatch.setattr(daemon.asyncio, "create_subprocess_exec", spawn)
    settings = Settings(crawl_article_limit=123, database_url="postgresql://test/test")
    if code:
        with pytest.raises(RuntimeError, match="crawl process exited with status 1"):
            await daemon.run_crawl(settings)
    else:
        await daemon.run_crawl(settings)
    assert not child.terminated


@pytest.mark.parametrize("stuck", [False, True])
async def test_crawl_cancellation_reaps_child_and_kills_stuck_group(monkeypatch, stuck):
    child = ChildProcess(stuck=stuck)
    started = asyncio.Event()
    killed = []

    async def spawn(*args, **kwargs):
        started.set()
        return child

    async def shutdown_wait(awaitable, *, timeout):
        assert timeout == 60
        if stuck:
            awaitable.close()
            raise TimeoutError
        return await awaitable

    def killpg(pid, sig):
        killed.append((pid, sig))
        child.returncode = -9
        child.finished.set()

    monkeypatch.setattr(daemon.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(daemon.asyncio, "wait_for", shutdown_wait)
    monkeypatch.setattr(daemon.os, "killpg", killpg)
    task = asyncio.create_task(daemon.run_crawl(Settings()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child.terminated
    assert killed == ([(child.pid, signal.SIGKILL)] if stuck else [])


async def test_repeat_waits_after_success_and_lock_contention(monkeypatch):
    calls = 0
    waits = []

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AlreadyRunningError
        if calls == 3:
            raise RuntimeError("job failed")

    async def sleep(interval):
        waits.append(interval)

    monkeypatch.setattr(daemon.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="job failed"):
        await daemon.repeat("test", operation, 17)
    assert calls == 3
    assert waits == [17, 17]


async def test_jobs_run_independently_and_cancel_cleanly(monkeypatch):
    crawling = asyncio.Event()
    indexed_twice = asyncio.Event()
    cleaned = set()
    index_calls = 0

    async def crawl(*args):
        crawling.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.add("crawl")

    async def index(*args):
        nonlocal index_calls
        await crawling.wait()
        index_calls += 1
        if index_calls == 2:
            indexed_twice.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.add("index")

    async def sleep(interval):
        assert interval == 13

    monkeypatch.setattr(daemon, "run_crawl", crawl)
    monkeypatch.setattr(daemon, "run_index", index)
    monkeypatch.setattr(daemon.asyncio, "sleep", sleep)
    task = asyncio.create_task(
        daemon.run_daemon(cast(Any, object()), Settings(index_interval_seconds=13))
    )
    try:
        await asyncio.wait_for(indexed_twice.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleaned == {"crawl", "index"}


async def test_unexpected_job_failure_stops_sibling(monkeypatch):
    crawling = asyncio.Event()
    stopped = asyncio.Event()

    async def crawl(*args):
        crawling.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def index(*args):
        await crawling.wait()
        raise RuntimeError("database lost")

    monkeypatch.setattr(daemon, "run_crawl", crawl)
    monkeypatch.setattr(daemon, "run_index", index)
    with pytest.raises(ExceptionGroup) as caught:
        await asyncio.wait_for(
            daemon.run_daemon(cast(Any, object()), Settings()), timeout=1
        )
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert stopped.is_set()
