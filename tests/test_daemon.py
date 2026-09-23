import asyncio
from typing import Any, cast

import pytest

from semsearch.cli import daemon
from semsearch.cli.locks import AlreadyRunningError
from semsearch.share.config import Settings


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
