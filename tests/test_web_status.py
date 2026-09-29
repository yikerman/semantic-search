import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from semsearch.share.status import IndexStats
from semsearch.web import app as web_app
from semsearch.web import status
from semsearch.web.app import create_app


def snapshot(count):
    return status.StatusSnapshot(
        IndexStats(1, count, count, 0, 0, 0),
        (),
        (),
        datetime(2026, 9, 26, 12, 0, tzinfo=UTC),
    )


@pytest.mark.parametrize("populated", [False, True])
async def test_status_http_does_not_wait_for_refresh(monkeypatch, populated):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_collect(pool):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(status, "collect_status", slow_collect)
    app = create_app()
    state = app.state.status
    if populated:
        state.snapshot = snapshot(123)
    task = asyncio.create_task(status.refresh_status(cast(Any, object()), state))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await asyncio.wait_for(client.get("/status"), timeout=2)
        assert response.status_code == 200
        assert not task.done()
        if populated:
            assert "<td>123</td>" in response.text
            assert "Updated" in response.text
        else:
            assert "Collecting status" in response.text
            assert "<caption>Index totals</caption>" not in response.text
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


async def test_refresh_retains_snapshot_on_failure_then_recovers(monkeypatch):
    state = status.StatusState()
    first, second = snapshot(123), snapshot(456)
    calls = 0
    waits = []

    async def collect(pool):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("database unavailable")
        return first if calls == 1 else second

    async def sleep(interval):
        waits.append(interval)
        if calls == 1:
            assert state.snapshot is first and not state.refresh_failed
        elif calls == 2:
            assert state.snapshot is first and state.refresh_failed
        else:
            assert state.snapshot is second and not state.refresh_failed
            raise asyncio.CancelledError

    monkeypatch.setattr(status, "collect_status", collect)
    monkeypatch.setattr(status.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await status.refresh_status(cast(Any, object()), state, interval=17)
    assert waits == [17, 17, 17]


@pytest.mark.parametrize("populated", [False, True])
async def test_status_reports_failed_refresh_without_inventing_totals(populated):
    app = create_app()
    app.state.status.refresh_failed = True
    if populated:
        app.state.status.snapshot = snapshot(123)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/status")
    assert response.status_code == 200
    assert "Status could not be refreshed" in response.text
    if populated:
        assert "Showing the last available update" in response.text
        assert "<td>123</td>" in response.text
    else:
        assert "Retrying shortly" in response.text
        assert "<caption>Index totals</caption>" not in response.text


async def test_refresh_timeout_preserves_previous_snapshot(monkeypatch):
    previous = snapshot(123)
    state = status.StatusState(snapshot=previous)
    timeout = asyncio.timeout
    cancelled = asyncio.Event()

    async def collect(pool):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def sleep(interval):
        assert state.snapshot is previous
        assert state.refresh_failed
        assert cancelled.is_set()
        raise asyncio.CancelledError

    monkeypatch.setattr(status, "collect_status", collect)
    monkeypatch.setattr(status.asyncio, "timeout", lambda _: timeout(0))
    monkeypatch.setattr(status.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await status.refresh_status(cast(Any, object()), state)


async def test_lifespan_starts_refresh_and_cancels_before_closing_pool(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()
    languages_stopped = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def pool(settings):
        try:
            yield object()
        finally:
            closed.set()

    @asynccontextmanager
    async def embeddings(settings):
        yield SimpleNamespace(embed_query=object())

    async def refresh(pool, state):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert not closed.is_set()
            stopped.set()

    async def refresh_languages(pool, state):
        try:
            await asyncio.Event().wait()
        finally:
            assert not closed.is_set()
            languages_stopped.set()

    monkeypatch.setattr(web_app, "refresh_languages", refresh_languages)
    monkeypatch.setattr(web_app, "create_pool", pool)
    monkeypatch.setattr(web_app, "create_embeddings", embeddings)
    monkeypatch.setattr(web_app, "refresh_status", refresh)
    app = create_app()
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not stopped.is_set()
    assert stopped.is_set()
    assert languages_stopped.is_set()
    assert closed.is_set()


@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_detail_failure_still_publishes_fresh_totals(monkeypatch, failure):
    class Connection:
        async def execute(self, query):
            assert query == "SET LOCAL statement_timeout = '5s'"

    class Pool:
        @asynccontextmanager
        async def connection(self):
            yield Connection()

    async def stats(conn):
        return IndexStats(1, 123, 120, 0, 0, 0)

    async def activity(conn):
        if failure == "timeout":
            raise TimeoutError
        raise OSError("details unavailable")

    monkeypatch.setattr(status, "fetch_index_stats", stats)
    monkeypatch.setattr(status, "list_recent_activity", activity)
    result = await status.collect_status(cast(Any, Pool()))
    assert result.stats.page_count == 123
    assert result.details_failed
    app = create_app()
    app.state.status.snapshot = result
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/status")
    assert "<td>123</td>" in response.text
    assert "Recent activity could not be refreshed" in response.text
    assert "No recent activity" not in response.text
    assert "Status could not be refreshed" not in response.text
