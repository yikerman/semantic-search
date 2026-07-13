import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from semsearch.cli.ingest.fetch import FetchError
from semsearch.cli.ingest.lease import LeaseLostError
from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.ingest.pipeline import IngestError
from semsearch.cli.ingest.worker import (
    WORKER_LOCK_ID,
    BusySites,
    WorkerAlreadyRunningError,
    advisory_lock,
    process_one_job,
    run_worker,
)
from semsearch.cli.models import CrawlJob


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


def _run_failing_job(monkeypatch, *, attempt_count: int, error: Exception):
    job = CrawlJob(1, 2, "https://example.com/post", "feed", attempt_count, uuid4())
    failed: list[str] = []
    retries: list[int] = []

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        return job

    async def ingest(*args):
        raise error

    async def fail(conn, *, job_id, lease_token, error):
        failed.append(error)

    async def retry(conn, *, job_id, lease_token, error, delay_seconds):
        retries.append(delay_seconds)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.fail_crawl_job", fail)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.retry_crawl_job", retry)

    async def run():
        outcome = await process_one_job(
            cast(Any, FakePool()), cast(Any, None), cast(Any, None), cast(Any, None)
        )
        assert outcome is not None
        return outcome, failed, retries

    return run()


async def test_unextractable_page_is_dropped_immediately(monkeypatch):
    outcome, failed, retries = await _run_failing_job(
        monkeypatch,
        attempt_count=0,
        error=IngestError("no extractable article text"),
    )

    assert outcome.detail.startswith("dropped:")
    assert failed == ["no extractable article text"]
    assert retries == []


@pytest.mark.parametrize(
    ("attempt_count", "permanent", "dropped", "delays"),
    [
        (0, True, False, [300]),
        (2, True, True, []),
        (4, False, False, [86400]),
        (9, False, True, []),
    ],
)
async def test_fetch_failure_attempt_budget(
    monkeypatch, attempt_count, permanent, dropped, delays
):
    outcome, failed, retries = await _run_failing_job(
        monkeypatch,
        attempt_count=attempt_count,
        error=FetchError("GET failed", permanent=permanent),
    )

    assert outcome.detail.startswith("dropped:" if dropped else "will retry:")
    assert failed == (["GET failed"] if dropped else [])
    assert retries == delays


async def test_lease_loss_leaves_job_untouched(monkeypatch):
    outcome, failed, retries = await _run_failing_job(
        monkeypatch,
        attempt_count=0,
        error=LeaseLostError("database lease was lost"),
    )

    assert outcome.detail == "lease lost"
    assert failed == []
    assert retries == []


def _run_busy_site_job(monkeypatch, *, claim, busy_sites: BusySites):
    seen_busy: list[set[int]] = []

    async def ingest(pool, embed_documents, fetcher, chunker, job):
        seen_busy.append(set(busy_sites.site_ids()))
        return IndexOutcome(job.url, "indexed", chunk_count=1)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)

    async def run():
        outcome = await process_one_job(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, None),
            cast(Any, None),
            busy_sites=busy_sites,
        )
        return outcome, seen_busy

    return run()


async def test_claims_exclude_sites_other_loops_are_working(monkeypatch):
    exclusions: list[tuple[int, ...]] = []
    busy_sites = BusySites()
    busy_sites.add(4)

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        return CrawlJob(1, 7, "https://example.com/post", "feed", 0, uuid4())

    outcome, seen_busy = await _run_busy_site_job(
        monkeypatch, claim=claim, busy_sites=busy_sites
    )

    assert outcome is not None and outcome.status == "indexed"
    assert exclusions == [(4,)]
    assert seen_busy == [{4, 7}]
    assert busy_sites.site_ids() == (4,)


async def test_claim_falls_back_to_busy_sites_when_nothing_else_is_ready(monkeypatch):
    exclusions: list[tuple[int, ...]] = []
    busy_sites = BusySites()
    busy_sites.add(4)

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        if exclude_site_ids:
            return None
        return CrawlJob(1, 4, "https://example.com/post", "feed", 0, uuid4())

    outcome, seen_busy = await _run_busy_site_job(
        monkeypatch, claim=claim, busy_sites=busy_sites
    )

    assert outcome is not None and outcome.status == "indexed"
    assert exclusions == [(4,), ()]
    assert seen_busy == [{4}]


async def test_site_stays_busy_while_another_loop_still_works_it(monkeypatch):
    # Two loops share one site (the fallback allows it). The first to finish
    # must not unmark the site while the other is still fetching, or claims
    # keep piling onto the queue-head origin.
    busy_sites = BusySites()
    busy_sites.add(7)

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        if exclude_site_ids:
            return None
        return CrawlJob(1, 7, "https://example.com/post", "feed", 0, uuid4())

    outcome, seen_busy = await _run_busy_site_job(
        monkeypatch, claim=claim, busy_sites=busy_sites
    )

    assert outcome is not None and outcome.status == "indexed"
    assert busy_sites.site_ids() == (7,)


async def test_concurrent_claims_see_each_other_before_choosing_a_site(monkeypatch):
    # Claims are serialized: a loop claiming while another loop's claim is in
    # flight must wait and then exclude the site that claim just took, so
    # loops never herd onto the queue-head site at startup.
    busy_sites = BusySites()
    exclusions: list[tuple[int, ...]] = []
    first_claim_started = asyncio.Event()
    release_first_claim = asyncio.Event()
    second_claim_done = asyncio.Event()

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        if len(exclusions) == 1:
            first_claim_started.set()
            await release_first_claim.wait()
            return CrawlJob(1, 7, "https://a.example/post", "feed", 0, uuid4())
        second_claim_done.set()
        return CrawlJob(2, 9, "https://b.example/post", "feed", 0, uuid4())

    async def ingest(pool, embed_documents, fetcher, chunker, job):
        if job.site_id == 7:
            await second_claim_done.wait()
        return IndexOutcome(job.url, "indexed", chunk_count=1)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)

    async def one():
        return await process_one_job(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, None),
            cast(Any, None),
            busy_sites=busy_sites,
        )

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(one())
        await first_claim_started.wait()
        tasks.create_task(one())
        release_first_claim.set()

    assert exclusions == [(), (7,)]
    assert busy_sites.site_ids() == ()


async def test_busy_site_is_released_when_ingest_fails(monkeypatch):
    busy_sites = BusySites()
    job = CrawlJob(1, 7, "https://example.com/post", "feed", 0, uuid4())

    async def claim(conn, *, site_id=None, exclude_site_ids=()):
        return job

    async def ingest(*args):
        raise IngestError("no extractable article text")

    async def fail(conn, *, job_id, lease_token, error):
        return None

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.fail_crawl_job", fail)

    outcome = await process_one_job(
        cast(Any, FakePool()),
        cast(Any, None),
        cast(Any, None),
        cast(Any, None),
        busy_sites=busy_sites,
    )

    assert outcome is not None and outcome.detail.startswith("dropped:")
    assert busy_sites.site_ids() == ()


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

    with pytest.raises(WorkerAlreadyRunningError):
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


def _worker_settings() -> Any:
    return cast(
        Any,
        SimpleNamespace(
            site_poll_interval_seconds=3600,
            site_poll_concurrency=1,
            ingest_concurrency=1,
        ),
    )


async def test_run_worker_holds_lock_around_supervision(monkeypatch):
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

    async def claim_crawl_job(conn, *, site_id=None, exclude_site_ids=()):
        return None

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.scatter_poll_schedule", scatter)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_due_site", claim_due_site)
    monkeypatch.setattr(
        "semsearch.cli.ingest.worker.db.claim_crawl_job", claim_crawl_job
    )

    task = asyncio.create_task(
        run_worker(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, None),
            cast(Any, None),
            _worker_settings(),
            lock=lock,
        )
    )
    await loops_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [
        ("acquired", WORKER_LOCK_ID),
        ("scattered", 3600),
        ("released", WORKER_LOCK_ID),
    ]


async def test_run_worker_starts_nothing_when_lock_is_unavailable(monkeypatch):
    scattered: list[int] = []

    @asynccontextmanager
    async def lock(pool, lock_id):
        raise WorkerAlreadyRunningError("another semsearch worker is already running")
        yield

    async def scatter(conn, *, interval_seconds):
        scattered.append(interval_seconds)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.scatter_poll_schedule", scatter)

    with pytest.raises(WorkerAlreadyRunningError):
        await run_worker(
            cast(Any, FakePool()),
            cast(Any, None),
            cast(Any, None),
            cast(Any, None),
            _worker_settings(),
            lock=lock,
        )

    assert scattered == []
