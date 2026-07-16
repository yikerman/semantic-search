import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from uuid import uuid4

import pytest

from semsearch.cli.ingest import crawl_jobs
from semsearch.cli.ingest.chunk import Chunk
from semsearch.cli.ingest.crawl_jobs import create_crawl_job_processor
from semsearch.cli.ingest.extract import ExtractedPage
from semsearch.cli.ingest.fetch import FetchError, FetchResponse
from semsearch.cli.ingest.lease import LeaseLostError
from semsearch.cli.models import CrawlAttempt


class Connection(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.in_transaction = False

    async def __aenter__(self):
        self.in_transaction = True
        return self

    async def __aexit__(self, *exc_info):
        self.in_transaction = False

    def transaction(self):
        return self


class Pool:
    def __init__(self) -> None:
        self.conn = Connection()

    def connection(self):
        return self.conn


def attempt(*, attempt_count: int = 0, site_id: int = 2, job_id: int = 1):
    return CrawlAttempt(
        job_id,
        site_id,
        "https://example.com/post",
        "feed",
        attempt_count,
        uuid4(),
    )


def response(url: str = "https://example.com/post") -> FetchResponse:
    return FetchResponse(b"<html></html>", {"content-type": "text/html"}, 200, url)


async def embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


def processor(pool: Pool, fetch_page):
    return create_crawl_job_processor(
        pool=cast(Any, pool),
        fetch_page=fetch_page,
        embed_documents=embed,
        chunker=lambda text: [Chunk(0, text, len(text))],
    )


async def test_idle_queue_returns_none_without_fetching(monkeypatch):
    async def claim(conn, *, exclude_site_ids=()):
        return None

    fetched = False

    async def fetch_page(url):
        nonlocal fetched
        fetched = True
        return response(url)

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)

    assert await processor(Pool(), fetch_page)() is None
    assert not fetched


async def test_indexed_attempt_fences_before_atomic_page_and_chunk_writes(
    monkeypatch,
):
    claimed = attempt()
    exists = iter((False, False))
    writes: list[str] = []

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def page_exists(conn, *, url):
        return next(exists)

    async def complete(conn, *, job_id, lease_token):
        assert conn.in_transaction and lease_token == claimed.lease_token
        writes.append("complete")
        return True

    async def insert_page(conn, **kwargs):
        assert conn.in_transaction and kwargs["language"] == "en"
        writes.append("page")
        return 3

    async def insert_chunks(conn, **kwargs):
        assert conn.in_transaction
        writes.append("chunks")

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs.db, "page_exists", page_exists)
    monkeypatch.setattr(crawl_jobs.db, "complete_crawl_job", complete)
    monkeypatch.setattr(crawl_jobs.db, "insert_page", insert_page)
    monkeypatch.setattr(crawl_jobs.db, "insert_page_chunks", insert_chunks)
    monkeypatch.setattr(
        crawl_jobs,
        "extract_page",
        lambda html, url: ExtractedPage("Title", "body", None, "en"),
    )

    outcome = await processor(
        Pool(), lambda url: asyncio.sleep(0, result=response(url))
    )()

    assert outcome is not None and outcome.status == "indexed"
    assert outcome.chunk_count == 1
    assert writes == ["complete", "page", "chunks"]


async def test_existing_page_skips_fetch_after_fenced_completion(monkeypatch):
    claimed = attempt()
    fetched = False

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def page_exists(conn, *, url):
        return True

    async def complete(conn, *, job_id, lease_token):
        return True

    async def fetch_page(url):
        nonlocal fetched
        fetched = True
        return response(url)

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs.db, "page_exists", page_exists)
    monkeypatch.setattr(crawl_jobs.db, "complete_crawl_job", complete)

    outcome = await processor(Pool(), fetch_page)()

    assert outcome is not None and outcome.status == "skipped"
    assert not fetched


async def test_concurrent_page_insert_skips_without_writing_chunks(monkeypatch):
    claimed = attempt()
    chunks_written = False

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def page_exists(conn, *, url):
        return False

    async def complete(conn, *, job_id, lease_token):
        return True

    async def insert_page(conn, **kwargs):
        return None

    async def insert_chunks(conn, **kwargs):
        nonlocal chunks_written
        chunks_written = True

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs.db, "page_exists", page_exists)
    monkeypatch.setattr(crawl_jobs.db, "complete_crawl_job", complete)
    monkeypatch.setattr(crawl_jobs.db, "insert_page", insert_page)
    monkeypatch.setattr(crawl_jobs.db, "insert_page_chunks", insert_chunks)
    monkeypatch.setattr(
        crawl_jobs,
        "extract_page",
        lambda html, url: ExtractedPage("Title", "body", None, "en"),
    )

    outcome = await processor(
        Pool(), lambda url: asyncio.sleep(0, result=response(url))
    )()

    assert outcome is not None and outcome.status == "skipped"
    assert not chunks_written


async def test_stale_completion_returns_lease_lost_before_page_writes(monkeypatch):
    claimed = attempt()
    inserted = False

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def page_exists(conn, *, url):
        return False

    async def complete(conn, *, job_id, lease_token):
        return False

    async def insert_page(conn, **kwargs):
        nonlocal inserted
        inserted = True
        return 3

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs.db, "page_exists", page_exists)
    monkeypatch.setattr(crawl_jobs.db, "complete_crawl_job", complete)
    monkeypatch.setattr(crawl_jobs.db, "insert_page", insert_page)
    monkeypatch.setattr(
        crawl_jobs,
        "extract_page",
        lambda html, url: ExtractedPage("Title", "body", None, "en"),
    )

    outcome = await processor(
        Pool(), lambda url: asyncio.sleep(0, result=response(url))
    )()

    assert outcome is not None and outcome.status == "lease_lost"
    assert not inserted


async def _run_failure(
    monkeypatch,
    *,
    error: Exception | None = None,
    attempt_count: int = 0,
    transition_succeeds: bool = True,
):
    claimed = attempt(attempt_count=attempt_count)
    retries: list[int] = []
    failures: list[str] = []

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def fetch_page(url):
        if error is not None:
            raise error
        return response(url)

    async def retry(conn, *, job_id, lease_token, error, delay_seconds):
        retries.append(delay_seconds)
        return transition_succeeds

    async def fail(conn, *, job_id, lease_token, error):
        failures.append(error)
        return transition_succeeds

    async def page_exists(conn, *, url):
        return False

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs.db, "page_exists", page_exists)
    monkeypatch.setattr(crawl_jobs.db, "retry_crawl_job", retry)
    monkeypatch.setattr(crawl_jobs.db, "fail_crawl_job", fail)
    if error is None:
        monkeypatch.setattr(crawl_jobs, "extract_page", lambda html, url: None)

    outcome = await processor(Pool(), fetch_page)()
    assert outcome is not None
    return outcome, retries, failures


async def test_unextractable_page_fails_immediately(monkeypatch):
    outcome, retries, failures = await _run_failure(monkeypatch)

    assert outcome.status == "failed"
    assert retries == []
    assert failures == ["no extractable article text"]


@pytest.mark.parametrize(
    ("attempt_count", "permanent", "status", "delays"),
    [
        (0, True, "retrying", [300]),
        (2, True, "failed", []),
        (4, False, "retrying", [86400]),
        (9, False, "failed", []),
    ],
)
async def test_fetch_failure_attempt_budget(
    monkeypatch, attempt_count, permanent, status, delays
):
    outcome, retries, _failures = await _run_failure(
        monkeypatch,
        attempt_count=attempt_count,
        error=FetchError("GET failed", permanent=permanent),
    )

    assert outcome.status == status
    assert retries == delays


@pytest.mark.parametrize(
    ("attempt_count", "expected_transition"), [(0, "retry"), (9, "fail")]
)
async def test_stale_failure_transition_becomes_lease_lost(
    monkeypatch, attempt_count, expected_transition
):
    outcome, retries, failures = await _run_failure(
        monkeypatch,
        attempt_count=attempt_count,
        error=FetchError("GET failed"),
        transition_succeeds=False,
    )

    assert outcome.status == "lease_lost"
    assert bool(retries) is (expected_transition == "retry")
    assert bool(failures) is (expected_transition == "fail")


async def test_heartbeat_lease_loss_becomes_explicit_outcome(monkeypatch):
    claimed = attempt()

    async def claim(conn, *, exclude_site_ids=()):
        return claimed

    async def lose_lease(operation, renew):
        raise LeaseLostError("database lease was lost")

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs, "run_with_lease", lose_lease)

    outcome = await processor(
        Pool(), lambda url: asyncio.sleep(0, result=response(url))
    )()

    assert outcome is not None and outcome.status == "lease_lost"


async def test_concurrent_calls_prefer_different_sites(monkeypatch):
    exclusions: list[tuple[int, ...]] = []
    second_claimed = asyncio.Event()
    release_first = asyncio.Event()

    async def claim(conn, *, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        if len(exclusions) == 1:
            return attempt(site_id=7, job_id=1)
        if len(exclusions) == 2:
            second_claimed.set()
            return attempt(site_id=9, job_id=2)
        return None

    async def run_attempt(pool, fetch_page, embed_documents, chunker, claimed):
        if claimed.site_id == 7:
            await second_claimed.wait()
            await release_first.wait()
        return crawl_jobs.CrawlAttemptOutcome(claimed.url, "indexed")

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs, "_run_crawl_attempt", run_attempt)
    process_next = processor(Pool(), lambda url: asyncio.sleep(0, result=response(url)))

    first = asyncio.ensure_future(process_next())
    await asyncio.sleep(0)
    second = asyncio.ensure_future(process_next())
    await second_claimed.wait()
    release_first.set()
    await asyncio.gather(first, second)

    assert exclusions == [(), (7,)]
    assert await process_next() is None
    assert exclusions[-1] == ()


async def test_claim_falls_back_to_a_busy_site(monkeypatch):
    exclusions: list[tuple[int, ...]] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def claim(conn, *, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        if len(exclusions) == 1:
            return attempt(site_id=7, job_id=1)
        if len(exclusions) == 2:
            return None
        if len(exclusions) == 3:
            return attempt(site_id=7, job_id=2)
        return None

    async def run_attempt(pool, fetch_page, embed_documents, chunker, claimed):
        if claimed.id == 1:
            first_started.set()
            await release_first.wait()
        return crawl_jobs.CrawlAttemptOutcome(claimed.url, "indexed")

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs, "_run_crawl_attempt", run_attempt)
    process_next = processor(Pool(), lambda url: asyncio.sleep(0, result=response(url)))

    first = asyncio.ensure_future(process_next())
    await first_started.wait()
    second = await process_next()
    assert second is not None
    release_first.set()
    await first

    assert exclusions[:3] == [(), (7,), ()]


async def test_busy_site_is_released_when_processing_raises(monkeypatch):
    exclusions: list[tuple[int, ...]] = []

    async def claim(conn, *, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        return attempt(site_id=7) if len(exclusions) == 1 else None

    async def run_attempt(*args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs, "_run_crawl_attempt", run_attempt)
    process_next = processor(Pool(), lambda url: asyncio.sleep(0, result=response(url)))

    with pytest.raises(RuntimeError, match="database unavailable"):
        await process_next()
    assert await process_next() is None
    assert exclusions == [(), ()]


async def test_cancellation_releases_busy_site(monkeypatch):
    exclusions: list[tuple[int, ...]] = []
    started = asyncio.Event()

    async def claim(conn, *, exclude_site_ids=()):
        exclusions.append(tuple(exclude_site_ids))
        return attempt(site_id=7) if len(exclusions) == 1 else None

    async def run_attempt(*args):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(crawl_jobs.db, "claim_crawl_job", claim)
    monkeypatch.setattr(crawl_jobs, "_run_crawl_attempt", run_attempt)
    process_next = processor(Pool(), lambda url: asyncio.sleep(0, result=response(url)))

    task = asyncio.ensure_future(process_next())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await process_next() is None
    assert exclusions == [(), ()]
