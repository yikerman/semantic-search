from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from uuid import uuid4

from semsearch.cli.ingest.fetch import FetchError
from semsearch.cli.ingest.pipeline import IngestError
from semsearch.cli.ingest.worker import process_one_job
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


async def test_repeated_permanent_failure_is_retained_as_terminal(monkeypatch):
    job = CrawlJob(1, 2, "https://example.com/post", "feed", 2, uuid4())
    failed: list[str] = []

    async def claim(conn, *, site_id=None):
        return job

    async def ingest(*args):
        raise IngestError("no extractable article text")

    async def fail(conn, *, job_id, lease_token, error):
        failed.append(error)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.fail_crawl_job", fail)

    outcome = await process_one_job(
        cast(Any, FakePool()), cast(Any, None), cast(Any, None), cast(Any, None)
    )

    assert outcome is not None and outcome.detail.startswith("permanent:")
    assert failed == ["no extractable article text"]


async def test_transient_failure_remains_scheduled(monkeypatch):
    job = CrawlJob(1, 2, "https://example.com/post", "feed", 4, uuid4())
    retries: list[int] = []

    async def claim(conn, *, site_id=None):
        return job

    async def ingest(*args):
        raise FetchError("temporary network failure")

    async def retry(conn, *, job_id, lease_token, error, delay_seconds):
        retries.append(delay_seconds)

    monkeypatch.setattr("semsearch.cli.ingest.worker.db.claim_crawl_job", claim)
    monkeypatch.setattr("semsearch.cli.ingest.worker.ingest_job", ingest)
    monkeypatch.setattr("semsearch.cli.ingest.worker.db.retry_crawl_job", retry)

    outcome = await process_one_job(
        cast(Any, FakePool()), cast(Any, None), cast(Any, None), cast(Any, None)
    )

    assert outcome is not None and outcome.detail.startswith("will retry:")
    assert retries == [86400]
