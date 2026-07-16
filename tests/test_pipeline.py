from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from uuid import uuid4

from semsearch.cli.ingest.chunk import Chunk
from semsearch.cli.ingest.extract import ExtractedPage
from semsearch.cli.ingest.fetch import FetchResponse
from semsearch.cli.ingest.pipeline import ingest_job
from semsearch.cli.models import CrawlJob


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


class Fetcher:
    async def fetch_response(self, url: str):
        return FetchResponse(
            b"<html></html>",
            {"content-type": "text/html"},
            200,
            url,
        )


async def test_successful_ingest_stores_page_chunks_and_completes_job_atomically(
    monkeypatch,
):
    pool = Pool()
    token = uuid4()
    job = CrawlJob(1, 2, "https://example.com/post", "feed", 0, token)
    exists = iter((False, False))
    writes: list[str] = []

    async def page_exists(conn, *, url):
        return next(exists)

    async def insert_page(conn, **kwargs):
        assert conn.in_transaction
        assert kwargs["language"] == "en"
        writes.append("page")
        return 3

    async def replace_chunks(conn, **kwargs):
        assert conn.in_transaction
        writes.append("chunks")

    async def complete(conn, *, job_id, lease_token):
        assert conn.in_transaction and lease_token == token
        writes.append("complete")

    monkeypatch.setattr("semsearch.cli.ingest.pipeline.db.page_exists", page_exists)
    monkeypatch.setattr("semsearch.cli.ingest.pipeline.db.insert_page", insert_page)
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.db.replace_page_chunks", replace_chunks
    )
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.db.complete_existing_job", complete
    )
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.extract_page",
        lambda html, url: ExtractedPage("Title", "body", None, "en"),
    )

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    outcome = await ingest_job(
        cast(Any, pool),
        embed,
        cast(Any, Fetcher()),
        lambda text: [Chunk(0, text, len(text))],
        job,
    )

    assert outcome.status == "indexed"
    assert writes == ["page", "chunks", "complete"]


async def test_ingest_skips_when_page_inserted_concurrently(monkeypatch):
    pool = Pool()
    token = uuid4()
    job = CrawlJob(1, 2, "https://example.com/post", "feed", 0, token)
    # page_exists is False at the pre-check, but the INSERT conflicts (a racing
    # worker committed first), so insert_page returns None.
    writes: list[str] = []

    async def page_exists(conn, *, url):
        return False

    async def insert_page(conn, **kwargs):
        writes.append("page")
        return None

    async def replace_chunks(conn, **kwargs):
        writes.append("chunks")

    async def complete(conn, *, job_id, lease_token):
        assert lease_token == token
        writes.append("complete")

    monkeypatch.setattr("semsearch.cli.ingest.pipeline.db.page_exists", page_exists)
    monkeypatch.setattr("semsearch.cli.ingest.pipeline.db.insert_page", insert_page)
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.db.replace_page_chunks", replace_chunks
    )
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.db.complete_existing_job", complete
    )
    monkeypatch.setattr(
        "semsearch.cli.ingest.pipeline.extract_page",
        lambda html, url: ExtractedPage("Title", "body", None, "en"),
    )

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    outcome = await ingest_job(
        cast(Any, pool),
        embed,
        cast(Any, Fetcher()),
        lambda text: [Chunk(0, text, len(text))],
        job,
    )

    # Append-only: the existing page's chunks must not be overwritten.
    assert outcome.status == "skipped"
    assert "chunks" not in writes
    assert writes == ["page", "complete"]
