from contextlib import AbstractAsyncContextManager
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from semsearch.cli import db


class Cursor:
    def __init__(self, row=None, *, rowcount: int = 1) -> None:
        self.row = row
        self.rowcount = rowcount

    async def fetchone(self):
        return self.row


class ClaimConnection:
    def __init__(self) -> None:
        self.query = ""
        self.params: tuple[object, ...] = ()

    async def execute(self, query, params):
        self.query = query
        self.params = params
        token = cast(UUID, params[-1])
        return Cursor((1, 2, "https://example.com/post", "feed", 0, token))


async def test_claim_crawl_job_returns_ownership_token():
    conn = ClaimConnection()

    job = await db.claim_crawl_job(cast(Any, conn))

    assert job is not None and job.lease_token == conn.params[-1]
    assert "FOR UPDATE SKIP LOCKED" in conn.query
    assert "next_attempt_at IS NOT NULL" in conn.query
    assert "lease_token = %s" in conn.query


async def test_claim_crawl_job_can_exclude_sites():
    conn = ClaimConnection()

    job = await db.claim_crawl_job(cast(Any, conn), exclude_site_ids=(4, 9))

    assert job is not None
    assert "site_id != ALL(%s::bigint[])" in conn.query
    assert conn.params[0] == [4, 9]


class InvalidClaimConnection:
    async def execute(self, query, params):
        return Cursor((1, 2, "https://example.com/post", "feed", -1, params[-1]))


async def test_claim_crawl_job_validates_database_row():
    with pytest.raises(ValueError, match="invalid crawl attempt database row"):
        await db.claim_crawl_job(cast(Any, InvalidClaimConnection()))


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query, params):
        self.calls.append((query, params))
        return Cursor()


async def test_job_state_updates_require_a_live_lease_and_report_ownership():
    conn = RecordingConnection()
    token = uuid4()

    retried = await db.retry_crawl_job(
        cast(Any, conn),
        job_id=7,
        lease_token=token,
        error="temporary",
        delay_seconds=300,
    )
    failed = await db.fail_crawl_job(
        cast(Any, conn), job_id=8, lease_token=token, error="permanent"
    )
    completed = await db.complete_crawl_job(
        cast(Any, conn), job_id=9, lease_token=token
    )
    renewed = await db.renew_crawl_lease(cast(Any, conn), job_id=10, lease_token=token)

    assert all("lease_token = %s" in query for query, _ in conn.calls)
    assert all("lease_until >= now()" in query for query, _ in conn.calls)
    assert all(params[-1] == token for _, params in conn.calls)
    assert retried and failed and completed and renewed


async def test_job_state_update_reports_lost_lease():
    class LostConnection:
        async def execute(self, query, params):
            return Cursor(rowcount=0)

    assert not await db.complete_crawl_job(
        cast(Any, LostConnection()), job_id=9, lease_token=uuid4()
    )


class ChunkCursor(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.rows = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def executemany(self, query, rows):
        self.rows = rows


class ChunkConnection:
    def __init__(self) -> None:
        self.cur = ChunkCursor()

    def cursor(self):
        return self.cur

    async def execute(self, query, params):
        raise AssertionError("append-only chunk insertion must not delete")


async def test_insert_page_chunks_never_replaces_existing_chunks():
    conn = ChunkConnection()

    await db.insert_page_chunks(
        cast(Any, conn),
        page_id=3,
        chunks=[db.ChunkInsert(0, "content", 7, (1.0, 0.0))],
    )

    assert len(conn.cur.rows) == 1
