from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from semsearch.cli.daemon import queue


class Cursor:
    def __init__(self, row=None, *, rows=None, rowcount: int = 1) -> None:
        self.row = row
        self.rows = rows or []
        self.rowcount = rowcount

    async def fetchone(self):
        return self.row

    async def fetchall(self):
        return self.rows


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

    job = await queue.claim_crawl_job(cast(Any, conn))

    assert job is not None and job.lease_token == conn.params[-1]
    assert "FOR UPDATE SKIP LOCKED" in conn.query
    assert "next_attempt_at IS NOT NULL" in conn.query
    assert "lease_token = %s" in conn.query


async def test_claim_crawl_job_can_exclude_sites():
    conn = ClaimConnection()

    job = await queue.claim_crawl_job(cast(Any, conn), exclude_site_ids=(4, 9))

    assert job is not None
    assert "site_id != ALL(%s::bigint[])" in conn.query
    assert conn.params[0] == [4, 9]


class InvalidClaimConnection:
    async def execute(self, query, params):
        return Cursor((1, 2, "https://example.com/post", "feed", -1, params[-1]))


async def test_claim_crawl_job_validates_database_row():
    with pytest.raises(ValueError, match="invalid crawl attempt database row"):
        await queue.claim_crawl_job(cast(Any, InvalidClaimConnection()))


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query, params):
        self.calls.append((query, params))
        return Cursor()


async def test_job_state_updates_require_a_live_lease_and_report_ownership():
    conn = RecordingConnection()
    token = uuid4()

    retried = await queue.retry_crawl_job(
        cast(Any, conn),
        job_id=7,
        lease_token=token,
        error="temporary",
        delay_seconds=300,
    )
    failed = await queue.fail_crawl_job(
        cast(Any, conn), job_id=8, lease_token=token, error="permanent"
    )
    completed = await queue.complete_crawl_job(
        cast(Any, conn), job_id=9, lease_token=token
    )
    renewed = await queue.renew_crawl_lease(
        cast(Any, conn), job_id=10, lease_token=token
    )

    assert all("lease_token = %s" in query for query, _ in conn.calls)
    assert all("lease_until >= now()" in query for query, _ in conn.calls)
    assert all(params[-1] == token for _, params in conn.calls)
    assert retried and failed and completed and renewed


async def test_job_state_update_reports_lost_lease():
    class LostConnection:
        async def execute(self, query, params):
            return Cursor(rowcount=0)

    assert not await queue.complete_crawl_job(
        cast(Any, LostConnection()), job_id=9, lease_token=uuid4()
    )
