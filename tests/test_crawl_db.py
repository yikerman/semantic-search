from typing import Any, cast
from uuid import UUID, uuid4

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


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query, params):
        self.calls.append((query, params))
        return Cursor()


async def test_job_state_updates_are_fenced_by_lease_token():
    conn = RecordingConnection()
    token = uuid4()

    await db.retry_crawl_job(
        cast(Any, conn),
        job_id=7,
        lease_token=token,
        error="temporary",
        delay_seconds=300,
    )
    await db.fail_crawl_job(
        cast(Any, conn), job_id=8, lease_token=token, error="permanent"
    )
    await db.complete_existing_job(cast(Any, conn), job_id=9, lease_token=token)

    assert all("lease_token = %s" in query for query, _ in conn.calls)
    assert all(params[-1] == token for _, params in conn.calls)
