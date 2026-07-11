from contextlib import asynccontextmanager
from typing import Any, cast

from semsearch.web import db
from semsearch.web.search.base import RetrievalRequest
from semsearch.web.search.bm25 import retrieve_bm25
from semsearch.web.search.filters import SqlPredicate


class FakePool:
    @asynccontextmanager
    async def connection(self):
        yield object()


async def test_retrieve_bm25_returns_named_run_with_native_scores(monkeypatch):
    calls: list[dict[str, object]] = []

    async def fetch_rows(conn, **kwargs):
        calls.append({"conn": conn, **kwargs})
        return [
            db.Bm25CandidateRecord(
                chunk_id=7,
                page_id=3,
                url="https://example.com/post",
                title="Post",
                content="matching content",
                rank=0.25,
            )
        ]

    monkeypatch.setattr(db, "fetch_bm25_candidate_rows", fetch_rows)
    request = RetrievalRequest("matching", (1.0, 0.0), (), 12)

    result = await retrieve_bm25(request, pool=cast(Any, FakePool()))

    assert result.name == "bm25"
    assert result.candidates[0].scores == {"bm25": 0.25}
    assert calls[0]["query"] == "matching"
    assert calls[0]["limit"] == 12
    predicate = cast(SqlPredicate, calls[0]["predicate"])
    assert predicate.clause.as_string() == "TRUE"
