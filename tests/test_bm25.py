from contextlib import asynccontextmanager
from typing import Any, cast

from semsearch.web import db
from semsearch.web.search.filters import SqlPredicate
from semsearch.web.search.models import RetrievalRequest
from semsearch.web.search.retrievers import retrieve_bm25


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
                rank=0.25,
            )
        ]

    monkeypatch.setattr(db, "fetch_bm25_candidate_rows", fetch_rows)
    request = RetrievalRequest("matching", (1.0, 0.0), (), 12)

    result = await retrieve_bm25(request, pool=cast(Any, FakePool()))

    assert result.name == "bm25"
    assert result.weight == 0.5
    assert result.candidates[0].scores == {"bm25": 0.25}
    assert calls[0]["query"] == "matching"
    assert calls[0]["limit"] == 12
    predicate = cast(SqlPredicate, calls[0]["predicate"])
    assert predicate.clause.as_string() == "TRUE"
