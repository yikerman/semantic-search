from contextlib import asynccontextmanager
from typing import Any, cast

from semsearch.web import db
from semsearch.web.search.filters import SqlPredicate
from semsearch.web.search.models import RetrievalRequest
from semsearch.web.search.retrievers import retrieve_dense


class FakePool:
    @asynccontextmanager
    async def connection(self):
        yield object()


async def test_retrieve_dense_returns_named_run_with_native_scores(monkeypatch):
    calls: list[dict[str, object]] = []

    async def fetch_rows(conn, **kwargs):
        calls.append({"conn": conn, **kwargs})
        return [db.DenseCandidateRecord(chunk_id=7, page_id=3, similarity=0.75)]

    monkeypatch.setattr(db, "fetch_dense_candidate_rows", fetch_rows)
    request = RetrievalRequest("matching", (1.0, 0.0), (), 12)

    result = await retrieve_dense(request, pool=cast(Any, FakePool()))

    assert result.name == "dense"
    assert result.weight == 2.0
    assert result.candidates[0].scores == {"dense": 0.75}
    assert calls[0]["query_embedding"] == (1.0, 0.0)
    assert calls[0]["limit"] == 12
    predicate = cast(SqlPredicate, calls[0]["predicate"])
    assert predicate.clause.as_string() == "TRUE"
