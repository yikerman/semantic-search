from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from semsearch.web import db
from semsearch.web.search.fusion import (
    reciprocal_rank_fusion,
    union_chunk_candidates,
)
from semsearch.web.search.models import (
    ChunkCandidate,
    PageCandidate,
    RankedRun,
    RetrievalRequest,
    Retriever,
    make_run,
)
from semsearch.web.search.pipeline import (
    aggregate_page_run,
    rerank_by_length,
    search,
)


def chunk(chunk_id: int, page_id: int, **scores: float) -> ChunkCandidate:
    return ChunkCandidate(
        chunk_id=chunk_id,
        page_id=page_id,
        scores=scores,
    )


def page(page_id: int, content: str = "content", **scores: float) -> PageCandidate:
    return PageCandidate(
        page_id=page_id,
        url=f"https://blog.example/p{page_id}",
        title=f"Post {page_id}",
        content=content,
        scores=scores,
    )


def chunk_run(
    name: str, weight: float, *candidates: ChunkCandidate
) -> RankedRun[ChunkCandidate]:
    return RankedRun(name, weight, candidates)


def page_run(
    name: str, weight: float, *candidates: PageCandidate
) -> RankedRun[PageCandidate]:
    return RankedRun(name, weight, candidates)


def test_make_run_writes_named_score_and_merges_existing_scores():
    run = make_run("length", 1.0, [(page(1, dense=0.9), 42.0)])

    assert run.name == "length"
    assert run.weight == 1.0
    assert run.candidates[0].scores == {"dense": 0.9, "length": 42.0}


def test_make_run_orders_by_score_descending_with_stable_ties():
    run = make_run(
        "dense",
        2.0,
        [(chunk(1, 1), 0.5), (chunk(2, 1), 0.9), (chunk(3, 2), 0.5)],
    )

    assert [candidate.chunk_id for candidate in run.candidates] == [2, 1, 3]
    assert run.candidates[0].scores == {"dense": 0.9}


def test_union_chunk_candidates_combines_scores_without_mutating_inputs():
    dense = chunk(1, 1, dense=0.9)
    lexical = chunk(1, 1, bm25=4.2)

    merged = union_chunk_candidates(
        [chunk_run("dense", 2.0, dense), chunk_run("bm25", 0.5, lexical)]
    )

    assert merged[0].scores == {"dense": 0.9, "bm25": 4.2}
    assert dense.scores == {"dense": 0.9}
    assert lexical.scores == {"bm25": 4.2}
    with pytest.raises(TypeError):
        cast(dict[str, float], merged[0].scores)["other"] = 1.0


def test_weighted_rrf_uses_page_runs_and_preserves_native_scores():
    dense = page_run("dense", 2.0, page(1, dense=0.9), page(2, dense=0.8))
    lexical = page_run("bm25", 0.5, page(2, bm25=4.0), page(3, bm25=3.0))

    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    assert [candidate.page_id for candidate in fused] == [2, 1, 3]
    assert fused[0].scores == {
        "dense": 0.8,
        "bm25": 4.0,
        "rrf": pytest.approx(2 / 62 + 0.5 / 61),
    }


def test_rrf_reads_each_weight_off_its_run():
    dense = page_run("dense", 2.0, page(1, dense=0.9), page(2, dense=0.8))
    lexical = page_run("bm25", 0.5, page(2, bm25=4.0), page(1, bm25=3.0))
    length = page_run("length", 1.0, page(2, length=2000), page(1, length=1000))

    fused = reciprocal_rank_fusion([dense, lexical, length], k=60)

    assert [candidate.page_id for candidate in fused] == [1, 2]
    assert fused[0].scores["rrf"] == pytest.approx(2 / 61 + 0.5 / 62 + 1 / 62)
    assert fused[1].scores["rrf"] == pytest.approx(2 / 62 + 0.5 / 61 + 1 / 61)


def test_page_run_rewards_top_three_chunk_scores_including_negative_scores():
    pages = {1: page(1), 2: page(2), 3: page(3)}
    run = chunk_run(
        "dense",
        2.0,
        chunk(1, 1, dense=0.9),
        chunk(2, 1, dense=0.8),
        chunk(3, 1, dense=0.7),
        chunk(4, 1, dense=0.6),
        chunk(5, 2, dense=0.95),
        chunk(6, 3, dense=-0.1),
        chunk(7, 3, dense=-0.2),
        chunk(8, 3, dense=-0.3),
    )

    aggregated = aggregate_page_run(run, pages)

    assert aggregated.weight == run.weight
    assert [candidate.page_id for candidate in aggregated.candidates] == [1, 2, 3]
    assert aggregated.candidates[0].scores["dense"] == pytest.approx(0.987)
    assert aggregated.candidates[2].scores["dense"] == pytest.approx(-0.123)


async def test_length_reranker_scores_and_orders_full_page_content():
    candidates = [page(1, "short"), page(2, "a much longer page")]

    ranked = await rerank_by_length("ignored query", candidates)

    assert [candidate.page_id for candidate in ranked.candidates] == [2, 1]
    assert ranked.candidates[0].scores["length"] == 18.0
    assert "length" not in candidates[0].scores


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeConnection(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class FakePool:
    def connection(self):
        return FakeConnection()


def fake_retriever(
    name: str,
    weight: float,
    candidates: Sequence[ChunkCandidate],
    calls: list[tuple[RetrievalRequest, object]],
) -> Retriever:
    async def retrieve(request: RetrievalRequest, pool: object):
        calls.append((request, pool))
        return RankedRun(name, weight, tuple(candidates))

    return cast(Retriever, retrieve)


async def test_search_materializes_and_fuses_unique_pages(monkeypatch):
    pool = FakePool()
    embedder = FakeEmbedder()
    calls: list[tuple[RetrievalRequest, object]] = []
    fetches: list[tuple[object, list[int]]] = []
    dense = fake_retriever(
        "dense",
        2.0,
        [
            chunk(1, 1, dense=0.9),
            chunk(2, 1, dense=0.8),
            chunk(3, 2, dense=0.85),
        ],
        calls,
    )
    bm25 = fake_retriever("bm25", 0.5, [chunk(3, 2, bm25=2.0)], calls)

    async def fetch_pages(conn, *, page_ids):
        fetches.append((conn, page_ids))
        return {
            1: db.PageRecord(
                1,
                "https://blog.example/p1",
                "Post 1",
                "alpha beta gamma",
                datetime(2025, 1, 2, tzinfo=UTC),
            ),
            2: db.PageRecord(2, "https://blog.example/p2", "Post 2", "short", None),
        }

    monkeypatch.setattr(db, "fetch_pages", fetch_pages)

    results = await search(
        "query",
        pool=cast(Any, pool),
        embed_query=embedder.embed_query,
        retrievers=(dense, bm25),
        retriever_limit=12,
    )

    assert embedder.queries == ["query"]
    assert calls[0][0] is calls[1][0]
    assert all(call_pool is pool for _, call_pool in calls)
    assert calls[0][0].limit == 12
    assert fetches[0][1] == [1, 2]
    assert [result.page_id for result in results] == [2, 1]
    assert len({result.page_id for result in results}) == len(results)
    assert results[1].content == "alpha beta gamma"
    assert results[1].published_at == datetime(2025, 1, 2, tzinfo=UTC)
    assert results[0].published_at is None
    assert results[1].scores["dense"] == pytest.approx(0.98)
    assert "length" not in results[0].scores
