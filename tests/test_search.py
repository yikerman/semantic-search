from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from semsearch.web.search.fusion import (
    reciprocal_rank_fusion,
    union_page_candidates,
)
from semsearch.web.search.models import (
    PageCandidate,
    RankedRun,
    RetrievalRequest,
    Retriever,
    make_run,
)
from semsearch.web.search.pipeline import (
    rerank_by_length,
    search,
)


def page(page_id: int, content: str = "content", **scores: float) -> PageCandidate:
    return PageCandidate(
        page_id=page_id,
        url=f"https://blog.example/p{page_id}",
        title=f"Post {page_id}",
        content=content,
        scores=scores,
    )


def page_run(name: str, weight: float, *candidates: PageCandidate) -> RankedRun:
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
        [(page(1), 0.5), (page(2), 0.9), (page(3), 0.5)],
    )

    assert [candidate.page_id for candidate in run.candidates] == [2, 1, 3]
    assert run.candidates[0].scores == {"dense": 0.9}


def test_union_page_candidates_combines_scores_without_mutating_inputs():
    dense = page(1, dense=0.9)
    lexical = page(1, bm25=4.2)

    merged = union_page_candidates(
        [page_run("dense", 2.0, dense), page_run("bm25", 0.5, lexical)]
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


def fake_retriever(
    name: str,
    weight: float,
    candidates: Sequence[PageCandidate],
    calls: list[tuple[RetrievalRequest, object]],
) -> Retriever:
    async def retrieve(request: RetrievalRequest, pool: object):
        calls.append((request, pool))
        return RankedRun(name, weight, tuple(candidates))

    return cast(Retriever, retrieve)


async def test_search_fuses_unique_pages_and_passes_merged_scores_to_rerankers():
    pool = object()
    embedder = FakeEmbedder()
    calls: list[tuple[RetrievalRequest, object]] = []
    published = datetime(2025, 1, 2, tzinfo=UTC)
    first = PageCandidate(
        1,
        "https://blog.example/p1",
        "Post 1",
        "Whole first post",
        published,
        {"dense": 0.9},
    )
    dense = fake_retriever("dense", 2.0, [first, page(2, dense=0.85)], calls)
    bm25 = fake_retriever("bm25", 0.5, [page(2, bm25=2.0)], calls)
    reranked = []

    async def reranker(query, candidates):
        reranked.extend(candidates)
        return make_run("extra", 0.0, ((candidate, 1.0) for candidate in candidates))

    results = await search(
        "query",
        pool=cast(Any, pool),
        embed_query=embedder.embed_query,
        retrievers=(dense, bm25),
        rerankers=(reranker,),
        retriever_limit=12,
    )
    assert embedder.queries == ["query"]
    assert calls[0][0] is calls[1][0]
    assert all(call_pool is pool for _, call_pool in calls)
    assert calls[0][0].limit == 12
    assert [result.page_id for result in results] == [2, 1]
    assert reranked[1].scores == {"dense": 0.85, "bm25": 2.0}
    assert results[1].content == "Whole first post"
    assert results[1].published_at == published
    assert results[1].scores["dense"] == 0.9
    assert results[0].scores["rrf"] == pytest.approx(2 / 62 + 0.5 / 61)


async def test_empty_results():
    async def forbidden(query, candidates):
        pytest.fail("empty retrieval must not call a reranker")

    results = await search(
        "query",
        pool=cast(Any, object()),
        embed_query=FakeEmbedder().embed_query,
        retrievers=(fake_retriever("dense", 2.0, [], []),),
        rerankers=(forbidden,),
    )
    assert results == []
