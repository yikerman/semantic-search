from collections.abc import Sequence
from functools import partial
from typing import cast

import pytest

from semsearch.web.search.base import RankedRun, RetrievalRequest, Retriever
from semsearch.web.search.fusion import reciprocal_rank_fusion, union_candidates
from semsearch.web.search.models import Candidate
from semsearch.web.search.pipeline import group_by_page, search


def cand(chunk_id: int, page_id: int, **scores: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        page_id=page_id,
        url=f"https://blog.example/p{page_id}",
        title=f"Post {page_id}",
        content=f"chunk {chunk_id} of page {page_id}",
        scores=scores,
    )


def run(name: str, *candidates: Candidate) -> RankedRun:
    return RankedRun(name, candidates)


def test_union_candidates_combines_scores_without_mutating_inputs():
    dense = cand(1, 1, dense=0.9)
    lexical = cand(1, 1, bm25=4.2)

    merged = union_candidates([run("dense", dense), run("bm25", lexical)])

    assert merged[0].scores == {"dense": 0.9, "bm25": 4.2}
    assert dense.scores == {"dense": 0.9}
    assert lexical.scores == {"bm25": 4.2}
    with pytest.raises(TypeError):
        cast(dict[str, float], merged[0].scores)["other"] = 1.0


def test_rrf_uses_ranked_runs_and_preserves_native_scores():
    dense = run("dense", cand(1, 1, dense=0.9), cand(2, 2, dense=0.8))
    lexical = run("bm25", cand(2, 2, bm25=4.0), cand(3, 3, bm25=3.0))

    fused = reciprocal_rank_fusion([dense, lexical], k=60)

    assert [candidate.chunk_id for candidate in fused] == [2, 1, 3]
    assert fused[0].scores == {
        "dense": 0.8,
        "bm25": 4.0,
        "rrf": pytest.approx(1 / 62 + 1 / 61),
    }
    assert fused[1].scores["rrf"] == pytest.approx(1 / 61)


def test_group_by_page_keeps_highest_rrf_chunk():
    results = group_by_page(
        [
            cand(1, 1, dense=0.99, rrf=0.01),
            cand(2, 1, dense=0.10, rrf=0.02),
            cand(3, 2, dense=0.80, rrf=0.015),
        ],
        limit=10,
    )

    assert [(result.page_id, result.scores["rrf"]) for result in results] == [
        (1, 0.02),
        (2, 0.015),
    ]
    assert results[0].content == "chunk 2 of page 1"


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


def fake_retriever(
    name: str,
    candidates: Sequence[Candidate],
    requests: list[RetrievalRequest] | None = None,
) -> Retriever:
    async def retrieve(request: RetrievalRequest) -> RankedRun:
        if requests is not None:
            requests.append(request)
        return RankedRun(name, tuple(candidates))

    return retrieve


async def reverse_reranker(query: str, candidates: Sequence[Candidate]) -> RankedRun:
    ranked = tuple(
        candidate.with_scores(
            {**candidate.scores, "cross_encoder": float(candidate.chunk_id)}
        )
        for candidate in reversed(candidates)
    )
    return RankedRun("cross_encoder", ranked)


async def test_search_embeds_once_and_passes_same_request_to_every_retriever():
    embedder = FakeEmbedder()
    dense_requests: list[RetrievalRequest] = []
    bm25_requests: list[RetrievalRequest] = []
    dense = fake_retriever("dense", [cand(1, 1, dense=0.9)], dense_requests)
    bm25 = fake_retriever("bm25", [cand(2, 2, bm25=3.0)], bm25_requests)
    pipeline = partial(
        search,
        embed_query=embedder.embed_query,
        retrievers=[dense, bm25],
    )

    await pipeline("query", fetch_k=12)

    assert embedder.queries == ["query"]
    assert dense_requests[0] is bm25_requests[0]
    assert dense_requests[0].query_embedding == (1.0, 0.0)
    assert dense_requests[0].limit == 12


async def test_reranker_run_contributes_to_rrf_ordering():
    retriever = fake_retriever(
        "dense",
        [cand(1, 1, dense=0.9), cand(2, 2, dense=0.8), cand(3, 3, dense=0.7)],
    )
    pipeline = partial(
        search,
        embed_query=FakeEmbedder().embed_query,
        retrievers=[retriever],
        rerankers=[reverse_reranker],
    )

    results = await pipeline("query")

    assert [result.page_id for result in results] == [1, 3, 2]
    assert results[0].scores["cross_encoder"] == 1.0
    assert results[0].scores["rrf"] == pytest.approx(1 / 61 + 1 / 63)
