from collections.abc import Sequence

import pytest

from semsearch.config import Settings
from semsearch.models import Candidate, SearchResult
from semsearch.search.base import RankedRun, RetrievalRequest
from semsearch.search.fusion import ReciprocalRankFusion, union_candidates
from semsearch.search.service import SearchService, group_by_page


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
        merged[0].scores["other"] = 1.0  # type: ignore[index]


def test_rrf_uses_ranked_runs_and_preserves_native_scores():
    dense = run("dense", cand(1, 1, dense=0.9), cand(2, 2, dense=0.8))
    lexical = run("bm25", cand(2, 2, bm25=4.0), cand(3, 3, bm25=3.0))

    fused = ReciprocalRankFusion(k=60).fuse([dense, lexical])

    assert [candidate.chunk_id for candidate in fused] == [2, 1, 3]
    assert fused[0].scores == {
        "dense": 0.8,
        "bm25": 4.0,
        "rrf": pytest.approx(1 / 62 + 1 / 61),
    }
    assert fused[1].scores["rrf"] == pytest.approx(1 / 61)


def test_rrf_ignores_duplicate_chunks_within_one_run():
    duplicate = cand(1, 1, dense=0.9)

    fused = ReciprocalRankFusion(k=0).fuse(
        [run("dense", duplicate, duplicate, cand(2, 2, dense=0.8))]
    )

    assert [candidate.scores["rrf"] for candidate in fused] == [1.0, 0.5]


def test_group_by_page_keeps_highest_rrf_chunk_and_requires_rrf():
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
    assert results[0].snippet == "chunk 2 of page 1"
    with pytest.raises(KeyError):
        group_by_page([cand(4, 4, dense=1.0)], limit=10)


class FakeCursor:
    def __init__(self, row):
        self.row = row

    async def fetchone(self):
        return self.row


class FakeConn:
    def __init__(self, meta_row):
        self.meta_row = meta_row

    async def execute(self, sql, params=None):
        return FakeCursor(self.meta_row)


class FakePool:
    def __init__(self, meta_row):
        self._conn = FakeConn(meta_row)

    def connection(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool._conn

            async def __aexit__(self, *exc_info):
                return None

        return _Ctx()


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


class FakeRetriever:
    def __init__(self, name: str, candidates: Sequence[Candidate]) -> None:
        self.name = name
        self.candidates = candidates
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> RankedRun:
        self.requests.append(request)
        return RankedRun(self.name, tuple(self.candidates))


class ReverseReranker:
    name = "cross_encoder"

    async def rerank(self, query: str, candidates: Sequence[Candidate]) -> RankedRun:
        ranked = tuple(
            candidate.with_scores(
                {**candidate.scores, self.name: float(candidate.chunk_id)}
            )
            for candidate in reversed(candidates)
        )
        return RankedRun(self.name, ranked)


class InvalidReranker:
    name = "invalid"

    async def rerank(self, query: str, candidates: Sequence[Candidate]) -> RankedRun:
        return RankedRun(self.name, tuple(candidates[:-1]))


def make_settings() -> Settings:
    return Settings(embedding_model="test-model", embedding_dim=2)


async def test_search_embeds_once_and_passes_same_request_to_every_retriever():
    embedder = FakeEmbedder()
    dense = FakeRetriever("dense", [cand(1, 1, dense=0.9)])
    bm25 = FakeRetriever("bm25", [cand(2, 2, bm25=3.0)])
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        embedder,
        make_settings(),
        retrievers=[dense, bm25],
    )

    results = await service.search("  query  ", fetch_k=12)

    assert embedder.queries == ["query"]
    assert dense.requests[0] is bm25.requests[0]
    assert dense.requests[0].query_embedding == (1.0, 0.0)
    assert dense.requests[0].limit == 12
    assert all(isinstance(result, SearchResult) for result in results)


async def test_reranker_run_contributes_to_rrf_ordering():
    retriever = FakeRetriever(
        "dense",
        [cand(1, 1, dense=0.9), cand(2, 2, dense=0.8), cand(3, 3, dense=0.7)],
    )
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        FakeEmbedder(),
        make_settings(),
        retrievers=[retriever],
        rerankers=[ReverseReranker()],
    )

    results = await service.search("query")

    assert [result.page_id for result in results] == [1, 3, 2]
    assert results[0].scores["cross_encoder"] == 1.0
    assert results[0].scores["rrf"] == pytest.approx(1 / 61 + 1 / 63)


async def test_reranker_must_return_every_candidate_once():
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        FakeEmbedder(),
        make_settings(),
        retrievers=[
            FakeRetriever("dense", [cand(1, 1, dense=0.9), cand(2, 2, dense=0.8)])
        ],
        rerankers=[InvalidReranker()],
    )

    with pytest.raises(ValueError, match="every candidate exactly once"):
        await service.search("query")


def test_source_names_must_be_unique_and_cannot_use_rrf():
    pool = FakePool(("test-model", 2))  # type: ignore[assignment]
    embedder = FakeEmbedder()
    settings = make_settings()

    with pytest.raises(ValueError, match="unique"):
        SearchService(
            pool,  # type: ignore[arg-type]
            embedder,
            settings,
            retrievers=[FakeRetriever("dense", []), FakeRetriever("dense", [])],
        )
    with pytest.raises(ValueError, match="reserved"):
        SearchService(
            pool,  # type: ignore[arg-type]
            embedder,
            settings,
            retrievers=[FakeRetriever("rrf", [])],
        )


async def test_blank_query_short_circuits():
    embedder = FakeEmbedder()
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        embedder,
        make_settings(),
        retrievers=[FakeRetriever("dense", [])],
    )

    assert await service.search("   ") == []
    assert embedder.queries == []
