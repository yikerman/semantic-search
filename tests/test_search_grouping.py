from semsearch.config import Settings
from semsearch.models import Candidate, SearchResult
from semsearch.search.base import final_score
from semsearch.search.service import SearchService, group_by_page, merge_candidates


def cand(chunk_id: int, page_id: int, **scores: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        page_id=page_id,
        url=f"https://blog.example/p{page_id}",
        title=f"Post {page_id}",
        content=f"chunk {chunk_id} of page {page_id}",
        scores=scores,
    )


def test_final_score_prefers_final_over_dense():
    assert final_score(cand(1, 1, dense=0.9, final=0.2)) == 0.2
    assert final_score(cand(1, 1, dense=0.9)) == 0.9
    assert final_score(cand(1, 1)) == 0.0


def test_merge_candidates_unions_scores_by_chunk():
    dense = [cand(1, 1, dense=0.9), cand(2, 1, dense=0.8)]
    bm25 = [cand(1, 1, bm25=0.5), cand(3, 2, bm25=0.7)]
    merged = {c.chunk_id: c for c in merge_candidates([dense, bm25])}
    assert set(merged) == {1, 2, 3}
    assert merged[1].scores == {"dense": 0.9, "bm25": 0.5}


def test_group_by_page_keeps_best_chunk_per_page():
    results = group_by_page(
        [
            cand(1, 1, dense=0.7),
            cand(2, 1, dense=0.9),
            cand(3, 2, dense=0.8),
        ],
        limit=10,
    )
    assert [(r.page_id, r.score) for r in results] == [(1, 0.9), (2, 0.8)]
    assert results[0].snippet == "chunk 2 of page 1"


def test_group_by_page_applies_limit():
    candidates = [cand(i, i, dense=i / 10) for i in range(1, 6)]
    results = group_by_page(candidates, limit=2)
    assert [r.page_id for r in results] == [5, 4]


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
    async def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        return [1.0, 0.0]


class FakeRetriever:
    name = "dense"

    def __init__(self, candidates):
        self.candidates = candidates

    async def retrieve(self, query, query_embedding, k):
        return self.candidates


class InvertRanker:
    async def rank(self, query, candidates):
        for candidate in candidates:
            candidate.scores["final"] = 1.0 - candidate.scores["dense"]
        return candidates


def make_settings() -> Settings:
    return Settings(embedding_model="test-model", embedding_dim=2)


async def test_search_pipeline_with_fake_retriever():
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        FakeEmbedder(),
        make_settings(),
        retrievers=[FakeRetriever([cand(1, 1, dense=0.9), cand(2, 2, dense=0.4)])],
    )
    results = await service.search("query")
    assert [r.page_id for r in results] == [1, 2]
    assert all(isinstance(r, SearchResult) for r in results)


async def test_rankers_override_ordering_via_final_score():
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        FakeEmbedder(),
        make_settings(),
        retrievers=[FakeRetriever([cand(1, 1, dense=0.9), cand(2, 2, dense=0.4)])],
        rankers=[InvertRanker()],
    )
    results = await service.search("query")
    assert [r.page_id for r in results] == [2, 1]


async def test_blank_query_short_circuits():
    service = SearchService(
        FakePool(("test-model", 2)),  # type: ignore[arg-type]
        FakeEmbedder(),
        make_settings(),
        retrievers=[FakeRetriever([])],
    )
    assert await service.search("   ") == []
