from collections.abc import Sequence

from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.config import Settings
from semsearch.embeddings.base import EmbeddingProvider
from semsearch.models import Candidate, SearchResult
from semsearch.search.base import (
    Fusion,
    RankedRun,
    Reranker,
    RetrievalRequest,
    Retriever,
    SearchFilter,
)
from semsearch.search.dense import DenseRetriever
from semsearch.search.fusion import ReciprocalRankFusion, union_candidates
from semsearch.util import map_concurrently


def group_by_page(candidates: Sequence[Candidate], limit: int) -> list[SearchResult]:
    best: dict[int, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.page_id)
        if current is None or candidate.scores["rrf"] > current.scores["rrf"]:
            best[candidate.page_id] = candidate
    ranked = sorted(
        best.values(), key=lambda candidate: candidate.scores["rrf"], reverse=True
    )[:limit]
    return [
        SearchResult(
            page_id=candidate.page_id,
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.content,
            scores=candidate.scores,
        )
        for candidate in ranked
    ]


class SearchService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: EmbeddingProvider,
        settings: Settings,
        *,
        retrievers: list[Retriever] | None = None,
        rerankers: list[Reranker] | None = None,
        fusion: Fusion | None = None,
        meta_guard: db.IndexMetaGuard | None = None,
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.settings = settings
        self.retrievers = (
            retrievers if retrievers is not None else [DenseRetriever(pool)]
        )
        self.rerankers = rerankers if rerankers is not None else []
        self.fusion = fusion or ReciprocalRankFusion()
        _validate_source_names([*self.retrievers, *self.rerankers])
        self.meta_guard = meta_guard or db.IndexMetaGuard(settings)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        fetch_k: int = 50,
        filters: Sequence[SearchFilter] = (),
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        async with self.pool.connection() as conn:
            await self.meta_guard.ensure(conn)

        query_embedding = await self.embedder.embed_query(query)
        request = RetrievalRequest(
            query, tuple(query_embedding), tuple(filters), fetch_k
        )
        retrieval_runs = await map_concurrently(
            self.retrievers,
            limit=len(self.retrievers),
            func=lambda retriever: _retrieve(retriever, request),
        )
        candidates = union_candidates(retrieval_runs)
        reranker_runs: list[RankedRun] = []
        for reranker in self.rerankers:
            reranker_runs.append(await _rerank(reranker, query, candidates))
        return group_by_page(self.fusion.fuse([*retrieval_runs, *reranker_runs]), limit)


def _validate_source_names(sources: Sequence[Retriever | Reranker]) -> None:
    names = [source.name for source in sources]
    if "rrf" in names:
        raise ValueError("'rrf' is reserved for fusion scores")
    if len(names) != len(set(names)):
        raise ValueError("Retriever and reranker names must be unique")


async def _retrieve(retriever: Retriever, request: RetrievalRequest) -> RankedRun:
    run = await retriever.retrieve(request)
    _validate_run_name(retriever.name, run)
    return run


async def _rerank(
    reranker: Reranker, query: str, candidates: Sequence[Candidate]
) -> RankedRun:
    run = await reranker.rerank(query, candidates)
    _validate_run_name(reranker.name, run)
    expected = {candidate.chunk_id for candidate in candidates}
    actual = [candidate.chunk_id for candidate in run.candidates]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            f"Reranker {reranker.name!r} must return every candidate exactly once"
        )
    return run


def _validate_run_name(expected: str, run: RankedRun) -> None:
    if run.name != expected:
        raise ValueError(f"Source {expected!r} returned a run named {run.name!r}")
