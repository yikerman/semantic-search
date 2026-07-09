from collections.abc import Sequence
from itertools import chain

from psycopg_pool import AsyncConnectionPool

from semsearch.config import Settings
from semsearch.db import check_index_meta
from semsearch.embeddings.base import EmbeddingProvider
from semsearch.models import Candidate, SearchResult
from semsearch.search.base import Ranker, Retriever, final_score
from semsearch.search.dense import DenseRetriever
from semsearch.util import map_concurrently


def merge_candidates(candidate_lists: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    merged: dict[int, Candidate] = {}
    for candidate in chain.from_iterable(candidate_lists):
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
        else:
            existing.scores.update(candidate.scores)
    return list(merged.values())


def group_by_page(candidates: list[Candidate], limit: int) -> list[SearchResult]:
    best: dict[int, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.page_id)
        if current is None or final_score(candidate) > final_score(current):
            best[candidate.page_id] = candidate
    ranked = sorted(best.values(), key=final_score, reverse=True)[:limit]
    return [
        SearchResult(
            page_id=candidate.page_id,
            url=candidate.url,
            title=candidate.title,
            score=final_score(candidate),
            snippet=candidate.content,
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
        rankers: list[Ranker] | None = None,
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.settings = settings
        self.retrievers = (
            retrievers if retrievers is not None else [DenseRetriever(pool)]
        )
        self.rankers = rankers if rankers is not None else []
        self._meta_checked = False

    async def search(
        self, query: str, *, limit: int = 10, fetch_k: int = 50
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        if not self._meta_checked:
            async with self.pool.connection() as conn:
                await check_index_meta(conn, self.settings)
            self._meta_checked = True

        query_embedding = await self.embedder.embed_query(query)
        candidate_lists = await map_concurrently(
            self.retrievers,
            limit=len(self.retrievers),
            func=lambda retriever: retriever.retrieve(query, query_embedding, fetch_k),
        )
        candidates = merge_candidates(candidate_lists)
        for ranker in self.rankers:
            candidates = await ranker.rank(query, candidates)
        return group_by_page(candidates, limit)
