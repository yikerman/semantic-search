from collections.abc import Sequence

from semsearch.embeddings.base import EmbedQuery
from semsearch.models import Candidate
from semsearch.search.base import (
    Fusion,
    Reranker,
    RetrievalRequest,
    Retriever,
    SearchFilter,
)
from semsearch.search.fusion import reciprocal_rank_fusion, union_candidates
from semsearch.util import map_concurrently


def group_by_page(candidates: Sequence[Candidate], limit: int) -> list[Candidate]:
    best: dict[int, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.page_id)
        if current is None or candidate.scores["rrf"] > current.scores["rrf"]:
            best[candidate.page_id] = candidate
    return sorted(
        best.values(), key=lambda candidate: candidate.scores["rrf"], reverse=True
    )[:limit]


async def search(
    query: str,
    *,
    embed_query: EmbedQuery,
    retrievers: Sequence[Retriever],
    rerankers: Sequence[Reranker] = (),
    fusion: Fusion = reciprocal_rank_fusion,
    limit: int = 10,
    fetch_k: int = 50,
    filters: Sequence[SearchFilter] = (),
) -> list[Candidate]:
    query_embedding = await embed_query(query)
    request = RetrievalRequest(query, tuple(query_embedding), tuple(filters), fetch_k)
    retrieval_runs = await map_concurrently(
        retrievers,
        limit=len(retrievers),
        func=lambda retrieve: retrieve(request),
    )
    candidates = union_candidates(retrieval_runs)
    reranker_runs = [await reranker(query, candidates) for reranker in rerankers]
    runs = [*retrieval_runs, *reranker_runs]
    return group_by_page(fusion(runs), limit)
