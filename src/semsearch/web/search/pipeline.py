from collections.abc import Sequence

from psycopg_pool import AsyncConnectionPool

from semsearch.share.embeddings import EmbedQuery
from semsearch.share.util import map_concurrently
from semsearch.web.search.filters import SearchFilter
from semsearch.web.search.fusion import (
    reciprocal_rank_fusion,
    union_page_candidates,
)
from semsearch.web.search.models import (
    Fusion,
    PageCandidate,
    RankedRun,
    Reranker,
    RetrievalRequest,
    Retriever,
    make_run,
)


async def rerank_by_length(
    query: str, candidates: Sequence[PageCandidate]
) -> RankedRun:
    del query
    return make_run(
        "length",
        1.0,
        ((candidate, float(len(candidate.content))) for candidate in candidates),
    )


async def search(
    query: str,
    *,
    pool: AsyncConnectionPool,
    embed_query: EmbedQuery,
    retrievers: Sequence[Retriever],
    rerankers: Sequence[Reranker] = (),
    fusion: Fusion = reciprocal_rank_fusion,
    limit: int = 64,
    retriever_limit: int = 64,
    filters: Sequence[SearchFilter] = (),
) -> list[PageCandidate]:
    query_embedding = await embed_query(query)
    request = RetrievalRequest(
        query, tuple(query_embedding), tuple(filters), retriever_limit
    )
    retrieval_runs = await map_concurrently(
        retrievers,
        limit=len(retrievers),
        func=lambda retrieve: retrieve(request, pool),
    )
    merged_pages = union_page_candidates(retrieval_runs)
    if not merged_pages:
        return []
    reranker_runs = [await reranker(query, merged_pages) for reranker in rerankers]
    return fusion([*retrieval_runs, *reranker_runs])[:limit]
