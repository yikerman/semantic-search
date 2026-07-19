from collections.abc import Sequence

from psycopg_pool import AsyncConnectionPool

from semsearch.share.embeddings import EmbedQuery
from semsearch.share.util import map_concurrently
from semsearch.web import db
from semsearch.web.search.filters import SearchFilter
from semsearch.web.search.fusion import (
    reciprocal_rank_fusion,
    union_chunk_candidates,
    union_page_candidates,
)
from semsearch.web.search.models import (
    ChunkCandidate,
    Fusion,
    PageCandidate,
    RankedRun,
    Reranker,
    RetrievalRequest,
    Retriever,
)


def aggregate_page_run(
    run: RankedRun[ChunkCandidate], pages: dict[int, PageCandidate]
) -> RankedRun[PageCandidate]:
    scores_by_page: dict[int, list[float]] = {}
    for candidate in run.candidates:
        scores_by_page.setdefault(candidate.page_id, []).append(
            candidate.scores[run.name]
        )

    candidates = []
    for page_id, native_scores in scores_by_page.items():
        top_scores = sorted(native_scores, reverse=True)[:3]
        aggregate = sum(score * (0.1**index) for index, score in enumerate(top_scores))
        candidates.append(pages[page_id].with_scores({run.name: aggregate}))
    candidates.sort(key=lambda candidate: candidate.scores[run.name], reverse=True)
    return RankedRun(run.name, tuple(candidates))


async def rerank_by_length(
    query: str, candidates: Sequence[PageCandidate]
) -> RankedRun[PageCandidate]:
    del query
    scored = [
        candidate.with_scores(
            {**candidate.scores, "length": float(len(candidate.content))}
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda candidate: candidate.scores["length"], reverse=True)
    return RankedRun("length", tuple(scored))


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
    chunk_candidates = union_chunk_candidates(retrieval_runs)
    if not chunk_candidates:
        return []

    page_ids = list(dict.fromkeys(candidate.page_id for candidate in chunk_candidates))
    async with pool.connection() as conn:
        page_records = await db.fetch_pages(conn, page_ids=page_ids)
    pages = {
        page_id: PageCandidate(
            page_id=page_id,
            url=record.url,
            title=record.title,
            content=record.content,
            published_at=record.published_at,
        )
        for page_id, record in page_records.items()
    }
    page_retrieval_runs = [aggregate_page_run(run, pages) for run in retrieval_runs]
    merged_pages = union_page_candidates(page_retrieval_runs)
    reranker_runs = [await reranker(query, merged_pages) for reranker in rerankers]
    return fusion([*page_retrieval_runs, *reranker_runs])[:limit]
