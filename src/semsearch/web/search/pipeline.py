from collections.abc import Sequence

from psycopg_pool import AsyncConnectionPool

from semsearch.share.embeddings import EmbedQuery
from semsearch.share.util import map_concurrently
from semsearch.web import db
from semsearch.web.search.base import (
    Fusion,
    RankedRun,
    Reranker,
    RetrievalRequest,
    Retriever,
    SearchFilter,
)
from semsearch.web.search.fusion import (
    reciprocal_rank_fusion,
    union_chunk_candidates,
    union_page_candidates,
)
from semsearch.web.search.models import ChunkCandidate, PageCandidate


def _suffix_prefix_overlap(left: str, right: str) -> int:
    if not left or not right:
        return 0

    prefix_lengths = [0] * len(right)
    for index in range(1, len(right)):
        matched = prefix_lengths[index - 1]
        while matched and right[index] != right[matched]:
            matched = prefix_lengths[matched - 1]
        if right[index] == right[matched]:
            matched += 1
        prefix_lengths[index] = matched

    matched = 0
    for character in left[-len(right) :]:
        while matched and character != right[matched]:
            matched = prefix_lengths[matched - 1]
        if character == right[matched]:
            matched += 1
    while matched:
        left_starts_at_word = (
            matched == len(left) or left[len(left) - matched - 1].isspace()
        )
        right_ends_at_word = matched == len(right) or right[matched].isspace()
        if left_starts_at_word and right_ends_at_word:
            break
        matched = prefix_lengths[matched - 1]
    return matched


def compose_chunks(chunks: Sequence[str]) -> str:
    if not chunks:
        return ""
    content = chunks[0]
    for chunk in chunks[1:]:
        overlap = _suffix_prefix_overlap(content, chunk)
        if overlap:
            content += chunk[overlap:]
        elif content and chunk:
            content += f" {chunk}"
        else:
            content += chunk
    return content


def compose_page_candidates(
    candidates: Sequence[ChunkCandidate],
    chunks_by_page: dict[int, tuple[str, ...]],
) -> list[PageCandidate]:
    pages: dict[int, ChunkCandidate] = {}
    for candidate in candidates:
        pages.setdefault(candidate.page_id, candidate)

    composed: list[PageCandidate] = []
    for page_id, candidate in pages.items():
        try:
            chunks = chunks_by_page[page_id]
        except KeyError as exc:
            raise ValueError(f"retrieved page {page_id} has no chunks") from exc
        composed.append(
            PageCandidate(
                page_id=page_id,
                url=candidate.url,
                title=candidate.title,
                content=compose_chunks(chunks),
            )
        )
    return composed


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
        chunks_by_page = await db.fetch_page_chunks(conn, page_ids=page_ids)
    page_candidates = compose_page_candidates(chunk_candidates, chunks_by_page)
    pages = {candidate.page_id: candidate for candidate in page_candidates}
    page_retrieval_runs = [aggregate_page_run(run, pages) for run in retrieval_runs]
    merged_pages = union_page_candidates(page_retrieval_runs)
    reranker_runs = [await reranker(query, merged_pages) for reranker in rerankers]
    return fusion([*page_retrieval_runs, *reranker_runs])[:limit]
