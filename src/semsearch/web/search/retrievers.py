from psycopg_pool import AsyncConnectionPool

from semsearch.web import db
from semsearch.web.search.filters import compile_filters
from semsearch.web.search.models import (
    ChunkCandidate,
    RankedRun,
    RetrievalRequest,
    make_run,
)


async def retrieve_bm25(
    request: RetrievalRequest, pool: AsyncConnectionPool
) -> RankedRun[ChunkCandidate]:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_bm25_candidate_rows(
            conn,
            query=request.query,
            predicate=predicate,
            limit=request.limit,
        )
    return make_run(
        "bm25",
        0.5,
        (
            (ChunkCandidate(chunk_id=row.chunk_id, page_id=row.page_id), row.rank)
            for row in rows
        ),
    )


async def retrieve_dense(
    request: RetrievalRequest, pool: AsyncConnectionPool
) -> RankedRun[ChunkCandidate]:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_dense_candidate_rows(
            conn,
            query_embedding=request.query_embedding,
            predicate=predicate,
            limit=request.limit,
        )
    return make_run(
        "dense",
        2.0,
        (
            (ChunkCandidate(chunk_id=row.chunk_id, page_id=row.page_id), row.similarity)
            for row in rows
        ),
    )
