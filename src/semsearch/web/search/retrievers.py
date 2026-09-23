from psycopg_pool import AsyncConnectionPool

from semsearch.web import db
from semsearch.web.search.filters import compile_filters
from semsearch.web.search.models import (
    RankedRun,
    RetrievalRequest,
    make_run,
)


async def retrieve_bm25(
    request: RetrievalRequest, pool: AsyncConnectionPool
) -> RankedRun:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_bm25_candidate_rows(
            conn,
            query=request.query,
            predicate=predicate,
            limit=request.limit,
        )
    return make_run("bm25", 0.5, rows)


async def retrieve_dense(
    request: RetrievalRequest, pool: AsyncConnectionPool
) -> RankedRun:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_dense_candidate_rows(
            conn,
            query_embedding=request.query_embedding,
            predicate=predicate,
            limit=request.limit,
        )
    return make_run("dense", 2.0, rows)
