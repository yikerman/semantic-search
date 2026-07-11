from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.models import Candidate
from semsearch.search.base import RankedRun, RetrievalRequest
from semsearch.search.filters import compile_filters


async def retrieve_dense(
    request: RetrievalRequest, *, pool: AsyncConnectionPool
) -> RankedRun:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_dense_candidate_rows(
            conn,
            query_embedding=request.query_embedding,
            predicate=predicate,
            limit=request.limit,
        )
    return RankedRun(
        "dense",
        tuple(
            Candidate(
                chunk_id=row.chunk_id,
                page_id=row.page_id,
                url=row.url,
                title=row.title,
                content=row.content,
                scores={"dense": row.similarity},
            )
            for row in rows
        ),
    )
