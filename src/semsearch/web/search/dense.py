from psycopg_pool import AsyncConnectionPool

from semsearch.web import db
from semsearch.web.search.base import RankedRun, RetrievalRequest
from semsearch.web.search.filters import compile_filters
from semsearch.web.search.models import ChunkCandidate


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
    return RankedRun(
        "dense",
        tuple(
            ChunkCandidate(
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
