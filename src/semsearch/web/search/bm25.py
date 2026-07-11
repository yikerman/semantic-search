from psycopg_pool import AsyncConnectionPool

from semsearch.web import db
from semsearch.web.search.base import RankedRun, RetrievalRequest
from semsearch.web.search.filters import compile_filters
from semsearch.web.search.models import Candidate


async def retrieve_bm25(
    request: RetrievalRequest, *, pool: AsyncConnectionPool
) -> RankedRun:
    predicate = compile_filters(request.filters, page_alias="p")
    async with pool.connection() as conn:
        rows = await db.fetch_bm25_candidate_rows(
            conn,
            query=request.query,
            predicate=predicate,
            limit=request.limit,
        )
    return RankedRun(
        "bm25",
        tuple(
            Candidate(
                chunk_id=row.chunk_id,
                page_id=row.page_id,
                url=row.url,
                title=row.title,
                content=row.content,
                scores={"bm25": row.rank},
            )
            for row in rows
        ),
    )
