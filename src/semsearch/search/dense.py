from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.models import Candidate
from semsearch.search.base import RankedRun, RetrievalRequest
from semsearch.search.filters import compile_filters


class DenseRetriever:
    name = "dense"

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def retrieve(self, request: RetrievalRequest) -> RankedRun:
        predicate = compile_filters(request.filters, page_alias="p")
        async with self.pool.connection() as conn:
            rows = await db.fetch_dense_candidate_rows(
                conn,
                query_embedding=request.query_embedding,
                predicate=predicate,
                limit=request.limit,
            )
        return RankedRun(
            self.name,
            tuple(
                Candidate(
                    chunk_id=row.chunk_id,
                    page_id=row.page_id,
                    url=row.url,
                    title=row.title,
                    content=row.content,
                    scores={self.name: row.similarity},
                )
                for row in rows
            ),
        )
