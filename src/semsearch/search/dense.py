from psycopg_pool import AsyncConnectionPool

from semsearch import db
from semsearch.models import Candidate


class DenseRetriever:
    name = "dense"

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def retrieve(
        self, query: str, query_embedding: list[float], k: int
    ) -> list[Candidate]:
        async with self.pool.connection() as conn:
            rows = await db.fetch_dense_candidate_rows(
                conn,
                query_embedding=query_embedding,
                limit=k,
            )
        return [
            Candidate(
                chunk_id=chunk_id,
                page_id=page_id,
                url=url,
                title=title,
                content=content,
                scores={self.name: similarity},
            )
            for chunk_id, page_id, url, title, content, similarity in rows
        ]
