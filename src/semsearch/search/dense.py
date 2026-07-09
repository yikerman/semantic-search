from pgvector import HalfVector
from psycopg_pool import AsyncConnectionPool

from semsearch.models import Candidate


class DenseRetriever:
    name = "dense"

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def retrieve(
        self, query: str, query_embedding: list[float], k: int
    ) -> list[Candidate]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT c.id, c.page_id, p.url, p.title, c.content,
                       1 - (c.embedding <=> %s) AS similarity
                FROM chunks c
                JOIN pages p ON p.id = c.page_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (HalfVector(query_embedding), HalfVector(query_embedding), k),
            )
            rows = await cur.fetchall()
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
