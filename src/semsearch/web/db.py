from collections.abc import Sequence
from dataclasses import dataclass

import psycopg
from pgvector import HalfVector
from psycopg import sql

from semsearch.web.search.filters import SqlPredicate


@dataclass(frozen=True, slots=True)
class DenseCandidateRecord:
    chunk_id: int
    page_id: int
    url: str
    title: str | None
    content: str
    similarity: float


async def ping(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SELECT 1")


async def fetch_dense_candidate_rows(
    conn: psycopg.AsyncConnection,
    *,
    query_embedding: Sequence[float],
    predicate: SqlPredicate,
    limit: int,
) -> list[DenseCandidateRecord]:
    embedding = HalfVector(list(query_embedding))
    cur = await conn.execute(
        sql.SQL(
            """
        SELECT c.id, c.page_id, p.url, p.title, c.content,
               1 - (c.embedding <=> %s) AS similarity
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        WHERE {predicate}
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """
        ).format(predicate=predicate.clause),
        (embedding, *predicate.params, embedding, limit),
    )
    return [DenseCandidateRecord(*row) for row in await cur.fetchall()]
