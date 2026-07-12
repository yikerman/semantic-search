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


@dataclass(frozen=True, slots=True)
class Bm25CandidateRecord:
    chunk_id: int
    page_id: int
    url: str
    title: str | None
    content: str
    rank: float


async def ping(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SELECT 1")


async def fetch_lead_chunks(
    conn: psycopg.AsyncConnection, *, page_ids: Sequence[int]
) -> dict[int, str]:
    cur = await conn.execute(
        """
        SELECT page_id, content FROM chunks
        WHERE page_id = ANY(%s) AND chunk_index = 0
        """,
        (list(page_ids),),
    )
    return {page_id: content for page_id, content in await cur.fetchall()}


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


async def fetch_bm25_candidate_rows(
    conn: psycopg.AsyncConnection,
    *,
    query: str,
    predicate: SqlPredicate,
    limit: int,
) -> list[Bm25CandidateRecord]:
    cur = await conn.execute(
        sql.SQL(
            """
        WITH search_query AS (
            SELECT websearch_to_tsquery('simple', %s) AS value
        )
        SELECT c.id, c.page_id, p.url, p.title, c.content,
               ts_rank_cd(c.search_vector, search_query.value) AS rank
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        CROSS JOIN search_query
        WHERE c.search_vector @@ search_query.value
          AND {predicate}
        ORDER BY rank DESC, c.id
        LIMIT %s
        """
        ).format(predicate=predicate.clause),
        (query, *predicate.params, limit),
    )
    return [Bm25CandidateRecord(*row) for row in await cur.fetchall()]
