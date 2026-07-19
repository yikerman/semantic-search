from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import psycopg
from pgvector import HalfVector
from psycopg import sql

from semsearch.web.search.filters import SqlPredicate


@dataclass(frozen=True, slots=True)
class DenseCandidateRecord:
    chunk_id: int
    page_id: int
    similarity: float


@dataclass(frozen=True, slots=True)
class Bm25CandidateRecord:
    chunk_id: int
    page_id: int
    rank: float


@dataclass(frozen=True, slots=True)
class PageRecord:
    page_id: int
    url: str
    title: str | None
    content: str


@dataclass(frozen=True, slots=True)
class RecentActivity:
    url: str
    status: Literal["success", "failure"]
    occurred_at: datetime
    attempt_count: int | None
    detail: str | None


def _page_record_from_row(row: tuple[object, ...]) -> PageRecord:
    if len(row) != 4:
        raise ValueError("invalid page database row")
    page_id, url, title, content = row
    if (
        not isinstance(page_id, int)
        or isinstance(page_id, bool)
        or not isinstance(url, str)
        or (title is not None and not isinstance(title, str))
        or not isinstance(content, str)
    ):
        raise ValueError("invalid page database row")
    return PageRecord(page_id, url, title, content)


def _dense_candidate_from_row(row: tuple[object, ...]) -> DenseCandidateRecord:
    if len(row) != 3:
        raise ValueError("invalid dense candidate database row")
    chunk_id, page_id, similarity = row
    if (
        not isinstance(chunk_id, int)
        or isinstance(chunk_id, bool)
        or not isinstance(page_id, int)
        or isinstance(page_id, bool)
        or not isinstance(similarity, (int, float))
        or isinstance(similarity, bool)
    ):
        raise ValueError("invalid dense candidate database row")
    return DenseCandidateRecord(chunk_id, page_id, float(similarity))


def _bm25_candidate_from_row(row: tuple[object, ...]) -> Bm25CandidateRecord:
    if len(row) != 3:
        raise ValueError("invalid BM25 candidate database row")
    chunk_id, page_id, rank = row
    if (
        not isinstance(chunk_id, int)
        or isinstance(chunk_id, bool)
        or not isinstance(page_id, int)
        or isinstance(page_id, bool)
        or not isinstance(rank, (int, float))
        or isinstance(rank, bool)
    ):
        raise ValueError("invalid BM25 candidate database row")
    return Bm25CandidateRecord(chunk_id, page_id, float(rank))


def _recent_activity_from_row(row: tuple[object, ...]) -> RecentActivity:
    if len(row) != 5:
        raise ValueError("invalid recent activity database row")
    url, status, occurred_at, attempt_count, detail = row
    if not isinstance(url, str):
        raise ValueError("invalid recent activity database row")
    checked_status: Literal["success", "failure"]
    if status == "success":
        checked_status = "success"
    elif status == "failure":
        checked_status = "failure"
    else:
        raise ValueError("invalid recent activity database row")
    if not isinstance(occurred_at, datetime):
        raise ValueError("invalid recent activity database row")
    if attempt_count is not None and not isinstance(attempt_count, int):
        raise ValueError("invalid recent activity database row")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("invalid recent activity database row")
    return RecentActivity(url, checked_status, occurred_at, attempt_count, detail)


async def ping(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SELECT 1")


async def list_available_languages(conn: psycopg.AsyncConnection) -> list[str]:
    cur = await conn.execute(
        """
        SELECT DISTINCT language
        FROM pages
        WHERE language IS NOT NULL
        ORDER BY language
        """
    )
    languages: list[str] = []
    for row in await cur.fetchall():
        if len(row) != 1:
            raise ValueError("invalid page language database row")
        language = row[0]
        if (
            not isinstance(language, str)
            or len(language) != 2
            or not language.isascii()
            or not language.isalpha()
            or not language.islower()
        ):
            raise ValueError("invalid page language database row")
        languages.append(language)
    return languages


async def fetch_pages(
    conn: psycopg.AsyncConnection, *, page_ids: Sequence[int]
) -> dict[int, PageRecord]:
    cur = await conn.execute(
        """
        SELECT id, url, title, content
        FROM pages
        WHERE id = ANY(%s)
        """,
        (list(page_ids),),
    )
    records = [_page_record_from_row(row) for row in await cur.fetchall()]
    return {record.page_id: record for record in records}


async def list_recent_activity(
    conn: psycopg.AsyncConnection, *, limit: int = 10
) -> list[RecentActivity]:
    cur = await conn.execute(
        """
        SELECT url, status, occurred_at, attempt_count, detail
        FROM (
            SELECT url, 'success' AS status, fetched_at AS occurred_at,
                   NULL::int AS attempt_count, NULL::text AS detail
            FROM pages
            UNION ALL
            SELECT url, 'failure' AS status, failed_at AS occurred_at,
                   attempt_count, last_error AS detail
            FROM crawl_jobs
            WHERE failed_at IS NOT NULL
        ) AS activity
        ORDER BY occurred_at DESC, url
        LIMIT %s
        """,
        (limit,),
    )
    return [_recent_activity_from_row(row) for row in await cur.fetchall()]


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
        SELECT c.id, c.page_id, 1 - (c.embedding <=> %s) AS similarity
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        WHERE {predicate}
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """
        ).format(predicate=predicate.clause),
        (embedding, *predicate.params, embedding, limit),
    )
    return [_dense_candidate_from_row(row) for row in await cur.fetchall()]


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
        SELECT c.id, c.page_id,
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
    return [_bm25_candidate_from_row(row) for row in await cur.fetchall()]
