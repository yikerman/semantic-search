import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import psycopg
from pgvector import Vector
from psycopg import sql

from semsearch.web.search.filters import SqlPredicate
from semsearch.web.search.models import PageCandidate


@dataclass(frozen=True, slots=True)
class RecentActivity:
    url: str
    status: Literal["success", "failure"]
    occurred_at: datetime
    failed_batches: int | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class IndexingIssue:
    url: str
    detail: str
    rejected: bool


async def list_indexing_issues(
    conn: psycopg.AsyncConnection, *, limit: int = 10
) -> list[IndexingIssue]:
    cur = await conn.execute(
        """
        SELECT url, index_error, index_rejected
        FROM pages
        WHERE indexed_at IS NULL AND index_error IS NOT NULL
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,),
    )
    issues: list[IndexingIssue] = []
    for row in await cur.fetchall():
        if (
            len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or type(row[2]) is not bool
        ):
            raise ValueError("invalid indexing issue database row")
        issues.append(IndexingIssue(*row))
    return issues


def _scored_page_from_row(row: tuple[object, ...]) -> tuple[PageCandidate, float]:
    if len(row) != 6:
        raise ValueError("invalid scored page database row")
    page_id, url, title, content, published_at, score = row
    if (
        type(page_id) is not int
        or page_id <= 0
        or not isinstance(url, str)
        or (title is not None and not isinstance(title, str))
        or not isinstance(content, str)
        or (
            published_at is not None
            and (
                not isinstance(published_at, datetime)
                or published_at.utcoffset() is None
            )
        )
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
    ):
        raise ValueError("invalid scored page database row")
    return PageCandidate(page_id, url, title, content, published_at), float(score)


def _recent_activity_from_row(row: tuple[object, ...]) -> RecentActivity:
    if len(row) != 5:
        raise ValueError("invalid recent activity database row")
    url, status, occurred_at, failed_batches, detail = row
    if not isinstance(url, str):
        raise ValueError("invalid recent activity database row")  # noqa: TRY004
    checked_status: Literal["success", "failure"]
    if status == "success":
        checked_status = "success"
    elif status == "failure":
        checked_status = "failure"
    else:
        raise ValueError("invalid recent activity database row")
    if not isinstance(occurred_at, datetime):
        raise ValueError("invalid recent activity database row")  # noqa: TRY004
    if failed_batches is not None and not isinstance(failed_batches, int):
        raise ValueError("invalid recent activity database row")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("invalid recent activity database row")
    return RecentActivity(url, checked_status, occurred_at, failed_batches, detail)


async def ping(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SELECT 1")


async def list_available_languages(conn: psycopg.AsyncConnection) -> list[str]:
    cur = await conn.execute(
        """
        SELECT DISTINCT language
        FROM pages
        WHERE language IS NOT NULL AND indexed_at IS NOT NULL
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
            or len(language) not in (2, 3)
            or not language.isascii()
            or not language.isalpha()
            or not language.islower()
        ):
            raise ValueError("invalid page language database row")
        languages.append(language)
    return languages


async def list_recent_activity(
    conn: psycopg.AsyncConnection, *, limit: int = 10
) -> list[RecentActivity]:
    cur = await conn.execute(
        """
        SELECT url, status, occurred_at, failed_batches, detail
        FROM (
            SELECT url, 'success' AS status, fetched_at AS occurred_at,
                   NULL::int AS failed_batches, NULL::text AS detail
            FROM pages
            UNION ALL
            SELECT url, 'failure' AS status, updated_at AS occurred_at,
                   failed_batches, last_error AS detail
            FROM article_urls
            WHERE status IN ('failed', 'rejected')
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
) -> list[tuple[PageCandidate, float]]:
    embedding = Vector(list(query_embedding))
    # VectorChord 1.1.1's rabitq8 <=> returns negative cosine, not 1 - cosine.
    cur = await conn.execute(
        sql.SQL(
            """
        SELECT p.id, p.url, p.title, p.content, p.published_at,
               -(p.embedding <=> quantize_to_rabitq8(%s::vector)) AS similarity
        FROM pages p
        WHERE p.indexed_at IS NOT NULL AND {predicate}
        ORDER BY p.embedding <=> quantize_to_rabitq8(%s::vector)
        LIMIT %s
        """
        ).format(predicate=predicate.clause),
        (embedding, *predicate.params, embedding, limit),
    )
    return [_scored_page_from_row(row) for row in await cur.fetchall()]


async def fetch_bm25_candidate_rows(
    conn: psycopg.AsyncConnection,
    *,
    query: str,
    predicate: SqlPredicate,
    limit: int,
) -> list[tuple[PageCandidate, float]]:
    cur = await conn.execute(
        sql.SQL(
            """
        WITH search_query AS (
            SELECT to_bm25query(
                'pages_search_vector_bm25_idx'::regclass,
                tokenize(%s, 'semsearch_llmlingua2')::bm25vector
            ) AS value
        )
        SELECT p.id, p.url, p.title, p.content, p.published_at,
               -(p.search_vector <&> search_query.value) AS rank
        FROM pages p
        CROSS JOIN search_query
        WHERE p.indexed_at IS NOT NULL AND {predicate}
        ORDER BY p.search_vector <&> search_query.value
        LIMIT %s
        """
        ).format(predicate=predicate.clause),
        (query, *predicate.params, limit),
    )
    return [_scored_page_from_row(row) for row in await cur.fetchall()]
