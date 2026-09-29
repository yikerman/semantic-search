from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import psycopg


@dataclass(frozen=True, slots=True)
class IndexStats:
    site_count: int
    page_count: int
    indexed_count: int
    queued_count: int
    retrying_count: int
    failed_count: int
    rejected_count: int = 0
    pending_index_count: int = 0
    failed_index_count: int = 0
    rejected_index_count: int = 0


@dataclass(frozen=True, slots=True)
class FailedArticle:
    url: str
    failed_batches: int
    last_error: str


async def fetch_index_stats(conn: psycopg.AsyncConnection) -> IndexStats:
    cur = await conn.execute(
        """
        SELECT count(*),
               coalesce(sum(page_count), 0)::bigint, coalesce(sum(indexed_count), 0)::bigint,
               coalesce(sum(queued_count), 0)::bigint, coalesce(sum(retrying_count), 0)::bigint,
               coalesce(sum(failed_count), 0)::bigint, coalesce(sum(rejected_count), 0)::bigint,
               coalesce(sum(page_count - indexed_count - rejected_index_count), 0)::bigint,
               coalesce(sum(failed_index_count), 0)::bigint, coalesce(sum(rejected_index_count), 0)::bigint
        FROM site_stats
        """
    )
    return _index_stats_from_row(await cur.fetchone())


def _index_stats_from_row(row: object) -> IndexStats:
    if (
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes))
        or len(row) != 10
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in row
        )
    ):
        raise ValueError("invalid index stats database row")
    values = cast(tuple[int, int, int, int, int, int, int, int, int, int], tuple(row))
    return IndexStats(*values)


async def list_failed_articles(
    conn: psycopg.AsyncConnection, *, limit: int = 10
) -> list[FailedArticle]:
    cur = await conn.execute(
        """
        SELECT url, failed_batches, last_error
        FROM article_urls
        WHERE status = 'failed'
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return [FailedArticle(row[0], row[1], row[2]) for row in await cur.fetchall()]
