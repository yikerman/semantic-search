from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import psycopg


@dataclass(frozen=True, slots=True)
class IndexStats:
    site_count: int
    page_count: int
    chunk_count: int
    queued_count: int
    retrying_count: int
    failed_count: int


@dataclass(frozen=True, slots=True)
class FailedCrawlJob:
    url: str
    attempt_count: int
    last_error: str


async def fetch_index_stats(conn: psycopg.AsyncConnection) -> IndexStats:
    cur = await conn.execute(
        """
        SELECT (SELECT count(*) FROM sites),
               estimates.page_count,
               estimates.chunk_count,
               jobs.queued_count,
               jobs.retrying_count,
               jobs.failed_count
        FROM (
            SELECT COALESCE(
                       max(n_live_tup) FILTER (WHERE relname = 'pages'), 0
                   ) AS page_count,
                   COALESCE(
                       max(n_live_tup) FILTER (WHERE relname = 'chunks'), 0
                   ) AS chunk_count
            FROM pg_stat_user_tables
            WHERE schemaname = current_schema()
              AND relname IN ('pages', 'chunks')
        ) AS estimates
        CROSS JOIN (
            SELECT count(*) FILTER (WHERE failed_at IS NULL) AS queued_count,
                   count(*) FILTER (
                       WHERE failed_at IS NULL AND attempt_count > 0
                   ) AS retrying_count,
                   count(*) FILTER (
                       WHERE failed_at IS NOT NULL
                   ) AS failed_count
            FROM crawl_jobs
        ) AS jobs
        """
    )
    return _index_stats_from_row(await cur.fetchone())


def _index_stats_from_row(row: object) -> IndexStats:
    if (
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes))
        or len(row) != 6
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in row
        )
    ):
        raise ValueError("invalid index stats database row")
    values = cast(tuple[int, int, int, int, int, int], tuple(row))
    return IndexStats(*values)


async def list_failed_jobs(
    conn: psycopg.AsyncConnection, *, limit: int = 10
) -> list[FailedCrawlJob]:
    cur = await conn.execute(
        """
        SELECT url, attempt_count, last_error
        FROM crawl_jobs
        WHERE failed_at IS NOT NULL
        ORDER BY failed_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return [FailedCrawlJob(row[0], row[1], row[2]) for row in await cur.fetchall()]
