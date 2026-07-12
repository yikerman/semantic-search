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
               (SELECT count(*) FROM pages),
               (SELECT count(*) FROM chunks),
               (SELECT count(*) FROM crawl_jobs WHERE failed_at IS NULL),
               (SELECT count(*) FROM crawl_jobs
                WHERE failed_at IS NULL AND attempt_count > 0),
               (SELECT count(*) FROM crawl_jobs WHERE failed_at IS NOT NULL)
        """
    )
    return IndexStats(*cast(tuple, await cur.fetchone()))


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
