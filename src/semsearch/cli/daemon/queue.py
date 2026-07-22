from collections.abc import Sequence
from uuid import UUID, uuid4

import psycopg

from semsearch.cli.models import CrawlAttempt


async def known_urls(conn: psycopg.AsyncConnection, urls: Sequence[str]) -> set[str]:
    if not urls:
        return set()
    cur = await conn.execute(
        """
        SELECT url FROM pages WHERE url = ANY(%s)
        UNION
        SELECT url FROM crawl_jobs WHERE url = ANY(%s)
        """,
        (list(urls), list(urls)),
    )
    return {row[0] for row in await cur.fetchall()}


async def enqueue_urls(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    urls: Sequence[str],
    source: str,
) -> int:
    if not urls:
        return 0
    cur = await conn.execute(
        """
        INSERT INTO crawl_jobs (site_id, url, source)
        SELECT %s, candidate.url, %s
        FROM unnest(%s::text[]) AS candidate(url)
        WHERE NOT EXISTS (SELECT 1 FROM pages WHERE pages.url = candidate.url)
        ON CONFLICT (url) DO NOTHING
        """,
        (site_id, source, list(urls)),
    )
    return cur.rowcount


async def claim_crawl_job(
    conn: psycopg.AsyncConnection,
    *,
    exclude_site_ids: Sequence[int] = (),
    lease_seconds: int = 600,
) -> CrawlAttempt | None:
    token = uuid4()
    cur = await conn.execute(
        """
        WITH candidate AS (
            SELECT id
            FROM crawl_jobs
            WHERE next_attempt_at IS NOT NULL
              AND next_attempt_at <= now()
              AND (lease_until IS NULL OR lease_until < now())
              AND site_id != ALL(%s::bigint[])
            ORDER BY next_attempt_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE crawl_jobs
        SET lease_until = now() + make_interval(secs => %s),
            lease_token = %s
        FROM candidate
        WHERE crawl_jobs.id = candidate.id
        RETURNING crawl_jobs.id, crawl_jobs.site_id, crawl_jobs.url,
                  crawl_jobs.source, crawl_jobs.attempt_count,
                  crawl_jobs.lease_token
        """,
        (list(exclude_site_ids), lease_seconds, token),
    )
    row = await cur.fetchone()
    return None if row is None else _crawl_attempt_from_row(row)


def _crawl_attempt_from_row(row: object) -> CrawlAttempt:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 6:
        raise ValueError("invalid crawl attempt database row")
    job_id, site_id, url, source, attempt_count, lease_token = row
    if (
        not isinstance(job_id, int)
        or isinstance(job_id, bool)
        or not isinstance(site_id, int)
        or isinstance(site_id, bool)
        or not isinstance(url, str)
        or not isinstance(source, str)
        or not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 0
        or not isinstance(lease_token, UUID)
    ):
        raise ValueError("invalid crawl attempt database row")
    return CrawlAttempt(job_id, site_id, url, source, attempt_count, lease_token)


async def retry_crawl_job(
    conn: psycopg.AsyncConnection,
    *,
    job_id: int,
    lease_token: UUID,
    error: str,
    delay_seconds: int,
) -> bool:
    cur = await conn.execute(
        """
        UPDATE crawl_jobs
        SET attempt_count = attempt_count + 1,
            next_attempt_at = now() + make_interval(secs => %s),
            lease_until = NULL,
            lease_token = NULL,
            last_error = %s
        WHERE id = %s AND lease_token = %s AND lease_until >= now()
        """,
        (delay_seconds, error, job_id, lease_token),
    )
    return cur.rowcount == 1


async def fail_crawl_job(
    conn: psycopg.AsyncConnection,
    *,
    job_id: int,
    lease_token: UUID,
    error: str,
) -> bool:
    cur = await conn.execute(
        """
        UPDATE crawl_jobs
        SET attempt_count = attempt_count + 1,
            next_attempt_at = NULL,
            lease_until = NULL,
            lease_token = NULL,
            last_error = %s,
            failed_at = now()
        WHERE id = %s AND lease_token = %s AND lease_until >= now()
        """,
        (error, job_id, lease_token),
    )
    return cur.rowcount == 1


async def renew_crawl_lease(
    conn: psycopg.AsyncConnection,
    *,
    job_id: int,
    lease_token: UUID,
    lease_seconds: int = 600,
) -> bool:
    cur = await conn.execute(
        """
        UPDATE crawl_jobs
        SET lease_until = now() + make_interval(secs => %s)
        WHERE id = %s AND lease_token = %s AND lease_until >= now()
        """,
        (lease_seconds, job_id, lease_token),
    )
    return cur.rowcount == 1


async def complete_crawl_job(
    conn: psycopg.AsyncConnection, *, job_id: int, lease_token: UUID
) -> bool:
    cur = await conn.execute(
        """
        DELETE FROM crawl_jobs
        WHERE id = %s AND lease_token = %s AND lease_until >= now()
        """,
        (job_id, lease_token),
    )
    return cur.rowcount == 1
