import importlib.resources
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, LiteralString, cast
from uuid import UUID, uuid4

import psycopg
from pgvector import HalfVector
from psycopg.rows import dict_row

from semsearch.cli.models import CrawlAttempt, Site
from semsearch.share.config import Settings


@dataclass(frozen=True, slots=True)
class ChunkInsert:
    chunk_index: int
    content: str
    char_count: int
    embedding: Sequence[float]


# Rows are mapped to Site by column name; keep these names equal to Site's fields.
SITE_COLUMNS = """
sites.id, sites.base_url, sites.sitemap_url, sites.feed_url,
sites.last_polled_at, sites.next_poll_at, sites.feed_etag,
sites.feed_last_modified, sites.poll_failures, sites.sync_error,
sites.history_pending, sites.history_error
"""


def load_schema_sql(settings: Settings) -> LiteralString:
    raw = (
        importlib.resources.files("semsearch.share").joinpath("schema.sql").read_text()
    )
    return cast(LiteralString, raw.format(embedding_dim=settings.embedding_dim))


async def init_schema(settings: Settings) -> None:
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await conn.execute(load_schema_sql(settings))
        await conn.commit()


async def upsert_site_config(
    conn: psycopg.AsyncConnection,
    *,
    base_url: str,
    sitemap_url: str | None,
    feed_url: str,
    initial_poll_delay_seconds: int,
) -> Site:
    cur = conn.cursor(row_factory=dict_row)
    await cur.execute(
        f"""
        INSERT INTO sites (base_url, sitemap_url, feed_url, next_poll_at)
        VALUES (%s, %s, %s, now() + make_interval(secs => %s))
        ON CONFLICT (base_url) DO UPDATE SET
            sitemap_url = EXCLUDED.sitemap_url,
            feed_url = EXCLUDED.feed_url,
            feed_etag = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.feed_etag
                ELSE NULL
            END,
            feed_last_modified = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.feed_last_modified
                ELSE NULL
            END,
            next_poll_at = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.next_poll_at
                ELSE EXCLUDED.next_poll_at
            END,
            history_pending = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.history_pending
                ELSE false
            END,
            history_error = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.history_error
                ELSE NULL
            END,
            poll_failures = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.poll_failures
                ELSE 0
            END,
            sync_error = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.sync_error
                ELSE NULL
            END,
            poll_lease_until = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.poll_lease_until
                ELSE NULL
            END,
            poll_lease_token = CASE
                WHEN sites.feed_url = EXCLUDED.feed_url THEN sites.poll_lease_token
                ELSE NULL
            END
        RETURNING {SITE_COLUMNS}
        """,
        (base_url, sitemap_url, feed_url, initial_poll_delay_seconds),
    )
    return Site(**cast(dict[str, Any], await cur.fetchone()))


async def list_site_configs(conn: psycopg.AsyncConnection) -> list[Site]:
    cur = conn.cursor(row_factory=dict_row)
    await cur.execute(f"SELECT {SITE_COLUMNS} FROM sites ORDER BY base_url")
    return [Site(**row) for row in await cur.fetchall()]


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


async def mark_history_pending(
    conn: psycopg.AsyncConnection, *, site_id: int, lease_token: UUID
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET history_pending = true, history_error = NULL
        WHERE id = %s AND poll_lease_token = %s
        """,
        (site_id, lease_token),
    )


async def finish_history(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET history_pending = false, history_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (error, site_id, lease_token),
    )


async def mark_poll_succeeded(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    etag: str | None,
    modified: str | None,
    interval_seconds: int,
    lease_token: UUID,
    sync_error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET last_polled_at = now(),
            next_poll_at = now() + make_interval(secs => %s),
            feed_etag = COALESCE(%s, feed_etag),
            feed_last_modified = COALESCE(%s, feed_last_modified),
            poll_failures = 0,
            poll_lease_until = NULL,
            poll_lease_token = NULL,
            sync_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (interval_seconds, etag, modified, sync_error, site_id, lease_token),
    )


async def mark_poll_failed(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    error: str,
    interval_seconds: int,
) -> None:
    # Backoff caps at the healthy poll interval so a failing feed is never
    # polled more often than a working one.
    await conn.execute(
        """
        UPDATE sites
        SET poll_failures = poll_failures + 1,
            next_poll_at = now() + make_interval(
                secs => LEAST(%s, 300 * (2 ^ LEAST(poll_failures, 10)))
            ),
            poll_lease_until = NULL,
            poll_lease_token = NULL,
            sync_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (interval_seconds, error, site_id, lease_token),
    )


async def scatter_poll_schedule(
    conn: psycopg.AsyncConnection, *, interval_seconds: int
) -> None:
    await conn.execute(
        """
        WITH overdue AS (
            SELECT id,
                   row_number() OVER (ORDER BY id) - 1 AS position,
                   count(*) OVER () AS total
            FROM sites
            WHERE next_poll_at IS NULL OR next_poll_at <= now()
        )
        UPDATE sites
        SET next_poll_at = now() + make_interval(
            secs => (%s * overdue.position / GREATEST(overdue.total, 1))::int
        )
        FROM overdue
        WHERE sites.id = overdue.id
        """,
        (interval_seconds,),
    )


async def claim_due_site(
    conn: psycopg.AsyncConnection, *, lease_seconds: int = 600
) -> tuple[Site, UUID] | None:
    token = uuid4()
    cur = conn.cursor(row_factory=dict_row)
    await cur.execute(
        f"""
        WITH candidate AS (
            SELECT id
            FROM sites
            WHERE next_poll_at <= now()
              AND (poll_lease_until IS NULL OR poll_lease_until < now())
            ORDER BY next_poll_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE sites
        SET poll_lease_until = now() + make_interval(secs => %s),
            poll_lease_token = %s
        FROM candidate
        WHERE sites.id = candidate.id
        RETURNING {SITE_COLUMNS}, sites.poll_lease_token
        """,
        (lease_seconds, token),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    granted_token = cast(UUID, row.pop("poll_lease_token"))
    return Site(**row), granted_token


async def renew_poll_lease(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    lease_seconds: int = 600,
) -> bool:
    cur = await conn.execute(
        """
        UPDATE sites
        SET poll_lease_until = now() + make_interval(secs => %s)
        WHERE id = %s AND poll_lease_token = %s
        """,
        (lease_seconds, site_id, lease_token),
    )
    return cur.rowcount == 1


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


async def page_exists(conn: psycopg.AsyncConnection, *, url: str) -> bool:
    cur = await conn.execute("SELECT 1 FROM pages WHERE url = %s", (url,))
    return await cur.fetchone() is not None


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


async def insert_page(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    url: str,
    title: str | None,
    published_at: datetime | None,
    language: str,
) -> int | None:
    """Insert a new page, returning its id, or ``None`` if the URL already exists.

    URL is page identity and existing URLs are append-only, so a conflict means
    another writer already indexed this page; the caller must skip rather than
    overwrite its chunks.
    """
    cur = await conn.execute(
        """
        INSERT INTO pages
            (site_id, url, title, published_at, language, fetched_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (url) DO NOTHING
        RETURNING id
        """,
        (site_id, url, title, published_at, language),
    )
    row = await cur.fetchone()
    return None if row is None else row[0]


async def insert_page_chunks(
    conn: psycopg.AsyncConnection,
    *,
    page_id: int,
    chunks: Iterable[ChunkInsert],
) -> None:
    async with conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO chunks
                (page_id, chunk_index, content, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    page_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.char_count,
                    HalfVector(list(chunk.embedding)),
                )
                for chunk in chunks
            ],
        )
