from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.ingest.extract import ExtractedPage
from semsearch.cli.models import Site


async def discover(
    pool: AsyncConnectionPool, site_id: int, urls: Sequence[str], source: str
) -> list[str]:
    if not urls:
        return []
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """INSERT INTO article_urls (site_id, url, source)
            SELECT %s, unnest(%s::text[]), %s ON CONFLICT (url) DO NOTHING""",
            (site_id, list(urls), source),
        )
        cur = await conn.execute(
            """SELECT url FROM article_urls WHERE site_id = %s AND url = ANY(%s)
            AND status = 'pending' AND next_attempt_at <= now() ORDER BY id""",
            (site_id, list(urls)),
        )
        return _urls(await cur.fetchall())


async def pending(pool: AsyncConnectionPool, site_id: int, limit: int) -> list[str]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT url FROM article_urls WHERE site_id = %s AND status = 'pending'
            AND next_attempt_at <= now() ORDER BY id LIMIT %s""",
            (site_id, limit),
        )
        return _urls(await cur.fetchall())


def _urls(rows: Sequence[Sequence[object]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        if len(row) != 1 or not isinstance(row[0], str):
            raise ValueError("invalid article URL row")
        result.append(row[0])
    return result


async def store_page(
    pool: AsyncConnectionPool, site: Site, url: str, page: ExtractedPage
) -> None:
    async with pool.connection() as conn, conn.transaction():
        await db.insert_page(
            conn,
            site_id=site.id,
            url=url,
            title=page.title,
            content=page.text,
            published_at=page.published_at,
            language=page.language,
        )
        await conn.execute(
            """UPDATE article_urls SET status = 'stored', last_error = NULL,
            updated_at = now() WHERE url = %s""",
            (url,),
        )


async def outcome(
    pool: AsyncConnectionPool,
    url: str,
    status: Literal["rejected", "failed", "pending"],
    reason: str,
    interval: int,
) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """UPDATE article_urls SET
            failed_batches = failed_batches + CASE WHEN %s = 'pending' THEN 1 ELSE 0 END,
            status = CASE WHEN %s = 'pending' AND failed_batches >= 2 THEN 'failed' ELSE %s END,
            last_error = %s, next_attempt_at = now() + make_interval(secs => %s),
            updated_at = now() WHERE url = %s AND status = 'pending'""",
            (status, status, status, reason[:2000], interval, url),
        )


async def reset_failures(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("""UPDATE article_urls SET status = 'pending', failed_batches = 0,
            next_attempt_at = now(), last_error = NULL WHERE status IN ('rejected', 'failed')""")


async def finish_site(
    pool: AsyncConnectionPool, site: Site, *, history_complete: bool, error: str | None
) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """UPDATE sites SET last_crawled_at = now(), last_error = %s,
            history_complete = %s WHERE id = %s
            AND start_url = %s AND feed_url = %s AND sitemap_url = %s""",
            (
                error,
                history_complete,
                site.id,
                site.start_url,
                site.feed_url,
                site.sitemap_url,
            ),
        )


async def save_cooldown(
    pool: AsyncConnectionPool, origin: str, until: datetime
) -> None:
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            """INSERT INTO origin_cooldowns (origin, until_at) VALUES (%s, %s)
            ON CONFLICT (origin) DO UPDATE SET until_at = GREATEST(origin_cooldowns.until_at, EXCLUDED.until_at)""",
            (origin, until),
        )


async def load_cooldowns(pool: AsyncConnectionPool) -> dict[str, datetime]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT origin, until_at FROM origin_cooldowns WHERE until_at > now()"
        )
        result: dict[str, datetime] = {}
        for origin, until in await cur.fetchall():
            if (
                not isinstance(origin, str)
                or not isinstance(until, datetime)
                or until.utcoffset() is None
            ):
                raise ValueError("invalid cooldown row")
            result[origin] = until
        return result
