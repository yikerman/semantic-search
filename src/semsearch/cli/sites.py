from collections.abc import Sequence
from urllib.parse import urljoin

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.models import Site
from semsearch.cli.url import normalize_origin, normalize_url, same_site

SITE_COLUMNS = "id, base_url, start_url, sitemap_url, feed_url, history_complete, last_crawled_at, last_error"


class SiteError(ValueError):
    pass


async def add_site(
    pool: AsyncConnectionPool,
    url: str,
    sitemap_url: str = "auto",
    feed_url: str = "auto",
) -> Site:
    try:
        start = normalize_url(url)
        origin = normalize_origin(start)
        feed = _source(start, feed_url)
        sitemap = _source(start, sitemap_url)
        if sitemap not in ("auto", "none") and not same_site(sitemap, origin):
            raise ValueError("Sitemap URL must use the site origin")
        if feed == sitemap == "none":
            raise ValueError("Enable at least one feed or sitemap source")
    except ValueError as exc:
        raise SiteError(str(exc)) from exc
    async with pool.connection() as conn, conn.transaction():
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute(
            f"""INSERT INTO sites (base_url, start_url, feed_url, sitemap_url)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (base_url) DO UPDATE SET
                start_url = EXCLUDED.start_url, feed_url = EXCLUDED.feed_url,
                sitemap_url = EXCLUDED.sitemap_url, history_complete = false,
                last_error = NULL
            RETURNING {SITE_COLUMNS}""",
            (origin, start, feed, sitemap),
        )
        return Site.model_validate(await cur.fetchone())


def _source(start: str, value: str) -> str:
    return value if value in ("auto", "none") else normalize_url(urljoin(start, value))


async def list_sites(pool: AsyncConnectionPool) -> list[Site]:
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute(f"SELECT {SITE_COLUMNS} FROM sites ORDER BY id")
        return [Site.model_validate(row) for row in await cur.fetchall()]


async def remove_sites(
    pool: AsyncConnectionPool, base_urls: Sequence[str]
) -> list[str]:
    async with pool.connection() as conn, conn.transaction():
        return await db.delete_site_configs(conn, base_urls=base_urls)


async def remove_site(pool: AsyncConnectionPool, url: str) -> str:
    origin = normalize_origin(url)
    removed = await remove_sites(pool, (origin,))
    if not removed:
        raise SiteError(f"Site is not configured: {origin}")
    return removed[0]
