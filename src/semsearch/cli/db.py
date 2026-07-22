import importlib.resources
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, LiteralString, cast

import psycopg
from pgvector import HalfVector
from psycopg.rows import dict_row

from semsearch.cli.models import Site
from semsearch.share.config import Settings


@dataclass(frozen=True, slots=True)
class ChunkInsert:
    start_offset: int
    content: str
    embedding: Sequence[float]


# Rows are mapped to Site by column name; keep these names equal to Site's fields.
SITE_COLUMNS: LiteralString = """
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


async def delete_site_configs(
    conn: psycopg.AsyncConnection, *, base_urls: Sequence[str]
) -> list[str]:
    if not base_urls:
        return []
    cur = await conn.execute(
        "DELETE FROM sites WHERE base_url = ANY(%s) RETURNING base_url",
        (list(base_urls),),
    )
    rows = await cur.fetchall()
    if any(
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes))
        or len(row) != 1
        or not isinstance(row[0], str)
        for row in rows
    ):
        raise ValueError("invalid deleted site database row")
    return sorted(row[0] for row in rows)


async def page_exists(conn: psycopg.AsyncConnection, *, url: str) -> bool:
    cur = await conn.execute("SELECT 1 FROM pages WHERE url = %s", (url,))
    return await cur.fetchone() is not None


async def insert_page(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    url: str,
    title: str | None,
    content: str,
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
            (site_id, url, title, content, published_at, language, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (url) DO NOTHING
        RETURNING id
        """,
        (site_id, url, title, content, published_at, language),
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
                (page_id, start_offset, content_length, embedding, search_vector)
            VALUES (
                %s, %s, %s, %s,
                tokenize(%s, 'semsearch_llmlingua2')::bm25vector
            )
            """,
            [
                (
                    page_id,
                    chunk.start_offset,
                    len(chunk.content),
                    HalfVector(list(chunk.embedding)),
                    chunk.content,
                )
                for chunk in chunks
            ],
        )
