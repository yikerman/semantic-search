import importlib.resources
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import LiteralString, cast

import psycopg
from pgvector import HalfVector

from semsearch.cli.models import Site
from semsearch.share.config import Settings


@dataclass(frozen=True, slots=True)
class ChunkInsert:
    chunk_index: int
    content: str
    char_count: int
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class IndexStats:
    site_count: int
    page_count: int
    chunk_count: int


SITE_COLUMNS = """
id, base_url, sitemap_url, feed_url, last_indexed_at,
last_polled_at, feed_etag, feed_last_modified
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


async def fetch_index_stats(conn: psycopg.AsyncConnection) -> IndexStats:
    cur = await conn.execute(
        """
        SELECT (SELECT count(*) FROM sites),
               (SELECT count(*) FROM pages),
               (SELECT count(*) FROM chunks)
        """
    )
    row = cast(tuple, await cur.fetchone())
    return IndexStats(*row)


async def upsert_site_config(
    conn: psycopg.AsyncConnection,
    *,
    base_url: str,
    sitemap_url: str | None,
    feed_url: str | None,
) -> Site:
    cur = await conn.execute(
        f"""
        INSERT INTO sites (base_url, sitemap_url, feed_url)
        VALUES (%s, %s, %s)
        ON CONFLICT (base_url) DO UPDATE SET
            sitemap_url = EXCLUDED.sitemap_url,
            feed_url = EXCLUDED.feed_url
        RETURNING {SITE_COLUMNS}
        """,
        (base_url, sitemap_url, feed_url),
    )
    row = cast(tuple, await cur.fetchone())
    return Site(*row)


async def list_site_configs(conn: psycopg.AsyncConnection) -> list[Site]:
    cur = await conn.execute(
        f"""
        SELECT {SITE_COLUMNS}
        FROM sites
        ORDER BY base_url
        """
    )
    return [Site(*row) for row in await cur.fetchall()]


async def find_site_config(
    conn: psycopg.AsyncConnection, *, base_url: str
) -> Site | None:
    cur = await conn.execute(
        f"""
        SELECT {SITE_COLUMNS}
        FROM sites
        WHERE base_url = %s
        """,
        (base_url,),
    )
    row = await cur.fetchone()
    return None if row is None else Site(*row)


async def mark_site_indexed(conn: psycopg.AsyncConnection, *, site_id: int) -> None:
    await conn.execute(
        "UPDATE sites SET last_indexed_at = now() WHERE id = %s",
        (site_id,),
    )


async def mark_site_polled(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    feed_etag: str | None,
    feed_last_modified: str | None,
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET last_polled_at = now(),
            feed_etag = COALESCE(%s, feed_etag),
            feed_last_modified = COALESCE(%s, feed_last_modified)
        WHERE id = %s
        """,
        (feed_etag, feed_last_modified, site_id),
    )


async def ensure_site_origin(conn: psycopg.AsyncConnection, *, base_url: str) -> int:
    cur = await conn.execute(
        """
        INSERT INTO sites (base_url) VALUES (%s)
        ON CONFLICT (base_url) DO UPDATE SET base_url = EXCLUDED.base_url
        RETURNING id
        """,
        (base_url,),
    )
    row = cast(tuple, await cur.fetchone())
    return row[0]


async def page_exists(conn: psycopg.AsyncConnection, *, url: str) -> bool:
    cur = await conn.execute("SELECT 1 FROM pages WHERE url = %s", (url,))
    return await cur.fetchone() is not None


async def upsert_page(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    url: str,
    title: str | None,
    published_at: datetime | None,
) -> int:
    cur = await conn.execute(
        """
        INSERT INTO pages (site_id, url, title, published_at, fetched_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (url) DO UPDATE SET
            site_id = EXCLUDED.site_id,
            title = EXCLUDED.title,
            published_at = EXCLUDED.published_at,
            fetched_at = now()
        RETURNING id
        """,
        (site_id, url, title, published_at),
    )
    row = cast(tuple, await cur.fetchone())
    return row[0]


async def replace_page_chunks(
    conn: psycopg.AsyncConnection,
    *,
    page_id: int,
    chunks: Iterable[ChunkInsert],
) -> None:
    await conn.execute("DELETE FROM chunks WHERE page_id = %s", (page_id,))
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
                    HalfVector(chunk.embedding),
                )
                for chunk in chunks
            ],
        )
