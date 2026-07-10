import importlib.resources
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import LiteralString, TypeAlias, cast

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from semsearch.config import Settings


class IndexMetaError(RuntimeError):
    pass


SiteRow: TypeAlias = tuple[
    int,
    str,
    str | None,
    str | None,
    datetime | None,
    datetime | None,
    str | None,
    str | None,
]
ChunkInsert: TypeAlias = tuple[int, str, int, Sequence[float]]
DenseCandidateRow: TypeAlias = tuple[int, int, str, str | None, str, float]


@dataclass(slots=True)
class IndexStats:
    site_count: int
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dim: int


SITE_COLUMNS = """
id, base_url, sitemap_url, feed_url, last_indexed_at,
last_polled_at, feed_etag, feed_last_modified
"""


def load_schema_sql(settings: Settings) -> LiteralString:
    raw = importlib.resources.files("semsearch").joinpath("schema.sql").read_text()
    return cast(LiteralString, raw.format(embedding_dim=settings.embedding_dim))


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)
    await conn.commit()


def create_pool(settings: Settings) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        open=False,
        configure=_configure,
    )


async def init_schema(settings: Settings) -> None:
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await conn.execute(load_schema_sql(settings))
        await conn.execute(
            """
            INSERT INTO index_meta (id, embedding_model, embedding_dim)
            VALUES (1, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (settings.embedding_model, settings.embedding_dim),
        )
        await _verify_index_meta(conn, settings)
        await conn.commit()


async def check_index_meta(conn: psycopg.AsyncConnection, settings: Settings) -> None:
    try:
        await _verify_index_meta(conn, settings)
    except psycopg.errors.UndefinedTable:
        raise IndexMetaError(
            "Database schema not initialized. Run: semsearch init-db"
        ) from None


async def ping(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SELECT 1")


async def fetch_index_stats(conn: psycopg.AsyncConnection) -> IndexStats:
    cur = await conn.execute(
        """
        SELECT (SELECT count(*) FROM sites),
               (SELECT count(*) FROM pages),
               (SELECT count(*) FROM chunks),
               (SELECT embedding_model FROM index_meta WHERE id = 1),
               (SELECT embedding_dim FROM index_meta WHERE id = 1)
        """
    )
    row = await cur.fetchone()
    assert row is not None
    sites, pages, chunks, model, dim = row
    return IndexStats(sites, pages, chunks, model, dim)


async def upsert_site_config(
    conn: psycopg.AsyncConnection,
    *,
    base_url: str,
    sitemap_url: str | None,
    feed_url: str | None,
) -> SiteRow:
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
    row = await cur.fetchone()
    assert row is not None
    return cast(SiteRow, row)


async def list_site_configs(conn: psycopg.AsyncConnection) -> list[SiteRow]:
    cur = await conn.execute(
        f"""
        SELECT {SITE_COLUMNS}
        FROM sites
        ORDER BY base_url
        """
    )
    return [cast(SiteRow, row) for row in await cur.fetchall()]


async def find_site_config(
    conn: psycopg.AsyncConnection, *, base_url: str
) -> SiteRow | None:
    cur = await conn.execute(
        f"""
        SELECT {SITE_COLUMNS}
        FROM sites
        WHERE base_url = %s
        """,
        (base_url,),
    )
    row = await cur.fetchone()
    return None if row is None else cast(SiteRow, row)


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
    row = await cur.fetchone()
    assert row is not None
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
    row = await cur.fetchone()
    assert row is not None
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
                    chunk_index,
                    content,
                    char_count,
                    HalfVector(embedding),
                )
                for chunk_index, content, char_count, embedding in chunks
            ],
        )


async def fetch_dense_candidate_rows(
    conn: psycopg.AsyncConnection,
    *,
    query_embedding: Sequence[float],
    limit: int,
) -> list[DenseCandidateRow]:
    embedding = HalfVector(query_embedding)
    cur = await conn.execute(
        """
        SELECT c.id, c.page_id, p.url, p.title, c.content,
               1 - (c.embedding <=> %s) AS similarity
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """,
        (embedding, embedding, limit),
    )
    return [cast(DenseCandidateRow, row) for row in await cur.fetchall()]


async def _verify_index_meta(conn: psycopg.AsyncConnection, settings: Settings) -> None:
    cur = await conn.execute(
        "SELECT embedding_model, embedding_dim FROM index_meta WHERE id = 1"
    )
    row = await cur.fetchone()
    if row is None:
        raise IndexMetaError("index_meta is empty. Run: semsearch init-db")
    model, dim = row
    if model != settings.embedding_model or dim != settings.embedding_dim:
        raise IndexMetaError(
            f"Index was built with {model} ({dim} dims) but the configured model is "
            f"{settings.embedding_model} ({settings.embedding_dim} dims). "
            "Changing embedding models requires re-indexing from scratch "
            "(drop the database volume and run init-db again)."
        )
