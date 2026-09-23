import importlib.resources
from collections.abc import Sequence
from datetime import datetime
from typing import LiteralString, cast

import psycopg
from pgvector import Vector

from semsearch.share.config import Settings


def load_schema_sql(settings: Settings) -> LiteralString:
    raw = (
        importlib.resources.files("semsearch.share").joinpath("schema.sql").read_text()
    )
    return cast(LiteralString, raw.format(embedding_dim=settings.embedding_dim))


async def init_schema(settings: Settings) -> None:
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await conn.execute(load_schema_sql(settings))
        await conn.commit()


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
    another writer already stored this page; preserve its canonical content.
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


async def publish_page_index(
    conn: psycopg.AsyncConnection,
    *,
    page_id: int,
    text: str,
    embedding: Sequence[float],
) -> None:
    await conn.execute(
        """UPDATE pages SET embedding = quantize_to_rabitq8(%s::vector),
        search_vector = tokenize(%s, 'semsearch_llmlingua2')::bm25vector,
        indexed_at = now(), index_error = NULL
        WHERE id = %s AND indexed_at IS NULL AND NOT index_rejected""",
        (Vector(list(embedding)), text, page_id),
    )
