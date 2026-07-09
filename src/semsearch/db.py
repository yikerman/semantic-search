import importlib.resources
from typing import LiteralString, cast

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from semsearch.config import Settings


class IndexMetaError(RuntimeError):
    pass


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
