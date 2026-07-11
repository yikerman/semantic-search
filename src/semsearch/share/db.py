import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from semsearch.share.config import Settings


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
