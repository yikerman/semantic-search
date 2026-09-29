import asyncio
import logging
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from semsearch.web.db import list_available_languages

logger = logging.getLogger(__name__)


@dataclass
class LanguageState:
    codes: tuple[str, ...] = ()


async def refresh_languages(
    pool: AsyncConnectionPool, state: LanguageState, *, interval: float = 300
) -> None:
    """Keep dropdown options current without making page requests wait for SQL."""
    while True:
        try:
            async with asyncio.timeout(5), pool.connection() as conn:
                await conn.execute("SET LOCAL statement_timeout = '5s'")
                codes = tuple(await list_available_languages(conn))
        except Exception:
            logger.exception("Language refresh failed; retaining previous options")
        else:
            state.codes = codes
        await asyncio.sleep(interval)
