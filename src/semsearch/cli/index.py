import asyncio
import logging
from functools import partial

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.ingest.document import (
    DocumentTooLong,
    ValidateDocument,
    load_tokenizer,
    validate_document,
)
from semsearch.cli.locks import INDEX_LOCK, command_lock
from semsearch.cli.models import IndexPage
from semsearch.share.config import Settings
from semsearch.share.embeddings import EmbedDocument, create_embeddings

logger = logging.getLogger(__name__)


async def index_page(
    pool: AsyncConnectionPool,
    page: IndexPage,
    validate: ValidateDocument,
    embed: EmbedDocument,
) -> bool:
    text = f"{page.title}\n\n{page.content}" if page.title else page.content
    try:
        await asyncio.to_thread(validate, text)
        embedding = await embed(text)
    except Exception as exc:  # noqa: BLE001 -- isolate per-page tokenizer/provider failures
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE pages SET index_error = %s, index_rejected = %s WHERE id = %s AND indexed_at IS NULL",
                (str(exc)[:2000], isinstance(exc, DocumentTooLong), page.id),
            )
        logger.warning("Indexing page %s failed: %s", page.id, exc)
        return False
    # One statement atomically publishes both retrieval representations.
    async with pool.connection() as conn, conn.transaction():
        await db.publish_page_index(
            conn, page_id=page.id, text=text, embedding=embedding
        )
    return True


async def index_pending(
    pool: AsyncConnectionPool,
    validate: ValidateDocument,
    embed: EmbedDocument,
    concurrency: int,
) -> tuple[int, int]:
    # Snapshot IDs only. Load bounded batches of full text, not the whole corpus.
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id FROM pages WHERE indexed_at IS NULL AND NOT index_rejected ORDER BY id"
        )
        ids: list[int] = []
        for row in await cur.fetchall():
            if len(row) != 1 or type(row[0]) is not int or row[0] <= 0:
                raise ValueError("invalid page ID row")
            ids.append(row[0])
    succeeded = failed = 0
    for offset in range(0, len(ids), concurrency):
        async with pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                "SELECT id, title, content FROM pages WHERE id = ANY(%s) AND indexed_at IS NULL AND NOT index_rejected ORDER BY id",
                (ids[offset : offset + concurrency],),
            )
            pages = [IndexPage.model_validate(row) for row in await cur.fetchall()]
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(index_page(pool, page, validate, embed))
                for page in pages
            ]
        succeeded += sum(task.result() for task in tasks)
        failed += sum(not task.result() for task in tasks)
    return succeeded, failed


async def run_index(pool: AsyncConnectionPool, settings: Settings) -> tuple[int, int]:
    async with command_lock(pool, INDEX_LOCK):
        tokenizer = await asyncio.to_thread(
            load_tokenizer,
            settings.embedding_tokenizer,
            settings.embedding_tokenizer_revision,
        )
        validate = partial(
            validate_document,
            tokenizer=tokenizer,
            max_tokens=settings.embedding_max_tokens,
        )
        async with create_embeddings(settings) as embeddings:
            result = await index_pending(
                pool, validate, embeddings.embed_document, settings.index_concurrency
            )
        logger.info("Index batch: %s indexed, %s failed", *result)
        return result
