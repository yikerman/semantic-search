import asyncio
import logging
from functools import partial

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from semsearch.cli import db
from semsearch.cli.ingest.chunk import Chunker, load_tokenizer, token_chunks
from semsearch.cli.locks import INDEX_LOCK, command_lock
from semsearch.cli.models import IndexPage
from semsearch.share.config import Settings
from semsearch.share.embeddings import EmbedDocuments, create_embeddings

logger = logging.getLogger(__name__)
MAX_CHUNKS = 2000


async def index_page(
    pool: AsyncConnectionPool, page: IndexPage, chunker: Chunker, embed: EmbedDocuments
) -> bool:
    try:
        chunks = await asyncio.to_thread(chunker, page.content)
        if not chunks or len(chunks) > MAX_CHUNKS:
            raise ValueError("article exceeds chunk limit or produces no chunks")
        vectors = await embed(
            [
                f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
                for chunk in chunks
            ]
        )
        inserts = [
            db.ChunkInsert(chunk.start_offset, chunk.content, vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    except Exception as exc:  # noqa: BLE001 -- isolate per-page tokenizer/provider failures
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE pages SET index_error = %s WHERE id = %s AND indexed_at IS NULL",
                (str(exc)[:2000], page.id),
            )
        logger.warning("Indexing page %s failed: %s", page.id, exc)
        return False
    # Database failures propagate: the transaction rolls back all chunks.
    async with pool.connection() as conn, conn.transaction():
        cur = await conn.execute(
            "SELECT id FROM pages WHERE id = %s AND indexed_at IS NULL FOR UPDATE",
            (page.id,),
        )
        if await cur.fetchone() is None:
            return True
        await db.insert_page_chunks(conn, page_id=page.id, chunks=inserts)
        await conn.execute(
            "UPDATE pages SET indexed_at = now(), index_error = NULL WHERE id = %s",
            (page.id,),
        )
    return True


async def index_pending(
    pool: AsyncConnectionPool, chunker: Chunker, embed: EmbedDocuments, concurrency: int
) -> tuple[int, int]:
    # Snapshot IDs only. Load bounded batches of full text, not the whole corpus.
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id FROM pages WHERE indexed_at IS NULL ORDER BY id"
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
                "SELECT id, title, content FROM pages WHERE id = ANY(%s) AND indexed_at IS NULL ORDER BY id",
                (ids[offset : offset + concurrency],),
            )
            pages = [IndexPage.model_validate(row) for row in await cur.fetchall()]
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(index_page(pool, page, chunker, embed))
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
        chunker = partial(
            token_chunks,
            tokenizer=tokenizer,
            chunk_tokens=settings.chunk_tokens,
            chunk_token_overlap=settings.chunk_token_overlap,
        )
        async with create_embeddings(settings) as embeddings:
            result = await index_pending(
                pool, chunker, embeddings.embed_documents, settings.index_concurrency
            )
        logger.info("Index batch: %s indexed, %s failed", *result)
        return result
