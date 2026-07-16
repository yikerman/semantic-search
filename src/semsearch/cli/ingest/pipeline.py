import asyncio

from semsearch.cli import db
from semsearch.cli.ingest.chunk import Chunker
from semsearch.cli.ingest.extract import extract_page
from semsearch.cli.ingest.fetch import Fetcher
from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.models import CrawlJob
from semsearch.cli.url import same_site
from semsearch.share.embeddings import EmbedDocuments

from psycopg_pool import AsyncConnectionPool


class IngestError(RuntimeError):
    pass


async def ingest_job(
    pool: AsyncConnectionPool,
    embed_documents: EmbedDocuments,
    fetcher: Fetcher,
    chunker: Chunker,
    job: CrawlJob,
) -> IndexOutcome:
    async with pool.connection() as conn:
        if await db.page_exists(conn, url=job.url):
            async with conn.transaction():
                await db.complete_existing_job(
                    conn, job_id=job.id, lease_token=job.lease_token
                )
            return IndexOutcome(job.url, "skipped", "already indexed")

    response = await fetcher.fetch_response(job.url)
    if not same_site(response.url, job.url):
        raise IngestError("page redirected to a different origin")
    # trafilatura extraction is CPU-bound; run it off the event loop so it does
    # not stall concurrent ingest loops or their lease heartbeats.
    page = await asyncio.to_thread(extract_page, response.text, response.url)
    if page is None:
        raise IngestError("no extractable article text")

    chunks = chunker(page.text)
    if not chunks:
        raise IngestError("article produced no chunks")
    vectors = await embed_documents(
        [
            f"{page.title}\n\n{chunk.content}" if page.title else chunk.content
            for chunk in chunks
        ]
    )

    async with pool.connection() as conn, conn.transaction():
        if await db.page_exists(conn, url=job.url):
            await db.complete_existing_job(
                conn, job_id=job.id, lease_token=job.lease_token
            )
            return IndexOutcome(job.url, "skipped", "already indexed")
        page_id = await db.insert_page(
            conn,
            site_id=job.site_id,
            url=job.url,
            title=page.title,
            published_at=page.published_at,
            language=page.language,
        )
        if page_id is None:
            # Another writer inserted this URL between the check above and now;
            # existing pages are append-only, so skip rather than overwrite.
            await db.complete_existing_job(
                conn, job_id=job.id, lease_token=job.lease_token
            )
            return IndexOutcome(job.url, "skipped", "already indexed")
        await db.replace_page_chunks(
            conn,
            page_id=page_id,
            chunks=[
                db.ChunkInsert(
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_count=chunk.char_count,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        await db.complete_existing_job(conn, job_id=job.id, lease_token=job.lease_token)
    return IndexOutcome(job.url, "indexed", chunk_count=len(chunks))
