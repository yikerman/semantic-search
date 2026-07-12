from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from psycopg_pool import AsyncConnectionPool

from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError, create_embeddings
from semsearch.share.logging import configure_logging
from semsearch.share.status import fetch_index_stats, list_failed_jobs
from semsearch.web.db import fetch_lead_chunks, ping
from semsearch.web.search.bm25 import retrieve_bm25
from semsearch.web.search.dense import retrieve_dense
from semsearch.web.search.models import Candidate
from semsearch.web.search.pipeline import search

# Configure at import time: uvicorn loads this module before it logs its own
# startup lines, so even those render through our handler.
configure_logging(get_settings().log_level)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
logger = logging.getLogger(__name__)


async def prepare_display(
    pool: AsyncConnectionPool, results: list[Candidate]
) -> list[Candidate]:
    # Ranking matched a mid-article chunk window, which makes a poor snippet;
    # show the page's lead chunk instead.
    if not results:
        return results
    async with pool.connection() as conn:
        lead = await fetch_lead_chunks(
            conn, page_ids=[result.page_id for result in results]
        )
    return [
        replace(result, content=lead.get(result.page_id, result.content))
        for result in results
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting web application")
    async with create_embeddings(settings) as embedder, create_pool(settings) as pool:
        app.state.pool = pool
        run_search = partial(
            search,
            embed_query=embedder.embed_query,
            retrievers=(
                partial(retrieve_dense, pool=pool),
                partial(retrieve_bm25, pool=pool),
            ),
        )

        async def search_for_display(query: str) -> list[Candidate]:
            return await prepare_display(pool, await run_search(query))

        app.state.search = search_for_display
        logger.info(
            "Web application ready with embedding model %s (%d dimensions)",
            settings.embedding_model,
            settings.embedding_dim,
        )
        try:
            yield
        finally:
            logger.info("Stopping web application")


def create_app() -> FastAPI:
    app = FastAPI(title="semsearch", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, q: str = ""):
        results = None
        error = None
        query = q.strip()
        if query:
            started_at = perf_counter()
            try:
                results = await request.app.state.search(query)
            except EmbeddingError as exc:
                error = str(exc)
                logger.warning(
                    "Search failed after %.3f seconds due to an embedding error",
                    perf_counter() - started_at,
                )
            else:
                logger.info(
                    "Search completed in %.3f seconds with %d results",
                    perf_counter() - started_at,
                    len(results),
                )
        return templates.TemplateResponse(
            request, "index.html", {"q": q, "results": results, "error": error}
        )

    @app.get("/status", response_class=HTMLResponse)
    async def status(request: Request):
        async with request.app.state.pool.connection() as conn:
            stats = await fetch_index_stats(conn)
            failures = await list_failed_jobs(conn)
        settings = get_settings()
        return templates.TemplateResponse(
            request,
            "status.html",
            {
                "stats": stats,
                "failures": failures,
                "embedding_model": settings.embedding_model,
                "embedding_dim": settings.embedding_dim,
            },
        )

    @app.get("/healthz")
    async def healthz(request: Request):
        async with request.app.state.pool.connection() as conn:
            await ping(conn)
        return {"status": "ok"}

    return app


app = create_app()
