from contextlib import asynccontextmanager
from functools import partial
import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError, create_embeddings
from semsearch.share.logging import configure_logging
from semsearch.web.db import ping
from semsearch.web.search.bm25 import retrieve_bm25
from semsearch.web.search.dense import retrieve_dense
from semsearch.web.search.pipeline import search

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting web application")
    async with create_embeddings(settings) as embedder, create_pool(settings) as pool:
        app.state.pool = pool
        app.state.search = partial(
            search,
            embed_query=embedder.embed_query,
            retrievers=(
                partial(retrieve_dense, pool=pool),
                partial(retrieve_bm25, pool=pool),
            ),
        )
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

    @app.get("/healthz")
    async def healthz(request: Request):
        async with request.app.state.pool.connection() as conn:
            await ping(conn)
        return {"status": "ok"}

    return app


app = create_app()
