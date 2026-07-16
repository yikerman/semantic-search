from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, Query, Request, status as http_status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError, create_embeddings
from semsearch.share.logging import configure_logging
from semsearch.share.status import fetch_index_stats
from semsearch.web.db import list_available_languages, list_recent_activity, ping
from semsearch.web.search.bm25 import retrieve_bm25
from semsearch.web.search.dense import retrieve_dense
from semsearch.web.search.models import PageCandidate
from semsearch.web.search.pipeline import rerank_by_length, search
from semsearch.web.search.filters import filter_by_language

# Configure at import time: uvicorn loads this module before it logs its own
# startup lines, so even those render through our handler.
configure_logging(get_settings().log_level)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DisplayResult:
    page_id: int
    url: str
    title: str | None
    snippet: str
    is_truncated: bool
    scores: Mapping[str, float]


def prepare_language_options(
    languages: Sequence[str], *, selected: str | None
) -> list[str]:
    codes = set(languages)
    if selected:
        codes.add(selected)
    return sorted(codes)


def prepare_display(results: Sequence[PageCandidate]) -> list[DisplayResult]:
    return [
        DisplayResult(
            page_id=result.page_id,
            url=result.url,
            title=result.title,
            snippet=result.content[:500],
            is_truncated=len(result.content) > 500,
            scores=result.scores,
        )
        for result in results
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting web application")
    async with create_embeddings(settings) as embedder, create_pool(settings) as pool:
        app.state.pool = pool
        app.state.embed_query = embedder.embed_query
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
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        q: str = "",
        encourage_long_content: bool = False,
        lang: Annotated[str | None, Query(pattern=r"^[A-Za-z]{2}$")] = None,
    ):
        results = None
        error = None
        status_code = http_status.HTTP_200_OK
        query = q.strip()
        selected_language = lang.lower() if lang else None
        async with request.app.state.pool.connection() as conn:
            available_languages = await list_available_languages(conn)
        if query:
            rerankers = (rerank_by_length,) if encourage_long_content else ()
            filters = (
                (filter_by_language(selected_language),)
                if selected_language is not None
                else ()
            )
            run_search = partial(
                search,
                pool=request.app.state.pool,
                embed_query=request.app.state.embed_query,
                retrievers=(retrieve_dense, retrieve_bm25),
                rerankers=rerankers,
                filters=filters,
            )
            started_at = perf_counter()
            try:
                results = prepare_display(await run_search(query))
            except EmbeddingError:
                error = "The embedding service is temporarily unavailable."
                status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
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
            request,
            "index.html",
            {
                "active_page": "search",
                "q": q,
                "encourage_long_content": encourage_long_content,
                "lang": selected_language or "",
                "languages": prepare_language_options(
                    available_languages, selected=selected_language
                ),
                "results": results,
                "error": error,
            },
            status_code=status_code,
        )

    @app.get("/status", response_class=HTMLResponse)
    async def status(request: Request):
        async with request.app.state.pool.connection() as conn:
            stats = await fetch_index_stats(conn)
            activity = await list_recent_activity(conn)
        settings = get_settings()
        return templates.TemplateResponse(
            request,
            "status.html",
            {
                "active_page": "status",
                "stats": stats,
                "activity": activity,
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
