from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from semsearch.config import get_settings
from semsearch.db import IndexMetaError, IndexMetaGuard, create_pool, ping
from semsearch.embeddings import get_embedding_provider
from semsearch.embeddings.openai_compat import EmbeddingError
from semsearch.search.service import SearchService

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with AsyncExitStack() as stack:
        embedder = await stack.enter_async_context(get_embedding_provider(settings))
        pool = create_pool(settings)
        await stack.enter_async_context(pool)
        app.state.pool = pool
        app.state.search = SearchService(
            pool,
            embedder,
            settings,
            meta_guard=IndexMetaGuard(settings),
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="semsearch", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, q: str = ""):
        results = None
        error = None
        if q.strip():
            try:
                results = await request.app.state.search.search(q)
            except (IndexMetaError, EmbeddingError) as exc:
                error = str(exc)
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
