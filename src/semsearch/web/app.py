from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from semsearch.config import get_settings
from semsearch.db import IndexMetaError, create_pool
from semsearch.embeddings import get_embedding_provider
from semsearch.embeddings.openai_compat import EmbeddingError
from semsearch.search.service import SearchService

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    embedder = get_embedding_provider(settings)
    try:
        pool = create_pool(settings)
        await pool.open()
        try:
            app.state.pool = pool
            app.state.search = SearchService(pool, embedder, settings)
            yield
        finally:
            await pool.close()
    finally:
        await embedder.aclose()


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
            await conn.execute("SELECT 1")
        return {"status": "ok"}

    return app


app = create_app()
