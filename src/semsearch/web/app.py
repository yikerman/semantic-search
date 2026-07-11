from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.embeddings import EmbeddingError, create_embeddings
from semsearch.web.db import ping
from semsearch.web.search.dense import retrieve_dense
from semsearch.web.search.pipeline import search

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with create_embeddings(settings) as embedder, create_pool(settings) as pool:
        app.state.pool = pool
        app.state.search = partial(
            search,
            embed_query=embedder.embed_query,
            retrievers=(partial(retrieve_dense, pool=pool),),
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="semsearch", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, q: str = ""):
        results = None
        error = None
        query = q.strip()
        if query:
            try:
                results = await request.app.state.search(query)
            except EmbeddingError as exc:
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
