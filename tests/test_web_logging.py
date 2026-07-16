import logging
from contextlib import asynccontextmanager

import httpx

from semsearch.share.embeddings import EmbeddingError
from semsearch.web.app import create_app
from semsearch.web.search.pipeline import rerank_by_length


class Pool:
    @asynccontextmanager
    async def connection(self):
        yield object()


async def languages(conn):
    return ["en", "fr"]


async def test_search_logs_duration_and_result_count_without_query(caplog, monkeypatch):
    app = create_app()
    query = "private search terms"
    pool = Pool()
    app.state.pool = pool
    app.state.embed_query = object()

    async def fake_search(value: str, **kwargs):
        assert value == query
        assert kwargs["pool"] is pool
        assert kwargs["rerankers"] == (rerank_by_length,)
        assert len(kwargs["filters"]) == 1
        predicate = kwargs["filters"][0]("p")
        assert predicate.clause.as_string() == '"p".language = %s'
        assert predicate.params == ("en",)
        return []

    monkeypatch.setattr("semsearch.web.app.search", fake_search)
    monkeypatch.setattr("semsearch.web.app.list_available_languages", languages)
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.INFO, logger="semsearch.web.app"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/",
                params={
                    "q": query,
                    "encourage_long_content": "true",
                    "lang": "EN",
                },
            )

    assert response.status_code == 200
    assert 'name="encourage_long_content" value="true" checked' in response.text
    assert '<option value="en" selected>en</option>' in response.text
    assert not hasattr(app.state, "search")
    assert "Search completed in" in caplog.messages[-1]
    assert "with 0 results" in caplog.messages[-1]
    assert query not in caplog.text


async def test_search_logs_handled_embedding_error_without_query(caplog, monkeypatch):
    app = create_app()
    query = "another private query"
    app.state.pool = Pool()
    app.state.embed_query = object()

    async def fake_search(value: str, **kwargs):
        assert value == query
        assert kwargs["rerankers"] == ()
        assert kwargs["filters"] == ()
        raise EmbeddingError("service unavailable")

    monkeypatch.setattr("semsearch.web.app.search", fake_search)
    monkeypatch.setattr("semsearch.web.app.list_available_languages", languages)
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.WARNING, logger="semsearch.web.app"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/", params={"q": query})

    assert response.status_code == 503
    assert "The embedding service is temporarily unavailable." in response.text
    assert "service unavailable" not in response.text
    assert "Search failed after" in caplog.messages[-1]
    assert "embedding error" in caplog.messages[-1]
    assert "service unavailable" not in caplog.text
    assert query not in caplog.text


async def test_search_rejects_malformed_language_code():
    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", params={"q": "query", "lang": "english"})

    assert response.status_code == 422
