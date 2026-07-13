import logging

import httpx

from semsearch.share.embeddings import EmbeddingError
from semsearch.web.app import create_app
from semsearch.web.search.pipeline import rerank_by_length


async def test_search_logs_duration_and_result_count_without_query(caplog, monkeypatch):
    app = create_app()
    query = "private search terms"
    pool = object()
    app.state.pool = pool
    app.state.embed_query = object()

    async def fake_search(value: str, **kwargs):
        assert value == query
        assert kwargs["pool"] is pool
        assert kwargs["rerankers"] == (rerank_by_length,)
        return []

    monkeypatch.setattr("semsearch.web.app.search", fake_search)
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.INFO, logger="semsearch.web.app"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get(
                "/", params={"q": query, "encourage_long_content": "true"}
            )

    assert response.status_code == 200
    assert 'name="encourage_long_content" value="true" checked' in response.text
    assert not hasattr(app.state, "search")
    assert "Search completed in" in caplog.messages[-1]
    assert "with 0 results" in caplog.messages[-1]
    assert query not in caplog.text


async def test_search_logs_handled_embedding_error_without_query(caplog, monkeypatch):
    app = create_app()
    query = "another private query"
    app.state.pool = object()
    app.state.embed_query = object()

    async def fake_search(value: str, **kwargs):
        assert value == query
        assert kwargs["rerankers"] == ()
        raise EmbeddingError("service unavailable")

    monkeypatch.setattr("semsearch.web.app.search", fake_search)
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.WARNING, logger="semsearch.web.app"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/", params={"q": query})

    assert response.status_code == 200
    assert "Search failed after" in caplog.messages[-1]
    assert "embedding error" in caplog.messages[-1]
    assert "service unavailable" not in caplog.text
    assert query not in caplog.text
