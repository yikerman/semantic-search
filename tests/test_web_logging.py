import logging

import httpx

from semsearch.share.embeddings import EmbeddingError
from semsearch.web.app import create_app


async def test_search_logs_duration_and_result_count_without_query(caplog):
    app = create_app()
    query = "private search terms"

    async def search(value: str):
        assert value == query
        return []

    app.state.search = search
    transport = httpx.ASGITransport(app=app)

    with caplog.at_level(logging.INFO, logger="semsearch.web.app"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/", params={"q": query})

    assert response.status_code == 200
    assert "Search completed in" in caplog.messages[-1]
    assert "with 0 results" in caplog.messages[-1]
    assert query not in caplog.text


async def test_search_logs_handled_embedding_error_without_query(caplog):
    app = create_app()
    query = "another private query"

    async def search(value: str):
        assert value == query
        raise EmbeddingError("service unavailable")

    app.state.search = search
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
