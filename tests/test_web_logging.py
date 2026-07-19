import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
import pytest

from semsearch.share.embeddings import EmbeddingError
from semsearch.web.app import create_app
from semsearch.web.search.pipeline import rerank_by_length


class Pool:
    @asynccontextmanager
    async def connection(self):
        yield object()


async def languages(conn):
    return ["en", "fr"]


async def test_available_languages_are_cached(monkeypatch):
    calls = 0

    async def counting_languages(conn):
        nonlocal calls
        calls += 1
        return ["en", "fr"]

    app = create_app()
    app.state.pool = Pool()
    monkeypatch.setattr(
        "semsearch.web.app.list_available_languages", counting_languages
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/")
        second = await client.get("/")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == 1


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
        assert len(kwargs["filters"]) == 2
        language_predicate = kwargs["filters"][0]("p")
        assert language_predicate.clause.as_string() == '"p".language = %s'
        assert language_predicate.params == ("en",)
        date_predicate = kwargs["filters"][1]("p")
        assert date_predicate.clause.as_string() == (
            '"p".published_at >= %s AND "p".published_at < %s + INTERVAL \'24 hours\''
        )
        assert date_predicate.params == (
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 3, 4, tzinfo=UTC),
        )
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
                    "published_from": "2025-01-02",
                    "published_to": "2025-03-04",
                },
            )

    assert response.status_code == 200
    assert 'name="encourage_long_content" value="true" checked' in response.text
    assert '<option value="en" selected>en</option>' in response.text
    assert 'name="published_from" value="2025-01-02"' in response.text
    assert 'name="published_to" value="2025-03-04"' in response.text
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


async def test_search_accepts_empty_language_without_filter(monkeypatch):
    app = create_app()
    app.state.pool = Pool()
    app.state.embed_query = object()

    async def fake_search(value: str, **kwargs):
        assert value == "query"
        assert kwargs["filters"] == ()
        return []

    monkeypatch.setattr("semsearch.web.app.search", fake_search)
    monkeypatch.setattr("semsearch.web.app.list_available_languages", languages)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", params={"q": "query", "lang": ""})

    assert response.status_code == 200
    assert '<option value="">Any language</option>' in response.text


async def test_date_only_submission_preserves_filter_without_search(monkeypatch):
    app = create_app()
    app.state.pool = Pool()

    async def fail_search(value: str, **kwargs):
        raise AssertionError("date-only submission must not search")

    monkeypatch.setattr("semsearch.web.app.search", fail_search)
    monkeypatch.setattr("semsearch.web.app.list_available_languages", languages)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", params={"published_from": "2025-01-02"})

    assert response.status_code == 200
    assert 'name="published_from" value="2025-01-02"' in response.text
    assert '<section class="results"' not in response.text


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"q": "query", "published_from": "20250102"},
            "Published dates must use YYYY-MM-DD.",
        ),
        (
            {
                "q": "query",
                "published_from": "2025-03-04",
                "published_to": "2025-01-02",
            },
            "Published from must be on or before Published to.",
        ),
    ],
)
async def test_invalid_date_range_returns_html_without_search(
    monkeypatch, params, message
):
    app = create_app()
    app.state.pool = Pool()
    app.state.embed_query = object()

    async def fail_search(value: str, **kwargs):
        raise AssertionError("invalid date range must not search")

    monkeypatch.setattr("semsearch.web.app.search", fail_search)
    monkeypatch.setattr("semsearch.web.app.list_available_languages", languages)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/", params=params)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/html")
    assert '<p class="error" role="alert">' in response.text
    assert message in response.text
    assert '<section class="results"' not in response.text
