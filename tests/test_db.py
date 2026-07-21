from datetime import UTC, datetime
from typing import Any, cast

import pytest
from psycopg import sql

from semsearch.cli.db import load_schema_sql
from semsearch.share.config import Settings
from semsearch.web.db import (
    fetch_bm25_candidate_rows,
    fetch_dense_candidate_rows,
    fetch_pages,
    list_available_languages,
    list_recent_activity,
)
from semsearch.web.search.filters import SqlPredicate


def test_schema_uses_halfvec_hnsw_cosine_index():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "embedding halfvec(2) NOT NULL" in schema
    assert "USING hnsw (embedding halfvec_cosine_ops)" in schema
    assert "index_meta" not in schema


def test_schema_initializes_vectorchord_bm25_once():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "CREATE EXTENSION vector;" in schema
    assert "CREATE EXTENSION pg_tokenizer CASCADE;" in schema
    assert "CREATE EXTENSION vchord_bm25 CASCADE;" in schema
    assert "create_tokenizer('semsearch_llmlingua2'" in schema
    assert 'model = "llmlingua2"' in schema
    assert "IF NOT EXISTS" not in schema


def test_schema_keeps_canonical_page_content_and_derived_chunk_spans():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))
    chunks = schema.split("CREATE TABLE chunks", 1)[1].split(");", 1)[0]

    assert "content text NOT NULL" in schema
    assert "start_offset int NOT NULL CHECK (start_offset >= 0)" in chunks
    assert "content_length int NOT NULL CHECK (content_length > 0)" in chunks
    assert "search_vector bm25vector NOT NULL" in chunks
    assert "UNIQUE (page_id, start_offset)" in chunks
    assert "content text" not in chunks
    assert "GENERATED ALWAYS" not in chunks
    assert "USING bm25 (search_vector bm25_ops)" in schema


def test_schema_adds_durable_crawl_and_poll_state():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "CREATE TABLE crawl_jobs" in schema
    assert "url text UNIQUE NOT NULL" in schema
    assert "next_poll_at timestamptz" in schema
    assert "history_pending boolean NOT NULL DEFAULT false" in schema
    assert "feed_url text NOT NULL" in schema
    assert "site_id bigint NOT NULL REFERENCES sites" in schema
    assert "poll_lease_token uuid" in schema
    assert "lease_token uuid" in schema
    assert "failed_at timestamptz" in schema
    assert "WHERE next_attempt_at IS NOT NULL" in schema
    assert "ALTER TABLE" not in schema
    assert "last_indexed_at" not in schema


def test_schema_indexes_recent_activity():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "pages_recent_idx" in schema
    assert "ON pages (fetched_at DESC, url)" in schema
    assert "crawl_jobs_recent_failure_idx" in schema
    assert "ON crawl_jobs (failed_at DESC, url)" in schema
    assert "WHERE failed_at IS NOT NULL" in schema


def test_schema_adds_page_language_metadata():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "language text" in schema
    assert "pages_language_idx" in schema
    assert "WHERE language IS NOT NULL" in schema


def test_schema_indexes_known_publication_dates():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "pages_published_at_idx" in schema
    assert "ON pages (published_at)" in schema
    assert "WHERE published_at IS NOT NULL" in schema


class EmptyCursor:
    async def fetchall(self):
        return []


class LanguageCursor:
    async def fetchall(self):
        return [("en",), ("fr",)]


class LanguageConnection:
    async def execute(self, query):
        return LanguageCursor()


async def test_available_languages_are_read_from_page_metadata():
    languages = await list_available_languages(cast(Any, LanguageConnection()))

    assert languages == ["en", "fr"]


class DenseConnection:
    async def execute(self, query, params):
        return EmptyCursor()


async def test_dense_query_accepts_immutable_embedding_sequence():
    rows = await fetch_dense_candidate_rows(
        cast(Any, DenseConnection()),
        query_embedding=(1.0, 0.0),
        predicate=SqlPredicate(sql.SQL("TRUE")),
        limit=10,
    )

    assert rows == []


class RowCursor:
    async def fetchall(self):
        return [(7, 3, 0.25)]


class RecordingConnection:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return RowCursor()


async def test_bm25_query_uses_vectorchord_and_preserves_filter_params():
    conn = RecordingConnection()

    rows = await fetch_bm25_candidate_rows(
        cast(Any, conn),
        query='postgres "full text"',
        predicate=SqlPredicate(sql.SQL("p.site_id = %s"), (3,)),
        limit=12,
    )

    assert conn.query is not None
    query = conn.query.as_string()
    assert "to_bm25query(" in query
    assert "'chunks_search_vector_bm25_idx'::regclass" in query
    assert "tokenize(%s, 'semsearch_llmlingua2')::bm25vector" in query
    assert "-(c.search_vector <&> search_query.value) AS rank" in query
    assert "ORDER BY c.search_vector <&> search_query.value" in query
    assert "ORDER BY rank DESC" not in query
    assert "p.site_id = %s" in query
    assert conn.params == ('postgres "full text"', 3, 12)
    assert rows[0].chunk_id == 7
    assert rows[0].rank == 0.25


class PageCursor:
    async def fetchall(self):
        return [
            (
                3,
                "https://example.com/three",
                "Three",
                "full page three",
                datetime(2025, 1, 2, tzinfo=UTC),
            ),
            (5, "https://example.com/five", None, "full page five", None),
        ]


class PageConnection:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return PageCursor()


async def test_page_lookup_returns_validated_canonical_content():
    conn = PageConnection()

    pages = await fetch_pages(cast(Any, conn), page_ids=(3, 5))

    assert conn.query is not None
    assert "SELECT id, url, title, content, published_at" in conn.query
    assert "FROM pages" in conn.query
    assert conn.params == ([3, 5],)
    assert pages[3].content == "full page three"
    assert pages[3].published_at == datetime(2025, 1, 2, tzinfo=UTC)
    assert pages[5].title is None
    assert pages[5].published_at is None


async def test_page_lookup_rejects_invalid_database_rows():
    class InvalidPageCursor:
        async def fetchall(self):
            return [(True, "https://example.com", None, "content")]

    class InvalidPageConnection:
        async def execute(self, query, params):
            return InvalidPageCursor()

    with pytest.raises(ValueError, match="invalid page database row"):
        await fetch_pages(cast(Any, InvalidPageConnection()), page_ids=(1,))


async def test_page_lookup_rejects_naive_publication_timestamp():
    class InvalidPageCursor:
        async def fetchall(self):
            return [(1, "https://example.com", None, "content", datetime(2025, 1, 2))]

    class InvalidPageConnection:
        async def execute(self, query, params):
            return InvalidPageCursor()

    with pytest.raises(ValueError, match="invalid page database row"):
        await fetch_pages(cast(Any, InvalidPageConnection()), page_ids=(1,))


class ActivityCursor:
    async def fetchall(self):
        return [
            (
                "https://example.com/new",
                "success",
                datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
                None,
                None,
            ),
            (
                "https://example.com/broken",
                "failure",
                datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
                3,
                "GET returned 404",
            ),
        ]


class ActivityConnection:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return ActivityCursor()


async def test_recent_activity_combines_successes_and_failures():
    conn = ActivityConnection()

    activity = await list_recent_activity(cast(Any, conn))

    assert conn.query is not None
    assert "FROM pages" in conn.query
    assert "FROM crawl_jobs" in conn.query
    assert "ORDER BY occurred_at DESC, url" in conn.query
    assert conn.params == (10,)
    assert [item.status for item in activity] == ["success", "failure"]
    assert activity[1].attempt_count == 3
    assert activity[1].detail == "GET returned 404"


class InvalidActivityCursor:
    async def fetchall(self):
        return [("https://example.com/post", "pending", "not-a-datetime", None, None)]


class InvalidActivityConnection:
    async def execute(self, query, params):
        return InvalidActivityCursor()


async def test_recent_activity_validates_database_rows():
    with pytest.raises(ValueError, match="invalid recent activity database row"):
        await list_recent_activity(cast(Any, InvalidActivityConnection()))
