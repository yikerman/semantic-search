from datetime import UTC, datetime
from typing import Any, cast

import pytest
from psycopg import sql

from semsearch.cli.db import load_schema_sql
from semsearch.share.config import Settings
from semsearch.web.db import (
    fetch_bm25_candidate_rows,
    fetch_dense_candidate_rows,
    list_available_languages,
    list_recent_activity,
)
from semsearch.web.search.filters import SqlPredicate


def test_schema_uses_halfvec_hnsw_cosine_index():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "embedding halfvec(2)" in schema
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


def test_schema_stores_retrieval_data_on_pages():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))
    assert "CREATE TABLE chunks" not in schema
    assert "start_offset" not in schema
    assert "content text NOT NULL" in schema
    assert "embedding halfvec(2)" in schema
    assert "search_vector bm25vector" in schema
    assert "USING bm25 (search_vector bm25_ops)" in schema
    assert (
        "indexed_at IS NULL AND embedding IS NULL AND search_vector IS NULL" in schema
    )
    assert (
        "indexed_at IS NOT NULL AND embedding IS NOT NULL AND search_vector IS NOT NULL"
        in schema
    )


def test_schema_separates_discovery_storage_and_indexing():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))
    assert "CREATE TABLE article_urls" in schema
    assert "url text UNIQUE NOT NULL" in schema
    assert "ON DELETE CASCADE" in schema
    assert "history_complete boolean" in schema
    assert "indexed_at timestamptz" in schema
    assert "index_error text" in schema
    assert "lease" not in schema
    assert "crawl_jobs" not in schema
    assert "ALTER TABLE" not in schema


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
        return [(3, "https://example.com/post", "Post", "Whole post", None, 0.25)]


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
    assert "'pages_search_vector_bm25_idx'::regclass" in query
    assert "tokenize(%s, 'semsearch_llmlingua2')::bm25vector" in query
    assert "-(p.search_vector <&> search_query.value) AS rank" in query
    assert "ORDER BY p.search_vector <&> search_query.value" in query
    assert "ORDER BY rank DESC" not in query
    assert "p.site_id = %s" in query
    assert conn.params == ('postgres "full text"', 3, 12)
    assert rows[0][0].page_id == 3
    assert rows[0][1] == 0.25
    assert "FROM pages p" in query
    assert "JOIN pages" not in query
    assert "p.indexed_at IS NOT NULL" in query


async def test_dense_query_returns_full_pages_and_preserves_filter_params():
    conn = RecordingConnection()
    rows = await fetch_dense_candidate_rows(
        cast(Any, conn),
        query_embedding=(1.0, 0.0),
        predicate=SqlPredicate(sql.SQL("p.language = %s"), ("en",)),
        limit=12,
    )
    assert conn.query is not None
    assert conn.params is not None
    query = conn.query.as_string()
    assert "FROM pages p" in query
    assert "JOIN" not in query
    assert "p.indexed_at IS NOT NULL" in query
    assert "p.language = %s" in query
    assert "ORDER BY p.embedding <=> %s" in query
    assert conn.params[1] == "en"
    assert conn.params[-1] == 12
    page, score = rows[0]
    assert page.content == "Whole post"
    assert score == 0.25


@pytest.mark.parametrize(
    "row",
    [
        (True, "https://example.com", None, "content", None, 0.5),
        (1, "https://example.com", None, "content", datetime(2025, 1, 2), 0.5),  # noqa: DTZ001
        (1, "https://example.com", None, "content", None, float("nan")),
        (1, "https://example.com", None, "content", None, True),
        (1, "https://example.com", None, "content"),
    ],
)
@pytest.mark.parametrize("kind", ["dense", "bm25"])
async def test_retrieval_validates_database_rows(row, kind):
    class InvalidConnection:
        async def execute(self, query, params):
            return self

        async def fetchall(self):
            return [row]

    with pytest.raises(ValueError, match="invalid scored page database row"):
        if kind == "dense":
            await fetch_dense_candidate_rows(
                cast(Any, InvalidConnection()),
                query_embedding=(1.0, 0.0),
                predicate=SqlPredicate(sql.SQL("TRUE")),
                limit=1,
            )
        else:
            await fetch_bm25_candidate_rows(
                cast(Any, InvalidConnection()),
                query="test",
                predicate=SqlPredicate(sql.SQL("TRUE")),
                limit=1,
            )


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
    assert "FROM article_urls" in conn.query
    assert "ORDER BY occurred_at DESC, url" in conn.query
    assert conn.params == (10,)
    assert [item.status for item in activity] == ["success", "failure"]
    assert activity[1].failed_batches == 3
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
