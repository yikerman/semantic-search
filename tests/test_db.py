from typing import Any, cast

from psycopg import sql

from semsearch.cli.db import load_schema_sql
from semsearch.share.config import Settings
from semsearch.web.db import (
    fetch_bm25_candidate_rows,
    fetch_dense_candidate_rows,
    fetch_lead_chunks,
)
from semsearch.web.search.filters import SqlPredicate


def test_schema_uses_halfvec_hnsw_cosine_index():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "embedding halfvec(2) NOT NULL" in schema
    assert "USING hnsw (embedding halfvec_cosine_ops)" in schema
    assert "index_meta" not in schema


def test_schema_indexes_chunk_content_for_full_text_search():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "search_vector tsvector GENERATED ALWAYS AS" in schema
    assert "to_tsvector('simple', content)" in schema
    assert "USING gin (search_vector)" in schema


def test_schema_adds_durable_crawl_and_poll_state():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "CREATE TABLE IF NOT EXISTS crawl_jobs" in schema
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


class EmptyCursor:
    async def fetchall(self):
        return []


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
        return [
            (
                7,
                3,
                "https://example.com/post",
                "Post",
                "matching content",
                0.25,
            )
        ]


class RecordingConnection:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return RowCursor()


async def test_bm25_query_uses_search_vector_and_preserves_filter_params():
    conn = RecordingConnection()

    rows = await fetch_bm25_candidate_rows(
        cast(Any, conn),
        query='postgres "full text"',
        predicate=SqlPredicate(sql.SQL("p.site_id = %s"), (3,)),
        limit=12,
    )

    assert conn.query is not None
    query = conn.query.as_string()
    assert "websearch_to_tsquery('simple', %s)" in query
    assert "c.search_vector @@ search_query.value" in query
    assert "ts_rank_cd(c.search_vector, search_query.value)" in query
    assert "p.site_id = %s" in query
    assert conn.params == ('postgres "full text"', 3, 12)
    assert rows[0].chunk_id == 7
    assert rows[0].rank == 0.25


class LeadCursor:
    async def fetchall(self):
        return [(3, "lead three"), (5, "lead five")]


class LeadConnection:
    def __init__(self) -> None:
        self.query = None
        self.params = None

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return LeadCursor()


async def test_lead_chunk_lookup_maps_page_ids_to_first_chunk():
    conn = LeadConnection()

    lead = await fetch_lead_chunks(cast(Any, conn), page_ids=(3, 5))

    assert conn.query is not None
    assert "chunk_index = 0" in conn.query
    assert conn.params == ([3, 5],)
    assert lead == {3: "lead three", 5: "lead five"}
