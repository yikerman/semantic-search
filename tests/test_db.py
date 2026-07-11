from psycopg import sql

from semsearch.config import Settings
from semsearch.db import fetch_dense_candidate_rows, load_schema_sql
from semsearch.search.filters import SqlPredicate


def test_schema_uses_halfvec_hnsw_cosine_index():
    schema = load_schema_sql(Settings(embedding_model="test-model", embedding_dim=2))

    assert "embedding halfvec(2) NOT NULL" in schema
    assert "USING hnsw (embedding halfvec_cosine_ops)" in schema
    assert "index_meta" not in schema


class EmptyCursor:
    async def fetchall(self):
        return []


class DenseConnection:
    async def execute(self, query, params):
        return EmptyCursor()


async def test_dense_query_accepts_immutable_embedding_sequence():
    rows = await fetch_dense_candidate_rows(
        DenseConnection(),  # type: ignore[arg-type]
        query_embedding=(1.0, 0.0),
        predicate=SqlPredicate(sql.SQL("TRUE")),
        limit=10,
    )

    assert rows == []
