import sqlite3
from contextlib import closing
from typing import Any, cast

from semsearch.share.status import IndexStats, fetch_index_stats, list_failed_articles


async def test_status_aggregates_empty_and_mixed_outcomes():
    # These standard SQL aggregates can run locally without a Postgres service.
    with closing(sqlite3.connect(":memory:")) as database:
        database.executescript("""
            CREATE TABLE sites (id int);
            CREATE TABLE pages (indexed_at text, index_rejected boolean, index_error text);
            CREATE TABLE article_urls (status text, failed_batches int);
        """)

        class Connection:
            async def execute(self, query):
                self.row = database.execute(query).fetchone()
                return self

            async def fetchone(self):
                return self.row

        conn = cast(Any, Connection())
        assert await fetch_index_stats(conn) == IndexStats(0, 0, 0, 0, 0, 0)
        database.executescript("""
            INSERT INTO sites VALUES (1);
            INSERT INTO pages VALUES
                ('2026-01-01', false, NULL),
                (NULL, false, NULL),
                (NULL, false, 'provider unavailable'),
                (NULL, true, 'too long');
            INSERT INTO article_urls VALUES
                ('stored', 0), ('pending', 0), ('pending', 1),
                ('failed', 3), ('rejected', 0);
        """)
        assert await fetch_index_stats(conn) == IndexStats(1, 4, 1, 2, 1, 1, 1, 2, 1, 1)


class StatsCursor:
    async def fetchone(self):
        return (5, 100, 93, 20, 3, 2, 4, 5, 1, 2)


class StatsConnection:
    def __init__(self) -> None:
        self.query: str | None = None

    async def execute(self, query):
        self.query = query
        return StatsCursor()


async def test_index_stats_count_pages_and_separate_rejections():
    conn = StatsConnection()

    stats = await fetch_index_stats(cast(Any, conn))

    assert stats == IndexStats(5, 100, 93, 20, 3, 2, 4, 5, 1, 2)
    assert conn.query is not None
    assert "indexed_at IS NOT NULL" in conn.query
    assert "NOT index_rejected" in conn.query
    assert "WHERE index_rejected" in conn.query
    assert "indexed_at IS NULL" in conn.query
    assert "index_error IS NOT NULL" in conn.query
    assert conn.query.count("FROM article_urls") == 1
    assert conn.query.count("FROM pages") == 1


async def test_index_stats_reject_invalid_database_rows():
    class InvalidCursor:
        async def fetchone(self):
            return (5, 100, None, 20, 3, 2)

    class InvalidConnection:
        async def execute(self, query):
            return InvalidCursor()

    conn = InvalidConnection()

    try:
        await fetch_index_stats(cast(Any, conn))
    except ValueError as exc:
        assert str(exc) == "invalid index stats database row"
    else:
        raise AssertionError("invalid database row was accepted")


async def test_failure_status_reads_current_article_schema():
    class Connection:
        async def execute(self, query, params):
            assert "FROM article_urls" in query
            assert "status = 'failed'" in query
            assert "failed_batches" in query
            return self

        async def fetchall(self):
            return [("https://blog.example/post", 3, "timeout")]

    failures = await list_failed_articles(cast(Any, Connection()))
    assert failures[0].last_error == "timeout"
    assert failures[0].failed_batches == 3
