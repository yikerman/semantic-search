import sqlite3
from contextlib import closing
from typing import Any, cast

from semsearch.share.status import IndexStats, fetch_index_stats, list_failed_articles


async def test_status_aggregates_empty_and_multiple_sites():
    with closing(sqlite3.connect(":memory:")) as database:
        database.execute("""CREATE TABLE site_stats (
            site_id int, page_count int, indexed_count int, rejected_index_count int,
            failed_index_count int, queued_count int, retrying_count int,
            failed_count int, rejected_count int
        )""")

        class Connection:
            async def execute(self, query):
                self.row = database.execute(query.replace("::bigint", "")).fetchone()
                return self

            async def fetchone(self):
                return self.row

        conn = cast(Any, Connection())
        assert await fetch_index_stats(conn) == IndexStats(0, 0, 0, 0, 0, 0)
        database.executescript("""
            INSERT INTO site_stats VALUES (1, 4, 1, 1, 1, 2, 1, 1, 1);
            INSERT INTO site_stats VALUES (2, 7, 5, 1, 0, 3, 0, 0, 2);
            INSERT INTO site_stats VALUES (3, 0, 0, 0, 0, 0, 0, 0, 0);
        """)
        assert await fetch_index_stats(conn) == IndexStats(
            3, 11, 6, 5, 1, 1, 3, 3, 1, 2
        )


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
    assert "FROM site_stats" in conn.query
    assert "FROM pages" not in conn.query
    assert "FROM article_urls" not in conn.query


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
