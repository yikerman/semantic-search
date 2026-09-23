from typing import Any, cast

from semsearch.share.status import IndexStats, fetch_index_stats, list_failed_articles


class StatsCursor:
    async def fetchone(self):
        return (5, 100, 400, 20, 3, 2, 4, 5, 1)


class StatsConnection:
    def __init__(self) -> None:
        self.query: str | None = None

    async def execute(self, query):
        self.query = query
        return StatsCursor()


async def test_index_stats_estimate_chunks_and_scan_article_urls_once():
    conn = StatsConnection()

    stats = await fetch_index_stats(cast(Any, conn))

    assert stats == IndexStats(5, 100, 400, 20, 3, 2, 4, 5, 1)
    assert conn.query is not None
    assert "FROM pg_stat_user_tables" in conn.query
    assert "n_live_tup" in conn.query
    assert "indexed_at IS NULL" in conn.query
    assert "index_error IS NOT NULL" in conn.query
    assert conn.query.count("FROM article_urls") == 1


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
