from typing import Any, cast

from semsearch.share.status import IndexStats, fetch_index_stats


class StatsCursor:
    async def fetchone(self):
        return (5, 100, 400, 20, 3, 2)


class StatsConnection:
    def __init__(self) -> None:
        self.query: str | None = None

    async def execute(self, query):
        self.query = query
        return StatsCursor()


async def test_index_stats_estimate_index_totals_and_scan_crawl_jobs_once():
    conn = StatsConnection()

    stats = await fetch_index_stats(cast(Any, conn))

    assert stats == IndexStats(5, 100, 400, 20, 3, 2)
    assert conn.query is not None
    assert "FROM pg_stat_user_tables" in conn.query
    assert "n_live_tup" in conn.query
    assert "FROM pages" not in conn.query
    assert "FROM chunks" not in conn.query
    assert conn.query.count("FROM crawl_jobs") == 1
    assert conn.query.count("count(*) FILTER (") == 3
    assert "WHERE failed_at IS NULL AND attempt_count > 0" in conn.query


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
