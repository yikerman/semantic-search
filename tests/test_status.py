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


async def test_index_stats_aggregate_crawl_job_counts_in_one_scan():
    conn = StatsConnection()

    stats = await fetch_index_stats(cast(Any, conn))

    assert stats == IndexStats(5, 100, 400, 20, 3, 2)
    assert conn.query is not None
    assert conn.query.count("FROM crawl_jobs") == 1
    assert conn.query.count("FILTER (") == 3
    assert "WHERE failed_at IS NULL AND attempt_count > 0" in conn.query
