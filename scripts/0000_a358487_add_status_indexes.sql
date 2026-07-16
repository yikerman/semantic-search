-- Run with:
-- psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/0000_a358487_add_status_indexes.sql
--
-- Concurrent index operations cannot run inside an explicit transaction.

CREATE INDEX CONCURRENTLY IF NOT EXISTS pages_recent_idx
    ON pages (fetched_at DESC, url);

CREATE INDEX CONCURRENTLY IF NOT EXISTS crawl_jobs_recent_failure_idx
    ON crawl_jobs (failed_at DESC, url)
    WHERE failed_at IS NOT NULL;

-- The replacement index above covers queries ordered by failed_at alone.
DROP INDEX CONCURRENTLY IF EXISTS crawl_jobs_failed_idx;
