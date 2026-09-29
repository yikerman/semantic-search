-- Run against disposable PostgreSQL with psql -v ON_ERROR_STOP=1
-- -v stats_schema=/path/to/src/semsearch/share/stats.sql -f tests/sql/stats.sql.
-- All fixtures, functions, and triggers are rolled back at the end.
BEGIN;
CREATE SCHEMA stats_check;
SET LOCAL search_path = stats_check;
CREATE TABLE sites (id bigint PRIMARY KEY);
CREATE TABLE pages (
    id bigint PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    indexed_at timestamptz,
    index_rejected boolean NOT NULL DEFAULT false,
    index_error text
);
CREATE TABLE article_urls (
    id bigint PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'pending',
    failed_batches int NOT NULL DEFAULT 0
);
\i :stats_schema

CREATE VIEW expected_stats AS
SELECT s.id AS site_id,
       p.page_count, p.indexed_count, p.rejected_index_count, p.failed_index_count,
       u.queued_count, u.retrying_count, u.failed_count, u.rejected_count
FROM sites s
CROSS JOIN LATERAL (
    SELECT count(*) AS page_count,
           count(*) FILTER (WHERE indexed_at IS NOT NULL) AS indexed_count,
           count(*) FILTER (WHERE index_rejected) AS rejected_index_count,
           count(*) FILTER (WHERE indexed_at IS NULL AND NOT index_rejected AND index_error IS NOT NULL) AS failed_index_count
    FROM pages WHERE site_id = s.id
) p
CROSS JOIN LATERAL (
    SELECT count(*) FILTER (WHERE status = 'pending') AS queued_count,
           count(*) FILTER (WHERE status = 'pending' AND failed_batches > 0) AS retrying_count,
           count(*) FILTER (WHERE status = 'failed') AS failed_count,
           count(*) FILTER (WHERE status = 'rejected') AS rejected_count
    FROM article_urls WHERE site_id = s.id
) u;
CREATE FUNCTION assert_stats() RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (
        (TABLE site_stats EXCEPT TABLE expected_stats)
        UNION ALL
        (TABLE expected_stats EXCEPT TABLE site_stats)
    ) THEN
        RAISE EXCEPTION 'maintained counters differ from a full recount';
    END IF;
END;
$$;
SELECT assert_stats();
INSERT INTO sites VALUES (1), (2), (3);
INSERT INTO sites VALUES (1) ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id;
SELECT assert_stats();

-- Duplicate discovery, multiple sites, and a bulk insert.
INSERT INTO article_urls (id, site_id)
SELECT i, 1 + i % 2 FROM generate_series(1, 10000) i;
INSERT INTO article_urls (id, site_id) VALUES (1, 2) ON CONFLICT DO NOTHING;
SELECT assert_stats();
UPDATE article_urls SET failed_batches = 1 WHERE id < 100;
SELECT assert_stats();
UPDATE article_urls SET failed_batches = 2 WHERE id < 100;
SELECT assert_stats();
UPDATE article_urls SET status = 'failed' WHERE id < 50;
UPDATE article_urls SET status = 'rejected' WHERE id BETWEEN 50 AND 100;
SELECT assert_stats();
UPDATE article_urls SET status = 'pending', failed_batches = 0 WHERE status IN ('failed', 'rejected');
SELECT assert_stats();

-- Page publication and URL completion are one transaction.
INSERT INTO pages (id, site_id) SELECT id, site_id FROM article_urls WHERE id < 100;
UPDATE article_urls SET status = 'stored' WHERE id < 100;
INSERT INTO pages (id, site_id) VALUES (1, 2) ON CONFLICT DO NOTHING;
SELECT assert_stats();
UPDATE pages SET index_error = 'provider failure' WHERE id < 50;
SELECT assert_stats();
UPDATE pages SET index_error = 'another provider failure' WHERE id < 50;
SELECT assert_stats();
UPDATE pages SET indexed_at = now(), index_error = NULL WHERE id < 25;
UPDATE pages SET index_rejected = true, index_error = 'too long' WHERE id BETWEEN 25 AND 30;
SELECT assert_stats();
-- No-op UPDATE and INSERT statements must leave totals unchanged.
UPDATE pages SET indexed_at = indexed_at;
UPDATE pages SET index_error = NULL WHERE id = -1;
INSERT INTO pages (id, site_id) SELECT 10001, 1 WHERE false;
SELECT assert_stats();

-- Changes of ownership and independent deletion.
UPDATE pages SET site_id = 3 WHERE id < 20;
UPDATE article_urls SET site_id = 3 WHERE id < 20;
DELETE FROM pages WHERE id BETWEEN 31 AND 35;
DELETE FROM article_urls WHERE id BETWEEN 31 AND 35;
SELECT assert_stats();
SAVEPOINT before_changes;
DELETE FROM sites WHERE id = 3;
SELECT assert_stats();
ROLLBACK TO before_changes;
SELECT assert_stats();
DELETE FROM sites WHERE id IN (1, 3);
SELECT assert_stats();
TRUNCATE pages;
SELECT assert_stats();
TRUNCATE article_urls;
SELECT assert_stats();
TRUNCATE sites CASCADE;
SELECT assert_stats();
ROLLBACK;
