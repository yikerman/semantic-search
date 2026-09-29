-- Transactional per-site totals. Only the primary key is indexed so counter
-- updates can use HOT; reserve space for updates and vacuum this small table often.
CREATE TABLE site_stats (
    site_id bigint PRIMARY KEY REFERENCES sites(id) ON DELETE CASCADE,
    page_count bigint NOT NULL DEFAULT 0 CHECK (page_count >= 0),
    indexed_count bigint NOT NULL DEFAULT 0 CHECK (indexed_count >= 0),
    rejected_index_count bigint NOT NULL DEFAULT 0 CHECK (rejected_index_count >= 0),
    failed_index_count bigint NOT NULL DEFAULT 0 CHECK (failed_index_count >= 0),
    queued_count bigint NOT NULL DEFAULT 0 CHECK (queued_count >= 0),
    retrying_count bigint NOT NULL DEFAULT 0 CHECK (retrying_count >= 0),
    failed_count bigint NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
    rejected_count bigint NOT NULL DEFAULT 0 CHECK (rejected_count >= 0),
    CHECK (indexed_count + rejected_index_count <= page_count),
    CHECK (failed_index_count <= page_count - indexed_count - rejected_index_count),
    CHECK (retrying_count <= queued_count)
) WITH (fillfactor = 80, autovacuum_vacuum_scale_factor = 0.02);

CREATE FUNCTION initialize_site_stats() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO site_stats (site_id) SELECT id FROM new_sites;
    RETURN NULL;
END;
$$;
CREATE TRIGGER sites_stats_insert AFTER INSERT ON sites
    REFERENCING NEW TABLE AS new_sites
    FOR EACH STATEMENT EXECUTE FUNCTION initialize_site_stats();

CREATE FUNCTION pages_stats_insert() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(page_count) AS page_count,
                   sum(indexed_count) AS indexed_count,
                   sum(rejected_index_count) AS rejected_index_count,
                   sum(failed_index_count) AS failed_index_count
        FROM (
            SELECT site_id,
                   1 AS page_count,
                   (indexed_at IS NOT NULL)::int AS indexed_count,
                   index_rejected::int AS rejected_index_count,
                   (indexed_at IS NULL AND NOT index_rejected AND index_error IS NOT NULL)::int AS failed_index_count
            FROM new_rows
        ) changes
        GROUP BY site_id
        HAVING sum(page_count) <> 0
            OR sum(indexed_count) <> 0
            OR sum(rejected_index_count) <> 0
            OR sum(failed_index_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            page_count = s.page_count + delta.page_count,
            indexed_count = s.indexed_count + delta.indexed_count,
            rejected_index_count = s.rejected_index_count + delta.rejected_index_count,
            failed_index_count = s.failed_index_count + delta.failed_index_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER pages_stats_insert AFTER INSERT ON pages
    REFERENCING NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION pages_stats_insert();

CREATE FUNCTION pages_stats_update() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(page_count) AS page_count,
                   sum(indexed_count) AS indexed_count,
                   sum(rejected_index_count) AS rejected_index_count,
                   sum(failed_index_count) AS failed_index_count
        FROM (
            SELECT site_id,
                   1 AS page_count,
                   (indexed_at IS NOT NULL)::int AS indexed_count,
                   index_rejected::int AS rejected_index_count,
                   (indexed_at IS NULL AND NOT index_rejected AND index_error IS NOT NULL)::int AS failed_index_count
            FROM new_rows
            UNION ALL
            SELECT site_id,
                   -(1) AS page_count,
                   -((indexed_at IS NOT NULL)::int) AS indexed_count,
                   -(index_rejected::int) AS rejected_index_count,
                   -((indexed_at IS NULL AND NOT index_rejected AND index_error IS NOT NULL)::int) AS failed_index_count
            FROM old_rows
        ) changes
        GROUP BY site_id
        HAVING sum(page_count) <> 0
            OR sum(indexed_count) <> 0
            OR sum(rejected_index_count) <> 0
            OR sum(failed_index_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            page_count = s.page_count + delta.page_count,
            indexed_count = s.indexed_count + delta.indexed_count,
            rejected_index_count = s.rejected_index_count + delta.rejected_index_count,
            failed_index_count = s.failed_index_count + delta.failed_index_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER pages_stats_update AFTER UPDATE ON pages
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION pages_stats_update();

CREATE FUNCTION pages_stats_delete() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(page_count) AS page_count,
                   sum(indexed_count) AS indexed_count,
                   sum(rejected_index_count) AS rejected_index_count,
                   sum(failed_index_count) AS failed_index_count
        FROM (
            SELECT site_id,
                   -(1) AS page_count,
                   -((indexed_at IS NOT NULL)::int) AS indexed_count,
                   -(index_rejected::int) AS rejected_index_count,
                   -((indexed_at IS NULL AND NOT index_rejected AND index_error IS NOT NULL)::int) AS failed_index_count
            FROM old_rows
        ) changes
        GROUP BY site_id
        HAVING sum(page_count) <> 0
            OR sum(indexed_count) <> 0
            OR sum(rejected_index_count) <> 0
            OR sum(failed_index_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            page_count = s.page_count + delta.page_count,
            indexed_count = s.indexed_count + delta.indexed_count,
            rejected_index_count = s.rejected_index_count + delta.rejected_index_count,
            failed_index_count = s.failed_index_count + delta.failed_index_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER pages_stats_delete AFTER DELETE ON pages
    REFERENCING OLD TABLE AS old_rows
    FOR EACH STATEMENT EXECUTE FUNCTION pages_stats_delete();

CREATE FUNCTION pages_stats_truncate() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE site_stats SET page_count = 0, indexed_count = 0, rejected_index_count = 0, failed_index_count = 0;
    RETURN NULL;
END;
$$;
CREATE TRIGGER pages_stats_truncate AFTER TRUNCATE ON pages
    FOR EACH STATEMENT EXECUTE FUNCTION pages_stats_truncate();

CREATE FUNCTION article_urls_stats_insert() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(queued_count) AS queued_count,
                   sum(retrying_count) AS retrying_count,
                   sum(failed_count) AS failed_count,
                   sum(rejected_count) AS rejected_count
        FROM (
            SELECT site_id,
                   (status = 'pending')::int AS queued_count,
                   (status = 'pending' AND failed_batches > 0)::int AS retrying_count,
                   (status = 'failed')::int AS failed_count,
                   (status = 'rejected')::int AS rejected_count
            FROM new_rows
        ) changes
        GROUP BY site_id
        HAVING sum(queued_count) <> 0
            OR sum(retrying_count) <> 0
            OR sum(failed_count) <> 0
            OR sum(rejected_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            queued_count = s.queued_count + delta.queued_count,
            retrying_count = s.retrying_count + delta.retrying_count,
            failed_count = s.failed_count + delta.failed_count,
            rejected_count = s.rejected_count + delta.rejected_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER article_urls_stats_insert AFTER INSERT ON article_urls
    REFERENCING NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION article_urls_stats_insert();

CREATE FUNCTION article_urls_stats_update() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(queued_count) AS queued_count,
                   sum(retrying_count) AS retrying_count,
                   sum(failed_count) AS failed_count,
                   sum(rejected_count) AS rejected_count
        FROM (
            SELECT site_id,
                   (status = 'pending')::int AS queued_count,
                   (status = 'pending' AND failed_batches > 0)::int AS retrying_count,
                   (status = 'failed')::int AS failed_count,
                   (status = 'rejected')::int AS rejected_count
            FROM new_rows
            UNION ALL
            SELECT site_id,
                   -((status = 'pending')::int) AS queued_count,
                   -((status = 'pending' AND failed_batches > 0)::int) AS retrying_count,
                   -((status = 'failed')::int) AS failed_count,
                   -((status = 'rejected')::int) AS rejected_count
            FROM old_rows
        ) changes
        GROUP BY site_id
        HAVING sum(queued_count) <> 0
            OR sum(retrying_count) <> 0
            OR sum(failed_count) <> 0
            OR sum(rejected_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            queued_count = s.queued_count + delta.queued_count,
            retrying_count = s.retrying_count + delta.retrying_count,
            failed_count = s.failed_count + delta.failed_count,
            rejected_count = s.rejected_count + delta.rejected_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER article_urls_stats_update AFTER UPDATE ON article_urls
    REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
    FOR EACH STATEMENT EXECUTE FUNCTION article_urls_stats_update();

CREATE FUNCTION article_urls_stats_delete() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    delta record;
BEGIN
    -- One update per affected site, in a consistent lock order. Ignore no-ops.
    FOR delta IN
        SELECT site_id, sum(queued_count) AS queued_count,
                   sum(retrying_count) AS retrying_count,
                   sum(failed_count) AS failed_count,
                   sum(rejected_count) AS rejected_count
        FROM (
            SELECT site_id,
                   -((status = 'pending')::int) AS queued_count,
                   -((status = 'pending' AND failed_batches > 0)::int) AS retrying_count,
                   -((status = 'failed')::int) AS failed_count,
                   -((status = 'rejected')::int) AS rejected_count
            FROM old_rows
        ) changes
        GROUP BY site_id
        HAVING sum(queued_count) <> 0
            OR sum(retrying_count) <> 0
            OR sum(failed_count) <> 0
            OR sum(rejected_count) <> 0
        ORDER BY site_id
    LOOP
        UPDATE site_stats s SET
            queued_count = s.queued_count + delta.queued_count,
            retrying_count = s.retrying_count + delta.retrying_count,
            failed_count = s.failed_count + delta.failed_count,
            rejected_count = s.rejected_count + delta.rejected_count
        WHERE s.site_id = delta.site_id;
    END LOOP;
    RETURN NULL;
END;
$$;
CREATE TRIGGER article_urls_stats_delete AFTER DELETE ON article_urls
    REFERENCING OLD TABLE AS old_rows
    FOR EACH STATEMENT EXECUTE FUNCTION article_urls_stats_delete();

CREATE FUNCTION article_urls_stats_truncate() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE site_stats SET queued_count = 0, retrying_count = 0, failed_count = 0, rejected_count = 0;
    RETURN NULL;
END;
$$;
CREATE TRIGGER article_urls_stats_truncate AFTER TRUNCATE ON article_urls
    FOR EACH STATEMENT EXECUTE FUNCTION article_urls_stats_truncate();
