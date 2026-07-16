CREATE EXTENSION IF NOT EXISTS vector;

-- a site with several pages
CREATE TABLE IF NOT EXISTS sites (
    id bigserial PRIMARY KEY,
    base_url text UNIQUE NOT NULL,
    sitemap_url text,
    feed_url text NOT NULL,
    last_polled_at timestamptz,
    next_poll_at timestamptz,
    feed_etag text,
    feed_last_modified text,
    poll_failures int NOT NULL DEFAULT 0,
    poll_lease_until timestamptz,
    poll_lease_token uuid,
    sync_error text,
    history_pending boolean NOT NULL DEFAULT false,
    history_error text,
    added_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sites_next_poll_idx ON sites (next_poll_at);

-- a site has several to-crawl pages
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id bigserial PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    source text NOT NULL,
    attempt_count int NOT NULL DEFAULT 0,
    next_attempt_at timestamptz DEFAULT now(),
    lease_until timestamptz,
    lease_token uuid,
    last_error text,
    failed_at timestamptz,
    discovered_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS crawl_jobs_ready_idx
    ON crawl_jobs (next_attempt_at, lease_until)
    WHERE next_attempt_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS crawl_jobs_site_idx
    ON crawl_jobs (site_id, next_attempt_at);

CREATE INDEX IF NOT EXISTS crawl_jobs_recent_failure_idx
    ON crawl_jobs (failed_at DESC, url)
    WHERE failed_at IS NOT NULL;

-- pages divided into several chunks
CREATE TABLE IF NOT EXISTS pages (
    id bigserial PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    title text,
    published_at timestamptz,
    language text,
    fetched_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS pages_site_idx ON pages (site_id);

CREATE INDEX IF NOT EXISTS pages_recent_idx
    ON pages (fetched_at DESC, url);

CREATE INDEX IF NOT EXISTS pages_language_idx
    ON pages (language)
    WHERE language IS NOT NULL;

-- each chunk hold an embedding
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    page_id bigint NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    content text NOT NULL,
    char_count int NOT NULL,
    embedding halfvec({embedding_dim}) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED,
    UNIQUE (page_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding halfvec_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_search_vector_gin_idx
    ON chunks USING gin (search_vector);
