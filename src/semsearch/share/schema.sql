CREATE EXTENSION vector;
CREATE EXTENSION pg_tokenizer CASCADE;
CREATE EXTENSION vchord_bm25 CASCADE;

SELECT create_tokenizer('semsearch_llmlingua2', $$
model = "llmlingua2"
$$);

-- a site with several pages
CREATE TABLE sites (
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

CREATE INDEX sites_next_poll_idx ON sites (next_poll_at);

-- a site has several to-crawl pages
CREATE TABLE crawl_jobs (
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

CREATE INDEX crawl_jobs_ready_idx
    ON crawl_jobs (next_attempt_at, lease_until)
    WHERE next_attempt_at IS NOT NULL;

CREATE INDEX crawl_jobs_site_idx
    ON crawl_jobs (site_id, next_attempt_at);

CREATE INDEX crawl_jobs_recent_failure_idx
    ON crawl_jobs (failed_at DESC, url)
    WHERE failed_at IS NOT NULL;

-- canonical extracted pages divided into derived retrieval chunks
CREATE TABLE pages (
    id bigserial PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    title text,
    content text NOT NULL,
    published_at timestamptz,
    language text,
    fetched_at timestamptz NOT NULL
);

CREATE INDEX pages_site_idx ON pages (site_id);

CREATE INDEX pages_recent_idx
    ON pages (fetched_at DESC, url);

CREATE INDEX pages_published_at_idx
    ON pages (published_at)
    WHERE published_at IS NOT NULL;

CREATE INDEX pages_language_idx
    ON pages (language)
    WHERE language IS NOT NULL;

-- each chunk identifies a span of its page and holds its retrieval data
CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    page_id bigint NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    start_offset int NOT NULL CHECK (start_offset >= 0),
    content_length int NOT NULL CHECK (content_length > 0),
    embedding halfvec({embedding_dim}) NOT NULL,
    search_vector bm25vector NOT NULL,
    UNIQUE (page_id, start_offset)
);

CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding halfvec_cosine_ops);

CREATE INDEX chunks_search_vector_bm25_idx
    ON chunks USING bm25 (search_vector bm25_ops);
