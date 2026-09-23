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
    start_url text NOT NULL,
    feed_url text NOT NULL DEFAULT 'auto',
    sitemap_url text NOT NULL DEFAULT 'auto',
    history_complete boolean NOT NULL DEFAULT false,
    last_crawled_at timestamptz,
    last_error text,
    added_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE article_urls (
    id bigserial PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    source text NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'stored', 'rejected', 'failed')),
    failed_batches int NOT NULL DEFAULT 0 CHECK (failed_batches >= 0),
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    last_error text,
    discovered_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX article_urls_pending_idx ON article_urls (site_id, next_attempt_at, id)
    WHERE status = 'pending';
CREATE INDEX article_urls_failure_idx ON article_urls (updated_at DESC)
    WHERE status IN ('rejected', 'failed');

CREATE TABLE origin_cooldowns (
    origin text PRIMARY KEY,
    until_at timestamptz NOT NULL
);

-- canonical extracted pages divided into derived retrieval chunks
CREATE TABLE pages (
    id bigserial PRIMARY KEY,
    site_id bigint NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    title text,
    content text NOT NULL,
    published_at timestamptz,
    language text,
    fetched_at timestamptz NOT NULL,
    indexed_at timestamptz,
    index_error text
);

CREATE INDEX pages_pending_index_idx ON pages (id) WHERE indexed_at IS NULL;

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
