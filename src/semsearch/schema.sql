CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sites (
    id bigserial PRIMARY KEY,
    base_url text UNIQUE NOT NULL,
    sitemap_url text,
    feed_url text,
    last_indexed_at timestamptz,
    last_polled_at timestamptz,
    feed_etag text,
    feed_last_modified text,
    added_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    id bigserial PRIMARY KEY,
    site_id bigint REFERENCES sites(id) ON DELETE CASCADE,
    url text UNIQUE NOT NULL,
    title text,
    published_at timestamptz,
    fetched_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    page_id bigint NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    content text NOT NULL,
    char_count int NOT NULL,
    embedding halfvec({embedding_dim}) NOT NULL,
    UNIQUE (page_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding halfvec_cosine_ops);

CREATE TABLE IF NOT EXISTS index_meta (
    id int PRIMARY KEY CHECK (id = 1),
    embedding_model text NOT NULL,
    embedding_dim int NOT NULL
);
