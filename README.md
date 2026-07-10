*Current status: PoC, AI gen not fully manually reviewed*

# semsearch

Embeddings-first search for personal blogs and small sites. Ranking is semantic
similarity only; the project avoids traditional Google PageRank-style signals so
that indie sites can compete.

## Stack

- FastAPI + raw async psycopg3
- Typer admin CLI tool
- Postgres + pgvector
- OpenAI-compatible embeddings API
- Server-rendered HTML

## Setup

```sh
uv sync
podman compose up -d db
cp .env.example .env
uv run semsearch init-db
```

Set `EMBEDDING_API_KEY` in `.env` before indexing.

Run the web UI:

```sh
uv run uvicorn semsearch.web.app:app --reload
```

Run checks:

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyright
```

## CLI

```sh
uv run semsearch site add https://example.blog --sitemap auto --feed auto
uv run semsearch site add https://another.blog --sitemap auto --feed auto --index
uv run semsearch site index https://example.blog
uv run semsearch site poll --site https://example.blog
uv run semsearch site poll --all --concurrency 4
uv run semsearch site list
uv run semsearch site index https://example.blog --force
uv run semsearch status
```

`site add` stores crawl config, and `site index` runs sitemap ingest for that
configured site. Sites can be feed-only; use `--sitemap none` when no sitemap
exists, then `site poll` to ingest feed entries. `site poll --all` polls
configured sites concurrently, bounded by `SITE_POLL_CONCURRENCY` or
`--concurrency`.

Existing URLs are skipped without fetching. Use `--force` to refresh and
re-embed a page.

Search is available through the web page. The CLI is reserved for index and
site administration.

## Search pipeline status

Current stages:

1. compile search filters into bound SQL predicates
2. embed the query once
3. run retrievers concurrently (dense retrieval is currently the default)
4. build a deduplicated candidate pool and run optional rerankers
5. fuse retriever and reranker runs with reciprocal rank fusion (RRF)
6. keep the best chunk per page and render RRF plus native source scores

TODO:

- add concrete date and site filters using `pages.published_at` and site ids
- add a PostgreSQL full-text/BM25 retriever
- add a cross-encoder reranker that contributes a ranked run to RRF
- evaluate weighted fusion and tune the RRF constant against a relevance set
- add filter controls to the web form after concrete filters exist

## Configuration

All settings come from environment variables or `.env`; see `.env.example`.
Any OpenAI-compatible `/embeddings` endpoint works.

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` invalidates the index:

```sh
podman compose down db
podman volume rm semantic-search_pgdata
podman compose up -d db
uv run semsearch init-db
```

## Deployment

```sh
cp .env.example .env
podman compose --profile deploy up -d --build
```

The app listens on port 8000. For public deployment, change the Postgres
password and put TLS in front of the app.
