# semsearch

*Current status: PoC, AI gen not fully manually reviewed*

## Goal

Embeddings-first search for personal blogs and small sites. Ranking is semantic
similarity only in order to avoid Google PageRank-style large site monopoly.

## Implementation

- FastAPI + raw async psycopg3
- Typer admin CLI for site and index administration
- Postgres + pgvector (`halfvec` HNSW cosine ANN)
- Any OpenAI-compatible `/embeddings` endpoint

Search compiles filters into bound SQL, embeds the query once, runs dense and
Postgres full-text retrievers concurrently, fuses them with reciprocal rank
fusion (RRF), then keeps the best chunk per page. See CLAUDE.md for the score
contract and ingest pipeline. Configuration is environment variables or `.env`
(see `.env.example`); logs go to stderr as text, `LOG_LEVEL` default `INFO`.

## Structure

```text
src/semsearch/
|-- share/  # configuration, database pool, embeddings, shared utilities
|-- cli/    # Typer commands, site administration, crawling, and ingestion
`-- web/    # FastAPI application, search pipeline, and templates
```

## Development

```sh
uv sync
docker compose up -d db
cp .env.example .env            # set EMBEDDING_API_KEY before indexing
uv run semsearch init-db
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto
uv run semsearch worker         # long running daemon for feed fetching
uv run uvicorn semsearch.web.app:app --reload
```

Checks: `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`,
`uv run pyright`.

CLI commands:

```sh
uv run semsearch site add https://example.blog --sitemap auto --feed auto
uv run semsearch site poll https://example.blog
uv run semsearch site list
uv run semsearch worker
uv run semsearch status
```

`site add` requires an RSS or Atom feed and optionally stores a sitemap for
historical fallback. `worker` is the continuous process - it polls sites and
ingests discovered posts from a durable queue; run it under a process
supervisor in production. Bulk-import indieblog.page feeds with
`scripts/import_indieblog_feeds.py` (`--dry-run` first).

## Deployment

```sh
cp .env.example .env
docker compose --profile deploy up -d --build
docker compose exec app /app/.venv/bin/semsearch init-db   # first run only
```

Starts the web app (port 8000) and the continuous worker. For public
deployment, set `POSTGRES_PASSWORD` (optionally `POSTGRES_USER` and
`POSTGRES_DB`) in `.env` before the first start and put TLS in front of the
app. Postgres itself is only reachable from containers and `127.0.0.1`.

`EMBEDDING_API_BASE` must be reachable from inside the containers: a hosted API
works as-is; for an embedding server on the host, use
`http://host.docker.internal:<port>/v1`.

Run one-off admin commands inside the app container:

```sh
docker compose exec app /app/.venv/bin/semsearch status
docker compose exec app /app/.venv/bin/python scripts/import_indieblog_feeds.py --dry-run
```

The database defaults to a managed `pgdata` volume. To store it on a specific
disk, set `PGDATA_DIR` in `.env` to an absolute host path (NVMe, mounted
`noatime`); Compose bind-mounts it and Postgres keeps its data in a
major-version subdirectory inside it (e.g. `18/docker`). Changing it after the
database exists points Postgres at a fresh location, so move the old data first
or re-index.

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` invalidates the index - wipe and
re-index: TODO
