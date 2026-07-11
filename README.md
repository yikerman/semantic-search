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

## Code structure

Source code is organized by owning surface:

```text
src/semsearch/
|-- share/  # configuration, database pool, embeddings, shared utilities
|-- cli/    # Typer commands, site administration, crawling, and ingestion
`-- web/    # FastAPI application, search pipeline, and templates
```

## Setup

```sh
uv sync
podman compose up -d db
cp .env.example .env
uv run semsearch init-db
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto
uv run semsearch worker
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
uv run semsearch site poll https://example.blog
uv run semsearch site list
uv run semsearch worker
uv run semsearch status
```

`site add` requires an RSS or Atom feed and optionally stores a sitemap for
historical fallback. `site poll` immediately synchronizes one site. `worker` is
the continuous process: it scatters sites across twelve-hour polling windows,
uses conditional feed requests, and ingests discovered posts from a durable
queue. Run it under a process supervisor in production.

When a current feed contains only previously unseen URLs, historical discovery
follows RFC 5005 links, then WordPress feed pagination when applicable, and
finally the configured sitemap. A historical run stops and reports an error
after 2,000 unique post URLs. Sites without any usable historical source are
reported as potentially partial instead of being marked fully synchronized.

Existing URLs are append-only and are skipped before page fetching or
embedding. Transient page failures remain queued with bounded retry backoff;
repeated permanent failures are retained and reported by `semsearch status`.

Bulk-import the indieblog.page export with the standalone importer:

```sh
uv run python scripts/import_indieblog_feeds.py --dry-run
uv run python scripts/import_indieblog_feeds.py
```

The importer selects one feed per normalized origin, skips already configured
sites, and reports invalid, unreachable, and duplicate-origin feed rows.

Search is available through the web page. The CLI is reserved for index and
site administration.

## Search pipeline status

Current stages:

1. compile search filters into bound SQL predicates
2. embed the query once
3. run dense and PostgreSQL full-text retrievers concurrently
4. build a deduplicated candidate pool and run optional rerankers
5. fuse retriever and reranker runs with reciprocal rank fusion (RRF)
6. keep the best chunk per page and render RRF plus native source scores

## Configuration

All settings come from environment variables or `.env`; see `.env.example`.
Any OpenAI-compatible `/embeddings` endpoint works.

Application logs are written to stderr as readable text. Set `LOG_LEVEL` to
`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; the default is `INFO` for
both the web app and CLI. Uvicorn continues to provide HTTP access logs.

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

The deploy profile starts the web app and continuous ingestion worker. The app
listens on port 8000. For public deployment, change the Postgres password and
put TLS in front of the app.
