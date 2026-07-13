# semsearch

*Current status: PoC, AI gen not fully manually reviewed*

## Goal

Semsearch is an embedding-first and focused indexing and searching (ideas
heavily borrowed from AI RAG architecture) website.

## Implementation

FastAPI frontend & pgvector database

See `search(...)` from `semsearch.web.search.pipeline`, pretty
self-explaintory, hopefully.

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
cp .env.example .env  # set EMBEDDING_API_KEY before indexing
uv run semsearch init-db
# manually add a site. note that feed is mandatory
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto
uv run semsearch worker &  # long running daemon for feed fetching
uv run uvicorn semsearch.web.app:app --reload
```

Check with `pyright`, `pytest`, `ruff` and `ty`.

## Deployment

```sh
cp .env.example .env  # See .env.example for config keys
docker compose --profile deploy up -d --build
docker compose exec app /app/.venv/bin/semsearch init-db  # first run only
```

For an embedding server on the host, use
`http://host.docker.internal:<port>/some-api-endpoint` in `.env`.

Run admin commands inside the container:

```sh
docker compose exec app /app/.venv/bin/semsearch status
# if you decide to use it
docker compose exec app /app/.venv/bin/python scripts/import_indieblog_feeds.py --dry-run
```

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` invalidates existing data. To re-index: TODO
