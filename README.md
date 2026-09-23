# semsearch

*Current status: PoC, towards prod*

## Goal

Semsearch is an embedding-focused indexing and search engine (ideas heavily borrowed from agentic AI RAG architecture) that aims to aggregate and promote indie blogs.

## Implementation

FastAPI frontend & pgvector database

Refer to `search(...)` from `semsearch.web.search.pipeline`. Pretty self-explanatory code, hopefully.

## Structure

```text
src/semsearch/
|-- share/  # configuration, database pool, embeddings, shared utilities
|-- cli/    # Typer commands, site administration, crawling, and ingestion
`-- web/    # FastAPI application, search pipeline, and templates
```

## Development

Run the app with Compose (see below). For local checks:

```sh
uv sync
uv run pyright
uv run pytest
uv run ruff check
uv run ty check
```

## Deployment

Use Docker Compose or `podman compose`. Version 1.0 needs a fresh database.

```sh
cp .env.example .env  # See .env.example for config keys
docker compose build
docker compose up -d db
docker compose run --rm daemon init-db  # first run only
docker compose up -d
```

For an embedding server on the host, use
`http://host.docker.internal:<port>/some-api-endpoint` in `.env`.

Run admin commands inside the container:

```sh
docker compose run --rm daemon status
# manually add a site
docker compose run --rm daemon site add https://some.blog/ --sitemap auto --feed auto
# if you decide to use it
docker compose exec daemon /app/.venv/bin/python scripts/import_indieblog_feeds.py --dry-run
```

Changing the chunking algorithm, embedding dimension, or model requires re-indexing. TODO
