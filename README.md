# semsearch

*Current status: PoC; not fully manually reviewed AI codebase.*

## Goal

Semsearch is an embedding-focused indexing and search engine (ideas heavily borrowed from agentic AI RAG architecture) that aims to aggregate indie blogs.

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

Running db in container and python apps on host would be easier:

```sh
uv sync
docker compose up -d db
cp .env.example .env  # set EMBEDDING_API_KEY before indexing
uv run semsearch init-db
# manually add a site. note that feed is mandatory
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto
uv run semsearch daemon &  # long-running polling and ingestion process
uv run uvicorn semsearch.web.app:app --reload
```

Check with `pyright`, `pytest`, `ruff` and `ty`.

Apply migrations to an existing development database from the repository root:

```sh
docker compose exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' \
  < scripts/0000_a358487_add_status_indexes.sql
uv run python scripts/0001_2bfa077_add_page_language.py
```

## Deployment

```sh
cp .env.example .env  # See .env.example for config keys
docker compose up -d --build
docker compose exec app /app/.venv/bin/semsearch init-db  # first run only
```

Migrations example:

```sh
# plain sql
docker compose exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' \
  < scripts/0000_a358487_add_status_indexes.sql
# or py backfills
docker compose run --rm app \
  /app/.venv/bin/python scripts/0001_2bfa077_add_page_language.py
```

For an embedding server on the host, use
`http://host.docker.internal:<port>/some-api-endpoint` in `.env`.

Run admin commands inside the container:

```sh
docker compose exec app /app/.venv/bin/semsearch status
# if you decide to use it
docker compose exec app /app/.venv/bin/python scripts/import_indieblog_feeds.py --dry-run
```

Changing chunking algorithm, embedding dim or model requires re-indexing: TODO.
