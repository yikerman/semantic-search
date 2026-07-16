# semsearch

*Current status: PoC, AI gen not fully manually reviewed*

## Goal

Semsearch is an embedding-focused indexing and searching (ideas heavily
borrowed from agentic AI RAG architecture) website.

## Implementation

FastAPI frontend & pgvector database

See `search(...)` from `semsearch.web.search.pipeline`, pretty
self-explanatory code, hopefully.

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
docker compose --profile deploy up -d --build
docker compose exec app /app/.venv/bin/semsearch init-db  # first run only
```

For an existing production database container, run the same migration without
wrapping it in a transaction:

```sh
docker compose exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' \
  < scripts/0000_a358487_add_status_indexes.sql
docker compose build app daemon
docker compose run --rm app \
  /app/.venv/bin/python scripts/0001_2bfa077_add_page_language.py
docker compose --profile deploy up -d
docker compose run --rm app \
  /app/.venv/bin/python scripts/0001_2bfa077_add_page_language.py
```

The language migration is online and resumable. Its second run catches pages
that an older daemon may have inserted during the first backfill.

When upgrading from a release with the old `worker` Compose service, stop or
remove that container before starting `daemon` (or deploy with
`--remove-orphans`). Both names use the same advisory lock id, so they cannot
process the queue concurrently during the transition.

For an embedding server on the host, use
`http://host.docker.internal:<port>/some-api-endpoint` in `.env`.

Run admin commands inside the container:

```sh
docker compose exec app /app/.venv/bin/semsearch status
# if you decide to use it
docker compose exec app /app/.venv/bin/python scripts/import_indieblog_feeds.py --dry-run
```

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` invalidates existing data. To re-index: TODO
