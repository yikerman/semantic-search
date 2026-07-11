# semsearch

Search for personal blogs and small sites. Ranking is semantic relevance only:
no link graph, no global popularity, no SEO-derived signals.

## Shape

Two entrypoints share one service layer:

- `semsearch.cli`: Typer admin commands
- `semsearch.web.app`: FastAPI search page
- `db.py`: async psycopg3, raw SQL, schema init, pool lifecycle
- `sites.py`: site config and admin lifecycle
- `ingest/`: fetch, extract, chunk, embed, store
- `search/`: retrievers, rankers, page grouping

## Search

`search()` compiles filters, embeds the query, runs retrievers,
builds a candidate union, runs optional rerankers, applies RRF, then returns the
best chunk per page.

Score contract:

- retrievers write named scores, e.g. `scores["dense"]`
- rerankers return named ranked runs and may write a native diagnostic score
- RRF writes `scores["rrf"]`; it is the only final ordering score

Add BM25 as a `Retriever`, cross-encoder or preference ordering as a `Reranker`,
and filtering as SQL-backed `SearchFilter` implementations. Retriever and
reranker runs are inputs to the final RRF fusion.

## Ingest

The pipeline is:

1. fetch HTML with `curl-cffi`
2. extract main text with `trafilatura`
3. split with `char_chunks`
4. embed document chunks
5. store pages and chunks

URL is page identity. Existing URLs are skipped unless `--force` is set.
`robots.txt` is used for sitemap discovery; `Disallow` is not enforced yet.

Configured sites use normalized origins as human-readable ids and surrogate
`sites.id` values for foreign keys. Sitemap is optional; feed-only sites are
indexed by `site poll`. Public indexing commands operate on configured sites
only; keep arbitrary URL and sitemap ingest behind the service layer. `site poll
--all` polls sites concurrently with a bounded concurrency limit; feed entries
within one site remain sequential.

## Constraints

- One database holds one embedding space. Changing `EMBEDDING_MODEL` or
  `EMBEDDING_DIM` means wiping and re-indexing.
- Query embeddings use `QUERY_INSTRUCTION`; document embeddings use page title
  plus chunk text.
- Chunk windows count characters, not tokens. Keep it that way unless the
  retrieval strategy changes.
- Dense retrieval uses a pgvector `halfvec` HNSW cosine ANN index. Keep
  embeddings within pgvector halfvec limits; use MRL truncation if a model
  exceeds them.
- Tests use fakes; keep them hermetic.

## Dev Loop

```sh
podman compose up -d db
cp .env.example .env
uv run semsearch init-db
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto --index
uv run semsearch search "query"
uv run uvicorn semsearch.web.app:app --reload
uv run pytest
uv run ruff check
uv run pyright
```
