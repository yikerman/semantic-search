# semsearch

Search for personal blogs and small sites. Ranking is semantic relevance only:
no link graph, no global popularity, no SEO-derived signals.

## Shape

Code is organized by ownership under three packages:

- `semsearch.share`: settings, database pool setup, embeddings, schema, and
  utilities used by both surfaces
- `semsearch.cli`: Typer commands, CLI database operations, site lifecycle,
  crawling, and ingestion
- `semsearch.web`: FastAPI app, web database reads, search pipeline, and
  templates

Both surfaces may import `semsearch.share`. Shared code must not import either
surface, and `semsearch.cli` and `semsearch.web` must not import each other.
Put code in `share` when both surfaces use it or when its functionality is
general-purpose and independent of either surface.

## Code standards

- Work functional-first. Prefer module-level functions, callable type aliases,
  `partial`, and explicit arguments over service objects and protocols. Use a
  class when it owns real mutable state or a resource lifecycle.
- Validate at external input boundaries: settings, web queries, crawler/network
  responses, embedding API responses, and database rows. Internal typed code
  should trust its callers and should not repeat checks for valid function
  arguments.
- Keep package initializers minimal. Import concrete modules directly instead of
  building broad re-export surfaces.
- Keep tests hermetic and use fakes for external systems.

## Search

`semsearch.web.search.pipeline.search()` compiles filters, embeds the query,
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

The `semsearch.cli.ingest` pipeline is:

1. fetch HTML with `curl-cffi`
2. extract main text with `trafilatura`
3. split into token windows with the configured embedding tokenizer
4. embed document chunks
5. store pages and chunks

URL is page identity. Existing URLs are append-only and skipped.
`robots.txt` is used for sitemap discovery; `Disallow` is not enforced yet.

Configured sites use normalized origins as human-readable ids and surrogate
`sites.id` values for foreign keys. Sitemap is optional; feed-only sites are
indexed by the continuous `daemon`, which scatters polling and discovers
current and historical URLs into a durable queue, and ingests queued pages with
bounded concurrency and retry backoff; concurrent ingest loops prefer sites no
other loop is working so fetches spread across origins. When a feed shows only
unseen URLs, historical discovery follows RFC 5005 links, then WordPress feed
pagination, then the configured sitemap, stopping after `HISTORY_POST_LIMIT`
URLs.

## Constraints

- One database holds one embedding space. Changing `EMBEDDING_MODEL` or
  `EMBEDDING_DIM` means wiping and re-indexing.
- Query embeddings use `QUERY_INSTRUCTION`; document embeddings use page title
  plus chunk text.
- Document chunks use fixed token windows from the pinned embedding tokenizer.
  Query and document embedding inputs are sent to the embedding API as text.
- Dense retrieval uses a pgvector `halfvec` HNSW cosine ANN index. Keep
  embeddings within pgvector halfvec limits; use MRL truncation if a model
  exceeds them.
- Tests use fakes; keep them hermetic.

## Dev Loop

Start only the database with Compose; run the daemon and web app on the host.

```sh
podman compose up -d db
cp .env.example .env
uv run semsearch init-db
uv run semsearch site add https://some.blog/ --sitemap auto --feed auto
uv run semsearch daemon
uv run uvicorn semsearch.web.app:app --reload
uv run pytest
uv run ruff format
uv run ruff check
uv run pyright
uv run ty check
```
