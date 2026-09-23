# semsearch

Search for personal blogs and small sites. Ranking is semantic relevance only:
no link graph, no global popularity, no SEO-derived signals.

## Shape

Code is organized by ownership under three packages:

- `semsearch.share`: settings, database pool setup, embeddings, schema, and
  utilities used by both surfaces
- `semsearch.cli`: Typer commands, CLI database operations, site lifecycle,
  crawling, and indexing; `semsearch.cli.crawl` owns Scrapy batches,
  `semsearch.cli.index` indexes canonical pages, `semsearch.cli.daemon` schedules
  recurring jobs, and `semsearch.cli.ingest`
  holds parsing, extraction and document validation functions
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
ranked pages.

Score contract:

- retrievers write named scores, e.g. `scores["dense"]`
- rerankers return named ranked runs and may write a native diagnostic score
- every run carries its fusion weight; `make_run` writes the named score and
  orders candidates by it, best first
- RRF reads each run's weight and writes `scores["rrf"]`; it is the only final
  ordering score

Dense ANN and BM25 (VectorChord-bm25) ship as `Retriever`s. Add new retrieval
signals as `Retriever`s, cross-encoder or preference ordering as a `Reranker`,
and filtering as SQL-backed `SearchFilter` implementations. Retriever and
reranker runs are inputs to the final RRF fusion.

## Ingest

`semsearch crawl` runs a finite Scrapy batch. It discovers current RSS/Atom and
sitemap entries plus initial RFC 5005/WordPress history. Every eligible article URL
is committed to `article_urls` before scheduling; limits constrain downloads, not
persistence of discoveries. URL is append-only page identity. Robots rules are
enforced. Scrapy owns request concurrency, domain pacing, retries and redirects.
A public-only connection resolver and scoped redirect policy bound destinations.

Article responses must be HTML/XHTML, not feeds, sitemap XML or JSON. Trafilatura
and language detection run in a bounded Pebble process pool with killable tasks.
Canonical page insertion and URL completion are atomic. Source/body/text limits
produce recorded outcomes. Origin cooldowns survive batch restarts.

`semsearch index` takes a snapshot of unindexed canonical pages and embeds each
complete article (title plus body) once. The pinned tokenizer enforces the configured
input limit, reserving 32 tokens for provider formatting. Oversized posts retain
canonical text and are marked `index_rejected` rather than truncated or retried.
The embedding, page-level BM25 vector and `indexed_at` are published atomically.
Both retrievers return pages directly; there is no chunk storage or aggregation.
Embedding failure never requires another crawl. Search sees only indexed pages.

One crawl and one index command may run concurrently; advisory locks prevent
same-command overlap and cancel work if their connection is lost. There are no
per-request leases, custom request scheduler or persistent Scrapy job directory.
The container daemon runs crawl and index jobs independently, immediately at startup
and periodically after each batch. It also owns future recurring maintenance such
as partition rebuilding for `vchordrq`; that job is not implemented.
Unexpected job failures stop sibling jobs and let the container restart the daemon.
Podman/Docker Compose is the only supported deployment path. Site registration
is configuration-only, and both feed-only and sitemap-only sites are supported.
Version 1.0 requires a fresh database; do not add legacy migrations or adapters.

## Constraints

- One database holds one embedding space. Changing `EMBEDDING_MODEL` or
  `EMBEDDING_DIM` means wiping and re-indexing.
- Query embeddings use `QUERY_INSTRUCTION`; document embeddings use page title
  plus the full article text.
- The pinned embedding tokenizer checks whole-document input length without
  truncation. Query and document embedding inputs are sent to the API as text.
- Dense retrieval uses VectorChord `rabitq8` storage and a `vchordrq` cosine index.
  PostgreSQL quantizes document and query vectors from float32 with
  `quantize_to_rabitq8`; no unquantized embedding copy is stored.
  The pinned VectorChord 1.1.1 `rabitq8` cosine operator returns negative cosine;
  negate it for the dense similarity score while ordering by the operator itself.
  Inputs are limited to pgvector's 16,000-dimensional `vector` type.
  The initial index is unpartitioned so it can be built on an empty database.
- Tests use fakes; keep them hermetic.

## Dev Loop

Run the application with Podman/Docker Compose; run hermetic checks on the host.

```sh
cp .env.example .env
podman compose build
podman compose up -d db
podman compose run --rm daemon init-db
podman compose run --rm daemon site add https://some.blog/ --sitemap auto --feed auto
podman compose up -d
uv run pytest
uv run ruff format
uv run ruff check
uv run pyright
uv run ty check
```
