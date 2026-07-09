# TODO

## Next?

1. BM25 + RRF: generated `tsvector`, GIN index, `Bm25Retriever`, fusion ranker.
   Needed for exact names and quoted phrases.
2. Cross-encoder reranker: rescore the top candidates into `scores["final"]`.
3. Other rankers?

## Design Questions?

- Moderation: honor code? public submission?
- Site quality votes: human voting as site quality and reputation?
- Query filters: date range, site include/exclude, etc.?

## Gaps

- No migrations.
- No re-embedding pipeline.
- No scheduled RSS polling.
