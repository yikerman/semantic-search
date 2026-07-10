# TODO

## Next?

1. BM25: generated `tsvector`, GIN index, `Bm25Retriever`, fusion ranker?
2. Cross-encoder reranker: rescore the top candidates into `scores["final"]`?
3. Moderation: honor code? public submission?
4. Site quality votes: human voting as site quality and reputation?
5. Query filters: date range, site include/exclude, etc.?

## Gaps

- No migrations.
- No re-embedding pipeline.
- No scheduled RSS polling.
- No systematic RSS feed discovery.
