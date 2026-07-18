# TODO

## Next?

- Refactor `scores["scorer-name"]` into verifiable code.
- Cross-encoder reranker: rescore the top candidates into `scores["final"]`?
- Moderation: honor code? public submission?
- Site quality votes: human voting as site quality and reputation?
- Query filters: date range, site include/exclude, etc.?
- Language selector: replace its O(pages) scan with a catalog or cache if needed.
- Re-embedding pipeline: store canonical extracted page text, recrawl legacy
  pages once, replace duplicated chunk text with source offsets plus derived
  embeddings and `tsvector`, and atomically rebuild chunks when the embedding
  model, tokenizer, dimensions, or chunk settings change.
- More sophisticated chunk score aggregate?
