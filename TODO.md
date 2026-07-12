# TODO

## Next?

1. Cross-encoder reranker: rescore the top candidates into `scores["final"]`?
2. Moderation: honor code? public submission?
3. Site quality votes: human voting as site quality and reputation?
4. Query filters: date range, site include/exclude, etc.?

## Gaps

- No re-embedding pipeline.
- Chunk embeddings lack context hints: prepend the page URL and title to each
  chunk's embedding input (today it is title + chunk text). Requires
  re-embedding and updating every chunk, so defer until a re-embedding
  pipeline exists.
