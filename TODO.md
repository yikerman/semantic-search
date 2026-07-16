# TODO

## Next?

- Cross-encoder reranker: rescore the top candidates into `scores["final"]`?
- Moderation: honor code? public submission?
- Site quality votes: human voting as site quality and reputation?
- Query filters: date range, site include/exclude, etc.?
- Language selector: replace its O(pages) scan with a catalog or cache if needed.
- Re-embedding pipeline to test more embedding?
- More sophisticated chunk score aggregate?
