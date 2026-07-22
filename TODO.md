# TODO

1. Re-embedding pipeline: rebuild all chunks from canonical page content.
   1. Better chunking method? para-based? also look into https://www.anthropic.com/engineering/contextual-retrieval and https://jina.ai/news/late-chunking-in-long-context-embedding-models/ (especially late chunking, need bidirectional attention)
2. Refactor the suck-ass daemon and harden it from pathelogical pages.
3. Get rid of pathelogical pages such as some feed.xml/json got from sitemap.
4. Cross-encoder reranker? how heavy would that be?
5. Moderation: honor code? public submission?
6. Swap pgvector HNSW index for vchordrq or vchordg? If adopted, should daemon periodically rebuild that index?
7. Duplicate page language and publication date onto chunks so retrieval can apply those filters within its indexed candidate path.
