# TODO

1. Re-embedding pipeline: rebuild all chunks from canonical page content.
2. Better chunking method? para-based? also look into https://www.anthropic.com/engineering/contextual-retrieval and https://jina.ai/news/late-chunking-in-long-context-embedding-models/
   - And/Or maybe more sophisticated chunk score aggregate?
3. Cross-encoder reranker? how heavy would that be?
4. Moderation: honor code? public submission?
5. Evaluate replacing the pgvector HNSW index with vchordrq or vchordg. If adopted, decide whether the daemon should periodically rebuild that index.
6. Duplicate page language and publication date onto chunks so retrieval can apply those filters within its indexed candidate path.
