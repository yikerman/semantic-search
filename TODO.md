# TODO

1. Re-embedding pipeline: rebuild page embeddings from canonical content.
2. Cross-encoder reranker? how heavy would that be?
3. Moderation: honor code? public submission?
4. Daemon maintenance: partition and rebuild the vchordrq index as the corpus grows, within the 32 GB memory budget.
5. Filter homepages and article listings before storage and embedding. Combine narrow URL rules, main-content article previews, and supporting metadata; keep discovery working and accept uncertain pages without requiring metadata. Record rejection reasons and test plain HTML posts, archives, and articles with related links. Implementation deferred.
