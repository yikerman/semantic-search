# TODO

1. Refactor `scores["scorer-name"]` into verifiable code.
2. Re-embedding pipeline (0.2.0)
  - Preparation: new db structure: pages keep track of canonical single source of truth text. chunks track offset, length, embedding and tsvector.
  - Migration: keep track of all pages urls, and their parent site, erase pages and chunks table completely.
  - Actual re-embedding: wipe all chunks and repopulate.
3. Better chunking method? para-based?
4. Cross-encoder reranker? how heavy would that be?
5. Moderation: honor code? public submission?
6. Site quality votes: human voting as site quality and reputation?
7. More sophisticated chunk score aggregate?
