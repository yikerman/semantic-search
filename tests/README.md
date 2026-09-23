# Tests

Run `uv run pytest`. Tests are hermetic: no live database, external HTTP requests,
embedding API or downloaded tokenizer is needed. Crawler tests exercise callbacks,
policy and fake HTTP responses; indexing tests use transactional database fakes.
The extraction-pool test spawns real local processes to verify that timeouts kill
workers and replacement workers can accept subsequent tasks.
