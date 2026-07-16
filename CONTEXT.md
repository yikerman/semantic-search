# semsearch

Semsearch discovers and indexes Pages from configured Sites so they can be
ranked by semantic relevance.

## Language

**Crawl job**:
A durable request to fetch and index one Page URL. It remains until the Page is
known to the index or its retry budget is exhausted.
_Avoid_: Queue item, task

**Crawl attempt**:
One effort to fulfill a Crawl job. A Crawl job may have several attempts before
it is completed or fails.
_Avoid_: Job run, execution

**Daemon**:
The long-running process that discovers Pages from Sites and fulfills Crawl
jobs.
_Avoid_: Worker
