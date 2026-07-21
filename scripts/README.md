Consider these less-reviewed and tested.

`remove_robots_disallowed_sites.py` is destructive by default. It removes sites
whose current `robots.txt` fully disallows the configured crawler, including all
owned pages, chunks, and crawl jobs. Stop the daemon and use `--dry-run` to
preview the candidates.

`migrate_to_vchord_bm25.py` performs the one-time offline migration from the
old `tsvector` schema to VectorChord BM25. Stop the web app, daemon, and all
other database clients first. The script requires an exported absolute
`PGDATA_DIR` and a new absolute backup directory:

```sh
export PGDATA_DIR=/mnt/nvme/semsearch-pg
uv run python scripts/migrate_to_vchord_bm25.py /mnt/backups/semsearch-vchord
```

It writes a full logical dump and projected chunk export, stops the database,
moves the old PGDATA into the backup directory, initializes a fresh database,
and restores the existing data without crawling or embedding. A failure after
the database is stopped leaves it stopped for manual recovery.
