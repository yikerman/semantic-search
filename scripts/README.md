# Site imports

Use the running daemon container for imports (substitute `docker` for `podman`
when using Docker Compose):

```sh
podman compose exec daemon /app/.venv/bin/python scripts/import_indieblog_feeds.py
podman compose exec daemon /app/.venv/bin/python scripts/import_chinese_independent_blogs.py
podman compose exec daemon /app/.venv/bin/python scripts/import_smallweb.py
```

The first imports indieblog.page's JSON export; the second imports the
chinese-independent-blogs CSV; the third imports Kagi Small Web's blog feed list.
All accept `--dry-run`, `--limit`,
`--concurrency`, and `--refresh-existing`.

The Small Web importer keeps the first feed for each origin, skips existing sites
by default, and enables automatic sitemap discovery. Use `--source-url` to read a
different copy of the list.

Imports fetch the source list and register configuration only. Websites are not
contacted or DNS-checked during registration; the Scrapy downloader enforces
public destinations, robots rules and source discovery during `semsearch crawl`.

The previous robots-based site removal script has been removed. The crawler now
honors robots rules for individual requests and records rejected article URLs.
