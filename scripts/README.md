Consider these less-reviewed and tested.

`remove_robots_disallowed_sites.py` is destructive by default. It removes sites
whose current `robots.txt` fully disallows the configured crawler, including all
owned pages, chunks, and crawl jobs. Stop the daemon and use `--dry-run` to
preview the candidates.
