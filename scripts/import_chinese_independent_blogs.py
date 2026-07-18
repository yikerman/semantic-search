#!/usr/bin/env python3
import argparse
import asyncio
import csv
import io
import itertools
import logging
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit

from psycopg_pool import AsyncConnectionPool

from semsearch.cli.ingest.fetch import Fetcher, create_fetcher
from semsearch.cli.sites import add_site, list_sites
from semsearch.cli.url import normalize_origin, try_normalize_url
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.logging import configure_logging
from semsearch.share.util import map_concurrently

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/timqian/chinese-independent-blogs/"
    "refs/heads/master/blogs-original.csv"
)

logger = logging.getLogger("semsearch.import_chinese_independent_blogs")


@dataclass(frozen=True, slots=True)
class BlogFeed:
    origin: str
    homepage: str
    feed_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import feed-backed sites from chinese-independent-blogs."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--concurrency", type=_positive_int, default=16)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Validate and update sites that are already configured.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse the CSV without changing the database.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    async with create_fetcher(settings) as fetcher:
        rows = await fetch_rows(fetcher, args.source_url)
        feeds, duplicate_count = select_feeds(rows)
        if args.limit is not None:
            feeds = feeds[: args.limit]
        logger.info(
            "CSV contains %d rows; selected %d feed-backed origins and discarded "
            "%d duplicate-origin rows",
            len(rows),
            len(feeds),
            duplicate_count,
        )
        if args.dry_run:
            return 0
        return await import_feeds(
            settings,
            fetcher,
            feeds,
            concurrency=args.concurrency,
            refresh_existing=args.refresh_existing,
        )


async def fetch_rows(fetcher: Fetcher, url: str) -> list[dict[str, str | None]]:
    response = await fetcher.fetch_response(url)
    try:
        text = response.body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV is not valid UTF-8: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    required = {"Address", "RSS feed"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("CSV must contain Address and RSS feed columns")
    try:
        return list(reader)
    except csv.Error as exc:
        raise ValueError(f"CSV is invalid: {exc}") from exc


def select_feeds(
    rows: list[dict[str, str | None]],
) -> tuple[list[BlogFeed], int]:
    selected: dict[str, BlogFeed] = {}
    valid_rows = 0
    for line_number, row in enumerate(rows, start=2):
        feed = parse_feed(row, line_number=line_number)
        if feed is None:
            continue
        valid_rows += 1
        selected.setdefault(feed.origin, feed)
    return sorted(selected.values(), key=lambda feed: feed.origin), valid_rows - len(
        selected
    )


def parse_feed(row: dict[str, str | None], *, line_number: int) -> BlogFeed | None:
    raw_feed = (row.get("RSS feed") or "").strip()
    if not raw_feed:
        return None

    homepage = _http_url(row.get("Address"))
    feed_url = _http_url(raw_feed)
    if homepage is None:
        logger.warning("Skipping CSV line %d: invalid Address", line_number)
        return None
    if feed_url is None:
        logger.warning("Skipping CSV line %d: invalid RSS feed", line_number)
        return None
    return BlogFeed(
        origin=normalize_origin(homepage),
        homepage=homepage,
        feed_url=feed_url,
    )


async def import_feeds(
    settings: Settings,
    fetcher: Fetcher,
    feeds: list[BlogFeed],
    *,
    concurrency: int,
    refresh_existing: bool,
) -> int:
    async with create_pool(settings) as pool:
        configured = (
            {site.base_url for site in await list_sites(pool)}
            if not refresh_existing
            else set()
        )
        pending = [feed for feed in feeds if feed.origin not in configured]
        skipped = len(feeds) - len(pending)
        progress = itertools.count(1)

        async def import_one(feed: BlogFeed) -> bool:
            imported = await _import_feed(settings, fetcher, pool, feed)
            done = next(progress)
            if done % 100 == 0:
                logger.info("Processed %d/%d sites", done, len(pending))
            return imported

        results = await map_concurrently(pending, limit=concurrency, func=import_one)
        imported = sum(results)
        failed = len(results) - imported

    logger.info(
        "Import complete: %d imported, %d existing skipped, %d failed",
        imported,
        skipped,
        failed,
    )
    return 1 if failed else 0


async def _import_feed(
    settings: Settings,
    fetcher: Fetcher,
    pool: AsyncConnectionPool,
    feed: BlogFeed,
) -> bool:
    for url in (feed.feed_url, feed.homepage):
        if not await _resolves_public(url):
            logger.error("Skipping %s: non-public or unresolvable %s", feed.origin, url)
            return False
    try:
        await add_site(
            pool,
            fetcher,
            feed.homepage,
            "auto",
            feed.feed_url,
            poll_interval_seconds=settings.site_poll_interval_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed %s (%s): %s", feed.origin, feed.feed_url, exc)
        return False
    return True


async def _resolves_public(url: str) -> bool:
    host = urlsplit(url).hostname
    if host is None:
        return False
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    resolved = False
    for info in infos:
        try:
            address = ip_address(info[4][0])
        except ValueError:
            return False
        if not address.is_global:
            return False
        resolved = True
    return resolved


def _http_url(value: object) -> str | None:
    return try_normalize_url(value) if isinstance(value, str) else None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
