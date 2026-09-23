#!/usr/bin/env python3
import argparse
import asyncio
import itertools
import json
import logging
from dataclasses import dataclass

import httpx
from psycopg_pool import AsyncConnectionPool

from semsearch.cli.sites import add_site, list_sites
from semsearch.cli.url import normalize_origin, try_normalize_url
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.logging import configure_logging
from semsearch.share.util import map_concurrently

DEFAULT_EXPORT_URL = "https://indieblog.page/export"

logger = logging.getLogger("semsearch.import_indieblog")


@dataclass(frozen=True, slots=True)
class ExportFeed:
    origin: str
    homepage: str
    feed_url: str
    errors: int
    fetched: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import feed-backed sites from the indieblog.page JSON export."
    )
    parser.add_argument("--export-url", default=DEFAULT_EXPORT_URL)
    parser.add_argument("--concurrency", type=_positive_int, default=16)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Update configuration for sites that are already configured.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate the export without changing the database.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as fetcher:
        rows = await fetch_export(fetcher, args.export_url)
        feeds, duplicate_count = select_feeds(rows)
        if args.limit is not None:
            feeds = feeds[: args.limit]
        logger.info(
            "Export contains %d rows; selected %d origins and discarded %d "
            "duplicate-origin alternatives",
            len(rows),
            len(feeds),
            duplicate_count,
        )
        if args.dry_run:
            return 0
        return await import_feeds(
            settings,
            feeds,
            concurrency=args.concurrency,
            refresh_existing=args.refresh_existing,
        )


async def fetch_export(fetcher: httpx.AsyncClient, url: str) -> list[object]:
    response = await fetcher.get(url)
    response.raise_for_status()
    try:
        payload = json.loads(response.content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Export returned invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Export must be a JSON array")  # noqa: TRY004
    return payload


def select_feeds(rows: list[object]) -> tuple[list[ExportFeed], int]:
    selected: dict[str, ExportFeed] = {}
    valid_rows = 0
    for index, row in enumerate(rows):
        feed = parse_export_feed(row, index=index)
        if feed is None:
            continue
        valid_rows += 1
        current = selected.get(feed.origin)
        if current is None or _preference(feed) > _preference(current):
            selected[feed.origin] = feed
    feeds = sorted(selected.values(), key=lambda feed: feed.origin)
    return feeds, valid_rows - len(feeds)


def parse_export_feed(row: object, *, index: int) -> ExportFeed | None:
    if not isinstance(row, dict):
        logger.warning("Skipping export row %d: expected an object", index)
        return None
    feed_url = _http_url(row.get("feedurl"))
    homepage = _http_url(row.get("homepage"))
    if feed_url is None:
        logger.warning("Skipping export row %d: invalid feedurl", index)
        return None
    site_url = homepage or feed_url
    try:
        origin = normalize_origin(site_url)
    except ValueError:
        logger.warning("Skipping export row %d: invalid homepage", index)
        return None
    errors = row.get("errors")
    fetched = row.get("fetched")
    return ExportFeed(
        origin=origin,
        homepage=site_url,
        feed_url=feed_url,
        errors=errors if isinstance(errors, int) and errors >= 0 else 0,
        fetched=fetched if isinstance(fetched, int) and fetched >= 0 else 0,
    )


async def import_feeds(
    settings: Settings,
    feeds: list[ExportFeed],
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

        async def import_one(feed: ExportFeed) -> bool:
            imported = await _import_feed(pool, feed)
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
    pool: AsyncConnectionPool,
    feed: ExportFeed,
) -> bool:
    try:
        await add_site(
            pool,
            feed.homepage,
            "auto",
            feed.feed_url,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed %s (%s): %s", feed.origin, feed.feed_url, exc)
        return False
    return True


def _http_url(value: object) -> str | None:
    return try_normalize_url(value) if isinstance(value, str) else None


def _preference(feed: ExportFeed) -> tuple[int, int, str]:
    return (-feed.errors, feed.fetched, feed.feed_url)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
