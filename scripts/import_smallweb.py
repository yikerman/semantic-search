#!/usr/bin/env python3
import argparse
import asyncio
import logging

import httpx

from semsearch.cli.sites import add_site, list_sites
from semsearch.cli.url import normalize_origin, try_normalize_url
from semsearch.share.config import Settings, get_settings
from semsearch.share.db import create_pool
from semsearch.share.logging import configure_logging
from semsearch.share.util import map_concurrently

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/kagisearch/smallweb/main/smallweb.txt"
)

logger = logging.getLogger("semsearch.import_smallweb")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Kagi Small Web blog feeds.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--concurrency", type=positive_int, default=16)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Update configuration for sites that are already configured.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate the list without changing the database.",
    )
    return parser.parse_args()


def select_feeds(text: str) -> dict[str, str]:
    """Select the first valid feed per origin, matching site registration identity."""
    selected: dict[str, str] = {}
    duplicates = invalid = 0
    for number, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        feed = (
            try_normalize_url(value)
            if value.lower().startswith(("https://", "http://"))
            and not any(character.isspace() for character in value)
            else None
        )
        if feed is None:
            logger.warning("Skipping line %d: invalid HTTP(S) feed URL", number)
            invalid += 1
            continue
        origin = normalize_origin(feed)
        if origin in selected:
            duplicates += 1
        else:
            selected[origin] = feed
    logger.info(
        "Selected %d origins; skipped %d duplicate-origin feeds and %d invalid lines",
        len(selected),
        duplicates,
        invalid,
    )
    if not selected:
        raise ValueError("Small Web list contains no valid feed URLs")
    return selected


async def import_feeds(
    settings: Settings,
    feeds: dict[str, str],
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
        pending = [item for item in feeds.items() if item[0] not in configured]

        async def import_one(item: tuple[str, str]) -> bool:
            origin, feed = item
            try:
                await add_site(pool, origin + "/", "auto", feed)
            except Exception as exc:  # noqa: BLE001 -- report individual failures
                logger.error("Failed %s (%s): %s", origin, feed, exc)
                return False
            return True

        results = await map_concurrently(pending, limit=concurrency, func=import_one)
    imported = sum(results)
    failed = len(results) - imported
    logger.info(
        "Import complete: %d imported, %d existing skipped, %d failed",
        imported,
        len(feeds) - len(pending),
        failed,
    )
    return 1 if failed else 0


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as fetcher:
        response = await fetcher.get(args.source_url)
        response.raise_for_status()
        feeds = select_feeds(response.content.decode("utf-8-sig"))
    if args.limit is not None:
        feeds = dict(list(feeds.items())[: args.limit])
    logger.info("%d sites selected for import", len(feeds))
    if args.dry_run:
        return 0
    return await import_feeds(
        settings,
        feeds,
        concurrency=args.concurrency,
        refresh_existing=args.refresh_existing,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
