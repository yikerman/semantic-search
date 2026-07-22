#!/usr/bin/env python3
import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import psycopg
import psycopg_pool
from psycopg_pool import AsyncConnectionPool

from semsearch.cli.daemon.run import (
    DAEMON_LOCK_ID,
    DaemonAlreadyRunningError,
    advisory_lock,
)
from semsearch.cli.ingest.fetch import FetchError, Fetcher, create_fetcher
from semsearch.cli.models import Site
from semsearch.cli.sites import list_sites, remove_sites
from semsearch.share.config import get_settings
from semsearch.share.db import create_pool
from semsearch.share.logging import configure_logging
from semsearch.share.util import map_concurrently

logger = logging.getLogger("semsearch.remove_robots_disallowed_sites")


@dataclass(frozen=True, slots=True)
class RobotsFailure:
    base_url: str
    error: str


@dataclass(frozen=True, slots=True)
class CleanupResult:
    checked: int
    kept: int
    blocked: tuple[str, ...]
    removed: tuple[str, ...]
    failures: tuple[RobotsFailure, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove configured sites whose robots.txt fully disallows the crawler."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report sites that would be removed without changing the database.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        async with (
            create_pool(settings) as pool,
            create_fetcher(settings) as fetcher,
        ):
            if args.dry_run:
                result = await remove_fully_disallowed_sites(
                    pool,
                    fetcher,
                    user_agent=settings.user_agent,
                    concurrency=settings.fetch_concurrency,
                    dry_run=True,
                )
            else:
                async with advisory_lock(pool, DAEMON_LOCK_ID):
                    result = await remove_fully_disallowed_sites(
                        pool,
                        fetcher,
                        user_agent=settings.user_agent,
                        concurrency=settings.fetch_concurrency,
                        dry_run=False,
                    )
    except (
        DaemonAlreadyRunningError,
        psycopg.Error,
        psycopg_pool.PoolTimeout,
    ) as exc:
        logger.error("Cleanup failed: %s", exc)
        return 1

    report_cleanup(result, dry_run=args.dry_run)
    return 1 if result.failures else 0


async def remove_fully_disallowed_sites(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    *,
    user_agent: str,
    concurrency: int,
    dry_run: bool,
) -> CleanupResult:
    sites = await list_sites(pool)
    checks = await map_concurrently(
        sites,
        limit=concurrency,
        func=partial(_check_site, fetcher, user_agent),
    )
    blocked = tuple(
        check.site.base_url for check in checks if check.blocked and check.error is None
    )
    failures = tuple(
        RobotsFailure(check.site.base_url, check.error)
        for check in checks
        if check.error is not None
    )
    removed: Sequence[str] = ()
    if blocked and not dry_run:
        removed = await remove_sites(pool, blocked)
        removed_set = set(removed)
        failures += tuple(
            RobotsFailure(base_url, "site disappeared before it could be removed")
            for base_url in blocked
            if base_url not in removed_set
        )

    kept = sum(not check.blocked for check in checks if check.error is None)
    return CleanupResult(
        checked=len(sites),
        kept=kept,
        blocked=blocked,
        removed=tuple(removed),
        failures=failures,
    )


def fully_disallows(robots_txt: str, user_agent: str) -> bool:
    """Return whether the effective policy denies every non-robots path."""
    parser = RobotFileParser()
    parser.parse(robots_txt.splitlines())
    # Python 3.14 implements RFC 9309 group selection here but does not expose
    # the selected entry publicly. Keep that compatibility boundary local.
    entry = cast(Any, parser)._find_entry(user_agent)
    if entry is None:
        return False

    has_all_path_deny = any(
        not rule.allowance and _matches_every_path(rule.path, rule.fullmatch)
        for rule in entry.rulelines
    )
    has_allow_exception = any(
        rule.allowance and rule.path not in ("", "/robots.txt")
        for rule in entry.rulelines
    )
    return has_all_path_deny and not has_allow_exception


def report_cleanup(result: CleanupResult, *, dry_run: bool) -> None:
    selected = result.blocked if dry_run else result.removed
    action = "Would remove" if dry_run else "Removed"
    for base_url in selected:
        logger.info("%s %s", action, base_url)
    for failure in result.failures:
        logger.error("Kept %s: %s", failure.base_url, failure.error)
    selected_label = "would remove" if dry_run else "removed"
    logger.info(
        "Cleanup complete: %d checked, %d kept, %d %s, %d failed",
        result.checked,
        result.kept,
        len(selected),
        selected_label,
        len(result.failures),
    )


@dataclass(frozen=True, slots=True)
class _RobotsCheck:
    site: Site
    blocked: bool = False
    error: str | None = None


async def _check_site(fetcher: Fetcher, user_agent: str, site: Site) -> _RobotsCheck:
    robots_url = urljoin(site.base_url, "/robots.txt")
    try:
        robots_txt = await fetcher.fetch_text(robots_url)
    except FetchError as exc:
        return _RobotsCheck(site, error=str(exc))
    return _RobotsCheck(site, blocked=fully_disallows(robots_txt, user_agent))


def _matches_every_path(path: str, fullmatch: bool) -> bool:
    if path == "/":
        return not fullmatch
    return path in ("*", "/*")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
