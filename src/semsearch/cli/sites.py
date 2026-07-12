import asyncio
import hashlib
import logging
from dataclasses import dataclass
from functools import partial
from uuid import UUID
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from psycopg_pool import AsyncConnectionPool
from trafilatura.feeds import FeedParameters, determine_feed

from semsearch.cli import db
from semsearch.cli.ingest import sitemap
from semsearch.cli.ingest.feed import FeedError, ParsedFeed, parse_feed
from semsearch.cli.ingest.fetch import FetchError, Fetcher, FetchResponse
from semsearch.cli.ingest.lease import run_with_lease
from semsearch.cli.models import Site
from semsearch.cli.url import (
    canonicalize_url,
    normalize_origin,
    normalize_url,
    same_site,
)
from semsearch.share.config import Settings

logger = logging.getLogger(__name__)


class SiteError(RuntimeError):
    pass


class HistoryLimitError(RuntimeError):
    pass


class HistoryUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PollOutcome:
    site: Site
    discovered: int = 0
    not_modified: bool = False
    error: str | None = None


async def add_site(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    url: str,
    sitemap_url: str = "auto",
    feed_url: str = "auto",
    *,
    poll_interval_seconds: int = 43_200,
) -> Site:
    try:
        start_url = normalize_url(url)
    except ValueError as exc:
        raise SiteError(str(exc)) from exc

    resolved_feed, feed_home = await _resolve_feed(fetcher, start_url, feed_url)
    base_url = normalize_origin(feed_home or start_url)
    resolved_sitemap = await _resolve_sitemap(
        fetcher,
        feed_home or start_url,
        sitemap_url,
    )
    async with pool.connection() as conn, conn.transaction():
        return await db.upsert_site_config(
            conn,
            base_url=base_url,
            sitemap_url=resolved_sitemap,
            feed_url=resolved_feed,
            initial_poll_delay_seconds=_poll_offset(
                base_url,
                poll_interval_seconds,
            ),
        )


async def list_sites(pool: AsyncConnectionPool) -> list[Site]:
    async with pool.connection() as conn:
        return await db.list_site_configs(conn)


async def get_site(pool: AsyncConnectionPool, site: str) -> Site:
    try:
        base_url = normalize_origin(site)
    except ValueError as exc:
        raise SiteError(str(exc)) from exc
    async with pool.connection() as conn:
        row = await db.find_site_config(conn, base_url=base_url)
    if row is None:
        raise SiteError(f"Unknown feed-backed site: {base_url}")
    return row


async def poll_site(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    settings: Settings,
    site: str,
) -> PollOutcome:
    record = await get_site(pool, site)
    async with pool.connection() as conn, conn.transaction():
        claimed = await db.claim_site(conn, site_id=record.id)
    if claimed is None:
        raise SiteError(f"Site is already being polled: {record.base_url}")
    record, lease_token = claimed
    try:
        return await poll_site_record(pool, fetcher, settings, record, lease_token)
    except Exception as exc:
        async with pool.connection() as conn, conn.transaction():
            await db.mark_poll_failed(
                conn,
                site_id=record.id,
                lease_token=lease_token,
                error=str(exc),
                interval_seconds=settings.site_poll_interval_seconds,
            )
        raise


async def poll_site_record(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    settings: Settings,
    record: Site,
    lease_token: UUID,
) -> PollOutcome:
    return await run_with_lease(
        partial(_poll_site_record, pool, fetcher, settings, record, lease_token),
        partial(_renew_poll_lease, pool, record.id, lease_token),
    )


async def _poll_site_record(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    settings: Settings,
    record: Site,
    lease_token: UUID,
) -> PollOutcome:
    headers: dict[str, str] = {}
    if not record.history_pending and record.feed_etag:
        headers["If-None-Match"] = record.feed_etag
    if not record.history_pending and record.feed_last_modified:
        headers["If-Modified-Since"] = record.feed_last_modified

    response = await fetcher.fetch_response(
        record.feed_url,
        headers=headers or None,
        allow_not_modified=True,
    )
    if response.status == 304:
        async with pool.connection() as conn, conn.transaction():
            await db.mark_poll_succeeded(
                conn,
                site_id=record.id,
                etag=_header(response, "etag"),
                modified=_header(response, "last-modified"),
                interval_seconds=settings.site_poll_interval_seconds,
                lease_token=lease_token,
            )
        return PollOutcome(record, not_modified=True)

    parsed = _site_feed(await _parse_response(response), record)
    async with pool.connection() as conn, conn.transaction():
        known = await db.known_urls(conn, parsed.urls)
        all_new = bool(parsed.urls) and not known
        current_urls = (
            parsed.urls[: settings.history_post_limit]
            if (all_new or record.history_pending)
            else parsed.urls
        )
        discovered = await db.enqueue_urls(
            conn,
            site_id=record.id,
            urls=current_urls,
            source="feed",
        )
        if all_new and not record.history_pending:
            await db.mark_history_pending(
                conn, site_id=record.id, lease_token=lease_token
            )

    history_error: str | None = None
    if all_new or record.history_pending:
        try:
            discovered += await _discover_history(
                pool,
                fetcher,
                record,
                parsed,
                settings.history_post_limit,
            )
            async with pool.connection() as conn, conn.transaction():
                await db.finish_history(
                    conn, site_id=record.id, lease_token=lease_token
                )
        except HistoryLimitError as exc:
            history_error = str(exc)
            logger.error("%s: %s", record.base_url, history_error)
            async with pool.connection() as conn, conn.transaction():
                await db.finish_history(
                    conn,
                    site_id=record.id,
                    lease_token=lease_token,
                    error=history_error,
                )
        except (
            FetchError,
            FeedError,
            sitemap.SitemapError,
            HistoryUnavailableError,
        ) as exc:
            history_error = f"Historical discovery will retry: {exc}"
            logger.warning("%s: %s", record.base_url, history_error)

    async with pool.connection() as conn, conn.transaction():
        await db.mark_poll_succeeded(
            conn,
            site_id=record.id,
            etag=_header(response, "etag"),
            modified=_header(response, "last-modified"),
            interval_seconds=settings.site_poll_interval_seconds,
            lease_token=lease_token,
            sync_error=history_error,
        )
    return PollOutcome(record, discovered=discovered, error=history_error)


async def _discover_history(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    site: Site,
    current: ParsedFeed,
    limit: int,
) -> int:
    seen_posts = set(current.urls)
    if len(seen_posts) > limit:
        raise _history_limit(site, limit)

    if current.is_complete:
        return 0

    if current.history_url is not None:
        try:
            return await _walk_feed_history(
                pool,
                fetcher,
                site,
                current.history_url,
                seen_posts,
                limit,
            )
        except FetchError as exc:
            if not exc.permanent:
                raise
            logger.warning("%s: RFC 5005 history unavailable: %s", site.base_url, exc)
        except FeedError as exc:
            logger.warning("%s: invalid RFC 5005 history: %s", site.base_url, exc)

    if current.is_wordpress:
        discovered, usable = await _walk_wordpress_pages(
            pool,
            fetcher,
            site,
            seen_posts,
            limit,
        )
        if usable:
            return discovered

    return await _discover_sitemap(pool, fetcher, site, seen_posts, limit)


async def _walk_feed_history(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    site: Site,
    start_url: str,
    seen_posts: set[str],
    limit: int,
) -> int:
    visited_documents: set[str] = set()
    fingerprints: set[bytes] = set()
    discovered = 0
    url: str | None = start_url
    while url is not None and url not in visited_documents:
        visited_documents.add(url)
        response = await fetcher.fetch_response(url)
        fingerprint = hashlib.sha256(response.body).digest()
        if fingerprint in fingerprints:
            break
        fingerprints.add(fingerprint)
        parsed = _site_feed(await _parse_response(response), site)
        discovered += await _enqueue_fresh(pool, site, parsed.urls, seen_posts, limit)
        url = parsed.history_url
    return discovered


async def _walk_wordpress_pages(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    site: Site,
    seen_posts: set[str],
    limit: int,
) -> tuple[int, bool]:
    discovered = 0
    fingerprints: set[bytes] = set()
    page = 2
    while True:
        url = _with_query(site.feed_url, "paged", str(page))
        try:
            response = await fetcher.fetch_response(url)
        except FetchError as exc:
            if page == 2 and exc.status in (404, 410):
                return 0, False
            if exc.status in (404, 410):
                return discovered, True
            raise
        fingerprint = hashlib.sha256(response.body).digest()
        if fingerprint in fingerprints:
            return (0, False) if page == 2 else (discovered, True)
        fingerprints.add(fingerprint)
        parsed = _site_feed(await _parse_response(response), site)
        if all(candidate in seen_posts for candidate in parsed.urls):
            return (0, False) if page == 2 else (discovered, True)
        discovered += await _enqueue_fresh(pool, site, parsed.urls, seen_posts, limit)
        page += 1


async def _discover_sitemap(
    pool: AsyncConnectionPool,
    fetcher: Fetcher,
    site: Site,
    seen_posts: set[str],
    limit: int,
) -> int:
    if site.sitemap_url is None:
        raise HistoryUnavailableError(
            f"No historical feed or sitemap is available for {site.base_url}; "
            "the site may be only partially synchronized"
        )
    available = max(0, limit - len(seen_posts))
    pages = await sitemap.collect_page_urls(
        fetcher.fetch_text,
        site.sitemap_url,
        accept=lambda url: (
            same_site(url, site.base_url)
            and sitemap.is_post_url(url)
            and canonicalize_url(url, origin=site.base_url) not in seen_posts
        ),
        allow_sitemap=lambda url: same_site(url, site.base_url),
        limit=available + 1,
        strict=True,
    )
    candidates = [canonicalize_url(url, origin=site.base_url) for url in pages]
    return await _enqueue_fresh(
        pool, site, candidates, seen_posts, limit, source="sitemap"
    )


async def _enqueue_fresh(
    pool: AsyncConnectionPool,
    site: Site,
    candidates: list[str],
    seen_posts: set[str],
    limit: int,
    *,
    source: str = "history",
) -> int:
    """Enqueue not-yet-seen URLs, honoring the shared history-post budget.

    Truncates to the remaining budget and raises HistoryLimitError when the
    site's discovered posts would exceed ``limit`` so every discovery walker
    (feed, WordPress pages, sitemap) enforces the cap identically.
    """
    fresh = [url for url in candidates if url not in seen_posts]
    if not fresh:
        return 0
    seen_posts.update(fresh)
    exceeded = len(seen_posts) > limit
    if exceeded:
        keep = max(0, limit - (len(seen_posts) - len(fresh)))
        fresh = fresh[:keep]
    discovered = await _enqueue(pool, site.id, fresh, source)
    if exceeded:
        raise _history_limit(site, limit)
    return discovered


async def _enqueue(
    pool: AsyncConnectionPool, site_id: int, urls: list[str], source: str
) -> int:
    async with pool.connection() as conn, conn.transaction():
        return await db.enqueue_urls(conn, site_id=site_id, urls=urls, source=source)


async def _resolve_sitemap(fetcher: Fetcher, start_url: str, value: str) -> str | None:
    match value:
        case "none":
            return None
        case "auto":
            candidates = await sitemap.discover_sitemaps(fetcher.fetch_text, start_url)
            for candidate in candidates:
                if not same_site(candidate, start_url):
                    logger.warning("Ignoring cross-origin sitemap %s", candidate)
                    continue
                try:
                    xml = await fetcher.fetch_text(candidate)
                    pages, children = await asyncio.to_thread(
                        sitemap.parse_sitemap, xml
                    )
                except FetchError, ElementTree.ParseError:
                    continue
                if pages or children:
                    return candidate
            return None
        case _:
            candidate = normalize_url(urljoin(start_url, value))
            if not same_site(candidate, start_url):
                raise SiteError("Sitemap URL must use the site origin")
            return candidate


async def _resolve_feed(
    fetcher: Fetcher, start_url: str, value: str
) -> tuple[str, str | None]:
    if value == "none":
        raise SiteError("Every configured site must have an RSS or Atom feed")
    candidate = (
        await discover_feed_url(fetcher, start_url)
        if value == "auto"
        else urljoin(start_url, value)
    )
    if candidate is None:
        raise SiteError(f"No RSS/Atom feed found for {start_url}")
    try:
        response = await fetcher.fetch_response(candidate)
        parsed = await _parse_response(response)
    except (FetchError, FeedError) as exc:
        raise SiteError(f"Invalid feed {candidate}: {exc}") from exc
    return response.url, parsed.home_url


async def discover_feed_url(fetcher: Fetcher, site_url: str) -> str | None:
    response = await fetcher.fetch_response(site_url)
    try:
        await _parse_response(response)
    except FeedError:
        pass
    else:
        return response.url

    parts = urlsplit(response.url)
    params = FeedParameters(response.url, parts.hostname or "", response.url)
    feeds = determine_feed(response.text, params)
    return feeds[0] if feeds else None


async def _parse_response(response: FetchResponse) -> ParsedFeed:
    # feedparser is CPU-bound and feeds can be large; keep it off the event loop.
    return await asyncio.to_thread(
        parse_feed,
        response.body,
        url=response.url,
        headers=dict(response.headers),
    )


def _site_feed(parsed: ParsedFeed, site: Site) -> ParsedFeed:
    same = [url for url in parsed.urls if same_site(url, site.base_url)]
    dropped = len(parsed.urls) - len(same)
    if dropped:
        logger.warning("%s: ignored %d cross-origin feed URLs", site.base_url, dropped)
    if parsed.urls and not same:
        raise FeedError(f"Feed for {site.base_url} contains no same-site post URLs")
    # Fold scheme/host variants (http vs https, apex vs www) onto the configured
    # origin so the same post is one page identity across feed and sitemap.
    urls = list(
        dict.fromkeys(canonicalize_url(url, origin=site.base_url) for url in same)
    )
    feed_origin = normalize_origin(site.feed_url)
    history_url = (
        canonicalize_url(parsed.history_url, origin=feed_origin)
        if parsed.history_url is not None and same_site(parsed.history_url, feed_origin)
        else None
    )
    if parsed.history_url is not None and history_url is None:
        logger.warning("%s: ignored cross-origin feed history URL", site.base_url)
    return ParsedFeed(
        urls,
        parsed.home_url,
        history_url,
        parsed.is_wordpress,
        parsed.is_complete,
    )


def _with_query(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[key] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _history_limit(site: Site, limit: int) -> HistoryLimitError:
    return HistoryLimitError(
        f"Historical discovery exceeded {limit} posts for {site.base_url}; "
        "the site is only partially synchronized"
    )


def _header(response: FetchResponse, name: str) -> str | None:
    return response.headers.get(name.lower())


def _poll_offset(base_url: str, interval_seconds: int) -> int:
    digest = hashlib.sha256(base_url.encode()).digest()
    return int.from_bytes(digest[:8]) % interval_seconds


async def _renew_poll_lease(
    pool: AsyncConnectionPool, site_id: int, lease_token: UUID
) -> bool:
    async with pool.connection() as conn, conn.transaction():
        return await db.renew_poll_lease(conn, site_id=site_id, lease_token=lease_token)
