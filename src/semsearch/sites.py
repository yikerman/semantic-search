from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from psycopg_pool import AsyncConnectionPool
from trafilatura.feeds import FeedParameters, determine_feed, is_potential_feed

from semsearch.config import Settings
from semsearch.db import check_index_meta
from semsearch.ingest import sitemap
from semsearch.ingest.sitemap import TextFetcher
from semsearch.ingest.fetch import Fetcher, FetchResponse, FetchError
from semsearch.ingest.models import IndexOutcome
from semsearch.ingest.outcomes import collect_index_outcomes
from semsearch.url import normalize_origin, normalize_url
from semsearch.util import map_concurrently


class SiteError(RuntimeError):
    pass


class SitemapIndexer(Protocol):
    async def index_sitemap(
        self,
        url: str,
        *,
        include: str | None = None,
        exclude: str | None = None,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
    ) -> list[IndexOutcome]: ...


class UrlIndexer(Protocol):
    async def index_url(self, url: str, *, force: bool = False) -> IndexOutcome: ...


@dataclass(slots=True)
class Site:
    id: int
    base_url: str
    sitemap_url: str | None
    feed_url: str | None
    last_indexed_at: datetime | None
    last_polled_at: datetime | None
    feed_etag: str | None
    feed_last_modified: str | None


@dataclass(slots=True)
class PollOutcome:
    site: Site
    outcomes: list[IndexOutcome]
    not_modified: bool = False


class SiteService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        settings: Settings,
        *,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.fetcher = fetcher or Fetcher(
            user_agent=settings.user_agent,
            timeout=settings.fetch_timeout_seconds,
            delay_seconds=settings.fetch_delay_seconds,
            impersonate=settings.fetch_impersonate,
        )

    async def add_site(
        self,
        url: str,
        *,
        sitemap_url: str = "auto",
        feed_url: str = "auto",
    ) -> Site:
        try:
            start_url = normalize_url(url)
            base_url = normalize_origin(url)
        except ValueError as exc:
            raise SiteError(str(exc)) from exc

        resolved_sitemap = await self._resolve_sitemap(start_url, sitemap_url)
        resolved_feed = await self._resolve_feed(start_url, feed_url)
        async with self.pool.connection() as conn, conn.transaction():
            await check_index_meta(conn, self.settings)
            cur = await conn.execute(
                """
                INSERT INTO sites (base_url, sitemap_url, feed_url)
                VALUES (%s, %s, %s)
                ON CONFLICT (base_url) DO UPDATE SET
                    sitemap_url = EXCLUDED.sitemap_url,
                    feed_url = EXCLUDED.feed_url
                RETURNING id, base_url, sitemap_url, feed_url, last_indexed_at,
                          last_polled_at, feed_etag, feed_last_modified
                """,
                (base_url, resolved_sitemap, resolved_feed),
            )
            row = await cur.fetchone()
        assert row is not None
        return _site_from_row(row)

    async def list_sites(self) -> list[Site]:
        async with self.pool.connection() as conn:
            await check_index_meta(conn, self.settings)
            cur = await conn.execute(
                """
                SELECT id, base_url, sitemap_url, feed_url, last_indexed_at,
                       last_polled_at, feed_etag, feed_last_modified
                FROM sites
                ORDER BY base_url
                """
            )
            rows = await cur.fetchall()
        return [_site_from_row(row) for row in rows]

    async def get_site(self, site: str) -> Site:
        try:
            base_url = normalize_origin(site)
        except ValueError as exc:
            raise SiteError(str(exc)) from exc
        async with self.pool.connection() as conn:
            await check_index_meta(conn, self.settings)
            cur = await conn.execute(
                """
                SELECT id, base_url, sitemap_url, feed_url, last_indexed_at,
                       last_polled_at, feed_etag, feed_last_modified
                FROM sites
                WHERE base_url = %s
                """,
                (base_url,),
            )
            row = await cur.fetchone()
        if row is None:
            raise SiteError(f"Unknown site: {base_url}")
        return _site_from_row(row)

    async def index_site(
        self,
        site: str,
        indexer: SitemapIndexer,
        *,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
    ) -> list[IndexOutcome]:
        record = await self.get_site(site)
        if record.sitemap_url is None:
            raise SiteError(f"No sitemap configured for {record.base_url}")
        outcomes = await indexer.index_sitemap(
            record.sitemap_url,
            force=force,
            on_progress=on_progress,
        )
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE sites SET last_indexed_at = now() WHERE id = %s",
                (record.id,),
            )
        return outcomes

    async def poll_site(
        self,
        site: str,
        indexer: UrlIndexer,
        *,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
    ) -> PollOutcome:
        record = await self.get_site(site)
        if record.feed_url is None:
            raise SiteError(f"No feed configured for {record.base_url}")

        feed = await self._fetch_feed(record, use_cache=not force)
        if feed is None:
            await self._mark_polled(record.id, None)
            return PollOutcome(record, [], not_modified=True)

        urls = _dedupe_urls(
            [
                canonicalize_site_url(url, record.base_url)
                for url in extract_feed_urls(feed.text, record.feed_url)
            ]
        )

        async def index_one(url: str) -> IndexOutcome:
            return await indexer.index_url(url, force=force)

        outcomes = await collect_index_outcomes(
            urls,
            index_one,
            on_progress=on_progress,
        )
        await self._mark_polled(record.id, feed.headers)
        return PollOutcome(record, outcomes)

    async def poll_all(
        self,
        indexer: UrlIndexer,
        *,
        force: bool = False,
        on_progress: Callable[[IndexOutcome], None] | None = None,
        concurrency: int = 4,
    ) -> list[PollOutcome]:
        feed_sites = [site for site in await self.list_sites() if site.feed_url]

        async def poll_one(site: Site) -> PollOutcome:
            return await self.poll_site(
                site.base_url,
                indexer,
                force=force,
                on_progress=on_progress,
            )

        return await map_concurrently(feed_sites, limit=concurrency, func=poll_one)

    async def aclose(self) -> None:
        await self.fetcher.aclose()

    async def _resolve_sitemap(self, start_url: str, value: str) -> str | None:
        match value:
            case "none":
                return None
            case "auto":
                candidates = await sitemap.discover_sitemaps(self.fetcher, start_url)
                for candidate in candidates:
                    if await sitemap.collect_page_urls(
                        self.fetcher, candidate, warn=False
                    ):
                        return candidate
                return None
            case _:
                return urljoin(start_url, value)

    async def _resolve_feed(self, start_url: str, value: str) -> str | None:
        match value:
            case "none":
                return None
            case "auto":
                return await discover_feed_url(self.fetcher, start_url)
            case _:
                return urljoin(start_url, value)

    async def _fetch_feed(self, site: Site, *, use_cache: bool) -> FetchResponse | None:
        assert site.feed_url is not None
        headers: dict[str, str] = {}
        if use_cache and site.feed_etag:
            headers["If-None-Match"] = site.feed_etag
        if use_cache and site.feed_last_modified:
            headers["If-Modified-Since"] = site.feed_last_modified
        return await self.fetcher.fetch_response(
            site.feed_url,
            headers=headers or None,
            allow_not_modified=True,
        )

    async def _mark_polled(
        self, site_id: int, headers: Mapping[str, str] | None
    ) -> None:
        etag = _header(headers, "etag") if headers is not None else None
        last_modified = (
            _header(headers, "last-modified") if headers is not None else None
        )
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                """
                UPDATE sites
                SET last_polled_at = now(),
                    feed_etag = COALESCE(%s, feed_etag),
                    feed_last_modified = COALESCE(%s, feed_last_modified)
                WHERE id = %s
                """,
                (etag, last_modified, site_id),
            )


async def discover_feed_url(fetcher: TextFetcher, site_url: str) -> str | None:
    start_url = normalize_url(site_url)
    parts = urlsplit(start_url)
    try:
        html = await fetcher.fetch_text(start_url)
    except FetchError:
        return None

    if is_potential_feed(html):
        return start_url

    params = FeedParameters(start_url, parts.hostname or "", start_url)
    feeds = determine_feed(html, params)
    return feeds[0] if feeds else None


def extract_feed_urls(feed_text: str, feed_url: str) -> list[str]:
    text = feed_text.strip()
    if text.startswith("{"):
        return _extract_json_feed_urls(text, feed_url)
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return []
    urls: list[str] = []
    for item in root.iter():
        match _local_name(item.tag):
            case "item":
                urls.extend(_rss_item_urls(item, feed_url))
            case "entry":
                urls.extend(_atom_entry_urls(item, feed_url))
    return _dedupe_urls(urls)


def canonicalize_site_url(url: str, base_url: str) -> str:
    parts = urlsplit(url)
    base = urlsplit(base_url)
    if parts.hostname != base.hostname:
        return url
    return urlunsplit(
        (
            base.scheme,
            base.netloc,
            parts.path or "/",
            parts.query,
            "",
        )
    )


def _extract_json_feed_urls(feed_text: str, feed_url: str) -> list[str]:
    try:
        data = json.loads(feed_text)
    except json.JSONDecodeError:
        return []
    urls = [
        urljoin(feed_url, url)
        for item in data.get("items", [])
        for url in (item.get("url"), item.get("external_url"))
        if isinstance(url, str)
    ]
    return _dedupe_urls(urls)


def _rss_item_urls(item: ElementTree.Element, feed_url: str) -> list[str]:
    for child in item:
        if _local_name(child.tag) == "link" and child.text:
            return [urljoin(feed_url, child.text.strip())]
    return []


def _atom_entry_urls(entry: ElementTree.Element, feed_url: str) -> list[str]:
    links = [
        link for link in entry if _local_name(link.tag) == "link" and link.get("href")
    ]
    preferred = [
        link
        for link in links
        if link.get("rel") in (None, "", "alternate")
        and link.get("type") not in ("application/atom+xml", "application/rss+xml")
    ]
    return [urljoin(feed_url, link.get("href", "")) for link in preferred[:1]]


def _dedupe_urls(urls: list[str]) -> list[str]:
    return list(
        dict.fromkeys(url for url in urls if url.startswith(("http://", "https://")))
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return headers.get(name) or headers.get(name.title()) or headers.get(name.upper())


def _site_from_row(row) -> Site:
    (
        site_id,
        base_url,
        sitemap_url,
        feed_url,
        last_indexed_at,
        last_polled_at,
        feed_etag,
        feed_last_modified,
    ) = row
    return Site(
        id=site_id,
        base_url=base_url,
        sitemap_url=sitemap_url,
        feed_url=feed_url,
        last_indexed_at=last_indexed_at,
        last_polled_at=last_polled_at,
        feed_etag=feed_etag,
        feed_last_modified=feed_last_modified,
    )
