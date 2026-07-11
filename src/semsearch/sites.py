from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import json
from typing import cast
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from psycopg_pool import AsyncConnectionPool
from trafilatura.feeds import FeedParameters, determine_feed, is_potential_feed

from semsearch import db
from semsearch.ingest import sitemap
from semsearch.ingest.sitemap import FetchText
from semsearch.ingest.fetch import Fetcher, FetchResponse, FetchError
from semsearch.ingest.models import IndexOutcome
from semsearch.ingest.outcomes import collect_index_outcomes
from semsearch.models import Site
from semsearch.url import normalize_origin, normalize_url
from semsearch.util import map_concurrently


class SiteError(RuntimeError):
    pass


type ProgressCallback = Callable[[IndexOutcome], None]
type IndexSitemap = Callable[
    [str, bool, ProgressCallback | None], Awaitable[list[IndexOutcome]]
]
type IndexUrl = Callable[[str, bool], Awaitable[IndexOutcome]]


@dataclass(slots=True)
class PollOutcome:
    site: Site
    outcomes: list[IndexOutcome]
    not_modified: bool = False


class SiteService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        fetcher: Fetcher,
    ) -> None:
        self.pool = pool
        self.fetcher = fetcher

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
            row = await db.upsert_site_config(
                conn,
                base_url=base_url,
                sitemap_url=resolved_sitemap,
                feed_url=resolved_feed,
            )
        return row

    async def list_sites(self) -> list[Site]:
        async with self.pool.connection() as conn:
            return await db.list_site_configs(conn)

    async def get_site(self, site: str) -> Site:
        try:
            base_url = normalize_origin(site)
        except ValueError as exc:
            raise SiteError(str(exc)) from exc
        async with self.pool.connection() as conn:
            row = await db.find_site_config(conn, base_url=base_url)
        if row is None:
            raise SiteError(f"Unknown site: {base_url}")
        return row

    async def index_site(
        self,
        site: str,
        index_sitemap: IndexSitemap,
        *,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> list[IndexOutcome]:
        record = await self.get_site(site)
        if record.sitemap_url is None:
            raise SiteError(f"No sitemap configured for {record.base_url}")
        outcomes = await index_sitemap(record.sitemap_url, force, on_progress)
        async with self.pool.connection() as conn, conn.transaction():
            await db.mark_site_indexed(conn, site_id=record.id)
        return outcomes

    async def poll_site(
        self,
        site: str,
        index_url: IndexUrl,
        *,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
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
            return await index_url(url, force)

        outcomes = await collect_index_outcomes(
            urls,
            index_one,
            on_progress=on_progress,
        )
        await self._mark_polled(record.id, feed.headers)
        return PollOutcome(record, outcomes)

    async def poll_all(
        self,
        index_url: IndexUrl,
        *,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
        concurrency: int = 4,
    ) -> list[PollOutcome]:
        feed_sites = [site for site in await self.list_sites() if site.feed_url]

        async def poll_one(site: Site) -> PollOutcome:
            return await self.poll_site(
                site.base_url,
                index_url,
                force=force,
                on_progress=on_progress,
            )

        return await map_concurrently(feed_sites, limit=concurrency, func=poll_one)

    async def _resolve_sitemap(self, start_url: str, value: str) -> str | None:
        match value:
            case "none":
                return None
            case "auto":
                candidates = await sitemap.discover_sitemaps(
                    self.fetcher.fetch_text, start_url
                )
                for candidate in candidates:
                    if await sitemap.collect_page_urls(
                        self.fetcher.fetch_text, candidate, warn=False
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
                return await discover_feed_url(self.fetcher.fetch_text, start_url)
            case _:
                return urljoin(start_url, value)

    async def _fetch_feed(self, site: Site, *, use_cache: bool) -> FetchResponse | None:
        feed_url = cast(str, site.feed_url)
        headers: dict[str, str] = {}
        if use_cache and site.feed_etag:
            headers["If-None-Match"] = site.feed_etag
        if use_cache and site.feed_last_modified:
            headers["If-Modified-Since"] = site.feed_last_modified
        return await self.fetcher.fetch_response(
            feed_url,
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
            await db.mark_site_polled(
                conn,
                site_id=site_id,
                feed_etag=etag,
                feed_last_modified=last_modified,
            )


async def discover_feed_url(fetch_text: FetchText, site_url: str) -> str | None:
    start_url = normalize_url(site_url)
    parts = urlsplit(start_url)
    try:
        html = await fetch_text(start_url)
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
