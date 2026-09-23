import asyncio
import hashlib
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import lxml.etree
import psycopg
from pebble import ProcessExpired
from psycopg_pool import AsyncConnectionPool
from scrapy import Request, signals
from scrapy.crawler import Crawler
from scrapy.exceptions import DownloadCancelledError, IgnoreRequest
from scrapy.http import Response, TextResponse
from scrapy.spiders import SitemapSpider
from scrapy.utils.request import RequestFingerprinter
from scrapy.utils.sitemap import Sitemap, sitemap_urls_from_robots
from trafilatura.feeds import FeedParameters, determine_feed
from twisted.python.failure import Failure

from semsearch.cli.crawl import store
from semsearch.cli.crawl.extraction import ExtractionPool
from semsearch.cli.crawl.pipeline import Article
from semsearch.cli.crawl.policy import DestinationRejected, OriginDeferred
from semsearch.cli.ingest.feed import FeedError, ParsedFeed, parse_feed
from semsearch.cli.models import Site
from semsearch.cli.url import (
    canonicalize_url,
    normalize_origin,
    normalize_url,
    same_site,
    try_normalize_url,
)
from semsearch.share.config import Settings


@dataclass
class SiteProgress:
    site: Site
    source_urls: set[str] = field(default_factory=set)
    article_urls: set[str] = field(default_factory=set)
    history_sets: set[frozenset[str]] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    feed_url: str | None = None
    wordpress: bool = False
    usable_sources: int = 0
    history_exhausted: bool = False


class PurposeFingerprinter(RequestFingerprinter):
    def fingerprint(self, request: Request) -> bytes:
        # A source can also be (incorrectly) listed as an article. Process that
        # article separately so it receives a durable rejection, not a dupe drop.
        purpose = f"{request.meta.get('site_id')}:{request.meta.get('kind')}".encode()
        return hashlib.sha256(super().fingerprint(request) + purpose).digest()


class BlogSpider(SitemapSpider):
    name = "blogs"

    def __init__(
        self,
        *,
        pool: AsyncConnectionPool,
        config: Settings,
        sites: list[Site],
        extractor: ExtractionPool,
        cooldowns: dict[str, datetime],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.pool = pool
        self.config = config
        self.progress = {site.id: SiteProgress(site) for site in sites}
        self.extractor = extractor
        self.cooldowns = cooldowns
        self.fatal_error: BaseException | None = None

    @classmethod
    def from_crawler(cls, crawler: Crawler, *args: Any, **kwargs: Any) -> BlogSpider:
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.fatal, signal=signals.spider_error)
        crawler.signals.connect(spider.fatal, signal=signals.item_error)
        return spider

    def fatal(self, failure: Failure, **kwargs: Any) -> None:
        if self.fatal_error is None:
            self.fatal_error = failure.value
            assert self.crawler.engine is not None
            asyncio.create_task(
                self.crawler.engine.close_spider_async(reason="application_error")
            )

    async def start(self) -> AsyncIterator[Any]:
        try:
            async for request in self.initial_requests():
                yield request
        except Exception as exc:
            self.fatal(Failure(exc))
            raise

    async def initial_requests(self) -> AsyncIterator[Any]:
        for state in self.progress.values():
            site = state.site
            for url in await store.pending(
                self.pool, site.id, self.config.crawl_article_limit
            ):
                request = self.article_request(state, url)
                if request:
                    yield request
            if site.feed_url == "auto":
                request = self.source_request(state, site.start_url, "home")
            elif site.feed_url != "none":
                state.feed_url = site.feed_url
                request = self.source_request(state, site.feed_url, "feed")
            else:
                state.history_exhausted = True
                request = None
            if request:
                yield request
            if site.sitemap_url == "auto":
                # robots discovery and conventional fallback both use Scrapy requests.
                for url, kind in (
                    (urljoin(site.base_url, "/robots.txt"), "robots"),
                    (
                        urljoin(site.start_url.rstrip("/") + "/", "sitemap.xml"),
                        "sitemap_optional",
                    ),
                    (urljoin(site.base_url, "/sitemap.xml"), "sitemap_optional"),
                    (urljoin(site.base_url, "/wp-sitemap.xml"), "sitemap_optional"),
                ):
                    request = self.source_request(state, url, kind)
                    if request:
                        yield request
            elif site.sitemap_url != "none":
                request = self.source_request(state, site.sitemap_url, "sitemap")
                if request:
                    yield request

    def source_request(
        self, state: SiteProgress, url: str, kind: str, *, page: int = 1
    ) -> Request | None:
        normalized = try_normalize_url(url)
        if normalized is None:
            state.errors.append("invalid_source_url")
            return None
        scope = (
            normalize_origin(state.feed_url or normalized)
            if kind in ("feed", "history", "wordpress")
            else state.site.base_url
        )
        if not same_site(normalized, scope):
            state.errors.append("cross_site_source")
            return None
        if normalized in state.source_urls:
            if kind in ("history", "wordpress"):
                state.history_exhausted = True
            return None
        if len(state.source_urls) >= self.config.crawl_source_limit:
            if "source_limit" not in state.errors:
                state.errors.append("source_limit")
            return None
        state.source_urls.add(normalized)
        callbacks = {
            "home": self.parse_home,
            "feed": self.parse_feed_response,
            "history": self.parse_feed_response,
            "wordpress": self.parse_feed_response,
            "robots": self.parse_robots,
            "sitemap": self.parse_sitemap_response,
            "sitemap_optional": self.parse_sitemap_response,
        }
        return Request(
            normalized,
            callback=callbacks[kind],
            errback=self.request_failed,
            meta={
                "site_id": state.site.id,
                "kind": kind,
                "scope": scope,
                "page": page,
                "download_maxsize": 10 * 1024 * 1024,
            },
        )

    def article_request(self, state: SiteProgress, url: str) -> Request | None:
        if (
            url in state.article_urls
            or len(state.article_urls) >= self.config.crawl_article_limit
        ):
            return None
        state.article_urls.add(url)
        return Request(
            url,
            callback=self.parse_article,
            errback=self.request_failed,
            meta={
                "site_id": state.site.id,
                "kind": "article",
                "article_url": url,
                "scope": state.site.base_url,
                "download_maxsize": 5 * 1024 * 1024,
            },
        )

    async def discovered(
        self, state: SiteProgress, urls: list[str], source: str
    ) -> AsyncIterator[Request]:
        candidates = list(
            dict.fromkeys(
                canonicalize_url(url, origin=state.site.base_url)
                for url in urls
                if same_site(url, state.site.base_url)
            )
        )
        # Commit *all* candidates, even if only some fit this batch's budget.
        pending = await store.discover(self.pool, state.site.id, candidates, source)
        for url in pending:
            request = self.article_request(state, url)
            if request:
                yield request

    async def parsed_feed(self, response: Response) -> ParsedFeed:
        headers = {
            key.decode("latin1").lower(): (response.headers[key] or b"").decode(
                "latin1"
            )
            for key in response.headers
        }
        # The function is CPU-bound; share the bounded, killable parser pool.
        try:
            return await self.extractor.call(
                parse_feed_document, response.body, response.url, headers
            )
        except (TimeoutError, ProcessExpired) as exc:
            raise FeedError("feed_parse_limit") from exc

    async def parse_home(self, response: Response) -> AsyncIterator[Any]:
        state = self.progress[response.meta["site_id"]]
        try:
            parsed = await self.parsed_feed(response)
        except FeedError:
            if not isinstance(response, TextResponse):
                state.errors.append("homepage_not_text")
                return
            params = FeedParameters(
                response.url, urlsplit(response.url).hostname or "", response.url
            )
            feeds = determine_feed(response.text, params)
            if not feeds:
                # Sitemap-only sites are supported even with feed=auto.
                state.history_exhausted = True
                return
            state.feed_url = normalize_url(urljoin(response.url, feeds[0]))
            request = self.source_request(state, state.feed_url, "feed")
            if request:
                yield request
        else:
            state.feed_url = response.url
            async for request in self.feed_requests(state, response, parsed):
                yield request

    async def parse_feed_response(self, response: Response) -> AsyncIterator[Any]:
        state = self.progress[response.meta["site_id"]]
        try:
            parsed = await self.parsed_feed(response)
        except FeedError as exc:
            state.errors.append(str(exc))
            if response.meta["kind"] == "history":
                request = self.wordpress_request(state, 2)
                if request:
                    yield request
            return
        async for request in self.feed_requests(state, response, parsed):
            yield request

    async def feed_requests(
        self, state: SiteProgress, response: Response, parsed: ParsedFeed
    ) -> AsyncIterator[Request]:
        state.usable_sources += 1
        state.wordpress = state.wordpress or parsed.is_wordpress
        async for request in self.discovered(state, parsed.urls, "feed"):
            yield request
        if state.site.history_complete:
            return
        signature = frozenset(parsed.urls)
        if parsed.is_complete or not signature or signature in state.history_sets:
            state.history_exhausted = True
            return
        state.history_sets.add(signature)
        kind = response.meta["kind"]
        if kind == "wordpress":
            request = self.wordpress_request(state, response.meta["page"] + 1)
        elif parsed.history_url:
            request = self.source_request(state, parsed.history_url, "history")
        elif kind == "history":
            state.history_exhausted = True
            request = None
        else:
            request = self.wordpress_request(state, 2)
            if request is None:
                state.history_exhausted = True
        if request:
            yield request

    def wordpress_request(self, state: SiteProgress, page: int) -> Request | None:
        if not state.wordpress or state.feed_url is None:
            return None
        parts = urlsplit(state.feed_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["paged"] = str(page)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
        return self.source_request(state, url, "wordpress", page=page)

    def parse_robots(self, response: Response) -> Any:
        state = self.progress[response.meta["site_id"]]
        for url in sitemap_urls_from_robots(response.body, base_url=response.url):
            request = self.source_request(state, url, "sitemap")
            if request:
                yield request

    async def parse_sitemap_response(
        self, response: Response
    ) -> AsyncIterator[Request]:
        state = self.progress[response.meta["site_id"]]
        try:
            body = self._get_sitemap_body(response)
        except (OSError, EOFError, zlib.error) as exc:
            state.errors.append(f"invalid_sitemap_compression: {exc}")
            return
        if not body:
            if response.meta["kind"] != "sitemap_optional":
                state.errors.append("invalid_sitemap")
            return
        try:
            sitemap = Sitemap(body)
            entries = list(sitemap)
        except (ValueError, TypeError, StopIteration, lxml.etree.XMLSyntaxError) as exc:
            state.errors.append(f"invalid_sitemap: {exc}")
            return
        if sitemap.type not in ("sitemapindex", "urlset"):
            if response.meta["kind"] != "sitemap_optional":
                state.errors.append("invalid_sitemap")
            return
        state.usable_sources += 1
        urls = [
            url
            for entry in entries
            if (url := try_normalize_url(entry["loc"])) is not None
        ]
        if sitemap.type == "sitemapindex":
            for url in urls:
                request = self.source_request(state, url, "sitemap")
                if request:
                    yield request
        else:
            async for request in self.discovered(state, urls, "sitemap"):
                yield request

    def parse_article(self, response: Response) -> Article:
        state = self.progress[response.meta["site_id"]]
        content_type = (response.headers.get(b"Content-Type") or b"").decode("latin1")
        return Article(
            state.site,
            response.meta["article_url"],
            response.url,
            response.body,
            content_type,
        )

    async def request_failed(self, failure: Failure) -> AsyncIterator[Request]:
        request = getattr(failure, "request", None)
        if not isinstance(request, Request):
            raise TypeError("request failure has no Scrapy request")
        state = self.progress[request.meta["site_id"]]
        kind = request.meta["kind"]
        exc = failure.value
        if isinstance(exc, psycopg.Error):
            raise exc
        if isinstance(exc, OriginDeferred):
            if kind != "article":
                state.errors.append("origin_cooldown")
            return
        response = getattr(exc, "response", None)
        status = response.status if response is not None else None
        if kind == "article":
            terminal = (
                isinstance(
                    exc, (DestinationRejected, IgnoreRequest, DownloadCancelledError)
                )
                and status is None
            )
            permanent_http = (
                status is not None
                and 400 <= status < 500
                and status not in (408, 425, 429)
            )
            outcome = (
                "rejected" if terminal else "failed" if permanent_http else "pending"
            )
            await store.outcome(
                self.pool,
                request.meta["article_url"],
                outcome,
                str(exc),
                self.config.crawl_interval_seconds,
            )
            return
        if kind == "wordpress" and status in (404, 410):
            state.history_exhausted = True
            return
        if kind in ("robots", "sitemap_optional") and status in (404, 410):
            return
        state.errors.append(f"{kind}: {str(exc)[:300]}")
        if kind == "history":
            fallback = self.wordpress_request(state, 2)
            if fallback:
                yield fallback

    async def finish(self) -> None:
        for state in self.progress.values():
            if not state.usable_sources:
                state.errors.append("no_usable_feed_or_sitemap")
            complete = state.site.history_complete or (
                state.history_exhausted and not state.errors
            )
            await store.finish_site(
                self.pool,
                state.site,
                history_complete=complete,
                error="; ".join(dict.fromkeys(state.errors))[:2000] or None,
            )


def parse_feed_document(body: bytes, url: str, headers: dict[str, str]) -> ParsedFeed:
    return parse_feed(body, url=url, headers=headers)
