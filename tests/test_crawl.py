import asyncio
import gzip
import socket
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest
from scrapy import Request
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Response, XmlResponse
from scrapy.settings import Settings as ScrapySettings
from twisted.python.failure import Failure

from semsearch.cli.crawl import store
from semsearch.cli.crawl.extraction import ArticleRejected, extract_article
from semsearch.cli.crawl.pipeline import Article, ArticlePipeline
from semsearch.cli.crawl.policy import (
    CrawlPolicy,
    DestinationRejected,
    OriginDeferred,
    PublicResolver,
    public_address,
    retry_after,
)
from semsearch.cli.crawl.settings import scrapy_settings
from semsearch.cli.crawl.spider import BlogSpider, PurposeFingerprinter
from semsearch.cli.models import Site
from semsearch.share.config import Settings


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "169.254.169.254"])
async def test_connection_resolver_rejects_private_dns_answers(monkeypatch, address):
    async def lookup(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", lookup)
    resolver = PublicResolver(cast(Any, object()), 0, 1)
    with pytest.raises(DestinationRejected):
        await resolver.getHostByName("dns-policy.example").asFuture(loop)


async def test_connection_resolver_returns_checked_public_address(monkeypatch):
    async def lookup(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", lookup)
    resolver = PublicResolver(cast(Any, object()), 0, 1)
    assert (
        await resolver.getHostByName("dns-policy.example").asFuture(loop) == "8.8.8.8"
    )


class InlineExtractor:
    async def call(self, function, *args):
        return function(*args)


@pytest.fixture
def site():
    return Site(
        id=1,
        base_url="https://blog.example",
        start_url="https://blog.example/",
        feed_url="https://blog.example/feed",
        sitemap_url="none",
    )


@pytest.fixture
def spider(site):
    crawler = Crawler(BlogSpider, ScrapySettings(scrapy_settings(Settings())))
    result = BlogSpider.from_crawler(
        crawler,
        pool=cast(Any, object()),
        config=Settings(),
        sites=[site],
        extractor=cast(Any, InlineExtractor()),
        cooldowns={},
    )
    crawler.spider = result
    return result


class Discovery:
    def __init__(self):
        self.rows = {}

    async def discover(self, pool, site_id, urls, source):
        for url in urls:
            self.rows.setdefault(url, "pending")
        return [url for url in urls if self.rows[url] == "pending"]

    async def pending(self, pool, site_id, limit):
        return [url for url, state in self.rows.items() if state == "pending"][:limit]


async def test_discovery_commits_all_urls_before_scheduling_and_replays(
    monkeypatch, spider
):
    data = Discovery()
    monkeypatch.setattr(store, "discover", data.discover)
    monkeypatch.setattr(store, "pending", data.pending)
    spider.config.crawl_article_limit = 1
    state = spider.progress[1]
    urls = ["https://blog.example/a", "https://blog.example/b"]
    stream = spider.discovered(state, urls, "feed")
    first = await anext(stream)
    assert first.url == urls[0]
    assert data.rows == dict.fromkeys(urls, "pending")
    await stream.aclose()  # interrupted after persistence, before scheduling b
    state.article_urls.clear()
    replay = [request async for request in spider.start()]
    assert replay[0].url == urls[0]
    data.rows[urls[0]] = "stored"
    state.article_urls.clear()
    replay = [request async for request in spider.start()]
    assert replay[0].url == urls[1]


async def test_discovery_database_failure_yields_nothing(monkeypatch, spider):
    async def broken(*args):
        raise psycopg.OperationalError("database offline")

    monkeypatch.setattr(store, "discover", broken)
    with pytest.raises(psycopg.OperationalError):
        await anext(
            spider.discovered(spider.progress[1], ["https://blog.example/a"], "feed")
        )


@pytest.mark.parametrize("body", [b"\x1f\x8b\x08not-gzip", b"not XML"])
async def test_malformed_sitemap_records_source_error(spider, body):
    state = spider.progress[1]
    request = spider.source_request(state, "https://blog.example/map.xml.gz", "sitemap")
    response = Response(request.url, request=request, body=body)
    assert [r async for r in spider.parse_sitemap_response(response)] == []
    assert state.errors[0].startswith("invalid_sitemap")


async def test_feed_history_follows_archive_and_stops_cycle(monkeypatch, spider):
    data = Discovery()
    monkeypatch.setattr(store, "discover", data.discover)
    state = spider.progress[1]
    state.feed_url = state.site.feed_url
    request = spider.source_request(state, state.feed_url, "feed")
    body = b"""<feed xmlns="http://www.w3.org/2005/Atom"><title>Blog</title>
    <link rel="prev-archive" href="/old"/>
    <entry><link href="/one"/><title>One</title></entry></feed>"""
    response = XmlResponse(request.url, body=body, request=request)
    requests = [r async for r in spider.parse_feed_response(response)]
    assert [r.url for r in requests] == [
        "https://blog.example/one",
        "https://blog.example/old",
    ]
    archived = response.replace(url=requests[-1].url, request=requests[-1])
    assert [r async for r in spider.parse_feed_response(archived)] == []
    assert state.history_exhausted


async def test_wordpress_pagination_terminates_on_404(monkeypatch, spider):
    data = Discovery()
    monkeypatch.setattr(store, "discover", data.discover)
    state = spider.progress[1]
    state.feed_url = state.site.feed_url
    req = spider.source_request(state, state.feed_url, "feed")
    body = b'<rss version="2.0"><channel><generator>WordPress</generator><item><link>https://blog.example/a</link></item></channel></rss>'
    requests = [
        r
        async for r in spider.parse_feed_response(
            XmlResponse(req.url, request=req, body=body)
        )
    ]
    assert requests[-1].url.endswith("?paged=2")
    from scrapy.spidermiddlewares.httperror import HttpError

    error = Failure(HttpError(Response(requests[-1].url, status=404), "404"))
    cast(Any, error).request = requests[-1]
    assert [r async for r in spider.request_failed(error)] == []
    assert state.history_exhausted


async def test_nested_gzip_sitemap_and_source_budget(monkeypatch, spider):
    data = Discovery()
    monkeypatch.setattr(store, "discover", data.discover)
    state = spider.progress[1]
    spider.config.crawl_source_limit = 2
    req = spider.source_request(state, "https://blog.example/sitemap.xml", "sitemap")
    body = b"<sitemapindex><sitemap><loc>https://blog.example/posts.xml.gz</loc></sitemap><sitemap><loc>https://blog.example/other.xml</loc></sitemap></sitemapindex>"
    requests = [
        r
        async for r in spider.parse_sitemap_response(
            XmlResponse(req.url, request=req, body=body)
        )
    ]
    assert len(requests) == 1
    assert "source_limit" in state.errors
    body = gzip.compress(
        b"<urlset><url><loc>https://blog.example/a</loc></url></urlset>"
    )
    articles = [
        r
        async for r in spider.parse_sitemap_response(
            Response(requests[0].url, request=requests[0], body=body)
        )
    ]
    assert articles[0].url == "https://blog.example/a"


@pytest.mark.parametrize(
    "body,mime",
    [
        (b"<rss><channel><title>Hello</title></channel></rss>", "text/html"),
        (b'<feed xmlns="http://www.w3.org/2005/Atom"/>', "text/html"),
        (b"<urlset><url><loc>https://example.com</loc></url></urlset>", "text/html"),
        (b'{"items":[]}', "text/html"),
        (b"<html><body>Hello</body></html>", "application/json"),
    ],
)
def test_non_articles_are_rejected_even_with_wrong_mime(body, mime):
    with pytest.raises(ArticleRejected):
        extract_article(body, "https://blog.example/a", mime)


def test_xhtml_and_old_encoding_reach_extractor_without_utf8_corruption(monkeypatch):
    from semsearch.cli.crawl import extraction
    from semsearch.cli.ingest.extract import ExtractedPage

    bodies = []

    def extract(body, url):
        bodies.append(body)
        return ExtractedPage("Title", "body " * 100, None, "en")

    monkeypatch.setattr(extraction, "extract_page", extract)
    body = b'<?xml version="1.0" encoding="ISO-8859-1"?><html xmlns="http://www.w3.org/1999/xhtml"><body>caf\xe9</body></html>'
    assert (
        extract_article(body, "https://blog.example/a", "application/xhtml+xml").title
        == "Title"
    )
    assert bodies == [body]


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.1.1", "::1", "169.254.169.254"])
def test_non_public_dns_results_are_rejected(address):
    with pytest.raises(DestinationRejected):
        public_address(address)
    assert public_address("8.8.8.8") == "8.8.8.8"


def test_redirect_scope_is_checked_before_download(spider):
    policy = CrawlPolicy(spider.crawler)
    req = Request("https://elsewhere.example/a", meta={"scope": "https://blog.example"})
    with pytest.raises(DestinationRejected):
        policy.process_request(req)
    with pytest.raises(DestinationRejected):
        policy.process_request(Request("http://127.0.0.1/a"))


async def test_retry_after_persists_cooldown_and_defers_other_requests(
    monkeypatch, spider
):
    saved = []

    async def save(pool, origin, until):
        saved.append((origin, until))

    monkeypatch.setattr(store, "save_cooldown", save)
    req = Request("https://blog.example/a")
    policy = CrawlPolicy(spider.crawler)
    with pytest.raises(OriginDeferred):
        await policy.process_response(
            req, Response(req.url, status=429, headers={"Retry-After": "120"})
        )
    assert saved[0][0] == "https://blog.example"
    with pytest.raises(OriginDeferred):
        policy.process_request(Request("https://blog.example/b"))
    policy.process_request(Request("https://other.example/a"))


def test_retry_after_http_date_and_invalid_input():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert retry_after(b"120", now) == now + timedelta(seconds=120)
    assert retry_after(b"Thu, 01 Jan 2026 00:02:00 GMT", now) == now + timedelta(
        seconds=120
    )
    assert retry_after(b"garbage", now) is None


async def test_article_retry_classification(monkeypatch, spider):
    outcomes = []

    async def save(pool, url, status, reason, interval):
        outcomes.append(status)

    monkeypatch.setattr(store, "outcome", save)
    from scrapy.spidermiddlewares.httperror import HttpError

    req = spider.article_request(spider.progress[1], "https://blog.example/a")
    for exc in (
        HttpError(Response(req.url, status=404), "404"),
        HttpError(Response(req.url, status=503), "503"),
        IgnoreRequest("Forbidden by robots.txt"),
    ):
        failure = Failure(exc)
        cast(Any, failure).request = req
        assert [r async for r in spider.request_failed(failure)] == []
    assert outcomes == ["failed", "pending", "rejected"]


async def test_pipeline_stores_extraction_without_embedding(monkeypatch, spider):
    from semsearch.cli.ingest.extract import ExtractedPage

    page = ExtractedPage("Title", "Body", None, "en")

    class Extract:
        async def call(self, *args):
            return page

    spider.extractor = cast(Any, Extract())
    saved = []

    async def save(*args):
        saved.append(args)

    monkeypatch.setattr(store, "store_page", save)
    crawler = SimpleNamespace(
        spider=spider, stats=SimpleNamespace(inc_value=lambda *args: None)
    )
    item = Article(
        spider.progress[1].site,
        "https://blog.example/a",
        "https://blog.example/a",
        b"html",
        "text/html",
    )
    assert await ArticlePipeline(cast(Any, crawler)).process_item(item) is item
    assert saved[0][-1] is page


def test_fingerprint_distinguishes_sources_and_articles():
    printer = PurposeFingerprinter()
    url = "https://blog.example/feed"
    assert printer.fingerprint(
        Request(url, meta={"kind": "feed"})
    ) != printer.fingerprint(Request(url, meta={"kind": "article"}))
