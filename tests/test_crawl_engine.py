"""Exercise the real Scrapy engine/middleware with an entirely fake HTTP transport."""

import asyncio
import gzip
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import psycopg
import pytest
from scrapy.http import HtmlResponse, Response, TextResponse

from semsearch.cli.crawl import extraction, run, store
from semsearch.cli.ingest.extract import ExtractedPage
from semsearch.cli.models import Site
from semsearch.share.config import Settings


class FixtureHTTP:
    lazy = True
    calls: ClassVar[Counter[str]] = Counter()
    crawlers: ClassVar[list[Any]] = []

    @classmethod
    def from_crawler(cls, crawler):
        cls.crawlers.append(crawler)
        return cls()

    async def download_request(self, request):
        self.calls[request.url] += 1
        path = request.url.partition("blog.example")[2]
        if path == "/robots.txt":
            return TextResponse(
                request.url,
                request=request,
                body=b"User-agent: *\nDisallow: /blocked\n",
                encoding="utf-8",
            )
        if path == "/feed":
            urls = ["/good", "/blocked", "/flaky", "/bomb", "/feed"]
            body = (
                '<rss version="2.0"><channel>'
                + "".join(
                    f"<item><link>https://blog.example{url}</link></item>"
                    for url in urls
                )
                + "</channel></rss>"
            ).encode()
            return TextResponse(
                request.url,
                request=request,
                headers={"Content-Type": "application/rss+xml"},
                body=body,
                encoding="utf-8",
            )
        if path == "/bomb":
            return HtmlResponse(
                request.url,
                request=request,
                headers={"Content-Type": "text/html", "Content-Encoding": "gzip"},
                body=gzip.compress(b"x" * (6 * 1024 * 1024)),
            )
        if path == "/flaky" and self.calls[request.url] == 1:
            return Response(request.url, request=request, status=503)
        return HtmlResponse(
            request.url,
            request=request,
            headers={"Content-Type": "text/html"},
            body=b"<html><body><article>Good article</article></body></html>",
        )

    async def close(self):
        pass


class InlinePool:
    closed: ClassVar[int] = 0

    def __init__(self, *args):
        pass

    async def call(self, function, *args):
        return function(*args)

    async def close(self):
        type(self).closed += 1


@asynccontextmanager
async def unlocked(*args):
    yield


async def test_real_engine_crawl_replay_and_fatal_database_error(monkeypatch):
    rows = {}
    pages = {}
    finishes = []
    site = Site(
        id=1,
        base_url="https://blog.example",
        start_url="https://blog.example/",
        feed_url="https://blog.example/feed",
        sitemap_url="none",
    )

    async def sites(*args):
        return [site]

    async def cooldowns(*args):
        return {}

    async def discover(pool, site_id, urls, source):
        for url in urls:
            rows.setdefault(url, "pending")
        return [url for url in urls if rows[url] == "pending"]

    async def pending(pool, site_id, limit):
        return [url for url, state in rows.items() if state == "pending"][:limit]

    async def save(pool, site, url, page):
        pages[url] = page
        rows[url] = "stored"

    async def outcome(pool, url, status, *args):
        rows[url] = status

    async def finish(*args, **kwargs):
        finishes.append(kwargs)

    monkeypatch.setattr(run, "command_lock", unlocked)
    monkeypatch.setattr(run, "list_sites", sites)
    monkeypatch.setattr(run, "ExtractionPool", InlinePool)
    monkeypatch.setattr(store, "load_cooldowns", cooldowns)
    monkeypatch.setattr(store, "discover", discover)
    monkeypatch.setattr(store, "pending", pending)
    monkeypatch.setattr(store, "store_page", save)
    monkeypatch.setattr(store, "outcome", outcome)
    monkeypatch.setattr(store, "finish_site", finish)
    monkeypatch.setattr(
        extraction,
        "extract_page",
        lambda *args: ExtractedPage("Title", "Body " * 100, None, "en"),
    )
    original = run.scrapy_settings

    def settings(config):
        value = original(config)
        value["DOWNLOAD_HANDLERS"] = {"http": FixtureHTTP, "https": FixtureHTTP}
        value["AUTOTHROTTLE_ENABLED"] = False
        value["LOG_ENABLED"] = False
        return value

    monkeypatch.setattr(run, "scrapy_settings", settings)
    config = Settings(crawl_delay_seconds=0)
    FixtureHTTP.calls.clear()
    FixtureHTTP.crawlers.clear()
    await asyncio.wait_for(run.run_crawl(cast(Any, object()), config), timeout=15)
    assert set(pages) == {"https://blog.example/good", "https://blog.example/flaky"}
    assert rows["https://blog.example/blocked"] == "rejected"
    assert rows["https://blog.example/bomb"] == "rejected"
    assert rows["https://blog.example/feed"] == "rejected"
    assert FixtureHTTP.calls["https://blog.example/blocked"] == 0
    assert FixtureHTTP.calls["https://blog.example/flaky"] == 2
    assert finishes[-1]["error"] is None
    crawler = FixtureHTTP.crawlers[-1]
    assert crawler.stats.get_value("scheduler/enqueued/disk", 0) > 0
    assert crawler.stats.get_value("scheduler/enqueued/memory", 0) == 0
    assert not Path(crawler.settings["JOBDIR"]).exists()
    previous = FixtureHTTP.calls["https://blog.example/good"]
    await asyncio.wait_for(run.run_crawl(cast(Any, object()), config), timeout=15)
    assert FixtureHTTP.calls["https://blog.example/good"] == previous

    async def broken(*args):
        raise psycopg.OperationalError("database lost during discovery")

    monkeypatch.setattr(store, "discover", broken)
    with pytest.raises(RuntimeError, match="application failure"):
        await asyncio.wait_for(run.run_crawl(cast(Any, object()), config), timeout=15)
    assert len(finishes) == 2  # failed batch cannot mark history complete

    monkeypatch.setattr(store, "discover", discover)
    monkeypatch.setattr(store, "pending", broken)
    with pytest.raises(RuntimeError, match="application failure"):
        await asyncio.wait_for(run.run_crawl(cast(Any, object()), config), timeout=15)
    assert len(finishes) == 2  # errors from start() must also propagate

    monkeypatch.setattr(store, "pending", pending)
    rows.clear()
    monkeypatch.setattr(store, "store_page", broken)
    with pytest.raises(RuntimeError, match="application failure"):
        await asyncio.wait_for(run.run_crawl(cast(Any, object()), config), timeout=15)
    assert len(finishes) == 2  # pipeline failures cannot look like success

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_download(self, request):
        entered.set()
        await release.wait()
        return Response(request.url, request=request, status=503)

    monkeypatch.setattr(FixtureHTTP, "download_request", blocked_download)
    closed_before = InlinePool.closed
    task = asyncio.create_task(run.run_crawl(cast(Any, object()), config))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert InlinePool.closed == closed_before + 1
    assert len(finishes) == 2
    for crawler in FixtureHTTP.crawlers:
        assert not Path(crawler.settings["JOBDIR"]).exists()
