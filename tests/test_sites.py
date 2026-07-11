import asyncio

from semsearch.ingest.fetch import FetchError, FetchResponse
from semsearch.ingest.models import IndexOutcome
from semsearch.models import Site
from semsearch.sites import (
    PollOutcome,
    SiteService,
    canonicalize_site_url,
    discover_feed_url,
    extract_feed_urls,
)
from semsearch.url import normalize_origin, normalize_url


class FakeFetcher:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    async def fetch_text(self, url: str) -> str:
        try:
            return self.responses[url]
        except KeyError as exc:
            raise FetchError(url) from exc


class FakeUrlIndexer:
    def __init__(self, failing_url: str) -> None:
        self.failing_url = failing_url
        self.indexed: list[tuple[str, bool]] = []

    async def index_url(self, url: str, force: bool = False) -> IndexOutcome:
        self.indexed.append((url, force))
        if url == self.failing_url:
            raise FetchError("stale feed entry")
        return IndexOutcome(url, "indexed", chunk_count=1)


class FakePollingSiteService(SiteService):
    def __init__(self, feed: FetchResponse) -> None:
        self.feed = feed
        self.polled_headers = None

    async def get_site(self, site: str) -> Site:
        return Site(
            id=1,
            base_url="https://example.com",
            sitemap_url=None,
            feed_url="https://example.com/feed.xml",
            last_indexed_at=None,
            last_polled_at=None,
            feed_etag=None,
            feed_last_modified=None,
        )

    async def _fetch_feed(self, site: Site, *, use_cache: bool) -> FetchResponse | None:
        return self.feed

    async def _mark_polled(self, site_id: int, headers) -> None:
        self.polled_headers = headers


class FakeConcurrentPollingSiteService(SiteService):
    def __init__(self) -> None:
        self.sites = [
            Site(
                id=1,
                base_url="https://a.example",
                sitemap_url=None,
                feed_url="https://a.example/feed.xml",
                last_indexed_at=None,
                last_polled_at=None,
                feed_etag=None,
                feed_last_modified=None,
            ),
            Site(
                id=2,
                base_url="https://b.example",
                sitemap_url=None,
                feed_url="https://b.example/feed.xml",
                last_indexed_at=None,
                last_polled_at=None,
                feed_etag=None,
                feed_last_modified=None,
            ),
            Site(
                id=3,
                base_url="https://c.example",
                sitemap_url=None,
                feed_url="https://c.example/feed.xml",
                last_indexed_at=None,
                last_polled_at=None,
                feed_etag=None,
                feed_last_modified=None,
            ),
        ]
        self.release = asyncio.Event()
        self.two_active = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def list_sites(self) -> list[Site]:
        return self.sites

    async def poll_site(
        self,
        site: str,
        index_url,
        *,
        force: bool = False,
        on_progress=None,
    ) -> PollOutcome:
        record = next(record for record in self.sites if record.base_url == site)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.two_active.set()
        try:
            await self.release.wait()
            return PollOutcome(record, [IndexOutcome(site, "skipped")])
        finally:
            self.active -= 1


def test_normalize_origin_defaults_to_https_and_drops_path():
    assert normalize_origin("Example.COM/blog/?x=1") == "https://example.com"
    assert (
        normalize_origin("http://Example.COM:8080/blog/") == "http://example.com:8080"
    )


def test_normalize_url_keeps_path_without_query():
    assert normalize_url("Example.COM/blog/?x=1") == "https://example.com/blog/"
    assert normalize_url("https://example.com") == "https://example.com/"


async def test_discover_feed_url_uses_alternate_link():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" href="/blog/feed/" />
      </head>
    </html>
    """
    fetcher = FakeFetcher({"https://example.com/blog/": html})
    feed = await discover_feed_url(fetcher.fetch_text, "https://example.com/blog/")
    assert feed == "https://example.com/blog/feed/"


async def test_discover_feed_url_resolves_relative_links_from_page_url():
    html = """
    <html>
      <head>
        <link rel="alternate" type="application/rss+xml" href="feed/" />
      </head>
    </html>
    """
    fetcher = FakeFetcher({"https://example.com/blog/": html})
    feed = await discover_feed_url(fetcher.fetch_text, "https://example.com/blog/")
    assert feed == "https://example.com/blog/feed/"


async def test_discover_feed_url_accepts_feed_url():
    rss = """
    <rss version="2.0">
      <channel><title>Example</title><link>https://example.com/</link></channel>
    </rss>
    """
    fetcher = FakeFetcher({"https://example.com/feed.xml": rss})
    feed = await discover_feed_url(fetcher.fetch_text, "https://example.com/feed.xml")
    assert feed == "https://example.com/feed.xml"


def test_extract_feed_urls_from_rss():
    rss = """
    <rss version="2.0">
      <channel>
        <item><link>https://example.com/a</link></item>
        <item><link>https://example.com/b</link></item>
        <item><link>https://example.com/a</link></item>
      </channel>
    </rss>
    """
    assert extract_feed_urls(rss, "https://example.com/feed.xml") == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_extract_feed_urls_from_atom():
    atom = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><link href="https://example.com/a" /></entry>
      <entry><link href="https://example.com/b" /></entry>
    </feed>
    """
    assert extract_feed_urls(atom, "https://example.com/feed.xml") == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_extract_feed_urls_from_json_feed():
    feed = """
    {
      "version": "https://jsonfeed.org/version/1.1",
      "items": [
        {"url": "https://example.com/a"},
        {"external_url": "https://example.com/b"}
      ]
    }
    """
    assert extract_feed_urls(feed, "https://example.com/feed.json") == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_canonicalize_site_url_keeps_configured_origin():
    assert (
        canonicalize_site_url(
            "http://karpathy.github.io/2026/02/12/microgpt/",
            "https://karpathy.github.io",
        )
        == "https://karpathy.github.io/2026/02/12/microgpt/"
    )
    assert (
        canonicalize_site_url("https://other.example/post", "https://example.com")
        == "https://other.example/post"
    )


async def test_poll_site_keeps_polling_after_url_failure():
    feed = FetchResponse(
        """
        <rss version="2.0">
          <channel>
            <item><link>https://example.com/a</link></item>
            <item><link>https://example.com/bad</link></item>
            <item><link>https://example.com/c</link></item>
          </channel>
        </rss>
        """,
        {"etag": '"feed-v1"'},
    )
    service = FakePollingSiteService(feed)
    indexer = FakeUrlIndexer("https://example.com/bad")
    progress: list[IndexOutcome] = []

    outcome = await service.poll_site(
        "https://example.com",
        indexer.index_url,
        force=True,
        on_progress=progress.append,
    )

    assert [entry.status for entry in outcome.outcomes] == [
        "indexed",
        "error",
        "indexed",
    ]
    assert outcome.outcomes[1].detail == "stale feed entry"
    assert indexer.indexed == [
        ("https://example.com/a", True),
        ("https://example.com/bad", True),
        ("https://example.com/c", True),
    ]
    assert progress == outcome.outcomes
    assert service.polled_headers == {"etag": '"feed-v1"'}


async def test_poll_all_polls_sites_concurrently_with_limit():
    service = FakeConcurrentPollingSiteService()
    indexer = FakeUrlIndexer("https://never.example")
    task = asyncio.create_task(service.poll_all(indexer.index_url, concurrency=2))

    await asyncio.wait_for(service.two_active.wait(), timeout=1)
    service.release.set()
    outcomes = await task

    assert service.max_active == 2
    assert [outcome.site.base_url for outcome in outcomes] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]
