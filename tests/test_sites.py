from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

import pytest

from semsearch.cli.ingest.feed import FeedError, parse_feed
from semsearch.cli.ingest.fetch import FetchError, FetchResponse
from semsearch.cli.models import Site
from semsearch.cli.sites import (
    HistoryLimitError,
    HistoryUnavailableError,
    _discover_history,
    _history_limit,
    _poll_offset,
    _site_feed,
    _walk_feed_history,
    _walk_wordpress_pages,
    _with_query,
    poll_site_record,
)
from semsearch.cli.url import try_normalize_url
from semsearch.share.config import Settings


def test_parse_rss_urls_with_feedparser():
    rss = b"""
    <rss version="2.0"><channel><title>Example</title>
      <item><guid>one</guid><link>https://example.com/a#fragment</link></item>
      <item><guid>two</guid><link>https://example.com/b?view=full</link></item>
      <item><guid>again</guid><link>https://example.com/a#other</link></item>
    </channel></rss>
    """

    parsed = parse_feed(
        rss,
        url="https://example.com/feed.xml",
        headers={"content-type": "application/rss+xml"},
    )

    assert parsed.urls == [
        "https://example.com/a",
        "https://example.com/b?view=full",
    ]
    assert parsed.history_url is None


def test_parse_atom_resolves_relative_urls_and_archive_link():
    atom = b"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Example</title><id>feed</id>
      <link rel="prev-archive" href="archive-1.atom" />
      <entry><id>one</id><title>One</title><link href="posts/one" /></entry>
    </feed>
    """

    parsed = parse_feed(
        atom,
        url="https://example.com/blog/feed.atom",
        headers={"content-type": "application/atom+xml"},
    )

    assert parsed.urls == ["https://example.com/blog/posts/one"]
    assert parsed.home_url is None
    assert parsed.history_url == "https://example.com/blog/archive-1.atom"


def test_parse_feed_detects_wordpress_generator():
    rss = b"""
    <rss version="2.0"><channel><title>Example</title>
      <generator>https://wordpress.org/?v=6.8</generator>
      <item><link>https://example.com/a</link></item>
    </channel></rss>
    """

    parsed = parse_feed(
        rss,
        url="https://example.com/feed/",
        headers={"content-type": "application/rss+xml"},
    )

    assert parsed.is_wordpress


def test_parse_feed_detects_rfc_5005_complete_marker():
    atom = b"""
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:fh="http://purl.org/syndication/history/1.0">
      <title>Complete</title><id>feed</id><fh:complete/>
      <entry><id>one</id><title>One</title>
        <link href="https://example.com/one" />
      </entry>
    </feed>
    """

    parsed = parse_feed(
        atom,
        url="https://example.com/feed.atom",
        headers={"content-type": "application/atom+xml"},
    )

    assert parsed.is_complete


def test_parse_feed_uses_url_shaped_id_when_link_is_missing():
    atom = b"""
    <feed xmlns="http://www.w3.org/2005/Atom"><title>x</title><id>feed</id>
      <entry><title>x</title><id>https://example.com/from-id</id></entry>
    </feed>
    """

    parsed = parse_feed(
        atom,
        url="https://example.com/feed",
        headers={"content-type": "application/atom+xml"},
    )

    assert parsed.urls == ["https://example.com/from-id"]


def test_json_feed_is_intentionally_rejected():
    try:
        parse_feed(
            b'{"version":"https://jsonfeed.org/version/1.1","items":[]}',
            url="https://example.com/feed.json",
            headers={"content-type": "application/feed+json"},
        )
    except FeedError as exc:
        assert str(exc) == "JSON Feed is not supported"
    else:
        raise AssertionError("JSON Feed accepted")


def test_try_normalize_url_preserves_query_and_drops_default_port():
    assert (
        try_normalize_url("HTTPS://Example.COM:443/post?p=1#comments")
        == "https://example.com/post?p=1"
    )
    assert try_normalize_url("mailto:someone@example.com") is None
    assert try_normalize_url("http://127.0.0.1/private") is None


def test_site_feed_drops_cross_origin_posts_and_history():
    parsed = parse_feed(
        b"""<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
        <channel><title>x</title>
          <atom:link rel="prev-archive" href="https://other.example/archive" />
          <item><link>https://example.com/post</link></item>
          <item><link>http://example.com/older-post</link></item>
          <item><link>https://other.example/post</link></item>
        </channel></rss>""",
        url="https://example.com/feed/",
        headers={"content-type": "application/rss+xml"},
    )

    filtered = _site_feed(parsed, _site())

    # Cross-origin posts are dropped; same-site posts are canonicalized onto the
    # configured origin's scheme (http://example.com -> https://example.com).
    assert filtered.urls == [
        "https://example.com/post",
        "https://example.com/older-post",
    ]
    assert filtered.history_url is None


def test_wordpress_page_query_preserves_existing_parameters():
    assert (
        _with_query("https://example.com/feed/?lang=en", "paged", "2")
        == "https://example.com/feed/?lang=en&paged=2"
    )


def test_poll_offsets_scatter_4000_sites_across_the_hour():
    offsets = [
        _poll_offset(f"https://site-{index}.example", 3600) for index in range(4000)
    ]

    assert min(offsets) < 10
    assert max(offsets) > 3590
    assert len(set(offsets)) > 2300
    assert offsets == [
        _poll_offset(f"https://site-{index}.example", 3600) for index in range(4000)
    ]


def test_history_limit_error_reports_partial_sync():
    site = Site(
        1,
        "https://example.com",
        "https://example.com/sitemap.xml",
        "https://example.com/feed.xml",
        None,
        None,
        None,
        None,
        0,
        None,
        True,
        None,
    )

    error = _history_limit(site, 2000)

    assert "exceeded 2000 posts" in str(error)
    assert "partially synchronized" in str(error)


class HistoryFetcher:
    def __init__(self, responses: dict[str, FetchResponse | Exception]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    async def fetch_response(self, url: str, **kwargs):
        self.requested.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def rss_response(url: str, item_urls: list[str], history_url: str | None = None):
    history = (
        f'<atom:link rel="prev-archive" href="{history_url}" />' if history_url else ""
    )
    items = "".join(f"<item><link>{item}</link></item>" for item in item_urls)
    body = (
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">'
        f"<channel><title>x</title>{history}{items}</channel></rss>"
    ).encode()
    return FetchResponse(body, {"content-type": "application/rss+xml"}, 200, url)


async def test_feed_history_stops_cycles_and_enqueues_each_url_once(monkeypatch):
    page = "https://example.com/archive.xml"
    fetcher = HistoryFetcher(
        {page: rss_response(page, ["https://example.com/older"], page)}
    )
    enqueued: list[str] = []

    async def enqueue(pool, site_id, urls, source):
        enqueued.extend(urls)
        return len(urls)

    monkeypatch.setattr("semsearch.cli.sites._enqueue", enqueue)
    site = _site()

    count = await _walk_feed_history(
        cast(Any, object()), cast(Any, fetcher), site, page, set(), 2000
    )

    assert count == 1
    assert enqueued == ["https://example.com/older"]
    assert fetcher.requested == [page]


async def test_feed_history_follows_older_link_past_duplicate_page(monkeypatch):
    first = "https://example.com/archive-1.xml"
    second = "https://example.com/archive-2.xml"
    current = "https://example.com/current"
    fetcher = HistoryFetcher(
        {
            first: rss_response(first, [current], second),
            second: rss_response(second, ["https://example.com/older"]),
        }
    )
    enqueued: list[str] = []

    async def enqueue(pool, site_id, urls, source):
        enqueued.extend(urls)
        return len(urls)

    monkeypatch.setattr("semsearch.cli.sites._enqueue", enqueue)

    count = await _walk_feed_history(
        cast(Any, object()), cast(Any, fetcher), _site(), first, {current}, 2000
    )

    assert count == 1
    assert enqueued == ["https://example.com/older"]
    assert fetcher.requested == [first, second]


async def test_feed_history_enqueues_only_up_to_limit_then_errors(monkeypatch):
    page = "https://example.com/archive.xml"
    fetcher = HistoryFetcher(
        {
            page: rss_response(
                page,
                ["https://example.com/older-a", "https://example.com/older-b"],
            )
        }
    )
    enqueued: list[str] = []

    async def enqueue(pool, site_id, urls, source):
        enqueued.extend(urls)
        return len(urls)

    monkeypatch.setattr("semsearch.cli.sites._enqueue", enqueue)
    seen = {f"https://example.com/current-{index}" for index in range(1999)}

    with pytest.raises(HistoryLimitError, match="exceeded 2000 posts"):
        await _walk_feed_history(
            cast(Any, object()),
            cast(Any, fetcher),
            _site(),
            page,
            seen,
            2000,
        )

    assert enqueued == ["https://example.com/older-a"]


async def test_wordpress_history_uses_paged_query_until_terminal_response(
    monkeypatch,
):
    page_2 = "https://example.com/feed/?paged=2"
    fetcher = HistoryFetcher(
        {
            page_2: rss_response(page_2, ["https://example.com/older"]),
            "https://example.com/feed/?paged=3": FetchError("not found", status=404),
        }
    )

    async def enqueue(pool, site_id, urls, source):
        return len(urls)

    monkeypatch.setattr("semsearch.cli.sites._enqueue", enqueue)

    count, usable = await _walk_wordpress_pages(
        cast(Any, object()), cast(Any, fetcher), _site(), set(), 2000
    )

    assert (count, usable) == (1, True)
    assert fetcher.requested == [page_2, "https://example.com/feed/?paged=3"]


def _site() -> Site:
    return Site(
        1,
        "https://example.com",
        "https://example.com/sitemap.xml",
        "https://example.com/feed/",
        None,
        None,
        None,
        None,
        0,
        None,
        True,
        None,
    )


async def test_history_without_any_historical_source_reports_partial_sync():
    current = parse_feed(
        b"""<rss version="2.0"><channel><title>x</title>
        <item><link>https://example.com/current</link></item>
        </channel></rss>""",
        url="https://example.com/feed/",
        headers={"content-type": "application/rss+xml"},
    )

    with pytest.raises(HistoryUnavailableError, match="partially synchronized"):
        await _discover_history(
            cast(Any, object()),
            cast(Any, object()),
            replace(_site(), sitemap_url=None),
            current,
            2000,
        )


class FakeConnectionContext(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def transaction(self):
        return self


class FakePool:
    def connection(self):
        return FakeConnectionContext()


async def test_pending_history_retries_after_recent_urls_are_already_known(
    monkeypatch,
):
    current_url = "https://example.com/current"
    response = rss_response("https://example.com/feed/", [current_url])
    fetcher = HistoryFetcher({"https://example.com/feed/": response})
    pending = False
    known = False
    history_attempts = 0

    async def known_urls(conn, urls):
        return {current_url} if known else set()

    async def enqueue_urls(conn, *, site_id, urls, source):
        nonlocal known
        known = True
        return len(urls)

    async def mark_pending(conn, *, site_id, lease_token):
        nonlocal pending
        pending = True

    async def finish_history(conn, *, site_id, lease_token, error=None):
        nonlocal pending
        pending = False

    async def mark_succeeded(conn, **kwargs):
        return None

    async def discover_history(pool, fetcher, site, parsed, limit):
        nonlocal history_attempts
        history_attempts += 1
        if history_attempts == 1:
            raise FetchError("archive temporarily unavailable")
        return 0

    monkeypatch.setattr("semsearch.cli.sites.db.known_urls", known_urls)
    monkeypatch.setattr("semsearch.cli.sites.db.enqueue_urls", enqueue_urls)
    monkeypatch.setattr("semsearch.cli.sites.db.mark_history_pending", mark_pending)
    monkeypatch.setattr("semsearch.cli.sites.db.finish_history", finish_history)
    monkeypatch.setattr("semsearch.cli.sites.db.mark_poll_succeeded", mark_succeeded)
    monkeypatch.setattr("semsearch.cli.sites._discover_history", discover_history)
    settings = Settings(embedding_model="test", embedding_dim=2)
    site = _site()

    first = await poll_site_record(
        cast(Any, FakePool()),
        cast(Any, fetcher),
        settings,
        replace(site, history_pending=False),
        uuid4(),
    )
    assert (
        first.error
        == "Historical discovery will retry: archive temporarily unavailable"
    )
    assert pending

    second = await poll_site_record(
        cast(Any, FakePool()),
        cast(Any, fetcher),
        settings,
        replace(site, history_pending=True),
        uuid4(),
    )
    assert second.error is None
    assert history_attempts == 2
    assert not pending
