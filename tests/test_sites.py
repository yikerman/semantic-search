from semsearch.cli.ingest.feed import FeedError, parse_feed
from semsearch.cli.url import try_normalize_url


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
