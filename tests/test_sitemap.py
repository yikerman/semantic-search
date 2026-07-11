import pytest

from xml.etree import ElementTree

from semsearch.cli.ingest.sitemap import (
    SitemapError,
    collect_page_urls,
    discover_sitemaps,
    parse_robots_sitemaps,
    parse_sitemap,
)

URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://blog.example/post-1</loc>
    <lastmod>2026-01-01</lastmod>
    <image:image><image:loc>https://blog.example/img.png</image:loc></image:image>
  </url>
  <url><loc> https://blog.example/post-2 </loc></url>
</urlset>
"""

URLSET_NO_NAMESPACE = """<urlset>
  <url><loc>https://blog.example/post-3</loc></url>
</urlset>
"""

SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://blog.example/sitemap-posts.xml</loc></sitemap>
  <sitemap><loc>https://blog.example/sitemap-pages.xml</loc></sitemap>
</sitemapindex>
"""


def test_parse_urlset_ignores_image_locs():
    pages, children = parse_sitemap(URLSET)
    assert pages == ["https://blog.example/post-1", "https://blog.example/post-2"]
    assert children == []


def test_parse_urlset_without_namespace():
    pages, children = parse_sitemap(URLSET_NO_NAMESPACE)
    assert pages == ["https://blog.example/post-3"]
    assert children == []


def test_parse_sitemap_index():
    pages, children = parse_sitemap(SITEMAP_INDEX)
    assert pages == []
    assert children == [
        "https://blog.example/sitemap-posts.xml",
        "https://blog.example/sitemap-pages.xml",
    ]


def test_parse_robots_sitemaps():
    robots = "\n".join(
        [
            "User-agent: *",
            "Disallow: /admin",
            "Sitemap: https://blog.example/sitemap.xml",
            "sitemap: /relative-sitemap.xml",
            "# Sitemap: https://blog.example/commented.xml",
        ]
    )
    result = parse_robots_sitemaps(robots, "https://blog.example")
    assert result == [
        "https://blog.example/sitemap.xml",
        "https://blog.example/relative-sitemap.xml",
    ]


def test_parse_sitemap_rejects_entity_expansion():
    billion_laughs = """<?xml version="1.0"?>
    <!DOCTYPE urlset [
      <!ENTITY lol "lol">
      <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
      <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
    ]>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>&lol3;</loc></url>
    </urlset>"""
    with pytest.raises(ElementTree.ParseError):
        parse_sitemap(billion_laughs)


class FakeFetcher:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    async def fetch_text(self, url: str) -> str:
        self.requested.append(url)
        try:
            return self.responses[url]
        except KeyError as exc:
            from semsearch.cli.ingest.fetch import FetchError

            raise FetchError(url) from exc


async def test_collect_page_urls_recurses_and_dedupes():
    index_xml = """<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://b.example/a.xml</loc></sitemap>
      <sitemap><loc>https://b.example/a.xml</loc></sitemap>
      <sitemap><loc>https://b.example/cycle.xml</loc></sitemap>
    </sitemapindex>"""
    cycle_xml = """<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://b.example/index.xml</loc></sitemap>
      <sitemap><loc>https://b.example/b.xml</loc></sitemap>
    </sitemapindex>"""
    a_xml = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://b.example/post-1</loc></url>
      <url><loc>https://b.example/post-2</loc></url>
    </urlset>"""
    b_xml = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://b.example/post-2</loc></url>
      <url><loc>https://b.example/post-3</loc></url>
    </urlset>"""
    fetcher = FakeFetcher(
        {
            "https://b.example/index.xml": index_xml,
            "https://b.example/a.xml": a_xml,
            "https://b.example/cycle.xml": cycle_xml,
            "https://b.example/b.xml": b_xml,
        }
    )
    pages = await collect_page_urls(fetcher.fetch_text, "https://b.example/index.xml")
    assert pages == [
        "https://b.example/post-1",
        "https://b.example/post-2",
        "https://b.example/post-3",
    ]
    assert sorted(fetcher.requested) == [
        "https://b.example/a.xml",
        "https://b.example/b.xml",
        "https://b.example/cycle.xml",
        "https://b.example/index.xml",
    ]


async def test_collect_page_urls_stops_at_eligible_limit():
    index_xml = """<sitemapindex>
      <sitemap><loc>https://b.example/a.xml</loc></sitemap>
      <sitemap><loc>https://b.example/b.xml</loc></sitemap>
    </sitemapindex>"""
    page_xml = """<urlset>
      <url><loc>https://b.example/about/</loc></url>
      <url><loc>https://b.example/post-1</loc></url>
      <url><loc>https://b.example/post-2</loc></url>
    </urlset>"""
    fetcher = FakeFetcher(
        {
            "https://b.example/index.xml": index_xml,
            "https://b.example/a.xml": page_xml,
            "https://b.example/b.xml": URLSET_NO_NAMESPACE,
        }
    )

    pages = await collect_page_urls(
        fetcher.fetch_text,
        "https://b.example/index.xml",
        accept=lambda url: "/post-" in url,
        limit=2,
        strict=True,
    )

    assert pages == ["https://b.example/post-1", "https://b.example/post-2"]
    assert "https://b.example/b.xml" not in fetcher.requested


async def test_collect_page_urls_strict_mode_propagates_child_failure():
    fetcher = FakeFetcher(
        {
            "https://b.example/index.xml": SITEMAP_INDEX.replace(
                "blog.example", "b.example"
            )
        }
    )

    with pytest.raises(SitemapError, match="sitemap-posts.xml"):
        await collect_page_urls(
            fetcher.fetch_text,
            "https://b.example/index.xml",
            strict=True,
        )


async def test_discover_sitemaps_uses_path_fallbacks_without_robots():
    fetcher = FakeFetcher({})
    pages = await discover_sitemaps(fetcher.fetch_text, "https://blog.example/blog/")
    assert pages == [
        "https://blog.example/blog/sitemap.xml",
        "https://blog.example/blog/wp-sitemap.xml",
        "https://blog.example/sitemap.xml",
    ]
