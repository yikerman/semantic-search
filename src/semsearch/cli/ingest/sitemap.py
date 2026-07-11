import logging
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from semsearch.cli.ingest.fetch import FetchError

logger = logging.getLogger(__name__)

MAX_SITEMAP_DEPTH = 5


_SITEMAP_NAMESPACES = (
    "",
    "{http://www.sitemaps.org/schemas/sitemap/0.9}",
    "{https://www.sitemaps.org/schemas/sitemap/0.9}",
)
_LOC_TAGS = frozenset(ns + "loc" for ns in _SITEMAP_NAMESPACES)


type FetchText = Callable[[str], Awaitable[str]]


def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    root = ElementTree.fromstring(xml)
    locs = [
        element.text.strip()
        for element in root.iter()
        if element.tag in _LOC_TAGS and element.text and element.text.strip()
    ]
    if _local_name(root.tag) == "sitemapindex":
        return [], locs
    return locs, []


def parse_robots_sitemaps(robots_txt: str, base_url: str) -> list[str]:
    sitemaps: list[str] = []
    for line in robots_txt.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "sitemap" and value.strip():
            sitemaps.append(urljoin(base_url, value.strip()))
    return sitemaps


def filter_urls(
    urls: list[str], *, include: str | None = None, exclude: str | None = None
) -> list[str]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None

    def keep(url: str) -> bool:
        return (
            url.startswith(("http://", "https://"))
            and (include_re is None or include_re.search(url) is not None)
            and (exclude_re is None or exclude_re.search(url) is None)
        )

    return [url for url in urls if keep(url)]


def is_site_root(url: str) -> bool:
    parts = urlsplit(url)
    return parts.path in ("", "/") and not parts.query


async def discover_sitemaps(fetch_text: FetchText, site_url: str) -> list[str]:
    parts = urlsplit(site_url)
    origin = f"{parts.scheme}://{parts.netloc}"
    try:
        robots = await fetch_text(urljoin(origin, "/robots.txt"))
        sitemaps = parse_robots_sitemaps(robots, origin)
        if sitemaps:
            return sitemaps
    except FetchError:
        pass
    base_path = parts.path.rstrip("/")
    candidates = []
    if base_path:
        candidates.extend(
            [
                urljoin(origin, f"{base_path}/sitemap.xml"),
                urljoin(origin, f"{base_path}/wp-sitemap.xml"),
            ]
        )
    candidates.append(urljoin(origin, "/sitemap.xml"))
    return list(dict.fromkeys(candidates))


async def collect_page_urls(
    fetch_text: FetchText, sitemap_url: str, *, warn: bool = True
) -> list[str]:
    seen_sitemaps: set[str] = set()
    pages: dict[str, None] = {}

    async def walk(url: str, depth: int) -> None:
        if url in seen_sitemaps or depth > MAX_SITEMAP_DEPTH:
            return
        seen_sitemaps.add(url)
        try:
            xml = await fetch_text(url)
            page_urls, children = parse_sitemap(xml)
        except (FetchError, ElementTree.ParseError) as exc:
            if warn:
                logger.warning("Skipping sitemap %s: %s", url, exc)
            return
        for page_url in page_urls:
            pages.setdefault(page_url)
        for child in children:
            await walk(child, depth + 1)

    await walk(sitemap_url, 0)
    return list(pages)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
