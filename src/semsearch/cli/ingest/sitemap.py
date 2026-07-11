import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

from semsearch.cli.ingest.fetch import FetchError
from semsearch.cli.url import normalize_origin, try_normalize_url

logger = logging.getLogger(__name__)

MAX_SITEMAP_DEPTH = 5


_SITEMAP_NAMESPACES = (
    "",
    "{http://www.sitemaps.org/schemas/sitemap/0.9}",
    "{https://www.sitemaps.org/schemas/sitemap/0.9}",
)
_LOC_TAGS = frozenset(ns + "loc" for ns in _SITEMAP_NAMESPACES)
_POST_SITEMAP_RE = re.compile(
    r"(?:^|[-_/])(post|posts|article|articles|blog)(?:[-_.?/]|$)", re.I
)
_NON_POST_PATH_RE = re.compile(
    r"/(?:about|contact|privacy|terms|tag|tags|category|categories|author|authors|"
    r"attachment|attachments|archive|archives)(?:/|$)",
    re.I,
)


type FetchText = Callable[[str], Awaitable[str]]
type UrlPredicate = Callable[[str], bool]


class SitemapError(RuntimeError):
    pass


def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    try:
        root = DefusedET.fromstring(xml)
    except DefusedXmlException as exc:
        raise ElementTree.ParseError(f"unsafe sitemap XML rejected: {exc}") from exc
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


async def discover_sitemaps(fetch_text: FetchText, site_url: str) -> list[str]:
    parts = urlsplit(site_url)
    origin = normalize_origin(site_url)
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
    fetch_text: FetchText,
    sitemap_url: str,
    *,
    accept: UrlPredicate | None = None,
    allow_sitemap: UrlPredicate | None = None,
    limit: int | None = None,
    strict: bool = False,
) -> list[str]:
    seen_sitemaps: set[str] = set()
    pages: dict[str, None] = {}

    async def walk(url: str, depth: int) -> None:
        if (
            url in seen_sitemaps
            or depth > MAX_SITEMAP_DEPTH
            or (limit is not None and len(pages) >= limit)
        ):
            return
        seen_sitemaps.add(url)
        try:
            xml = await fetch_text(url)
            page_urls, children = await asyncio.to_thread(parse_sitemap, xml)
        except (FetchError, ElementTree.ParseError) as exc:
            if strict:
                raise SitemapError(f"Failed sitemap {url}: {exc}") from exc
            logger.warning("Skipping sitemap %s: %s", url, exc)
            return
        for page_url in page_urls:
            candidate = try_normalize_url(page_url)
            if candidate is not None and (accept is None or accept(candidate)):
                pages.setdefault(candidate)
                if limit is not None and len(pages) >= limit:
                    return
        preferred = [child for child in children if _POST_SITEMAP_RE.search(child)]
        for child in preferred or children:
            candidate = try_normalize_url(child)
            if candidate is None or (
                allow_sitemap is not None and not allow_sitemap(candidate)
            ):
                continue
            await walk(candidate, depth + 1)
            if limit is not None and len(pages) >= limit:
                return

    await walk(sitemap_url, 0)
    return list(pages)


def is_post_url(url: str) -> bool:
    return not _NON_POST_PATH_RE.search(urlsplit(url).path)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
