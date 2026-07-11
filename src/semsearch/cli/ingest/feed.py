from dataclasses import dataclass
from typing import Any, cast
import feedparser

from semsearch.cli.url import try_normalize_url


class FeedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedFeed:
    urls: list[str]
    home_url: str | None
    history_url: str | None
    is_wordpress: bool
    is_complete: bool


def parse_feed(body: bytes, *, url: str, headers: dict[str, str]) -> ParsedFeed:
    content_type = headers.get("content-type", "").lower()
    if "json" in content_type or body.lstrip().startswith((b"{", b"[")):
        raise FeedError("JSON Feed is not supported")

    response_headers = {key.lower(): value for key, value in headers.items()}
    response_headers.setdefault("content-location", url)
    parsed = cast(
        Any,
        feedparser.parse(body, response_headers=response_headers),
    )
    version = str(parsed.get("version") or "")
    if not version.startswith(("rss", "atom")):
        detail = parsed.get("bozo_exception")
        raise FeedError(f"Not a valid RSS/Atom feed{f': {detail}' if detail else ''}")

    urls: list[str] = []
    for entry in parsed.entries:
        candidate = entry.get("link")
        if not isinstance(candidate, str) or not candidate:
            entry_id = entry.get("id")
            candidate = entry_id if isinstance(entry_id, str) else None
        normalized = try_normalize_url(candidate) if candidate else None
        if normalized is not None:
            urls.append(normalized)

    feed = parsed.feed
    links = feed.get("links", [])
    home_url = _feed_link(links, ("alternate",))
    history_url = _feed_link(links, ("prev-archive", "next"))
    generator = str(feed.get("generator") or "").lower()
    return ParsedFeed(
        urls=list(dict.fromkeys(urls)),
        home_url=home_url,
        history_url=history_url,
        is_wordpress="wordpress" in generator,
        is_complete="fh_complete" in feed,
    )


def _feed_link(links: object, relations: tuple[str, ...]) -> str | None:
    if not isinstance(links, list):
        return None
    for relation in relations:
        for link in links:
            if not isinstance(link, dict) or link.get("rel") != relation:
                continue
            href = link.get("href")
            if isinstance(href, str):
                normalized = try_normalize_url(href)
                if normalized is not None:
                    return normalized
    return None
