import asyncio
import multiprocessing
from collections.abc import Callable
from typing import Any, cast

from pebble import ProcessPool

from semsearch.cli.ingest.extract import ExtractedPage, extract_page

MAX_TEXT_CHARS = 500_000


class ArticleRejected(ValueError):
    pass


def extract_article(body: bytes, url: str, content_type: str) -> ExtractedPage:
    import lxml.etree
    import lxml.html

    mime = content_type.partition(";")[0].strip().lower()
    if mime not in ("text/html", "application/xhtml+xml"):
        raise ArticleRejected("not_html")
    prefix = body.lstrip()
    if prefix.startswith((b"{", b"[")):
        raise ArticleRejected("json_document")
    # Inspect XML before the forgiving HTML parser can turn feeds into HTML.
    parser = lxml.etree.XMLParser(
        resolve_entities=False, no_network=True, recover=False
    )
    try:
        root = lxml.etree.fromstring(body, parser)
    except lxml.etree.XMLSyntaxError:
        root = None
    if root is not None and lxml.etree.QName(root).localname.lower() != "html":
        raise ArticleRejected("non_article_xml")
    try:
        document = lxml.html.fromstring(
            body, parser=lxml.html.HTMLParser(huge_tree=False)
        )
    except (lxml.etree.ParserError, lxml.etree.XMLSyntaxError, ValueError) as exc:
        raise ArticleRejected("invalid_html") from exc
    if document.xpath(
        '//*[local-name()="rss" or local-name()="feed" or local-name()="urlset" or local-name()="sitemapindex"]'
    ):
        raise ArticleRejected("feed_or_sitemap")
    page = extract_page(body, url)
    if page is None:
        raise ArticleRejected("no_article_text")
    if len(page.text) > MAX_TEXT_CHARS:
        raise ArticleRejected("article_too_long")
    return page


class ExtractionPool:
    """Bound submitted work as well as processes; cancellation kills the task."""

    def __init__(self, workers: int = 2, timeout: float = 10):
        self.timeout = timeout
        self.slots = asyncio.Semaphore(workers)
        self.pool = ProcessPool(
            max_workers=workers,
            max_tasks=100,
            # Pebble annotates this as ModuleType but accepts a process context.
            context=cast(Any, multiprocessing.get_context("spawn")),
        )

    async def call[T](self, function: Callable[..., T], *args: Any) -> T:
        async with self.slots:
            future = self.pool.schedule(function, args=args, timeout=self.timeout)
            try:
                return await asyncio.wrap_future(future)
            except asyncio.CancelledError:
                future.cancel()
                raise

    async def close(self) -> None:
        self.pool.stop()
        await asyncio.to_thread(self.pool.join)
