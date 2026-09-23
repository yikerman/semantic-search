from dataclasses import dataclass, field
from typing import Any

from pebble import ProcessExpired
from scrapy.crawler import Crawler
from scrapy.exceptions import DropItem

from semsearch.cli.crawl import store
from semsearch.cli.crawl.extraction import ArticleRejected, extract_article
from semsearch.cli.models import Site


@dataclass(frozen=True, slots=True)
class Article:
    site: Site
    url: str
    fetched_url: str
    body: bytes = field(repr=False)
    content_type: str


class ArticlePipeline:
    def __init__(self, crawler: Crawler):
        self.crawler = crawler

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> ArticlePipeline:
        return cls(crawler)

    async def process_item(self, item: Article) -> Article:
        spider: Any = self.crawler.spider
        try:
            page = await spider.extractor.call(
                extract_article, item.body, item.fetched_url, item.content_type
            )
        except (ArticleRejected, TimeoutError, ProcessExpired) as exc:
            reason = str(exc) or type(exc).__name__
            await store.outcome(
                spider.pool,
                item.url,
                "rejected",
                reason,
                spider.config.crawl_interval_seconds,
            )
            self.crawler.stats.inc_value("articles/rejected")
            raise DropItem(reason) from exc
        await store.store_page(spider.pool, item.site, item.url, page)
        self.crawler.stats.inc_value("articles/stored")
        return item
