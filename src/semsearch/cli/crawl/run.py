import asyncio
import logging
from time import monotonic

from psycopg_pool import AsyncConnectionPool
from scrapy.crawler import AsyncCrawlerRunner
from scrapy.utils.reactor import install_reactor

from semsearch.cli.crawl import store
from semsearch.cli.crawl.extraction import ExtractionPool
from semsearch.cli.crawl.policy import PublicResolver
from semsearch.cli.crawl.settings import scrapy_settings
from semsearch.cli.crawl.spider import BlogSpider
from semsearch.cli.locks import CRAWL_LOCK, command_lock
from semsearch.cli.sites import list_sites
from semsearch.share.config import Settings

logger = logging.getLogger(__name__)


async def run_crawl(
    pool: AsyncConnectionPool, settings: Settings, *, retry_failed: bool = False
) -> None:
    async with command_lock(pool, CRAWL_LOCK):
        install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")
        from typing import cast

        from twisted.internet import reactor
        from twisted.internet.base import ReactorBase

        options = scrapy_settings(settings)
        options["REQUEST_FINGERPRINTER_CLASS"] = (
            "semsearch.cli.crawl.spider.PurposeFingerprinter"
        )
        runner = AsyncCrawlerRunner(options)
        # Runners share our event loop and do not install a resolver themselves.
        PublicResolver(
            cast(ReactorBase, reactor), 10000, settings.crawl_timeout_seconds
        ).install_on_reactor()
        if retry_failed:
            await store.reset_failures(pool)
        sites = await list_sites(pool)
        cooldowns = await store.load_cooldowns(pool)
        extractor = ExtractionPool(
            settings.extraction_workers, settings.extraction_timeout_seconds
        )
        crawler = runner.create_crawler(BlogSpider)
        started = monotonic()
        crawl_task = runner.crawl(
            crawler,
            pool=pool,
            config=settings,
            sites=sites,
            extractor=extractor,
            cooldowns=cooldowns,
        )
        try:
            # Scrapy consumes cancellation while closing its engine. Keep it
            # on the owner so interrupted batches never publish completion.
            await asyncio.shield(crawl_task)
            spider = crawler.spider
            assert isinstance(spider, BlogSpider)
            if spider.fatal_error:
                raise RuntimeError(
                    "crawl aborted after an application failure"
                ) from spider.fatal_error
            await spider.finish()
        except asyncio.CancelledError:
            await runner.stop()
            await crawl_task
            raise
        finally:
            await extractor.close()
        logger.info(
            "Crawl batch finished in %.1fs: %s",
            monotonic() - started,
            crawler.stats.get_stats() if crawler.stats else {},
        )
