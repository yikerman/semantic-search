from io import FileIO

import pytest
from scrapy import Request, Spider
from scrapy.crawler import Crawler
from scrapy.utils.misc import build_from_crawler, load_object

from semsearch.cli.crawl.settings import scrapy_settings
from semsearch.share.config import Settings


@pytest.mark.parametrize(
    "setting", ["SCHEDULER_DISK_QUEUE", "SCHEDULER_START_DISK_QUEUE"]
)
def test_unbuffered_request_queue_rollover_and_reopen(tmp_path, setting):
    crawler = Crawler(Spider)
    crawler.spider = Spider("test")
    queue_path = scrapy_settings(Settings())[setting]
    assert isinstance(queue_path, str)
    queue_class = load_object(queue_path)
    path = str(tmp_path / "queue")
    queue = build_from_crawler(queue_class, crawler, path)
    # Exercise file rollover without writing the default 100,000 requests.
    queue.chunksize = queue.info["chunksize"] = 2
    requests = [
        Request(
            f"https://blog.example/post/{i}",
            callback=crawler.spider.parse,
            meta={"site_id": 1, "kind": "article", "label": "café"},
            headers={"Accept": "text/html"},
        )
        for i in range(5)
    ]

    def assert_unbuffered():
        assert isinstance(queue.headf, FileIO)
        assert isinstance(queue.tailf, FileIO)

    try:
        assert_unbuffered()
        for request in requests:
            queue.push(request)
            assert_unbuffered()
        assert queue.peek().url == requests[0].url
        assert len(queue) == 5
        assert queue.pop().url == requests[0].url
        queue.close()
        queue = build_from_crawler(queue_class, crawler, path)
        assert_unbuffered()
        for expected in requests[1:]:
            actual = queue.pop()
            assert actual.to_dict(spider=crawler.spider) == expected.to_dict(
                spider=crawler.spider
            )
            assert_unbuffered()
        assert queue.peek() is None
        assert queue.pop() is None
        assert len(queue) == 0
    finally:
        queue.close()
    assert not (tmp_path / "queue").exists()
