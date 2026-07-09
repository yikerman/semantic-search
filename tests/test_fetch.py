from typing import Any, cast

from semsearch.ingest.fetch import Fetcher


class FakeResponse:
    status_code = 200
    text = "{}"
    headers = {"content-type": "application/feed+json; charset=utf-8"}


class FakeSession:
    async def get(self, url, headers=None):
        return FakeResponse()

    async def close(self):
        return None


async def test_fetch_response_allows_json_feed_content_type():
    fetcher = cast(Any, Fetcher.__new__(Fetcher))
    fetcher._delay = 0.0
    fetcher._last_request = 0.0
    fetcher._session = FakeSession()

    response = await fetcher.fetch_response("https://example.com/feed.json")

    assert response is not None
    assert response.text == "{}"
