import asyncio
from typing import Any, cast

from semsearch.cli.ingest.fetch import Fetcher


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
    fetcher._request_lock = asyncio.Lock()
    fetcher._session = FakeSession()

    response = await fetcher.fetch_response("https://example.com/feed.json")

    assert response is not None
    assert response.text == "{}"


async def test_concurrent_requests_reserve_spaced_start_times(monkeypatch):
    clock = 100.0
    starts: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        nonlocal clock
        clock += delay
        await real_sleep(0)

    class RecordingSession(FakeSession):
        async def get(self, url, headers=None):
            starts.append(clock)
            await real_sleep(0)
            return FakeResponse()

    monkeypatch.setattr("semsearch.cli.ingest.fetch.time.monotonic", lambda: clock)
    monkeypatch.setattr("semsearch.cli.ingest.fetch.asyncio.sleep", sleep)
    fetcher = cast(Any, Fetcher.__new__(Fetcher))
    fetcher._delay = 1.0
    fetcher._last_request = 0.0
    fetcher._request_lock = asyncio.Lock()
    fetcher._session = RecordingSession()

    await asyncio.gather(
        fetcher.fetch_text("https://a"), fetcher.fetch_text("https://b")
    )

    assert starts == [100.0, 101.0]
