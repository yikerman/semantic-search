import asyncio
from collections import defaultdict
from typing import Any, cast

from semsearch.cli.ingest.fetch import Fetcher


class FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status_code = status
        self.content = b"<rss></rss>" if status == 200 else b""
        self.headers = {"content-type": "application/rss+xml", "etag": '"v1"'}
        self.url = "https://example.com/feed.xml"


class FakeSession:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse()
        self.starts: list[str] = []

    async def get(self, url, headers=None):
        self.starts.append(url)
        return self.response

    async def close(self):
        return None


def fake_fetcher(session: FakeSession, *, delay: float = 0.0):
    fetcher = cast(Any, Fetcher.__new__(Fetcher))
    fetcher._delay = delay
    fetcher._last_origin_start = {}
    fetcher._origin_locks = defaultdict(asyncio.Lock)
    fetcher._semaphore = asyncio.Semaphore(16)
    fetcher._session = session
    return fetcher


async def test_fetch_response_returns_bytes_metadata_and_headers():
    fetcher = fake_fetcher(FakeSession())

    response = await fetcher.fetch_response("https://example.com/feed.xml")

    assert response.body == b"<rss></rss>"
    assert response.status == 200
    assert response.headers["etag"] == '"v1"'


async def test_fetch_response_exposes_not_modified_without_parsing():
    fetcher = fake_fetcher(FakeSession(FakeResponse(304)))

    response = await fetcher.fetch_response(
        "https://example.com/feed.xml", allow_not_modified=True
    )

    assert response.status == 304
    assert response.body == b""


async def test_distinct_origins_are_not_cross_throttled(monkeypatch):
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
            return self.response

    monkeypatch.setattr("semsearch.cli.ingest.fetch.time.monotonic", lambda: clock)
    monkeypatch.setattr("semsearch.cli.ingest.fetch.asyncio.sleep", sleep)
    fetcher = fake_fetcher(RecordingSession(), delay=1.0)

    await asyncio.gather(
        fetcher.fetch_response("https://a.example/feed"),
        fetcher.fetch_response("https://b.example/feed"),
    )

    # Politeness is per-origin only: two different hosts start together.
    assert starts == [100.0, 100.0]


async def test_same_origin_requests_observe_origin_delay(monkeypatch):
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
            return self.response

    monkeypatch.setattr("semsearch.cli.ingest.fetch.time.monotonic", lambda: clock)
    monkeypatch.setattr("semsearch.cli.ingest.fetch.asyncio.sleep", sleep)
    fetcher = fake_fetcher(RecordingSession(), delay=1.0)

    await asyncio.gather(
        fetcher.fetch_response("https://a.example/one"),
        fetcher.fetch_response("https://a.example/two"),
    )

    assert starts == [100.0, 101.0]
