import asyncio
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import pytest

from semsearch.web import languages
from semsearch.web.app import create_app


class Pool:
    @asynccontextmanager
    async def connection(self):
        class Connection:
            async def execute(self, query):
                assert query == "SET LOCAL statement_timeout = '5s'"

        yield Connection()


@pytest.mark.parametrize("codes", [(), ("en", "fr")])
async def test_homepage_does_not_wait_for_language_refresh(monkeypatch, codes):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def load(conn):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(languages, "list_available_languages", load)
    app = create_app()
    app.state.languages.codes = codes
    task = asyncio.create_task(
        languages.refresh_languages(cast(Any, Pool()), app.state.languages)
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await asyncio.wait_for(client.get("/?lang=pcm"), 1)
        assert response.status_code == 200
        assert '<option value="pcm" selected>pcm</option>' in response.text
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled.is_set()


async def test_language_refresh_preserves_options_and_recovers(monkeypatch):
    state = languages.LanguageState()
    calls = 0

    async def load(conn):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("database unavailable")
        return ["en"] if calls == 1 else ["en", "fr"]

    async def sleep(interval):
        assert interval == 300
        if calls < 3:
            assert state.codes == ("en",)
        else:
            assert state.codes == ("en", "fr")
            raise asyncio.CancelledError

    monkeypatch.setattr(languages, "list_available_languages", load)
    monkeypatch.setattr(languages.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await languages.refresh_languages(cast(Any, Pool()), state)


async def test_language_refresh_timeout_preserves_options(monkeypatch):
    state = languages.LanguageState(("en",))
    timeout = asyncio.timeout

    async def load(conn):
        await asyncio.Event().wait()

    async def sleep(interval):
        assert state.codes == ("en",)
        raise asyncio.CancelledError

    monkeypatch.setattr(languages, "list_available_languages", load)
    monkeypatch.setattr(languages.asyncio, "timeout", lambda _: timeout(0))
    monkeypatch.setattr(languages.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await languages.refresh_languages(cast(Any, Pool()), state)
