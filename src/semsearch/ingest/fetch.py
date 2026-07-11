import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestException

from semsearch.config import Settings


class FetchError(RuntimeError):
    pass


@dataclass(slots=True)
class FetchResponse:
    text: str
    headers: Mapping[str, str]


_TEXT_CONTENT_MARKERS = ("text/", "html", "xml", "json")


class Fetcher:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 20.0,
        delay_seconds: float = 1.0,
        impersonate: str = "chrome",
    ) -> None:
        self._delay = delay_seconds
        self._last_request = 0.0
        if impersonate:
            self._session = AsyncSession(
                impersonate=cast(Any, impersonate),
                timeout=timeout,
                allow_redirects=True,
            )
        else:
            self._session = AsyncSession(
                headers={"User-Agent": user_agent},
                timeout=timeout,
                allow_redirects=True,
            )

    async def fetch_text(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> str:
        response = cast(FetchResponse, await self.fetch_response(url, headers=headers))
        return response.text

    async def fetch_response(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        allow_not_modified: bool = False,
    ) -> FetchResponse | None:
        wait = self._delay - (time.monotonic() - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            resp = await self._session.get(url, headers=headers)
        except RequestException as exc:
            raise FetchError(f"GET {url} failed: {exc}") from exc
        finally:
            self._last_request = time.monotonic()
        if resp.status_code == 304 and allow_not_modified:
            return None
        if resp.status_code != 200:
            raise FetchError(f"GET {url} returned {resp.status_code}")
        content_type = resp.headers.get("content-type", "").lower()
        if not any(marker in content_type for marker in _TEXT_CONTENT_MARKERS):
            raise FetchError(f"GET {url}: unsupported content-type {content_type!r}")
        headers_out = {
            key: value for key, value in resp.headers.items() if value is not None
        }
        return FetchResponse(resp.text, headers_out)

    async def aclose(self) -> None:
        await self._session.close()

    async def __aenter__(self) -> "Fetcher":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def create_fetcher(settings: Settings) -> Fetcher:
    return Fetcher(
        user_agent=settings.user_agent,
        timeout=settings.fetch_timeout_seconds,
        delay_seconds=settings.fetch_delay_seconds,
        impersonate=settings.fetch_impersonate,
    )
