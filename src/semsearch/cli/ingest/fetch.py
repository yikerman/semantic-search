import asyncio
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestException

from semsearch.share.config import Settings
from semsearch.cli.url import normalize_origin, normalize_url


class FetchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        permanent: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.permanent = permanent


@dataclass(frozen=True, slots=True)
class FetchResponse:
    body: bytes
    headers: Mapping[str, str]
    status: int
    url: str

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


_TEXT_CONTENT_MARKERS = ("text/", "html", "xml", "json", "rss", "atom")


class Fetcher:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 20.0,
        delay_seconds: float = 1.0,
        concurrency: int = 16,
        impersonate: str = "chrome",
    ) -> None:
        self._delay = delay_seconds
        self._last_origin_start: dict[str, float] = {}
        self._origin_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._semaphore = asyncio.Semaphore(concurrency)
        session_args: dict[str, Any] = {
            "timeout": timeout,
            "allow_redirects": True,
        }
        # Impersonation ships its own browser User-Agent; overriding it with our
        # own would advertise "semsearch" atop a Chrome TLS fingerprint.
        if impersonate:
            session_args["impersonate"] = impersonate
        else:
            session_args["headers"] = {"User-Agent": user_agent}
        self._session = AsyncSession(**session_args)

    async def fetch_text(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> str:
        return (await self.fetch_response(url, headers=headers)).text

    async def fetch_response(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        allow_not_modified: bool = False,
    ) -> FetchResponse:
        try:
            url = normalize_url(url)
        except ValueError as exc:
            raise FetchError(str(exc), permanent=True) from exc
        origin = normalize_origin(url)
        async with self._origin_locks[origin], self._semaphore:
            await self._wait_for_start(origin)
            try:
                resp = await self._session.get(url, headers=headers)
            except RequestException as exc:
                raise FetchError(f"GET {url} failed: {exc}") from exc

        status = resp.status_code
        if status == 304 and allow_not_modified:
            return FetchResponse(b"", _headers(resp.headers), status, str(resp.url))
        if status != 200:
            permanent = 400 <= status < 500 and status not in (408, 425, 429)
            raise FetchError(
                f"GET {url} returned {status}",
                status=status,
                permanent=permanent,
            )
        content_type = resp.headers.get("content-type", "").lower()
        if content_type and not any(
            marker in content_type for marker in _TEXT_CONTENT_MARKERS
        ):
            raise FetchError(
                f"GET {url}: unsupported content-type {content_type!r}",
                status=status,
                permanent=True,
            )
        try:
            final_url = normalize_url(str(resp.url))
        except ValueError as exc:
            raise FetchError(
                f"GET {url} redirected to an unsafe URL",
                status=status,
                permanent=True,
            ) from exc
        return FetchResponse(
            bytes(resp.content),
            _headers(resp.headers),
            status,
            final_url,
        )

    async def _wait_for_start(self, origin: str) -> None:
        # Politeness is per-origin: the origin lock serializes same-host requests
        # and this delay spaces their starts. Distinct origins are bounded only by
        # the concurrency semaphore, so the crawl fans out across many small sites.
        now = time.monotonic()
        origin_wait = self._delay - (now - self._last_origin_start.get(origin, 0.0))
        if origin_wait > 0:
            await asyncio.sleep(origin_wait)
        self._last_origin_start[origin] = time.monotonic()

    async def aclose(self) -> None:
        await self._session.close()

    async def __aenter__(self) -> "Fetcher":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _headers(headers: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if value is not None
    }


def create_fetcher(settings: Settings) -> Fetcher:
    return Fetcher(
        user_agent=settings.user_agent,
        timeout=settings.fetch_timeout_seconds,
        delay_seconds=settings.fetch_delay_seconds,
        concurrency=settings.fetch_concurrency,
        impersonate=settings.fetch_impersonate,
    )
