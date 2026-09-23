import asyncio
import socket
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from ipaddress import ip_address
from typing import Any

from scrapy import Request
from scrapy.crawler import Crawler
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Response
from scrapy.resolver import CachingThreadedResolver, dnscache
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure

from semsearch.cli.crawl import store
from semsearch.cli.url import normalize_origin, normalize_url, same_site


class DestinationRejected(IgnoreRequest):
    pass


class OriginDeferred(IgnoreRequest):
    pass


class PublicResolver(CachingThreadedResolver):
    """Validate the address returned to the actual HTTP connection, not a preflight DNS lookup."""

    def getHostByName(self, name: str, timeout: Sequence[int] = ()) -> Deferred[str]:
        async def resolve() -> str:
            if name in dnscache:
                return public_address(dnscache[name])
            # AsyncCrawlerRunner shares our asyncio loop; do not depend on an
            # independently started Twisted DNS thread pool.
            async with asyncio.timeout(self.timeout):
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    name, 0, family=socket.AF_INET, type=socket.SOCK_STREAM
                )
            checked = [public_address(str(entry[4][0])) for entry in addresses]
            if not checked:
                raise DestinationRejected("unresolvable_destination")
            if dnscache.limit:
                dnscache[name] = checked[0]
            return checked[0]

        return Deferred.fromFuture(asyncio.create_task(resolve()))


def public_address(address: str) -> str:
    if not ip_address(address).is_global:
        raise DestinationRejected("non_public_destination")
    return address


def retry_after(value: bytes | None, now: datetime) -> datetime | None:
    if not value:
        return None
    try:
        raw = value.decode("ascii").strip()
        if raw.isdigit():
            # Avoid datetime overflow from hostile headers.
            return now + timedelta(seconds=min(int(raw), 31_536_000))
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(now, parsed)
    except ValueError, OverflowError, UnicodeError, TypeError:
        return None


class CrawlPolicy:
    def __init__(self, crawler: Crawler):
        self.crawler = crawler

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> CrawlPolicy:
        return cls(crawler)

    def process_request(self, request: Request) -> None:
        try:
            normalize_url(request.url)
        except ValueError as exc:
            raise DestinationRejected(str(exc)) from exc
        scope = request.meta.get("scope")
        if scope and not same_site(request.url, scope):
            raise DestinationRejected("cross_site_redirect")
        # Use the current destination's slot even after a redirect.
        request.meta["download_slot"] = normalize_origin(request.url)
        runtime: Any = self.crawler.spider
        until = runtime.cooldowns.get(normalize_origin(request.url))
        if until is not None and until > datetime.now(UTC):
            raise OriginDeferred("origin_cooldown")

    async def process_response(self, request: Request, response: Response) -> Response:
        if response.status in (429, 503):
            now = datetime.now(UTC)
            until = retry_after(response.headers.get(b"Retry-After"), now)
            if until is not None and until > now:
                runtime: Any = self.crawler.spider
                origin = normalize_origin(request.url)
                runtime.cooldowns[origin] = max(
                    until, runtime.cooldowns.get(origin, until)
                )
                try:
                    await store.save_cooldown(
                        runtime.pool, origin, runtime.cooldowns[origin]
                    )
                except Exception as exc:
                    # Robots middleware can swallow downloader errors; record
                    # storage failures here so they still abort the batch.
                    runtime.fatal(Failure(exc))
                    raise
                raise OriginDeferred("retry_after")
        return response
