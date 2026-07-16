import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
type RenewLease = Callable[[], Awaitable[bool]]


class LeaseLostError(RuntimeError):
    pass


async def run_with_lease(
    operation: Callable[[], Awaitable[T]],
    renew: RenewLease,
    *,
    interval_seconds: float = 60.0,
) -> T:
    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            # A raised renew is a transient database hiccup, not proof the lease
            # is gone; the lease still has most of its duration left, so log and
            # retry on the next tick. Only a clean False (our token no longer
            # owns the row) means another claimant took over.
            try:
                still_ours = await renew()
            except Exception:
                logger.warning("Lease renewal failed; will retry", exc_info=True)
                continue
            if not still_ours:
                raise LeaseLostError("database lease was lost")

    work = asyncio.ensure_future(operation())
    lease = asyncio.create_task(heartbeat())
    try:
        done, _ = await asyncio.wait((work, lease), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        await lease
        raise LeaseLostError("database lease heartbeat stopped")
    finally:
        work.cancel()
        lease.cancel()
        await asyncio.gather(work, lease, return_exceptions=True)
