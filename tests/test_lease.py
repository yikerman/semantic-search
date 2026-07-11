import asyncio

import pytest

from semsearch.cli.ingest.lease import LeaseLostError, run_with_lease


async def test_lease_loss_cancels_in_flight_operation():
    cancelled = asyncio.Event()

    async def operation() -> None:
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def renew() -> bool:
        return False

    with pytest.raises(LeaseLostError, match="lease was lost"):
        await run_with_lease(operation, renew, interval_seconds=0.001)

    assert cancelled.is_set()


async def test_completed_operation_stops_heartbeat():
    renewals = 0

    async def operation() -> int:
        return 42

    async def renew() -> bool:
        nonlocal renewals
        renewals += 1
        return True

    assert await run_with_lease(operation, renew, interval_seconds=10) == 42
    assert renewals == 0
