import asyncio

from semsearch.share.util import map_concurrently


async def test_map_concurrently_limits_work_and_preserves_order():
    release = asyncio.Event()
    two_active = asyncio.Event()
    active = 0
    max_active = 0

    async def work(value: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            two_active.set()
        try:
            await release.wait()
            return value * 10
        finally:
            active -= 1

    task = asyncio.create_task(map_concurrently([1, 2, 3], limit=2, func=work))

    await asyncio.wait_for(two_active.wait(), timeout=1)
    release.set()
    results = await task

    assert max_active == 2
    assert results == [10, 20, 30]
