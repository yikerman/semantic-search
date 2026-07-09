import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


async def map_concurrently(
    items: Iterable[ItemT],
    *,
    limit: int,
    func: Callable[[ItemT], Awaitable[ResultT]],
) -> list[ResultT]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(item: ItemT) -> ResultT:
        async with semaphore:
            return await func(item)

    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return list(await asyncio.gather(*tasks))
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
