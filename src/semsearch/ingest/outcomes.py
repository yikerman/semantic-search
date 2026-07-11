from collections.abc import Awaitable, Callable, Iterable

from semsearch.ingest.models import IndexOutcome

IndexOne = Callable[[str], Awaitable[IndexOutcome]]


async def index_url_outcome(url: str, index_one: IndexOne) -> IndexOutcome:
    try:
        return await index_one(url)
    except Exception as exc:  # noqa: BLE001
        return IndexOutcome(url, "error", str(exc))


async def collect_index_outcomes(
    urls: Iterable[str],
    index_one: IndexOne,
    *,
    on_progress: Callable[[IndexOutcome], None] | None = None,
) -> list[IndexOutcome]:
    outcomes: list[IndexOutcome] = []
    for url in urls:
        outcome = await index_url_outcome(url, index_one)
        outcomes.append(outcome)
        if on_progress is not None:
            on_progress(outcome)
    return outcomes
