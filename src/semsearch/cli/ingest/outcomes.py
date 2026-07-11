from collections.abc import Awaitable, Callable, Iterable
import logging

from semsearch.cli.ingest.models import IndexOutcome

logger = logging.getLogger(__name__)

IndexOne = Callable[[str], Awaitable[IndexOutcome]]


async def index_url_outcome(url: str, index_one: IndexOne) -> IndexOutcome:
    try:
        return await index_one(url)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to index %s", url)
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
