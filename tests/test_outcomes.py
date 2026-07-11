import logging

from semsearch.cli.ingest.models import IndexOutcome
from semsearch.cli.ingest.outcomes import collect_index_outcomes, index_url_outcome


async def test_index_url_outcome_converts_and_logs_page_error(caplog):
    async def index_one(url: str) -> IndexOutcome:
        raise RuntimeError("broken page")

    with caplog.at_level(logging.ERROR, logger="semsearch.cli.ingest.outcomes"):
        outcome = await index_url_outcome("https://example.com/bad", index_one)

    assert outcome == IndexOutcome("https://example.com/bad", "error", "broken page")
    record = caplog.records[-1]
    assert record.getMessage() == "Failed to index https://example.com/bad"
    assert record.exc_info is not None


async def test_collect_index_outcomes_reports_progress_in_order():
    async def index_one(url: str) -> IndexOutcome:
        return IndexOutcome(url, "skipped")

    progress: list[IndexOutcome] = []

    outcomes = await collect_index_outcomes(
        ["https://example.com/a", "https://example.com/b"],
        index_one,
        on_progress=progress.append,
    )

    assert outcomes == [
        IndexOutcome("https://example.com/a", "skipped"),
        IndexOutcome("https://example.com/b", "skipped"),
    ]
    assert progress == outcomes
