from contextlib import AbstractAsyncContextManager
from typing import Any, cast

import httpx
import semsearch.cli.app as cli_module
from typer.testing import CliRunner

from semsearch.share.config import Settings
from semsearch.share.status import FailedCrawlJob, IndexStats
from semsearch.web.app import create_app, prepare_display, templates
from semsearch.web.search.models import Candidate


def test_cli_is_admin_only():
    result = CliRunner().invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0
    assert " Search from the terminal" not in result.stdout
    assert "init-db" in result.stdout
    assert "site" in result.stdout


def test_removed_bulk_and_index_commands_are_not_exposed():
    runner = CliRunner()

    site_help = runner.invoke(cli_module.app, ["site", "--help"])
    poll_help = runner.invoke(cli_module.app, ["site", "poll", "--help"])
    root_help = runner.invoke(cli_module.app, ["--help"])

    assert site_help.exit_code == 0
    assert "index" not in site_help.stdout
    assert "--all" not in poll_help.stdout
    assert "--force" not in poll_help.stdout
    assert "worker" in root_help.stdout


def test_web_template_shows_rrf_and_native_scores_without_styling():
    template = templates.get_template("index.html")
    html = template.render(
        q="query",
        error=None,
        results=[
            Candidate(
                chunk_id=1,
                page_id=1,
                url="https://example.com/post",
                title="Post",
                content="Snippet",
                scores={"dense": 0.8765, "rrf": 0.01234567},
            )
        ],
    )

    assert "rrf 0.012346" in html
    assert "dense 0.876" in html
    assert "<style" not in html
    assert "stylesheet" not in html


class FakeConnection(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class FakePool:
    def connection(self):
        return FakeConnection()


async def test_prepare_display_swaps_matched_chunk_for_lead_chunk(monkeypatch):
    async def lead(conn, *, page_ids):
        assert page_ids == [1, 2]
        return {1: "Lead chunk"}

    monkeypatch.setattr("semsearch.web.app.fetch_lead_chunks", lead)
    results = [
        Candidate(
            chunk_id=5,
            page_id=1,
            url="https://example.com/a",
            title="A",
            content="mid-article window",
            scores={"rrf": 0.5},
        ),
        Candidate(
            chunk_id=9,
            page_id=2,
            url="https://example.com/b",
            title="B",
            content="matched fallback",
            scores={"rrf": 0.4},
        ),
    ]

    displayed = await prepare_display(cast(Any, FakePool()), results)

    assert [result.content for result in displayed] == [
        "Lead chunk",
        "matched fallback",
    ]
    assert displayed[0].scores["rrf"] == 0.5


async def test_status_page_mirrors_cli_status(monkeypatch):
    async def stats(conn):
        return IndexStats(
            site_count=5,
            page_count=100,
            chunk_count=400,
            queued_count=20,
            retrying_count=3,
            failed_count=2,
        )

    async def failures(conn):
        return [FailedCrawlJob("https://example.com/gone", 3, "GET returned 404")]

    monkeypatch.setattr("semsearch.web.app.fetch_index_stats", stats)
    monkeypatch.setattr("semsearch.web.app.list_failed_jobs", failures)
    monkeypatch.setattr(
        "semsearch.web.app.get_settings",
        lambda: Settings(embedding_model="test-model", embedding_dim=8),
    )
    app = create_app()
    app.state.pool = FakePool()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/status")

    assert response.status_code == 200
    assert "<th>pages</th><td>100</td>" in response.text
    assert "<th>retrying</th><td>3</td>" in response.text
    assert "https://example.com/gone (3 attempts): GET returned 404" in response.text
    assert "test-model (8 dims)" in response.text
    assert "<style" not in response.text
