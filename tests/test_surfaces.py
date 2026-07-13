from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import semsearch.cli.app as cli_module
from typer.testing import CliRunner

from semsearch.share.config import Settings
from semsearch.share.status import IndexStats
from semsearch.web.app import create_app, prepare_display, templates
from semsearch.web.db import RecentActivity
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


def test_web_template_shows_scores_with_shared_semantic_structure():
    template = templates.get_template("index.html")
    html = template.render(
        active_page="search",
        q="query",
        error="Embedding service unavailable",
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
    assert '<link rel="stylesheet" href="/static/style.css">' in html
    assert '<nav aria-label="Primary">' in html
    assert html.count('aria-current="page"') == 1
    assert ">Search</a>" in html
    assert '<form class="search-form" action="/" method="get" role="search">' in html
    assert '<label class="visually-hidden" for="query">' in html
    assert '<p class="error" role="alert">' in html
    assert '<article class="result">' in html
    assert "<style" not in html


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


async def test_status_page_shows_recent_activity(monkeypatch):
    async def stats(conn):
        return IndexStats(
            site_count=5,
            page_count=100,
            chunk_count=400,
            queued_count=20,
            retrying_count=3,
            failed_count=2,
        )

    async def activity(conn):
        return [
            RecentActivity(
                "https://example.com/new",
                "success",
                datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
                None,
                None,
            ),
            RecentActivity(
                "https://example.com/gone",
                "failure",
                datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
                3,
                "GET returned 404",
            ),
        ]

    monkeypatch.setattr("semsearch.web.app.fetch_index_stats", stats)
    monkeypatch.setattr("semsearch.web.app.list_recent_activity", activity)
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
    assert response.text.count('aria-current="page"') == 1
    assert ">Status</a>" in response.text
    assert "<caption>Index totals</caption>" in response.text
    assert '<th scope="row">pages</th>' in response.text
    assert "<td>100</td>" in response.text
    assert '<th scope="row">retrying</th>' in response.text
    assert "<td>3</td>" in response.text
    assert "Recent activity" in response.text
    assert "Recent failures" not in response.text
    assert '<strong class="activity-status">success</strong>' in response.text
    assert '<strong class="activity-status">failure</strong>' in response.text
    assert "https://example.com/new" in response.text
    assert "3 attempts &middot; GET returned 404" in response.text
    assert 'datetime="2026-07-13T10:00:00+00:00"' in response.text
    assert "test-model (8 dims)" in response.text
    assert '<footer class="status-footer">' in response.text
    assert "<style" not in response.text
