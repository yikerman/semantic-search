from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

import httpx
import semsearch.cli.app as cli_module
from typer.testing import CliRunner

from semsearch.share.config import Settings
from semsearch.share.status import IndexStats
from semsearch.web.app import (
    DisplayResult,
    create_app,
    prepare_display,
    prepare_language_options,
    templates,
)
from semsearch.web.db import RecentActivity
from semsearch.web.search.models import PageCandidate


def test_cli_is_admin_only():
    result = CliRunner().invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0
    assert " Search from the terminal" not in result.stdout
    assert "init-db" in result.stdout
    assert "site" in result.stdout


def test_removed_commands_are_not_exposed_and_daemon_is_canonical():
    runner = CliRunner()

    site_help = runner.invoke(cli_module.app, ["site", "--help"])
    poll_help = runner.invoke(cli_module.app, ["site", "poll", "--help"])
    root_help = runner.invoke(cli_module.app, ["--help"])

    assert site_help.exit_code == 0
    assert "index" not in site_help.stdout
    assert poll_help.exit_code != 0
    assert "daemon" in root_help.stdout
    assert "worker" not in root_help.stdout


def test_web_template_shows_scores_with_shared_semantic_structure():
    template = templates.get_template("index.html")
    html = template.render(
        active_page="search",
        q="query",
        encourage_long_content=True,
        lang="fr",
        languages=["en", "fr"],
        error="Embedding service unavailable",
        results=[
            DisplayResult(
                page_id=1,
                url="https://example.com/post",
                title="Post",
                snippet="Snippet",
                is_truncated=False,
                scores={
                    "dense": 0.8765,
                    "length": 12_345,
                    "rrf": 0.01234567,
                },
            )
        ],
    )

    assert "rrf 0.012346" in html
    assert "dense 0.876" in html
    assert "length 12,345 chars" in html
    assert '<link rel="stylesheet" href="/static/style.css">' in html
    assert '<nav aria-label="Primary">' in html
    assert html.count('aria-current="page"') == 1
    assert ">Search</a>" in html
    assert '<form class="search-form" action="/" method="get" role="search">' in html
    assert '<label class="visually-hidden" for="query">' in html
    assert 'name="encourage_long_content" value="true"' in html
    assert '<select id="language" name="lang">' in html
    assert '<option value="fr" selected>fr</option>' in html
    assert "checked" in html
    assert '<p class="error" role="alert">' in html
    assert '<article class="result">' in html
    assert "<style" not in html


def test_language_options_are_sorted_and_preserve_unknown_selection():
    options = prepare_language_options(["fr", "en"], selected="zz")

    assert options == ["en", "fr", "zz"]


class FakeConnection(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


class FakePool:
    def connection(self):
        return FakeConnection()


def test_prepare_display_extracts_bounded_page_lead():
    results = [
        PageCandidate(
            page_id=1,
            url="https://example.com/a",
            title="A",
            content="x" * 501,
            scores={"rrf": 0.5},
        ),
        PageCandidate(
            page_id=2,
            url="https://example.com/b",
            title="B",
            content="short page",
            scores={"rrf": 0.4},
        ),
    ]

    displayed = prepare_display(results)

    assert displayed[0].snippet == "x" * 500
    assert displayed[0].is_truncated is True
    assert displayed[1].snippet == "short page"
    assert displayed[1].is_truncated is False
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
