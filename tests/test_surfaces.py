import semsearch.cli.app as cli_module
from typer.testing import CliRunner

from semsearch.web.app import templates
from semsearch.web.search.models import Candidate


def test_cli_is_admin_only():
    result = CliRunner().invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0
    assert " Search from the terminal" not in result.stdout
    assert "init-db" in result.stdout
    assert "site" in result.stdout


def test_site_poll_validates_selector_before_opening_services(monkeypatch):
    def fail_if_opened():
        raise AssertionError("services opened")

    monkeypatch.setattr(cli_module, "open_services", fail_if_opened)
    runner = CliRunner()

    missing = runner.invoke(cli_module.app, ["site", "poll"])
    conflicting = runner.invoke(
        cli_module.app,
        ["site", "poll", "--site", "https://example.com", "--all"],
    )

    assert missing.exit_code == 1
    assert "Pass a site origin or --all" in missing.output
    assert conflicting.exit_code == 1
    assert "Use either a site origin or --all" in conflicting.output


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
