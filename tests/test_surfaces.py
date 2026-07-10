from semsearch.models import SearchResult
from typer.testing import CliRunner

from semsearch.cli import app
from semsearch.web.app import templates


def test_cli_is_admin_only():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert " Search from the terminal" not in result.stdout
    assert "init-db" in result.stdout
    assert "site" in result.stdout


def test_web_template_shows_rrf_and_native_scores_without_styling():
    template = templates.get_template("index.html")
    html = template.render(
        q="query",
        error=None,
        results=[
            SearchResult(
                page_id=1,
                url="https://example.com/post",
                title="Post",
                snippet="Snippet",
                scores={"dense": 0.8765, "rrf": 0.01234567},
            )
        ],
    )

    assert "rrf 0.012346" in html
    assert "dense 0.876" in html
    assert "<style" not in html
    assert "stylesheet" not in html
