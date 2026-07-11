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
