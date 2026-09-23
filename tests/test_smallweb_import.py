import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from semsearch.share.config import Settings


@pytest.fixture
def importer():
    path = Path(__file__).parents[1] / "scripts" / "import_smallweb.py"
    spec = importlib.util.spec_from_file_location("import_smallweb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feed_selection_normalizes_deduplicates_and_rejects_invalid_urls(importer):
    feeds = importer.select_feeds(
        """
        # Blog feeds
        https://EXAMPLE.com:443/blog/feed.xml#top
        https://example.com/other-feed
        https://second.example/feed?format=atom
        ftp://invalid.example/feed
        https://127.0.0.1/feed
        not a URL
        <html>server error</html>
        """
    )
    assert feeds == {
        "https://example.com": "https://example.com/blog/feed.xml",
        "https://second.example": "https://second.example/feed?format=atom",
    }


@pytest.mark.parametrize("text", ["", "# comment", "<html>error</html>"])
def test_empty_or_invalid_list_fails(importer, text):
    with pytest.raises(ValueError, match="no valid feed URLs"):
        importer.select_feeds(text)


@pytest.mark.parametrize("refresh_existing", [False, True])
async def test_import_skips_existing_unless_refresh_requested(
    importer, monkeypatch, refresh_existing
):
    pool = object()
    registered = []

    @asynccontextmanager
    async def create_pool(settings):
        yield pool

    async def list_sites(conn):
        assert conn is pool
        return [SimpleNamespace(base_url="https://existing.example")]

    async def add_site(conn, start, sitemap, feed):
        assert conn is pool
        registered.append((start, sitemap, feed))
        if "broken" in start:
            raise ValueError("failed registration")

    monkeypatch.setattr(importer, "create_pool", create_pool)
    monkeypatch.setattr(importer, "list_sites", list_sites)
    monkeypatch.setattr(importer, "add_site", add_site)
    feeds = {
        f"https://{name}.example": f"https://{name}.example/feed"
        for name in ("existing", "new", "broken")
    }
    result = await importer.import_feeds(
        Settings(), feeds, concurrency=2, refresh_existing=refresh_existing
    )
    assert result == 1
    expected = {"new", "broken", "existing"} if refresh_existing else {"new", "broken"}
    assert set(registered) == {
        (f"https://{name}.example/", "auto", f"https://{name}.example/feed")
        for name in expected
    }


@pytest.mark.parametrize("dry_run", [False, True])
async def test_main_fetches_only_source_and_applies_limit(
    importer, monkeypatch, dry_run
):
    args = SimpleNamespace(
        source_url=importer.DEFAULT_SOURCE_URL,
        concurrency=2,
        limit=1,
        refresh_existing=False,
        dry_run=dry_run,
    )
    imported = []

    def respond(request):
        assert str(request.url) == args.source_url
        return httpx.Response(
            200,
            content=b"\xef\xbb\xbfhttps://one.example/feed\nhttps://two.example/rss\n",
        )

    async def import_feeds(settings, feeds, **kwargs):
        imported.append(feeds)
        return 0

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(importer.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(importer, "parse_args", lambda: args)
    monkeypatch.setattr(importer, "import_feeds", import_feeds)
    assert await importer.main() == 0
    assert imported == (
        [] if dry_run else [{"https://one.example": "https://one.example/feed"}]
    )


async def test_source_http_failure_never_imports(importer, monkeypatch):
    args = SimpleNamespace(source_url=importer.DEFAULT_SOURCE_URL)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    )
    monkeypatch.setattr(importer.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(importer, "parse_args", lambda: args)
    with pytest.raises(httpx.HTTPStatusError):
        await importer.main()
