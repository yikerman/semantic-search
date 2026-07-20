from contextlib import AbstractAsyncContextManager
from typing import Any, cast

import pytest

from semsearch.cli.sites import SiteError, remove_site


class FakeConnection(AbstractAsyncContextManager):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def transaction(self):
        return self


class FakePool:
    def connection(self):
        return FakeConnection()


async def test_remove_site_normalizes_url_to_origin(monkeypatch):
    seen: list[list[str]] = []

    async def delete(conn, *, base_urls):
        seen.append(list(base_urls))
        return list(base_urls)

    monkeypatch.setattr("semsearch.cli.sites.db.delete_site_configs", delete)

    removed = await remove_site(
        cast(Any, FakePool()), "HTTPS://Example.COM:443/posts/one"
    )

    assert removed == "https://example.com"
    assert seen == [["https://example.com"]]


async def test_remove_site_rejects_missing_origin(monkeypatch):
    async def delete(conn, *, base_urls):
        return []

    monkeypatch.setattr("semsearch.cli.sites.db.delete_site_configs", delete)

    with pytest.raises(SiteError, match="Site is not configured"):
        await remove_site(cast(Any, FakePool()), "https://missing.example/path")
