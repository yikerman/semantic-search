from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, cast

import psycopg
import pytest

from semsearch.cli import db
from semsearch.cli.crawl import store
from semsearch.cli.index import index_page
from semsearch.cli.ingest.document import DocumentTooLong
from semsearch.cli.ingest.extract import ExtractedPage
from semsearch.cli.models import IndexPage, Site
from semsearch.share.embeddings import EmbeddingError


class Cursor:
    async def fetchone(self):
        return (1,)


class Database:
    def __init__(self):
        self.data = {
            "pages": [],
            "embedding": None,
            "search_text": None,
            "index_rejected": False,
            "stored": False,
            "indexed": False,
            "index_error": None,
        }
        self.fail_publication = False
        self.transaction_open = False

    @asynccontextmanager
    async def connection(self):
        yield self

    @asynccontextmanager
    async def transaction(self):
        before = deepcopy(self.data)
        self.transaction_open = True
        try:
            yield self
        except BaseException:
            self.data = before
            raise
        finally:
            self.transaction_open = False

    async def execute(self, query, params: tuple[Any, ...] = ()):
        assert self.transaction_open
        if "UPDATE article_urls" in query:
            if self.fail_publication:
                raise psycopg.OperationalError("lost database")
            self.data["stored"] = True
        if "SET embedding" in query:
            self.data["embedding"] = params[0]
            self.data["search_text"] = params[1]
            if self.fail_publication:
                raise psycopg.OperationalError("lost database")
            self.data["indexed"] = True
            self.data["index_error"] = None
        if "SET index_error" in query:
            self.data["index_error"] = params[0]
            self.data["index_rejected"] = params[1]
        return Cursor()


@pytest.fixture
def database(monkeypatch):
    database = Database()

    async def insert_page(conn, **kwargs):
        assert conn.transaction_open
        conn.data["pages"].append(kwargs)
        return 1

    monkeypatch.setattr(db, "insert_page", insert_page)
    return database


async def embed(text):
    return [1.0, 0.0]


def validate(text):
    pass


async def test_storage_and_url_completion_are_atomic(database):
    site = Site(
        id=1,
        base_url="https://example.com",
        start_url="https://example.com/",
        feed_url="none",
        sitemap_url="auto",
    )
    page = ExtractedPage("Title", "Body", None, "en")
    database.fail_publication = True
    with pytest.raises(psycopg.OperationalError):
        await store.store_page(cast(Any, database), site, "https://example.com/a", page)
    assert database.data["pages"] == []
    assert not database.data["stored"]
    database.fail_publication = False
    await store.store_page(cast(Any, database), site, "https://example.com/a", page)
    assert database.data["stored"]
    assert database.data["pages"][0]["content"] == "Body"


async def test_indexing_failure_preserves_text_and_can_retry(database):
    page = IndexPage(id=1, title="Title", content="Canonical text")
    database.data["pages"].append(page)

    async def broken(text):
        raise EmbeddingError("provider unavailable")

    assert not await index_page(cast(Any, database), page, validate, broken)
    assert database.data["pages"] == [page]
    assert database.data["embedding"] is None
    assert not database.data["indexed"]
    assert database.data["index_error"] == "provider unavailable"
    assert not database.data["index_rejected"]
    assert await index_page(cast(Any, database), page, validate, embed)
    assert database.data["indexed"]
    assert database.data["embedding"].to_list() == [1.0, 0.0]
    assert database.data["search_text"] == "Title\n\nCanonical text"
    assert database.data["index_error"] is None


async def test_failed_index_transaction_never_publishes_partial_index(database):
    page = IndexPage(id=1, title=None, content="Body")
    database.fail_publication = True
    with pytest.raises(psycopg.OperationalError):
        await index_page(cast(Any, database), page, validate, embed)
    assert database.data["embedding"] is None
    assert not database.data["indexed"]


async def test_oversized_post_is_rejected_before_embedding(database):
    seen = []

    def reject(text):
        seen.append(text)
        raise DocumentTooLong("document exceeds input budget")

    async def forbidden(text):
        pytest.fail("oversized article must not reach provider")

    page = IndexPage(id=1, title="Title", content="Body")
    database.data["pages"].append(page)
    assert not await index_page(cast(Any, database), page, reject, forbidden)
    assert seen == ["Title\n\nBody"]
    assert database.data["index_rejected"]
    assert database.data["pages"] == [page]
    assert database.data["embedding"] is None


@pytest.mark.parametrize("title", [None, "Title"])
async def test_embeds_whole_post_once_without_truncation(database, title):
    page = IndexPage(
        id=1, title=title, content="Beginning. " + "Body. " * 1000 + "End."
    )
    expected = f"{title}\n\n{page.content}" if title else page.content
    validated = []
    requests = []

    async def record(text):
        requests.append(text)
        return [1.0, 0.0]

    assert await index_page(cast(Any, database), page, validated.append, record)
    assert validated == [expected]
    assert requests == [expected]
    assert database.data["search_text"] == expected
