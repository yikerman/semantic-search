from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, cast

import psycopg
import pytest

from semsearch.cli import db
from semsearch.cli.crawl import store
from semsearch.cli.index import index_page
from semsearch.cli.ingest.chunk import Chunk
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
            "chunks": [],
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
        if "SET indexed_at" in query:
            if self.fail_publication:
                raise psycopg.OperationalError("lost database")
            self.data["indexed"] = True
        if "SET index_error" in query:
            self.data["index_error"] = params[0]
        return Cursor()


@pytest.fixture
def database(monkeypatch):
    database = Database()

    async def insert_page(conn, **kwargs):
        assert conn.transaction_open
        conn.data["pages"].append(kwargs)
        return 1

    async def insert_chunks(conn, **kwargs):
        assert conn.transaction_open
        conn.data["chunks"].extend(kwargs["chunks"])

    monkeypatch.setattr(db, "insert_page", insert_page)
    monkeypatch.setattr(db, "insert_page_chunks", insert_chunks)
    return database


async def embed(texts):
    return [[1.0, 0.0] for _ in texts]


def chunks(text):
    return [Chunk(0, text)]


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

    async def broken(texts):
        raise EmbeddingError("provider unavailable")

    assert not await index_page(cast(Any, database), page, chunks, broken)
    assert database.data["pages"] == [page]
    assert database.data["chunks"] == []
    assert not database.data["indexed"]
    assert database.data["index_error"] == "provider unavailable"
    assert await index_page(cast(Any, database), page, chunks, embed)
    assert database.data["indexed"]
    assert len(database.data["chunks"]) == 1


async def test_failed_index_transaction_never_publishes_partial_chunks(database):
    page = IndexPage(id=1, title=None, content="Body")
    database.fail_publication = True
    with pytest.raises(psycopg.OperationalError):
        await index_page(cast(Any, database), page, chunks, embed)
    assert database.data["chunks"] == []
    assert not database.data["indexed"]


async def test_chunk_budget_is_checked_before_embedding(database):
    async def forbidden(texts):
        pytest.fail("oversized article must not reach provider")

    page = IndexPage(id=1, title=None, content="Body")
    assert not await index_page(
        cast(Any, database), page, lambda text: [Chunk(0, text)] * 2001, forbidden
    )
    assert "chunk limit" in database.data["index_error"]
