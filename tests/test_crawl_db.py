from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from semsearch.cli import db


class Cursor:
    def __init__(self, row=None, *, rows=None, rowcount: int = 1) -> None:
        self.row = row
        self.rows = rows or []
        self.rowcount = rowcount

    async def fetchone(self):
        return self.row

    async def fetchall(self):
        return self.rows


async def test_delete_site_configs_deletes_origins_in_one_statement():
    class DeleteConnection:
        def __init__(self) -> None:
            self.query = ""
            self.params = ()

        async def execute(self, query, params):
            self.query = query
            self.params = params
            return Cursor(rows=[("https://b.example",), ("https://a.example",)])

    conn = DeleteConnection()

    removed = await db.delete_site_configs(
        cast(Any, conn),
        base_urls=("https://a.example", "https://b.example"),
    )

    assert "DELETE FROM sites" in conn.query
    assert "RETURNING base_url" in conn.query
    assert conn.params == (["https://a.example", "https://b.example"],)
    assert removed == ["https://a.example", "https://b.example"]


class ChunkCursor(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.query = ""
        self.rows = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def executemany(self, query, rows):
        self.query = query
        self.rows = rows


class ChunkConnection:
    def __init__(self) -> None:
        self.cur = ChunkCursor()

    def cursor(self):
        return self.cur

    async def execute(self, query, params):
        raise AssertionError("append-only chunk insertion must not delete")


async def test_insert_page_chunks_never_replaces_existing_chunks():
    conn = ChunkConnection()

    await db.insert_page_chunks(
        cast(Any, conn),
        page_id=3,
        chunks=[db.ChunkInsert(4, "content", (1.0, 0.0))],
    )

    assert len(conn.cur.rows) == 1
    assert "(page_id, start_offset, content_length, embedding, search_vector)" in (
        conn.cur.query
    )
    assert "tokenize(%s, 'semsearch_llmlingua2')::bm25vector" in conn.cur.query
    assert conn.cur.rows[0][1:3] == (4, 7)
    assert conn.cur.rows[0][-1] == "content"


class PageInsertConnection:
    def __init__(self) -> None:
        self.query = ""
        self.params: tuple[object, ...] = ()

    async def execute(self, query, params):
        self.query = query
        self.params = params
        return Cursor((9,))


async def test_insert_page_stores_canonical_content():
    conn = PageInsertConnection()

    page_id = await db.insert_page(
        cast(Any, conn),
        site_id=2,
        url="https://example.com/post",
        title="Post",
        content="canonical article body",
        published_at=None,
        language="en",
    )

    assert page_id == 9
    assert "(site_id, url, title, content, published_at, language, fetched_at)" in (
        conn.query
    )
    assert conn.params[3] == "canonical article body"
