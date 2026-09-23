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


async def test_index_publication_updates_only_unindexed_pages():
    conn = PageInsertConnection()
    await db.publish_page_index(
        cast(Any, conn), page_id=3, text="Title\n\nWhole post", embedding=(1.0, 0.0)
    )
    assert "UPDATE pages SET embedding = %s" in conn.query
    assert "tokenize(%s, 'semsearch_llmlingua2')::bm25vector" in conn.query
    assert "indexed_at = now(), index_error = NULL" in conn.query
    assert "WHERE id = %s AND indexed_at IS NULL AND NOT index_rejected" in conn.query
    assert conn.params[1:] == ("Title\n\nWhole post", 3)


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
