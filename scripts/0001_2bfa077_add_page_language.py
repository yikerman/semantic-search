#!/usr/bin/env python3
import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import logging

import psycopg
from psycopg import sql

from semsearch.cli.ingest.extract import detect_language
from semsearch.share.config import get_settings
from semsearch.share.logging import configure_logging

MIGRATION_NAME = "0001_2bfa077_add_page_language"
INDEX_NAME = "pages_language_idx"
DEFAULT_BATCH_SIZE = 1000

logger = logging.getLogger(MIGRATION_NAME)


@dataclass(frozen=True, slots=True)
class BackfillPage:
    id: int
    title: str | None
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add and backfill the detected page language."
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def page_from_row(row: Sequence[object]) -> BackfillPage:
    if len(row) != 3:
        raise ValueError("invalid language backfill database row")
    page_id, title, content = row
    if not isinstance(page_id, int):
        raise ValueError("invalid language backfill database row")
    if title is not None and not isinstance(title, str):
        raise ValueError("invalid language backfill database row")
    if not isinstance(content, str):
        raise ValueError("invalid language backfill database row")
    return BackfillPage(page_id, title, content)


def classify_pages(pages: Sequence[BackfillPage]) -> list[tuple[int, str]]:
    return [
        (page.id, detect_language(page.content, title=page.title)) for page in pages
    ]


def fetch_batch(
    conn: psycopg.Connection, *, after_id: int, batch_size: int
) -> list[BackfillPage]:
    rows = conn.execute(
        """
        SELECT p.id, p.title, c.content
        FROM pages AS p
        JOIN chunks AS c ON c.page_id = p.id AND c.chunk_index = 0
        WHERE p.language IS NULL AND p.id > %s
        ORDER BY p.id
        LIMIT %s
        """,
        (after_id, batch_size),
    ).fetchall()
    return [page_from_row(row) for row in rows]


def update_languages(
    conn: psycopg.Connection, detected: Sequence[tuple[int, str]]
) -> int:
    if not detected:
        return 0
    page_ids, languages = zip(*detected, strict=True)
    cur = conn.execute(
        """
        UPDATE pages AS p
        SET language = detected.language
        FROM unnest(%s::bigint[], %s::text[]) AS detected(id, language)
        WHERE p.id = detected.id AND p.language IS NULL
        """,
        (list(page_ids), list(languages)),
    )
    return cur.rowcount


def backfill_languages(conn: psycopg.Connection, *, batch_size: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    after_id = 0
    updated_total = 0
    while pages := fetch_batch(conn, after_id=after_id, batch_size=batch_size):
        detected = classify_pages(pages)
        updated_total += update_languages(conn, detected)
        counts.update(language for _page_id, language in detected)
        after_id = pages[-1].id
        logger.info("Backfilled %d pages through page id %d", updated_total, after_id)
    return counts


def ensure_language_index(conn: psycopg.Connection) -> None:
    row = conn.execute(
        """
        SELECT i.indisvalid
        FROM pg_class AS c
        JOIN pg_index AS i ON i.indexrelid = c.oid
        WHERE c.relnamespace = 'public'::regnamespace AND c.relname = %s
        """,
        (INDEX_NAME,),
    ).fetchone()
    if row is not None and row[0] is not True:
        logger.warning("Dropping invalid interrupted index %s", INDEX_NAME)
        conn.execute(
            sql.SQL("DROP INDEX CONCURRENTLY {}").format(sql.Identifier(INDEX_NAME))
        )
    conn.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS pages_language_idx
        ON pages (language)
        WHERE language IS NOT NULL
        """
    )


def run_migration(database_url: str, *, batch_size: int) -> Counter[str]:
    with psycopg.connect(database_url, autocommit=True) as conn:
        locked = conn.execute(
            "SELECT pg_try_advisory_lock(hashtext(%s))", (MIGRATION_NAME,)
        ).fetchone()
        if locked is None or locked[0] is not True:
            raise RuntimeError(f"migration {MIGRATION_NAME} is already running")
        try:
            conn.execute("ALTER TABLE pages ADD COLUMN IF NOT EXISTS language text")
            counts = backfill_languages(conn, batch_size=batch_size)
            ensure_language_index(conn)
            remaining = conn.execute(
                "SELECT count(*) FROM pages WHERE language IS NULL"
            ).fetchone()
            if remaining is not None and remaining[0]:
                logger.warning(
                    "%d pages still have no language; rerun after deploying the "
                    "updated worker",
                    remaining[0],
                )
            return counts
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (MIGRATION_NAME,))


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    counts = run_migration(settings.database_url, batch_size=args.batch_size)
    logger.info("Language migration complete: %s", dict(counts.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
