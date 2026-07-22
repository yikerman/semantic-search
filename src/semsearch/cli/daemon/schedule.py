from typing import cast
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from semsearch.cli.db import SITE_COLUMNS
from semsearch.cli.models import Site


async def scatter_poll_schedule(
    conn: psycopg.AsyncConnection, *, interval_seconds: int
) -> None:
    await conn.execute(
        """
        WITH overdue AS (
            SELECT id,
                   row_number() OVER (ORDER BY id) - 1 AS position,
                   count(*) OVER () AS total
            FROM sites
            WHERE next_poll_at IS NULL OR next_poll_at <= now()
        )
        UPDATE sites
        SET next_poll_at = now() + make_interval(
            secs => (%s * overdue.position / GREATEST(overdue.total, 1))::int
        )
        FROM overdue
        WHERE sites.id = overdue.id
        """,
        (interval_seconds,),
    )


async def claim_due_site(
    conn: psycopg.AsyncConnection, *, lease_seconds: int = 600
) -> tuple[Site, UUID] | None:
    token = uuid4()
    cur = conn.cursor(row_factory=dict_row)
    await cur.execute(
        f"""
        WITH candidate AS (
            SELECT id
            FROM sites
            WHERE next_poll_at <= now()
              AND (poll_lease_until IS NULL OR poll_lease_until < now())
            ORDER BY next_poll_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE sites
        SET poll_lease_until = now() + make_interval(secs => %s),
            poll_lease_token = %s
        FROM candidate
        WHERE sites.id = candidate.id
        RETURNING {SITE_COLUMNS}, sites.poll_lease_token
        """,
        (lease_seconds, token),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    granted_token = cast(UUID, row.pop("poll_lease_token"))
    return Site(**row), granted_token


async def renew_poll_lease(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    lease_seconds: int = 600,
) -> bool:
    cur = await conn.execute(
        """
        UPDATE sites
        SET poll_lease_until = now() + make_interval(secs => %s)
        WHERE id = %s AND poll_lease_token = %s
        """,
        (lease_seconds, site_id, lease_token),
    )
    return cur.rowcount == 1


async def mark_poll_succeeded(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    etag: str | None,
    modified: str | None,
    interval_seconds: int,
    lease_token: UUID,
    sync_error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET last_polled_at = now(),
            next_poll_at = now() + make_interval(secs => %s),
            feed_etag = COALESCE(%s, feed_etag),
            feed_last_modified = COALESCE(%s, feed_last_modified),
            poll_failures = 0,
            poll_lease_until = NULL,
            poll_lease_token = NULL,
            sync_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (interval_seconds, etag, modified, sync_error, site_id, lease_token),
    )


async def mark_poll_failed(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    error: str,
    interval_seconds: int,
) -> None:
    # Backoff caps at the healthy poll interval so a failing feed is never
    # polled more often than a working one.
    await conn.execute(
        """
        UPDATE sites
        SET poll_failures = poll_failures + 1,
            next_poll_at = now() + make_interval(
                secs => LEAST(%s, 300 * (2 ^ LEAST(poll_failures, 10)))
            ),
            poll_lease_until = NULL,
            poll_lease_token = NULL,
            sync_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (interval_seconds, error, site_id, lease_token),
    )


async def mark_history_pending(
    conn: psycopg.AsyncConnection, *, site_id: int, lease_token: UUID
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET history_pending = true, history_error = NULL
        WHERE id = %s AND poll_lease_token = %s
        """,
        (site_id, lease_token),
    )


async def finish_history(
    conn: psycopg.AsyncConnection,
    *,
    site_id: int,
    lease_token: UUID,
    error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE sites
        SET history_pending = false, history_error = %s
        WHERE id = %s AND poll_lease_token = %s
        """,
        (error, site_id, lease_token),
    )
