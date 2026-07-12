from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Site:
    id: int
    base_url: str
    sitemap_url: str | None
    feed_url: str
    last_polled_at: datetime | None
    next_poll_at: datetime | None
    feed_etag: str | None
    feed_last_modified: str | None
    poll_failures: int
    sync_error: str | None
    history_pending: bool
    history_error: str | None


@dataclass(frozen=True, slots=True)
class CrawlJob:
    id: int
    site_id: int
    url: str
    source: str
    attempt_count: int
    lease_token: UUID
