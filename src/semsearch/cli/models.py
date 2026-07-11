from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Site:
    id: int
    base_url: str
    sitemap_url: str | None
    feed_url: str | None
    last_indexed_at: datetime | None
    last_polled_at: datetime | None
    feed_etag: str | None
    feed_last_modified: str | None
