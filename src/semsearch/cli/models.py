from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    id: int
    base_url: str
    start_url: str
    sitemap_url: str
    feed_url: str
    history_complete: bool = False
    last_crawled_at: datetime | None = None
    last_error: str | None = None


class IndexPage(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    id: int
    title: str | None
    content: str
