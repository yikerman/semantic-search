from dataclasses import dataclass
from datetime import datetime

import trafilatura
from trafilatura.settings import Document

MIN_TEXT_CHARS = 150


@dataclass(slots=True)
class ExtractedPage:
    title: str | None
    text: str
    published_at: datetime | None


def extract_page(html: str, url: str) -> ExtractedPage | None:
    doc = trafilatura.bare_extraction(html, url=url, with_metadata=True)
    if not isinstance(doc, Document):
        return None
    text = (doc.text or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return ExtractedPage(
        title=doc.title or None,
        text=text,
        published_at=_parse_date(doc.date),
    )


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
