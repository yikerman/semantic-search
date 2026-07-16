from dataclasses import dataclass
from datetime import datetime, timezone

import py3langid
import trafilatura
from trafilatura.settings import Document

MIN_TEXT_CHARS = 150
LANGUAGE_SAMPLE_CHARS = 1600


def detect_language(text: str, *, title: str | None = None) -> str:
    sample = text[:LANGUAGE_SAMPLE_CHARS]
    if title:
        sample = f"{title}\n\n{sample}"
    language, _score = py3langid.classify(sample)
    if len(language) != 2 or not language.isascii() or not language.isalpha():
        raise ValueError(f"invalid language code from py3langid: {language!r}")
    return language.lower()


@dataclass(slots=True)
class ExtractedPage:
    title: str | None
    text: str
    published_at: datetime | None
    language: str


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
        language=detect_language(text, title=doc.title),
    )


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
