from dataclasses import dataclass
from typing import Literal

IndexStatus = Literal["indexed", "skipped", "no_content", "error"]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_index: int
    content: str
    char_count: int


@dataclass(slots=True)
class IndexOutcome:
    url: str
    status: IndexStatus
    detail: str = ""
    chunk_count: int = 0
