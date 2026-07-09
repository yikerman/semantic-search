from dataclasses import dataclass
from typing import Literal

IndexStatus = Literal["indexed", "skipped", "no_content", "error"]


@dataclass(slots=True)
class IndexOutcome:
    url: str
    status: IndexStatus
    detail: str = ""
    chunk_count: int = 0
