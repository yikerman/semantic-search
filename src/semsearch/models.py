from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_index: int
    content: str
    char_count: int


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk_id: int
    page_id: int
    url: str
    title: str | None
    content: str
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def with_scores(self, scores: Mapping[str, float]) -> "Candidate":
        return replace(self, scores=scores)


@dataclass(frozen=True, slots=True)
class SearchResult:
    page_id: int
    url: str
    title: str | None
    snippet: str
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
