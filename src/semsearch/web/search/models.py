from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType

from psycopg_pool import AsyncConnectionPool

from semsearch.web.search.filters import SearchFilter


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    chunk_id: int
    page_id: int
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def with_scores(self, scores: Mapping[str, float]) -> "ChunkCandidate":
        return replace(self, scores=scores)


@dataclass(frozen=True, slots=True)
class PageCandidate:
    page_id: int
    url: str
    title: str | None
    content: str
    published_at: datetime | None = None
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def with_scores(self, scores: Mapping[str, float]) -> "PageCandidate":
        return replace(self, scores=scores)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    query_embedding: tuple[float, ...]
    filters: tuple[SearchFilter, ...]
    limit: int


@dataclass(frozen=True, slots=True)
class RankedRun[T]:
    name: str
    candidates: tuple[T, ...]


type Retriever = Callable[
    [RetrievalRequest, AsyncConnectionPool],
    Awaitable[RankedRun[ChunkCandidate]],
]
type Reranker = Callable[
    [str, Sequence[PageCandidate]], Awaitable[RankedRun[PageCandidate]]
]
type Fusion = Callable[[Sequence[RankedRun[PageCandidate]]], list[PageCandidate]]
