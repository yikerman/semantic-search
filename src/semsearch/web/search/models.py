from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Protocol, Self

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


class Scored(Protocol):
    @property
    def scores(self) -> Mapping[str, float]: ...

    def with_scores(self, scores: Mapping[str, float]) -> Self: ...


@dataclass(frozen=True, slots=True)
class RankedRun[T]:
    name: str
    weight: float
    candidates: tuple[T, ...]


def make_run[T: Scored](
    name: str, weight: float, scored: Iterable[tuple[T, float]]
) -> RankedRun[T]:
    ranked = sorted(
        (
            (candidate.with_scores({**candidate.scores, name: score}), score)
            for candidate, score in scored
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return RankedRun(name, weight, tuple(candidate for candidate, _ in ranked))


type Retriever = Callable[
    [RetrievalRequest, AsyncConnectionPool],
    Awaitable[RankedRun[ChunkCandidate]],
]
type Reranker = Callable[
    [str, Sequence[PageCandidate]], Awaitable[RankedRun[PageCandidate]]
]
type Fusion = Callable[[Sequence[RankedRun[PageCandidate]]], list[PageCandidate]]
