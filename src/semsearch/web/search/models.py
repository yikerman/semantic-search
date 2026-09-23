from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType

from psycopg_pool import AsyncConnectionPool

from semsearch.web.search.filters import SearchFilter


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

    def with_scores(self, scores: Mapping[str, float]) -> PageCandidate:
        return replace(self, scores=scores)


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    query_embedding: tuple[float, ...]
    filters: tuple[SearchFilter, ...]
    limit: int


@dataclass(frozen=True, slots=True)
class RankedRun:
    name: str
    weight: float
    candidates: tuple[PageCandidate, ...]


def make_run(
    name: str, weight: float, scored: Iterable[tuple[PageCandidate, float]]
) -> RankedRun:
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
    Awaitable[RankedRun],
]
type Reranker = Callable[[str, Sequence[PageCandidate]], Awaitable[RankedRun]]
type Fusion = Callable[[Sequence[RankedRun]], list[PageCandidate]]
