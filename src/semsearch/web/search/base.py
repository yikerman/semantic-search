from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from semsearch.web.search.filters import SearchFilter
from semsearch.web.search.models import ChunkCandidate, PageCandidate


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
