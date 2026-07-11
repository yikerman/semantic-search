from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from semsearch.web.search.filters import SearchFilter
from semsearch.web.search.models import Candidate


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    query_embedding: tuple[float, ...]
    filters: tuple[SearchFilter, ...]
    limit: int


@dataclass(frozen=True, slots=True)
class RankedRun:
    name: str
    candidates: tuple[Candidate, ...]


type Retriever = Callable[[RetrievalRequest], Awaitable[RankedRun]]
type Reranker = Callable[[str, Sequence[Candidate]], Awaitable[RankedRun]]
type Fusion = Callable[[Sequence[RankedRun]], list[Candidate]]
