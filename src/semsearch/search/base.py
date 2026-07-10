from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from semsearch.models import Candidate
from semsearch.search.filters import SearchFilter


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


class Retriever(Protocol):
    name: str

    async def retrieve(self, request: RetrievalRequest) -> RankedRun: ...


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: Sequence[Candidate]
    ) -> RankedRun: ...


class Fusion(Protocol):
    def fuse(self, runs: Sequence[RankedRun]) -> list[Candidate]: ...
