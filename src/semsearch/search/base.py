from typing import Protocol

from semsearch.models import Candidate


class Retriever(Protocol):
    name: str

    async def retrieve(
        self, query: str, query_embedding: list[float], k: int
    ) -> list[Candidate]: ...


class Ranker(Protocol):
    async def rank(
        self, query: str, candidates: list[Candidate]
    ) -> list[Candidate]: ...


def final_score(candidate: Candidate) -> float:
    return candidate.scores.get("final", candidate.scores.get("dense", 0.0))
