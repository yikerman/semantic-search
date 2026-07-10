from collections.abc import Sequence
from dataclasses import dataclass

from semsearch.models import Candidate
from semsearch.search.base import RankedRun


def union_candidates(runs: Sequence[RankedRun]) -> list[Candidate]:
    candidates: dict[int, Candidate] = {}
    scores: dict[int, dict[str, float]] = {}
    for run in runs:
        for candidate in run.candidates:
            candidates.setdefault(candidate.chunk_id, candidate)
            scores.setdefault(candidate.chunk_id, {}).update(candidate.scores)
    return [
        candidate.with_scores(scores[chunk_id])
        for chunk_id, candidate in candidates.items()
    ]


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusion:
    k: int = 60

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("RRF constant must be non-negative")

    def fuse(self, runs: Sequence[RankedRun]) -> list[Candidate]:
        candidates = {
            candidate.chunk_id: candidate for candidate in union_candidates(runs)
        }
        rrf_scores = dict.fromkeys(candidates, 0.0)

        for run in runs:
            seen: set[int] = set()
            rank = 0
            for candidate in run.candidates:
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                rank += 1
                rrf_scores[candidate.chunk_id] += 1 / (self.k + rank)

        fused = [
            candidate.with_scores({**candidate.scores, "rrf": rrf_scores[chunk_id]})
            for chunk_id, candidate in candidates.items()
        ]
        return sorted(
            fused, key=lambda candidate: candidate.scores["rrf"], reverse=True
        )
