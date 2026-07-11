from collections.abc import Sequence

from semsearch.web.search.base import RankedRun
from semsearch.web.search.models import Candidate


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


def reciprocal_rank_fusion(
    runs: Sequence[RankedRun], *, k: int = 60
) -> list[Candidate]:
    candidates = {candidate.chunk_id: candidate for candidate in union_candidates(runs)}
    rrf_scores = dict.fromkeys(candidates, 0.0)

    for run in runs:
        for rank, candidate in enumerate(run.candidates, start=1):
            rrf_scores[candidate.chunk_id] += 1 / (k + rank)

    fused = [
        candidate.with_scores({**candidate.scores, "rrf": rrf_scores[chunk_id]})
        for chunk_id, candidate in candidates.items()
    ]
    return sorted(fused, key=lambda candidate: candidate.scores["rrf"], reverse=True)
