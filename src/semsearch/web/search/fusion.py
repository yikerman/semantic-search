from collections.abc import Sequence

from semsearch.web.search.models import PageCandidate, RankedRun


def union_page_candidates(
    runs: Sequence[RankedRun],
) -> list[PageCandidate]:
    candidates: dict[int, PageCandidate] = {}
    scores: dict[int, dict[str, float]] = {}
    for run in runs:
        for candidate in run.candidates:
            candidates.setdefault(candidate.page_id, candidate)
            scores.setdefault(candidate.page_id, {}).update(candidate.scores)
    return [
        candidate.with_scores(scores[page_id])
        for page_id, candidate in candidates.items()
    ]


def reciprocal_rank_fusion(
    runs: Sequence[RankedRun], *, k: int = 60
) -> list[PageCandidate]:
    candidates = {
        candidate.page_id: candidate for candidate in union_page_candidates(runs)
    }
    rrf_scores = dict.fromkeys(candidates, 0.0)

    for run in runs:
        for rank, candidate in enumerate(run.candidates, start=1):
            rrf_scores[candidate.page_id] += run.weight / (k + rank)

    fused = [
        candidate.with_scores({**candidate.scores, "rrf": rrf_scores[page_id]})
        for page_id, candidate in candidates.items()
    ]
    return sorted(fused, key=lambda candidate: candidate.scores["rrf"], reverse=True)
