from collections.abc import Mapping, Sequence
from types import MappingProxyType

from semsearch.web.search.models import ChunkCandidate, PageCandidate, RankedRun

RUN_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"dense": 2.0, "bm25": 1.0, "length": 1.0}
)


def union_chunk_candidates(
    runs: Sequence[RankedRun[ChunkCandidate]],
) -> list[ChunkCandidate]:
    candidates: dict[int, ChunkCandidate] = {}
    scores: dict[int, dict[str, float]] = {}
    for run in runs:
        for candidate in run.candidates:
            candidates.setdefault(candidate.chunk_id, candidate)
            scores.setdefault(candidate.chunk_id, {}).update(candidate.scores)
    return [
        candidate.with_scores(scores[chunk_id])
        for chunk_id, candidate in candidates.items()
    ]


def union_page_candidates(
    runs: Sequence[RankedRun[PageCandidate]],
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
    runs: Sequence[RankedRun[PageCandidate]], *, k: int = 60
) -> list[PageCandidate]:
    weights: list[float] = []
    for run in runs:
        try:
            weights.append(RUN_WEIGHTS[run.name])
        except KeyError as exc:
            raise ValueError(f"missing RRF weight for run {run.name!r}") from exc

    candidates = {
        candidate.page_id: candidate for candidate in union_page_candidates(runs)
    }
    rrf_scores = dict.fromkeys(candidates, 0.0)

    for run, weight in zip(runs, weights, strict=True):
        for rank, candidate in enumerate(run.candidates, start=1):
            rrf_scores[candidate.page_id] += weight / (k + rank)

    fused = [
        candidate.with_scores({**candidate.scores, "rrf": rrf_scores[page_id]})
        for page_id, candidate in candidates.items()
    ]
    return sorted(fused, key=lambda candidate: candidate.scores["rrf"], reverse=True)
