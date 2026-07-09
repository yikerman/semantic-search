from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    content: str
    char_count: int


@dataclass(slots=True)
class Candidate:
    chunk_id: int
    page_id: int
    url: str
    title: str | None
    content: str
    scores: dict[str, float] = field(default_factory=dict)

    def score(self, key: str = "dense") -> float:
        return self.scores.get(key, 0.0)


@dataclass(slots=True)
class SearchResult:
    page_id: int
    url: str
    title: str | None
    score: float
    snippet: str
