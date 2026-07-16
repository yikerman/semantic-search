import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_index: int
    content: str
    char_count: int


_WORD = re.compile(r"\S+\s*")


type Chunker = Callable[[str], list[Chunk]]


def char_chunks(
    text: str, *, chunk_chars: int = 1600, chunk_overlap: int = 240
) -> list[Chunk]:
    words = _WORD.findall(text)
    if not words:
        return []
    widths = [len(word) for word in words]

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end, size = start, 0
        while end < len(words) and (end == start or size + widths[end] <= chunk_chars):
            size += widths[end]
            end += 1
        content = "".join(words[start:end]).strip()
        chunks.append(Chunk(len(chunks), content, len(content)))
        if end == len(words):
            break
        back, overlap = end, 0
        while back > start + 1 and overlap < chunk_overlap:
            back -= 1
            overlap += widths[back]
        start = back
    return chunks
