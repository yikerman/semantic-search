import re

from semsearch.models import Chunk

_WORD = re.compile(r"\S+\s*")


class CharChunker:
    def __init__(self, *, chunk_chars: int = 1600, chunk_overlap: int = 240) -> None:
        if chunk_overlap >= chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[Chunk]:
        words = _WORD.findall(text)
        if not words:
            return []
        widths = [len(word) for word in words]

        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end, size = start, 0
            while end < len(words) and (
                end == start or size + widths[end] <= self.chunk_chars
            ):
                size += widths[end]
                end += 1
            content = "".join(words[start:end]).strip()
            if content:
                chunks.append(Chunk(len(chunks), content, len(content)))
            if end == len(words):
                break
            back, overlap = end, 0
            while back > start + 1 and overlap < self.chunk_overlap:
                back -= 1
                overlap += widths[back]
            start = back
        return chunks
