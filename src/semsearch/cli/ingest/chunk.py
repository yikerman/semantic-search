from collections.abc import Callable
from dataclasses import dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True, slots=True)
class Chunk:
    start_offset: int
    content: str


type Chunker = Callable[[str], list[Chunk]]


class TokenizerError(RuntimeError):
    pass


def load_tokenizer(identifier: str, revision: str) -> Tokenizer:
    try:
        return Tokenizer.from_pretrained(identifier, revision=revision)
    except Exception as exc:  # noqa: BLE001
        raise TokenizerError(
            f"Could not load tokenizer {identifier} at revision {revision}"
        ) from exc


def token_chunks(
    text: str,
    *,
    tokenizer: Tokenizer,
    chunk_tokens: int = 384,
    chunk_token_overlap: int = 64,
) -> list[Chunk]:
    if not text.strip():
        return []
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if not encoding.ids:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(encoding.ids):
        end = min(start + chunk_tokens, len(encoding.ids))
        start_char = encoding.offsets[start][0]
        end_char = encoding.offsets[end - 1][1]
        content = text[start_char:end_char]
        chunks.append(Chunk(start_char, content))
        if end == len(encoding.ids):
            break
        start = end - chunk_token_overlap
    return chunks
