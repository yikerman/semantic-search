from dataclasses import dataclass
from typing import Any, cast

import pytest
from tokenizers import Tokenizer

from semsearch.cli.ingest import chunk
from semsearch.cli.ingest.chunk import (
    TokenizerError,
    load_tokenizer,
    token_chunks,
)


@dataclass
class Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> Encoding:
        assert not add_special_tokens
        return Encoding(
            list(range(len(text))),
            [(i, i + 1) for i in range(len(text))],
        )


@pytest.fixture
def tokenizer() -> Tokenizer:
    return cast(Any, CharacterTokenizer())


def test_empty_text_yields_no_chunks(tokenizer):
    assert token_chunks("", tokenizer=tokenizer, chunk_tokens=12) == []
    assert token_chunks("   \n  ", tokenizer=tokenizer, chunk_tokens=12) == []


def test_short_text_yields_single_chunk(tokenizer):
    chunks = token_chunks("abcdefgh", tokenizer=tokenizer, chunk_tokens=12)

    assert len(chunks) == 1
    assert chunks[0].start_offset == 0
    assert chunks[0].content == "abcdefgh"


def test_strict_windows_overlap_and_cover_everything(tokenizer):
    chunks = token_chunks(
        "abcdefghijklmnopqrst",
        tokenizer=tokenizer,
        chunk_tokens=8,
        chunk_token_overlap=2,
    )

    assert [item.content for item in chunks] == [
        "abcdefgh",
        "ghijklmn",
        "mnopqrst",
    ]
    assert [item.start_offset for item in chunks] == [0, 6, 12]


def test_trailing_tokens_get_their_own_overlapping_chunk(tokenizer):
    chunks = token_chunks(
        "abcdefghijklmnopq",
        tokenizer=tokenizer,
        chunk_tokens=8,
        chunk_token_overlap=2,
    )

    assert chunks[-1].content == "mnopq"


def test_no_redundant_trailing_window(tokenizer):
    chunks = token_chunks(
        "abcdefgh",
        tokenizer=tokenizer,
        chunk_tokens=8,
        chunk_token_overlap=2,
    )

    assert len(chunks) == 1


def test_unspaced_cjk_is_split_by_tokens(tokenizer):
    text = "中文输入没有空格也必须正确分块"
    chunks = token_chunks(
        text,
        tokenizer=tokenizer,
        chunk_tokens=6,
        chunk_token_overlap=2,
    )

    assert len(chunks) > 1
    assert all(len(item.content) <= 6 for item in chunks)
    assert [item.start_offset for item in chunks[:2]] == [0, 4]
    assert chunks[0].content == text[:6]
    assert chunks[1].content == text[4:10]


def test_load_tokenizer_uses_pinned_revision(monkeypatch):
    calls: list[tuple[str, str]] = []
    expected = tokenizer = cast(Any, CharacterTokenizer())

    class Factory:
        @staticmethod
        def from_pretrained(identifier: str, *, revision: str):
            calls.append((identifier, revision))
            return expected

    monkeypatch.setattr(chunk, "Tokenizer", Factory)

    assert load_tokenizer("org/model", "commit") is tokenizer
    assert calls == [("org/model", "commit")]


def test_load_tokenizer_wraps_provider_errors(monkeypatch):
    class Factory:
        @staticmethod
        def from_pretrained(identifier: str, *, revision: str):
            raise OSError("offline")

    monkeypatch.setattr(chunk, "Tokenizer", Factory)

    with pytest.raises(TokenizerError, match="org/model.*commit"):
        load_tokenizer("org/model", "commit")
