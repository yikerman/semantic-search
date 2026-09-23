from types import SimpleNamespace
from typing import Any, cast

import pytest

from semsearch.cli.ingest import document
from semsearch.cli.ingest.document import (
    TOKEN_RESERVE,
    DocumentTooLong,
    TokenizerError,
    load_tokenizer,
    validate_document,
)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens
        return SimpleNamespace(ids=list(range(len(text) + 1)))

    def no_truncation(self):
        pass

    def no_padding(self):
        pass


@pytest.mark.parametrize("text", ["Title\n\nWhole article", "没有空格的中文文章"])
def test_input_limit_includes_full_text_and_special_tokens(text):
    tokenizer = cast(Any, CharacterTokenizer())
    limit = len(text) + 1 + TOKEN_RESERVE
    validate_document(text, tokenizer=tokenizer, max_tokens=limit)
    with pytest.raises(DocumentTooLong, match="input budget"):
        validate_document(text, tokenizer=tokenizer, max_tokens=limit - 1)


def test_load_tokenizer_uses_pinned_revision(monkeypatch):
    calls: list[tuple[str, str]] = []
    expected = tokenizer = cast(Any, CharacterTokenizer())

    class Factory:
        @staticmethod
        def from_pretrained(identifier: str, *, revision: str):
            calls.append((identifier, revision))
            return expected

    monkeypatch.setattr(document, "Tokenizer", Factory)

    assert load_tokenizer("org/model", "commit") is tokenizer
    assert calls == [("org/model", "commit")]


def test_load_tokenizer_wraps_provider_errors(monkeypatch):
    class Factory:
        @staticmethod
        def from_pretrained(identifier: str, *, revision: str):
            raise OSError("offline")

    monkeypatch.setattr(document, "Tokenizer", Factory)

    with pytest.raises(TokenizerError, match="org/model.*commit"):
        load_tokenizer("org/model", "commit")


def test_loading_disables_configured_truncation_and_padding(tmp_path):
    from tokenizers import Tokenizer, models

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=128)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    from unittest.mock import patch

    with patch.object(
        document.Tokenizer,
        "from_pretrained",
        return_value=Tokenizer.from_file(str(path)),
    ):
        loaded = load_tokenizer("test/model", "revision")
    assert loaded.truncation is None
    assert loaded.padding is None
