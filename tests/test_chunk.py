import pytest

from semsearch.ingest.chunk import CharChunker


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_yields_no_chunks():
    assert CharChunker(chunk_chars=12, chunk_overlap=3).chunk("") == []
    assert CharChunker(chunk_chars=12, chunk_overlap=3).chunk("   \n  ") == []


def test_short_text_yields_single_chunk():
    chunks = CharChunker(chunk_chars=30, chunk_overlap=6).chunk(words(4))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == words(4)
    assert chunks[0].char_count == 11


def test_windows_overlap_and_cover_everything():
    chunks = CharChunker(chunk_chars=12, chunk_overlap=3).chunk(words(10))
    assert [chunk.content for chunk in chunks] == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.char_count for chunk in chunks] == [11, 11, 11]


def test_trailing_words_get_their_own_chunk():
    chunks = CharChunker(chunk_chars=12, chunk_overlap=3).chunk(words(11))
    assert chunks[-1].content == "w9 w10"
    assert chunks[-1].char_count == 6


def test_no_redundant_trailing_window():
    chunks = CharChunker(chunk_chars=12, chunk_overlap=3).chunk(words(4))
    assert len(chunks) == 1


def test_word_longer_than_window_is_kept_whole():
    chunks = CharChunker(chunk_chars=5, chunk_overlap=2).chunk(
        "supercalifragilistic tiny"
    )
    assert [chunk.content for chunk in chunks] == ["supercalifragilistic", "tiny"]


def test_windows_never_split_a_word():
    text = " ".join(f"word{i:03d}" for i in range(50))
    vocabulary = set(text.split())
    for chunk in CharChunker(chunk_chars=40, chunk_overlap=10).chunk(text):
        assert set(chunk.content.split()) <= vocabulary


def test_internal_whitespace_is_preserved():
    text = "alpha beta\n\ngamma delta"
    chunks = CharChunker(chunk_chars=100, chunk_overlap=10).chunk(text)
    assert [chunk.content for chunk in chunks] == [text]


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        CharChunker(chunk_chars=4, chunk_overlap=4)
