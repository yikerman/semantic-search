from typing import Any, cast

import pytest

from semsearch.share.embeddings import (
    EmbeddingError,
    OpenAICompatEmbeddings,
    _parse_embeddings,
)


def test_parse_embeddings_orders_and_normalizes_vectors():
    vectors = _parse_embeddings(
        {
            "data": [
                {"index": 1, "embedding": [3, 4]},
                {"index": 0, "embedding": [1, 2]},
            ]
        },
        expected_count=2,
        expected_dim=2,
        model="test",
    )

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"data": [None]},
        {"data": [{"index": True, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [float("nan")]}]},
        {"data": [{"index": 0, "embedding": [10**10000]}]},
        {"data": [{"index": 1, "embedding": [1.0]}]},
    ],
)
def test_parse_embeddings_rejects_invalid_provider_payloads(payload: object):
    with pytest.raises(EmbeddingError):
        _parse_embeddings(
            payload,
            expected_count=1,
            expected_dim=1,
            model="test",
        )


async def test_embedding_client_wraps_invalid_json():
    class Response:
        status_code = 200

        def json(self):
            raise ValueError("bad json")

    class Client:
        async def post(self, path, json):
            return Response()

    embedder = cast(Any, OpenAICompatEmbeddings.__new__(OpenAICompatEmbeddings))
    embedder.model = "test"
    embedder.batch_size = 1
    embedder.query_instruction = ""
    embedder.expected_dim = 1
    embedder.max_retries = 1
    embedder._client = Client()

    with pytest.raises(EmbeddingError, match="invalid JSON"):
        await embedder.embed_query("query")
