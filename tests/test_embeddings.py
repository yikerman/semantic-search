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
        text = "upstream returned HTML"

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

    with pytest.raises(EmbeddingError, match="invalid JSON") as raised:
        await embedder.embed_query("query")

    assert "model=test" in str(raised.value)
    assert "inputs=1" in str(raised.value)
    assert "upstream returned HTML" in str(raised.value)


class EmbeddingResponse:
    status_code = 200
    text = ""

    def __init__(self, count: int) -> None:
        self.count = count

    def json(self):
        return {
            "data": [
                {"index": index, "embedding": [1.0]} for index in range(self.count)
            ]
        }


class RecordingClient:
    def __init__(self) -> None:
        self.payloads: list[object] = []

    async def post(self, path, json):
        self.payloads.append(json)
        return EmbeddingResponse(len(json["input"]))


def recording_embedder(client: Any, *, query_instruction=""):
    embedder = cast(Any, OpenAICompatEmbeddings.__new__(OpenAICompatEmbeddings))
    embedder.model = "test"
    embedder.batch_size = 32
    embedder.query_instruction = query_instruction
    embedder.expected_dim = 1
    embedder.max_retries = 1
    embedder._client = client
    return embedder


async def test_embedding_client_reports_provider_error_payload():
    class Response:
        status_code = 200
        text = '{"error":{"message":"rate limited","code":429}}'
        headers = {"x-request-id": "request-123"}

        def json(self):
            return {"error": {"message": "rate limited", "code": 429}}

    class Client:
        async def post(self, path, json):
            return Response()

    embedder = recording_embedder(Client())

    with pytest.raises(EmbeddingError, match="has no data array") as raised:
        await embedder.embed_documents(["one", "two"])

    message = str(raised.value)
    assert "HTTP 200, model=test, inputs=2, request_id=request-123" in message
    assert "response_keys=['error']" in message
    assert "rate limited" in message


async def test_embedding_client_retains_error_after_http_retries():
    class Response:
        status_code = 429
        text = '{"error":{"message":"rate limited"}}'
        headers = {"x-request-id": "request-456"}

    class Client:
        async def post(self, path, json):
            return Response()

    embedder = recording_embedder(Client())
    embedder.max_retries = 1

    with pytest.raises(EmbeddingError, match="failed after 1 attempts") as raised:
        await embedder.embed_query("query")

    message = str(raised.value)
    assert "HTTP 429" in message
    assert "model=test, inputs=1, request_id=request-456" in message
    assert "rate limited" in message


async def test_document_embeddings_send_text():
    client = RecordingClient()
    embedder = recording_embedder(client)

    await embedder.embed_documents(["one", "three"])

    assert client.payloads == [{"model": "test", "input": ["one", "three"]}]


async def test_query_embeddings_send_instructed_text():
    client = RecordingClient()
    embedder = recording_embedder(
        client,
        query_instruction="find passages",
    )

    await embedder.embed_query("needles")

    assert client.payloads == [
        {
            "model": "test",
            "input": ["Instruct: find passages\nQuery: needles"],
        }
    ]
