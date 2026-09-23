import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Self

import httpx

from semsearch.share.config import Settings

type EmbedDocument = Callable[[str], Awaitable[list[float]]]
type EmbedQuery = Callable[[str], Awaitable[list[float]]]


class EmbeddingError(RuntimeError):
    pass


class OpenAICompatEmbeddings:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        query_instruction: str = "",
        expected_dim: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.query_instruction = query_instruction
        self.expected_dim = expected_dim
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def embed_document(self, text: str) -> list[float]:
        return await self._embed(text)

    async def embed_query(self, text: str) -> list[float]:
        if self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return await self._embed(text)

    async def _embed(self, text: str) -> list[float]:
        payload, request_id = await self._request(text)
        try:
            return _parse_embedding(
                payload,
                expected_dim=self.expected_dim,
                model=self.model,
            )
        except EmbeddingError as exc:
            context = _response_context(
                model=self.model,
                input_count=1,
                request_id=request_id,
            )
            summary = _payload_summary(payload)
            raise EmbeddingError(f"{exc} (HTTP 200, {context}; {summary})") from exc

    async def _request(self, text: str) -> tuple[object, str | None]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                await asyncio.sleep(2**attempt)
            try:
                resp = await self._client.post(
                    "/embeddings", json={"model": self.model, "input": [text]}
                )
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            if resp.status_code == 200:
                try:
                    return resp.json(), _request_id(resp)
                except ValueError as exc:
                    context = _response_context(
                        model=self.model,
                        input_count=1,
                        request_id=_request_id(resp),
                    )
                    raise EmbeddingError(
                        "Embedding API returned invalid JSON "
                        f"(HTTP 200, {context}; response={resp.text[:500]!r})"
                    ) from exc
            context = _response_context(
                model=self.model,
                input_count=1,
                request_id=_request_id(resp),
            )
            last_error = EmbeddingError(
                f"Embedding API returned HTTP {resp.status_code} "
                f"({context}; response={resp.text[:500]!r})"
            )
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise last_error
        raise EmbeddingError(
            f"Embedding request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _request_id(response: httpx.Response) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return headers.get("x-request-id") or headers.get("cf-ray")


def _response_context(*, model: str, input_count: int, request_id: str | None) -> str:
    context = f"model={model}, inputs={input_count}"
    if request_id:
        context += f", request_id={request_id}"
    return context


def _payload_summary(payload: object) -> str:
    if not isinstance(payload, dict):
        return f"response_type={type(payload).__name__}"

    keys = sorted(str(key) for key in payload)
    summary = f"response_keys={keys!r}"
    if "error" in payload:
        summary += f", error={payload.get('error')!r}"
    return summary[:500]


def _parse_embedding(
    payload: object, *, expected_dim: int | None, model: str
) -> list[float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise EmbeddingError("Embedding API response has no data array")
    data = payload["data"]
    if len(data) != 1:
        raise EmbeddingError(f"Requested one embedding, got {len(data)}")
    item = data[0]
    if not isinstance(item, dict):
        raise EmbeddingError("Embedding API returned an invalid data item")
    index = item.get("index")
    vector = item.get("embedding")
    if type(index) is not int or index != 0:
        raise EmbeddingError("Embedding API returned an invalid index")
    if not isinstance(vector, list):
        raise EmbeddingError("Embedding API returned an invalid vector")
    parsed: list[float] = []
    for value in vector:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EmbeddingError("Embedding API returned an invalid vector")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise EmbeddingError("Embedding API returned an invalid vector") from exc
        if not math.isfinite(number):
            raise EmbeddingError("Embedding API returned an invalid vector")
        parsed.append(number)
    if expected_dim is not None and len(parsed) != expected_dim:
        raise EmbeddingError(
            f"Model {model} returned {len(parsed)}-dim vectors, "
            f"expected {expected_dim} (check EMBEDDING_MODEL / EMBEDDING_DIM)"
        )
    return parsed


def create_embeddings(settings: Settings) -> OpenAICompatEmbeddings:
    return OpenAICompatEmbeddings(
        base_url=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        query_instruction=settings.query_instruction,
        expected_dim=settings.embedding_dim,
    )
