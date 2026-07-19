import asyncio
import math
from collections.abc import Awaitable, Callable

import httpx

from semsearch.share.config import Settings


type EmbedDocuments = Callable[[list[str]], Awaitable[list[list[float]]]]
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
        batch_size: int = 32,
        query_instruction: str = "",
        expected_dim: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.expected_dim = expected_dim
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(await self._embed(texts[start : start + self.batch_size]))
        return out

    async def embed_query(self, text: str) -> list[float]:
        if self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return (await self._embed([text]))[0]

    async def _embed(self, batch: list[str]) -> list[list[float]]:
        payload, request_id = await self._request(batch)
        try:
            return _parse_embeddings(
                payload,
                expected_count=len(batch),
                expected_dim=self.expected_dim,
                model=self.model,
            )
        except EmbeddingError as exc:
            context = _response_context(
                model=self.model,
                input_count=len(batch),
                request_id=request_id,
            )
            summary = _payload_summary(payload)
            raise EmbeddingError(f"{exc} (HTTP 200, {context}; {summary})") from exc

    async def _request(self, batch: list[str]) -> tuple[object, str | None]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                await asyncio.sleep(2**attempt)
            try:
                resp = await self._client.post(
                    "/embeddings", json={"model": self.model, "input": batch}
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
                        input_count=len(batch),
                        request_id=_request_id(resp),
                    )
                    raise EmbeddingError(
                        "Embedding API returned invalid JSON "
                        f"(HTTP 200, {context}; response={resp.text[:500]!r})"
                    ) from exc
            context = _response_context(
                model=self.model,
                input_count=len(batch),
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

    async def __aenter__(self) -> "OpenAICompatEmbeddings":
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


def _parse_embeddings(
    payload: object,
    *,
    expected_count: int,
    expected_dim: int | None,
    model: str,
) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise EmbeddingError("Embedding API response has no data array")
    data = payload.get("data")
    if not isinstance(data, list):
        raise EmbeddingError("Embedding API response has no data array")

    indexed: list[tuple[int, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingError("Embedding API returned an invalid data item")
        index = item.get("index")
        vector = item.get("embedding")
        if not isinstance(index, int) or isinstance(index, bool):
            raise EmbeddingError("Embedding API returned an invalid index")
        if not isinstance(vector, list):
            raise EmbeddingError("Embedding API returned an invalid vector")
        parsed_vector: list[float] = []
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EmbeddingError("Embedding API returned an invalid vector")
            try:
                parsed_value = float(value)
            except (OverflowError, ValueError) as exc:
                raise EmbeddingError(
                    "Embedding API returned an invalid vector"
                ) from exc
            if not math.isfinite(parsed_value):
                raise EmbeddingError("Embedding API returned an invalid vector")
            parsed_vector.append(parsed_value)
        indexed.append((index, parsed_vector))

    if len(indexed) != expected_count:
        raise EmbeddingError(
            f"Requested {expected_count} embeddings, got {len(indexed)}"
        )
    indexed.sort(key=lambda item: item[0])
    if [index for index, _ in indexed] != list(range(expected_count)):
        raise EmbeddingError("Embedding API returned invalid embedding indexes")

    vectors = [vector for _, vector in indexed]
    if expected_dim is not None:
        for vector in vectors:
            if len(vector) != expected_dim:
                raise EmbeddingError(
                    f"Model {model} returned {len(vector)}-dim vectors, "
                    f"expected {expected_dim} (check EMBEDDING_MODEL / EMBEDDING_DIM)"
                )
    return vectors


def create_embeddings(settings: Settings) -> OpenAICompatEmbeddings:
    return OpenAICompatEmbeddings(
        base_url=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.query_instruction,
        expected_dim=settings.embedding_dim,
    )
