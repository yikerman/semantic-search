import asyncio

import httpx


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
        if not batch:
            return []
        response = await self._request(batch)
        data = sorted(response["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"Requested {len(batch)} embeddings, got {len(vectors)}"
            )
        if self.expected_dim is not None:
            for vec in vectors:
                if len(vec) != self.expected_dim:
                    raise EmbeddingError(
                        f"Model {self.model} returned {len(vec)}-dim vectors, "
                        f"expected {self.expected_dim} (check EMBEDDING_MODEL / "
                        "EMBEDDING_DIM)"
                    )
        return vectors

    async def _request(self, batch: list[str]) -> dict:
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
                return resp.json()
            last_error = EmbeddingError(
                f"Embedding API returned {resp.status_code}: {resp.text[:500]}"
            )
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise last_error
        raise EmbeddingError(
            f"Embedding request failed after {self.max_retries} attempts"
        ) from last_error

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatEmbeddings":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
