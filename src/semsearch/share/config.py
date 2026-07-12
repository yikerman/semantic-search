from functools import lru_cache
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://semsearch:semsearch@localhost:5432/semsearch"
    database_pool_max_size: Annotated[int, Field(ge=2)] = 20
    log_level: LogLevel = "INFO"

    embedding_api_base: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""
    embedding_model: str = "qwen/qwen3-embedding-4b"
    embedding_dim: Annotated[int, Field(gt=0)] = 2560
    embedding_batch_size: Annotated[int, Field(gt=0)] = 32

    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    chunk_chars: Annotated[int, Field(gt=0)] = 1600
    chunk_overlap: Annotated[int, Field(ge=0)] = 240

    fetch_delay_seconds: Annotated[float, Field(ge=0)] = 1.0
    fetch_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0
    fetch_concurrency: Annotated[int, Field(gt=0)] = 16
    fetch_impersonate: str = "chrome"
    user_agent: str = "semsearch/0.1"

    site_poll_interval_seconds: Annotated[int, Field(gt=0)] = 43_200
    site_poll_concurrency: Annotated[int, Field(gt=0)] = 16
    ingest_concurrency: Annotated[int, Field(gt=0)] = 4
    history_post_limit: Annotated[int, Field(gt=0)] = 2000

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Self:
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_CHARS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
