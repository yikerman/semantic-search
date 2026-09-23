from functools import cache
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type NonBlankString = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://semsearch:semsearch@localhost:5432/semsearch"
    database_pool_max_size: Annotated[int, Field(ge=3)] = 40
    log_level: LogLevel = "INFO"

    embedding_api_base: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""
    embedding_model: str = "qwen/qwen3-embedding-4b"
    embedding_dim: Annotated[int, Field(gt=0, le=4000)] = 2560
    embedding_batch_size: Annotated[int, Field(gt=0)] = 32
    embedding_tokenizer: NonBlankString = "Qwen/Qwen3-Embedding-4B"
    embedding_tokenizer_revision: NonBlankString = (
        "5cf2132abc99cad020ac570b19d031efec650f2b"
    )

    query_instruction: str = "Given search query, retrieve relevant passages"

    chunk_tokens: Annotated[int, Field(gt=0)] = 384
    chunk_token_overlap: Annotated[int, Field(ge=0)] = 32

    user_agent: str = "semsearch/1.0"
    crawl_concurrency: Annotated[int, Field(gt=0)] = 16
    crawl_delay_seconds: Annotated[float, Field(ge=0)] = 2.0
    crawl_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0
    crawl_source_limit: Annotated[int, Field(gt=0)] = 100
    crawl_article_limit: Annotated[int, Field(gt=0)] = 2000
    crawl_interval_seconds: Annotated[int, Field(gt=0)] = 43_200
    extraction_workers: Annotated[int, Field(gt=0)] = 2
    extraction_timeout_seconds: Annotated[float, Field(gt=0)] = 10.0
    index_concurrency: Annotated[int, Field(gt=0)] = 4
    index_interval_seconds: Annotated[int, Field(gt=0)] = 300

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Self:
        if self.chunk_token_overlap >= self.chunk_tokens:
            raise ValueError("CHUNK_TOKEN_OVERLAP must be smaller than CHUNK_TOKENS")
        return self


@cache
def get_settings() -> Settings:
    return Settings()
