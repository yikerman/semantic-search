from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://semsearch:semsearch@localhost:5432/semsearch"

    embedding_api_base: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""
    embedding_model: str = "qwen/qwen3-embedding-4b"
    embedding_dim: int = 2560
    embedding_batch_size: int = 32

    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    chunk_chars: int = 1600
    chunk_overlap: int = 240

    fetch_delay_seconds: float = 1.0
    fetch_timeout_seconds: float = 20.0
    fetch_impersonate: str = "chrome"
    user_agent: str = "semsearch/0.1"

    site_poll_concurrency: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
