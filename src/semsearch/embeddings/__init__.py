from semsearch.config import Settings
from semsearch.embeddings.base import EmbedDocuments, EmbedQuery
from semsearch.embeddings.openai_compat import OpenAICompatEmbeddings

__all__ = [
    "EmbedDocuments",
    "EmbedQuery",
    "OpenAICompatEmbeddings",
    "get_embedding_provider",
]


def get_embedding_provider(settings: Settings) -> OpenAICompatEmbeddings:
    return OpenAICompatEmbeddings(
        base_url=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.query_instruction,
        expected_dim=settings.embedding_dim,
    )
