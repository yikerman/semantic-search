from typing import Any

import pytest
from pydantic import ValidationError

from semsearch.share.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dim", 0),
        ("embedding_dim", 16001),
        ("embedding_max_tokens", 32),
        ("crawl_delay_seconds", -1),
        ("crawl_timeout_seconds", 0),
        ("crawl_concurrency", 0),
        ("database_pool_max_size", 0),
        ("database_pool_max_size", 1),
        ("crawl_interval_seconds", 0),
        ("extraction_workers", 0),
        ("index_concurrency", 0),
        ("index_interval_seconds", 0),
        ("crawl_article_limit", 0),
    ],
)
def test_settings_reject_invalid_numeric_values(field: str, value: object):
    values: dict[str, Any] = {field: value}

    with pytest.raises(ValidationError):
        Settings(**values)


@pytest.mark.parametrize(
    "field", ["embedding_tokenizer", "embedding_tokenizer_revision"]
)
@pytest.mark.parametrize("value", ["", "   "])
def test_settings_reject_blank_tokenizer_values(field: str, value: str):
    values: dict[str, Any] = {field: value}

    with pytest.raises(ValidationError):
        Settings(**values)


def test_default_site_poll_interval_is_twelve_hours(tmp_path, monkeypatch):
    # Keep hermetic: chdir away from the repo so a developer's .env is not read,
    # and clear any exported override of the value under test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CRAWL_INTERVAL_SECONDS", raising=False)
    assert Settings().crawl_interval_seconds == 43_200


def test_embedding_dimensions_are_not_limited_by_the_old_hnsw_index():
    assert Settings(embedding_dim=4096).embedding_dim == 4096
    assert Settings(embedding_dim=16000).embedding_dim == 16000


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_settings_accept_log_levels(level: str):
    values: dict[str, Any] = {"log_level": level}

    assert Settings(**values).log_level == level


def test_settings_reject_invalid_log_level():
    values: dict[str, Any] = {"log_level": "TRACE"}

    with pytest.raises(ValidationError):
        Settings(**values)
