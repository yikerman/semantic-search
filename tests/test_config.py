from typing import Any

import pytest
from pydantic import ValidationError

from semsearch.share.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dim", 0),
        ("embedding_batch_size", 0),
        ("chunk_chars", 0),
        ("chunk_overlap", -1),
        ("fetch_delay_seconds", -1),
        ("fetch_timeout_seconds", 0),
        ("fetch_concurrency", 0),
        ("database_pool_max_size", 0),
        ("database_pool_max_size", 1),
        ("site_poll_interval_seconds", 0),
        ("site_poll_concurrency", 0),
        ("ingest_concurrency", 0),
        ("history_post_limit", 0),
    ],
)
def test_settings_reject_invalid_numeric_values(field: str, value: object):
    values: dict[str, Any] = {field: value}

    with pytest.raises(ValidationError):
        Settings(**values)


def test_settings_reject_chunk_overlap_at_least_window_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_chars=100, chunk_overlap=100)


def test_default_site_poll_interval_is_twelve_hours(tmp_path, monkeypatch):
    # Keep hermetic: chdir away from the repo so a developer's .env is not read,
    # and clear any exported override of the value under test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SITE_POLL_INTERVAL_SECONDS", raising=False)
    assert Settings().site_poll_interval_seconds == 43_200


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_settings_accept_log_levels(level: str):
    values: dict[str, Any] = {"log_level": level}

    assert Settings(**values).log_level == level


def test_settings_reject_invalid_log_level():
    values: dict[str, Any] = {"log_level": "TRACE"}

    with pytest.raises(ValidationError):
        Settings(**values)
