from typing import Any

import pytest
from pydantic import ValidationError

from semsearch.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dim", 0),
        ("embedding_batch_size", 0),
        ("chunk_chars", 0),
        ("chunk_overlap", -1),
        ("fetch_delay_seconds", -1),
        ("fetch_timeout_seconds", 0),
        ("site_poll_concurrency", 0),
    ],
)
def test_settings_reject_invalid_numeric_values(field: str, value: object):
    values: dict[str, Any] = {field: value}

    with pytest.raises(ValidationError):
        Settings(**values)


def test_settings_reject_chunk_overlap_at_least_window_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_chars=100, chunk_overlap=100)
