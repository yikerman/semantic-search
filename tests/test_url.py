import pytest

from semsearch.cli.url import (
    canonicalize_url,
    normalize_url,
    same_site,
    try_normalize_url,
)


def test_normalize_url_drops_default_ports():
    assert normalize_url("https://example.com:443/post") == "https://example.com/post"
    assert normalize_url("http://example.com:80/post") == "http://example.com/post"
    assert normalize_url("https://example.com:8443/x") == "https://example.com:8443/x"


def test_same_site_treats_apex_and_www_as_one_site():
    assert same_site("https://www.example.com/post", "https://example.com")
    assert same_site("https://example.com/post", "https://www.example.com")
    assert same_site("http://example.com/post", "https://example.com")
    assert not same_site("https://blog.example.com/post", "https://example.com")
    assert not same_site("https://example.org/post", "https://example.com")


def test_canonicalize_url_folds_scheme_and_host_onto_origin():
    assert (
        canonicalize_url(
            "http://www.example.com/post?a=1", origin="https://example.com"
        )
        == "https://example.com/post?a=1"
    )
    assert (
        canonicalize_url("https://example.com:443/post", origin="https://example.com")
        == "https://example.com/post"
    )


def test_try_normalize_url_rejects_non_public_and_non_http():
    assert try_normalize_url("http://127.0.0.1/x") is None
    assert try_normalize_url("http://localhost/x") is None
    assert try_normalize_url("mailto:a@example.com") is None


@pytest.mark.parametrize(
    "raw",
    [
        "http://metadata.google.internal/",
        "http://2130706433/",
        "http://0177.0.0.1/",
    ],
)
def test_normalize_url_does_not_resolve_dns_for_the_admin_path(raw: str):
    # The admin CLI trusts its input, so normalize_url only rejects IP literals
    # and localhost; SSRF sanitization for untrusted URLs lives in the import
    # script (host resolution), not here.
    assert normalize_url(raw) == raw
