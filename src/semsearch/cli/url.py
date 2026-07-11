from ipaddress import ip_address
from urllib.parse import SplitResult, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    raw = _with_scheme(url.strip())
    parts = _validate_parts(raw, url)
    path = parts.path or "/"
    return urlunsplit((_scheme(parts), _netloc(parts), path, parts.query, ""))


def normalize_origin(url: str) -> str:
    raw = _with_scheme(url.strip())
    parts = _validate_parts(raw, url)
    return urlunsplit((_scheme(parts), _netloc(parts), "", "", ""))


def try_normalize_url(url: str) -> str | None:
    try:
        return normalize_url(url)
    except ValueError:
        return None


def canonicalize_url(url: str, *, origin: str) -> str:
    """Rewrite a same-site URL onto ``origin``'s scheme and host.

    Feed, sitemap, and history sources link the same post under different
    scheme/host/port spellings (http vs https, apex vs www, explicit :443).
    Folding them onto the configured origin keeps ``url`` as a single page
    identity so append-only dedup does not store duplicates.
    """
    origin_parts = _validate_parts(_with_scheme(origin.strip()), origin)
    parts = _validate_parts(_with_scheme(url.strip()), url)
    path = parts.path or "/"
    return urlunsplit(
        (_scheme(origin_parts), _netloc(origin_parts), path, parts.query, "")
    )


def same_site(url: str, site: str) -> bool:
    try:
        left = _validate_parts(url, url)
        right = _validate_parts(site, site)
    except ValueError:
        return False
    left_host = left.hostname
    right_host = right.hostname
    return (
        left_host is not None
        and right_host is not None
        and _bare_host(left_host) == _bare_host(right_host)
        and _site_port(left) == _site_port(right)
    )


def _with_scheme(url: str) -> str:
    if "://" in url:
        return url
    _, separator, suffix = url.partition(":")
    if separator and not suffix.split("/", 1)[0].isdigit():
        return url
    return f"https://{url}"


def _validate_parts(raw: str, original: str) -> SplitResult:
    parts = urlsplit(raw)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {original}")
    hostname = parts.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError(f"Non-public URL: {original}")
    try:
        address = ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError(f"Non-public URL: {original}")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError(f"Invalid URL port: {original}") from exc
    return parts


def _scheme(parts: SplitResult) -> str:
    return parts.scheme.lower()


def _netloc(parts: SplitResult) -> str:
    host = parts.hostname.lower() if parts.hostname else ""
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is None or port == _default_port(parts.scheme):
        return host
    return f"{host}:{port}"


def _default_port(scheme: str) -> int:
    return 443 if scheme.lower() == "https" else 80


def _bare_host(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _site_port(parts: SplitResult) -> int | None:
    if parts.port in (None, 80, 443):
        return None
    return parts.port
