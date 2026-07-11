from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    raw = _with_scheme(url.strip())
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Invalid URL: {url}")
    path = parts.path or "/"
    return urlunsplit((_scheme(parts), _netloc(parts), path, "", ""))


def normalize_origin(url: str) -> str:
    raw = _with_scheme(url.strip())
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Invalid URL: {url}")
    return urlunsplit((_scheme(parts), _netloc(parts), "", "", ""))


def _with_scheme(url: str) -> str:
    if "://" in url:
        return url
    return f"https://{url}"


def _scheme(parts) -> str:
    return parts.scheme.lower()


def _netloc(parts) -> str:
    host = parts.hostname.lower() if parts.hostname else ""
    if parts.port is None:
        return host
    return f"{host}:{parts.port}"
