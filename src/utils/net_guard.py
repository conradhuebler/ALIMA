"""URL validation + guarded fetching for plugin/tool HTTP - Claude Generated.

Two postures, matching who supplied the URL:

* **Operator-configured base URLs** (catalog/finc/SRU endpoints from the
  settings): :func:`check_operator_url` returns *warnings* only. University
  catalogs legitimately live on intranet/private addresses, so private IPs are
  never blocked here — the operator configured the endpoint knowingly.
* **Runtime-supplied URLs** (LLM tool calls like ``scrape_url``, batch URL
  input): :func:`assert_public_http_url` / :func:`fetch_guarded` enforce a
  strict SSRF guard — http(s) only, every resolved address must be public
  (no loopback/private/link-local/reserved), redirects re-checked per hop,
  response size capped. Exceptions via ``SystemConfig.url_fetch_allowlist``.

Honest limit: the resolve-then-fetch pattern is not DNS-rebinding-proof (the
name may re-resolve between check and request). This raises the bar against
straightforward SSRF (file://, localhost, cloud metadata IPs); it is not a
network sandbox.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from typing import Dict, Iterable, List
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 15
DEFAULT_MAX_BYTES = 10_000_000
DEFAULT_MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = ("http", "https")


def check_operator_url(url: str) -> List[str]:
    """Warnings for an operator-configured base URL. Never raises, never blocks.

    An empty URL yields no warnings (unconfigured providers are handled by
    availability gating). Private/intranet hosts are deliberately NOT flagged.
    - Claude Generated
    """
    warnings: List[str] = []
    if not url or not str(url).strip():
        return warnings
    url = str(url).strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return [f"URL '{url}' ist nicht parsebar"]
    if parsed.scheme not in _ALLOWED_SCHEMES:
        warnings.append(
            f"URL '{url}': Schema '{parsed.scheme or '(leer)'}' — erwartet http/https"
        )
        return warnings
    if not parsed.hostname:
        warnings.append(f"URL '{url}' hat keinen Hostnamen")
    if parsed.scheme == "http":
        warnings.append(f"URL '{url}' ist unverschlüsselt (http statt https)")
    return warnings


def require_http_url(url: str, *, what: str = "URL") -> str:
    """Require an http(s) URL with a hostname; raise ``ValueError`` otherwise.

    The cheap scheme gate for configured endpoints at request time — kills
    ``file://``/``ftp://`` and scheme-less strings without blocking intranet
    hosts. Returns the url unchanged for call-site chaining. - Claude Generated
    """
    parsed = urlparse(str(url))
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
        raise ValueError(f"{what} '{url}' is not a valid http(s) URL")
    return url


def _host_allowed(hostname: str, allowlist: Iterable[str]) -> bool:
    host = hostname.lower().rstrip(".")
    for entry in allowlist or ():
        if host == str(entry).lower().strip().rstrip("."):
            return True
    return False


def assert_public_http_url(url: str, *, allowlist: Iterable[str] = ()) -> None:
    """Strict SSRF guard for runtime-supplied URLs; raises ``ValueError``.

    Requires http(s) + hostname, resolves the host and rejects the URL unless
    **every** resolved address is public (no private/loopback/link-local/
    reserved/multicast/unspecified). ``allowlist`` entries (hostnames or IP
    literals) bypass the address check, not the scheme check. - Claude Generated
    """
    parsed = urlparse(str(url))
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"URL '{url}': scheme '{parsed.scheme}' not allowed (http/https only)")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL '{url}' has no hostname")
    if _host_allowed(hostname, allowlist):
        return
    try:
        infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"URL '{url}': cannot resolve host '{hostname}': {exc}")
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            raise ValueError(f"URL '{url}': unparseable resolved address '{ip_str}'")
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                f"URL '{url}': host '{hostname}' resolves to non-public address {ip_str} — "
                "blocked (add the host to url_fetch_allowlist to permit intranet targets)"
            )


@dataclass
class GuardedResponse:
    """Minimal response surface returned by :func:`fetch_guarded`."""

    url: str  # final URL after redirects
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    content: bytes = b""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}")


def fetch_guarded(
    url: str,
    *,
    allowlist: Iterable[str] = (),
    timeout: int = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_agent: str = "ALIMA",
    headers: Dict[str, str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> GuardedResponse:
    """GET a runtime-supplied URL with the SSRF guard applied per redirect hop.

    Redirects are followed manually (≤ ``max_redirects``) so every hop passes
    :func:`assert_public_http_url`; the body is streamed and capped at
    ``max_bytes`` (raises ``ValueError`` beyond). - Claude Generated
    """
    import requests

    req_headers = {"User-Agent": user_agent}
    if headers:
        req_headers.update(headers)
    current = str(url)
    for _hop in range(max_redirects + 1):
        assert_public_http_url(current, allowlist=allowlist)
        resp = requests.get(
            current, headers=req_headers, timeout=timeout,
            stream=True, allow_redirects=False,
        )
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise RuntimeError(f"HTTP {resp.status_code} redirect without Location for {current}")
                current = urljoin(current, location)
                continue
            chunks: List[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"response for '{current}' exceeds max_bytes ({max_bytes}) — aborted"
                    )
                chunks.append(chunk)
            return GuardedResponse(
                url=current,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                content=b"".join(chunks),
            )
        finally:
            resp.close()
    raise RuntimeError(f"too many redirects (>{max_redirects}) fetching '{url}'")


def url_fetch_guard_settings() -> Dict[str, object]:
    """Read ``SystemConfig.url_fetch_allowlist`` / ``url_fetch_max_bytes``.

    Best-effort: any config failure returns the defaults, so the guard is
    always on. - Claude Generated
    """
    allowlist: List[str] = []
    max_bytes = DEFAULT_MAX_BYTES
    try:
        from src.utils.config_manager import ConfigManager

        system = ConfigManager().load_config().system_config
        allowlist = list(getattr(system, "url_fetch_allowlist", []) or [])
        max_bytes = int(getattr(system, "url_fetch_max_bytes", DEFAULT_MAX_BYTES))
    except Exception:
        pass
    return {"allowlist": allowlist, "max_bytes": max_bytes}
