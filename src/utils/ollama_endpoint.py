"""Ollama endpoint input: cleaning, hints and URL building - Claude Generated.

Qt-free so the provider dialog, the setup wizards and ``LlmService`` share one
reading of "what did the user mean by this host/port/SSL/key combination".
The failure that motivated it: host ``ollama.com`` with the dialog's defaults
(port 11434, no SSL) produced ``http://ollama.com:11434``, and the API key went
out in plain text to the wrong port.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlparse

DEFAULT_PORT = 11434
LOCAL_NAMES = ("localhost", "127.0.0.1", "::1")


@dataclass(frozen=True)
class Hint:
    level: str  # "error" | "warn" | "info"
    text: str


def is_ollama_cloud_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h == "ollama.com" or h.endswith(".ollama.com")


def normalize_ollama_cloud_url(url: str) -> str:
    """Ollama Cloud speaks HTTPS on 443 only; other hosts are returned unchanged."""
    parsed = urlparse(url if "://" in url else f"//{url}")
    host = (parsed.hostname or "").lower()
    if is_ollama_cloud_host(host):
        return f"https://{'ollama.com' if host == 'www.ollama.com' else host}"
    return url


def clean_api_key(key: str) -> str:
    """Strip whitespace/newlines (paste artefacts) and a leading ``Bearer ``."""
    key = "".join((key or "").split())
    if key.lower().startswith("bearer"):
        key = key[len("bearer"):].lstrip()
    return key


def parse_host(raw: str, port: int, use_ssl: bool) -> Tuple[str, int, bool]:
    """Reduce what a user typed into the host field to (host, port, use_ssl).

    ``https://gpu.example.org:8443/`` -> ``gpu.example.org``, 8443, SSL on. A
    scheme sets SSL; a ``:port`` suffix replaces the port field; paths go.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", port, use_ssl
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    if parsed.scheme in ("http", "https"):
        use_ssl = parsed.scheme == "https"
    try:
        if parsed.port:
            port = parsed.port
    except ValueError:
        pass
    return (parsed.hostname or raw), port, use_ssl


def base_url(host: str, port: int, use_ssl: bool) -> str:
    """The URL the native client connects to for a cleaned (host, port, ssl)."""
    host, port, use_ssl = parse_host(host, port, use_ssl)
    if is_ollama_cloud_host(host):
        return normalize_ollama_cloud_url(host)
    return f"{'https' if use_ssl else 'http'}://{host}:{port}"


def _is_local(host: str) -> bool:
    if host in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return "." not in host  # single-label name: intranet


def hints(host: str, port: int, use_ssl: bool, api_key: str) -> List[Hint]:
    """What to tell the user about the entered Ollama connection. Never blocks
    except for a missing host."""
    out: List[Hint] = []
    raw = (host or "").strip()
    if not raw:
        return [Hint("error", "Host fehlt.")]

    clean_host, port2, ssl2 = parse_host(raw, port, use_ssl)
    if raw != clean_host:
        out.append(Hint("info", f"Übernommen wird: Host {clean_host}, Port {port2}, "
                                f"{'HTTPS' if ssl2 else 'HTTP'}."))

    key = clean_api_key(api_key)
    if key != (api_key or ""):
        out.append(Hint("info", "Leerzeichen bzw. „Bearer“ im Schlüssel werden entfernt."))

    if is_ollama_cloud_host(clean_host):
        out.append(Hint("info", "Ollama Cloud: immer HTTPS auf Port 443, Port und SSL "
                                "werden ignoriert."))
        if not key:
            out.append(Hint("warn", "Ollama Cloud braucht einen API-Key."))
        return out

    if port2 == 443 and not ssl2:
        out.append(Hint("warn", "Port 443 ist üblicherweise HTTPS: SSL aktivieren?"))
    if ssl2 and port2 == DEFAULT_PORT:
        out.append(Hint("warn", "Ollama selbst spricht auf Port 11434 kein HTTPS. "
                                "Hinter einem Proxy ist es meist Port 443."))
    if key and not ssl2 and not _is_local(clean_host):
        out.append(Hint("warn", "Der Schlüssel würde unverschlüsselt über das Netz "
                                "gesendet. SSL aktivieren?"))
    return out
