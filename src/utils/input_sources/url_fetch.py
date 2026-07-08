"""URL fetch / web-scrape input source (BeautifulSoup) - Claude Generated.

The scraper used to live inline in ``batch_processor`` (Debt D-9), unreachable
from ``execute_input_extraction``. It is extracted here as a reusable input source
+ a plain :func:`scrape_url` function (batch calls the same function), so URL input
is available across GUI/CLI/webapp/batch and is per-instance configurable
(user-agent, timeout, minimum length).
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Tuple

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from .registry import register_input_source

_DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_guarded_response(
    url: str,
    *,
    user_agent: str = _DEFAULT_UA,
    timeout: int = 30,
    allowlist: Any = None,
    max_bytes: Any = None,
):
    """The single guarded HTTP-fetch entry point for runtime/LLM-supplied URLs.

    Returns the raw (SSRF-guarded, size-capped, redirect-re-checked) ``requests``
    response so callers can shape it themselves — HTML main-content text
    (:func:`scrape_url`), or full-page text + PDF detection (the MCP ``scrape_url``
    tool). ``allowlist``/``max_bytes`` default to ``SystemConfig.url_fetch_*``.
    - Claude Generated
    """
    from src.utils.net_guard import fetch_guarded, url_fetch_guard_settings

    if allowlist is None or max_bytes is None:
        guard = url_fetch_guard_settings()
        allowlist = guard["allowlist"] if allowlist is None else allowlist
        max_bytes = guard["max_bytes"] if max_bytes is None else max_bytes
    resp = fetch_guarded(
        url, allowlist=allowlist, timeout=timeout, max_bytes=int(max_bytes),
        user_agent=user_agent,
    )
    resp.raise_for_status()
    return resp


def scrape_url(
    url: str,
    *,
    user_agent: str = _DEFAULT_UA,
    timeout: int = 30,
    min_chars: int = 50,
    logger: Any = None,
    allowlist: Any = None,
    max_bytes: Any = None,
) -> str:
    """Fetch ``url`` and return cleaned main-content text.

    Verbatim behaviour of the former ``batch_processor`` URL branch (heuristic
    main/article/div.content extraction, tag stripping, whitespace cleanup, and
    the < ``min_chars`` guard), just parameterised.

    The URL is runtime-supplied (LLM tool / batch input), so the fetch goes
    through the strict SSRF guard (``net_guard.fetch_guarded``): http(s) only,
    no private/loopback targets unless allowlisted, redirects re-checked per
    hop, body capped. ``allowlist``/``max_bytes`` default to
    ``SystemConfig.url_fetch_allowlist`` / ``url_fetch_max_bytes``. - Claude Generated
    """
    import requests
    from bs4 import BeautifulSoup

    if logger:
        logger.info(f"Fetching URL: {url}")
    try:
        response = fetch_guarded_response(
            url, user_agent=user_agent, timeout=timeout,
            allowlist=allowlist, max_bytes=max_bytes,
        )
    except (requests.RequestException, ValueError, RuntimeError) as e:
        if logger:
            logger.error(f"Failed to fetch URL: {e}")
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    main_content = (
        soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    )
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        text = (
            body.get_text(separator="\n", strip=True)
            if body
            else soup.get_text(separator="\n", strip=True)
        )

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)

    if not text or len(text.strip()) < min_chars:
        raise ValueError(f"URL scraping resulted in too little text ({len(text)} chars)")
    if logger:
        logger.info(f"URL scraping completed: {len(text)} characters extracted")
    return text


@register_input_source
class UrlFetchInputSource:
    id = "url_fetch"
    label = "URL (Web-Scrape)"

    def __init__(self, **config: Any):
        self._config = config or {}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(key="user_agent", label="User-Agent", kind=TEXT, default=_DEFAULT_UA),
            ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=30),
            ConfigField(key="min_chars", label="Min. Zeichen", kind=INT, default=50),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Lädt eine Webseite und extrahiert den Hauptinhalt (BeautifulSoup; "
            "main/article/div.content-Heuristik, entfernt Skripte/Nav/Footer).",
            input="Eine http(s)-URL.",
            output="Bereinigter Text-Hauptinhalt der Seite.",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        if input_type == "url":
            return True
        return input_type == "auto" and str(source).strip().lower().startswith(("http://", "https://"))

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        text = scrape_url(
            source,
            user_agent=self._config.get("user_agent", _DEFAULT_UA) or _DEFAULT_UA,
            timeout=int(self._config.get("timeout", 30) or 30),
            min_chars=int(self._config.get("min_chars", 50) or 50),
            logger=logger,
        )
        return text, f"URL: {source}", "url_scrape"
