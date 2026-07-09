"""Webindex lookup plugin — website/URL keyword index as a retrieval source - Claude Generated.

Exposes three agent tools over a :class:`WebIndexStore` (its own SQLite DB):

* ``search_webindex`` — keyword-match a user question against the central keyword
  catalogue and return ranked pages with snippets.
* ``fetch_page``        — return a page's cached text, live-fetching + caching on miss
  (the "live-fetch or cache" leg).
* ``list_webindex_keywords`` — browse the central keyword catalogue.

The store is filled by the operator-driven crawler (``indexer.crawl_site``, wired to
the ``alima webindex crawl`` CLI), not by these tools — the agent only *reads* the
index. This is keyword-based retrieval (no embeddings): the question's terms are
matched against the inverted ``page_keywords`` index.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List

from src.core.plugins.schema import BOOL, INT, TEXT, URL, ConfigField, PluginDoc

from ..registry import LookupToolSpec, register_lookup
from .store import WebIndexStore, _normalize_keyword

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "der", "die", "das", "ein", "eine", "und", "oder", "ist", "in", "im", "mit",
    "von", "zu", "zur", "zum", "auf", "für", "the", "and", "or", "of", "for", "to",
    "a", "an", "in", "on", "with", "what", "how", "wie", "was", "wer", "wo",
}


def _tokenize_query(query: str) -> List[str]:
    """Split a natural-language question into normalised search terms.

    Individual tokens (len >= 3, non-stopword) plus the full normalised query as a
    phrase term — so multi-word heading keywords like "Öffnungszeiten Bibliothek"
    are also matchable. - Claude Generated
    """
    if not query:
        return []
    terms: List[str] = []
    seen = set()
    for raw in re.split(r"[^\wöäüßÖÄÜ-]+", query.lower()):
        t = raw.strip()
        if len(t) >= 3 and t not in _STOPWORDS and t not in seen:
            seen.add(t)
            terms.append(t)
    phrase = _normalize_keyword(query)
    if phrase and phrase not in seen:
        terms.append(phrase)
    return terms


def _make_snippet(text: str, matched_keywords: List[str], snippet_chars: int) -> str:
    """First ~``snippet_chars`` chars of ``text`` starting at the first matched
    keyword occurrence; falls back to the document start. - Claude Generated"""
    if not text:
        return ""
    snippet_chars = max(80, int(snippet_chars or 400))
    lower = text.lower()
    start = 0
    for kw in matched_keywords:
        idx = lower.find(kw)
        if idx >= 0:
            start = max(0, idx - 60)
            break
    snippet = text[start : start + snippet_chars]
    if start > 0:
        snippet = "…" + snippet
    if len(text) > start + snippet_chars:
        snippet = snippet + "…"
    return snippet.strip()


@register_lookup
class WebIndexLookup:
    """Lookup plugin over a :class:`WebIndexStore`. - Claude Generated"""

    id = "webindex"
    label = "Webseiten-Index (RAG)"

    def __init__(self, **config: Any):
        self._config = config or {}
        # Own DB file per instance (LocalGndStore pattern). The connection_name is
        # unique per instance so concurrent instances don't share QtSql connections.
        self._store = WebIndexStore(self._config, f"webindex_{id(self)}")

    @property
    def store(self) -> WebIndexStore:
        return self._store

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(key="base_url", label="Basis-URL (gecrawlter Scope)",
                        kind=URL, default="",
                        help="Haupt-URL der Website, die indexiert wird. Auch "
                             "Anker für relative fetch_page-URLs, die der Agent übergibt."),
            ConfigField(key="db_path", label="DB-Pfad (optional)", kind=TEXT, default="",
                        help="Leer → Sibling der Haupt-DB (z.B. ~/.config/alima/webindex.db)."),
            ConfigField(key="max_results", label="Max. Treffer (search)", kind=INT, default=10),
            ConfigField(key="snippet_chars", label="Snippet-Länge", kind=INT, default=400),
            ConfigField(key="fetch_on_miss", label="Bei Cache-Miss live fetchen",
                        kind=BOOL, default=True),
            ConfigField(key="fetch_timeout", label="Fetch-Timeout (s)", kind=INT, default=20),
            # --- crawl / indexing parameters (used by `alima webindex crawl` + GUI) ---
            ConfigField(key="llm_provider", label="Crawl-LLM Provider", kind=TEXT, default="",
                        help="Leer → globaler ALIMA-Default (agentic). Keyword-"
                             "Extraktion via webindex_keywords-Workflow."),
            ConfigField(key="llm_model", label="Crawl-LLM Modell", kind=TEXT, default="",
                        help="Leer → zum Provider passender Default / globaler Default."),
            ConfigField(key="max_depth", label="Crawl-Tiefe", kind=INT, default=2,
                        help="BFS-Tiefe ab der Basis-URL (0 = nur die Startseite)."),
            ConfigField(key="max_pages", label="Max. Seiten", kind=INT, default=50),
            ConfigField(key="include_re", label="Include-Regex", kind=TEXT, default="",
                        help="Nur Kinder, deren URL matched (leer = alle)."),
            ConfigField(key="exclude_re", label="Exclude-Regex", kind=TEXT, default="",
                        help="Kinder, deren URL matched, überspringen."),
            ConfigField(key="min_chars", label="Min. Textzeichen/Seite", kind=INT, default=50,
                        help="Seiten mit weniger Text werden nicht indiziert."),
            ConfigField(key="max_keywords", label="Max. Keywords/Seite", kind=INT, default=15),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Schlagwort-basiertes Retrieval über eine eigene "
            "Webseiten/URL-Datenbank. Eine Frage wird gegen den zentralen "
            "Keyword-Katalog gematcht; Trefferseiten werden aus dem Cache "
            "(oder per Live-Fetch) geholt und dem Chatbot als Kontext gereicht.",
            input="Eine natürlichsprachliche Frage (search) bzw. eine URL (fetch).",
            output="Gerankte Treffer-URLs mit Snippet + matched Keywords (search) "
            "bzw. der Seitentext (fetch).",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [
            LookupToolSpec(
                name="search_webindex",
                description="Keyword-search the local website/URL index for a user "
                "question. Returns ranked page URLs with title, snippet and the "
                "matched keywords. Use this FIRST to find relevant pages, then "
                "fetch_page to read them.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User question / keywords"},
                        "max_results": {"type": "integer", "default": 10,
                                        "description": "Max pages to return"},
                    },
                    "required": ["query"],
                },
                method="search_keyword",
                cache_key_param="query",
            ),
            LookupToolSpec(
                name="fetch_page",
                description="Fetch the cached text of an indexed page URL; "
                "live-fetches + caches on miss. Returns the page title + text "
                "(truncated flag if capped).",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Absolute page URL"},
                        "max_chars": {"type": "integer", "default": 0,
                                      "description": "Cap text length (0 = full cached text)"},
                    },
                    "required": ["url"],
                },
                method="fetch_page",
                cache_key_param="url",
            ),
            LookupToolSpec(
                name="list_webindex_keywords",
                description="Browse the central keyword catalogue of the website "
                "index (with per-keyword page counts). Useful to see what the index "
                "covers before searching.",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 200,
                                  "description": "Max keywords to return"},
                        "contains": {"type": "string", "default": "",
                                     "description": "Case-insensitive substring filter"},
                    },
                    "required": [],
                },
                method="list_keywords",
                cache_key_param="",
                cacheable=False,
            ),
        ]

    # --- tool methods (called by the generated handler) ------------------- #
    def search_keyword(self, query: str, max_results: int = 0) -> dict:
        terms = _tokenize_query(str(query or ""))
        limit = int(max_results or 0) or int(self._config.get("max_results", 10) or 10)
        snippet_chars = int(self._config.get("snippet_chars", 400) or 400)
        hits = self._store.get_pages_for_keywords(terms, max_results=limit)
        result = []
        for h in hits:
            result.append({
                "url": h["url"],
                "title": h["title"],
                "snippet": _make_snippet(h["text"], h["matched_keywords"], snippet_chars),
                "matched_keywords": h["matched_keywords"],
                "matched_count": h["matched_count"],
                "score": h["score"],
            })
        return {"query": str(query), "terms": terms, "count": len(result), "hits": result}

    def fetch_page(self, url: str, max_chars: int = 0) -> dict:
        url = self._resolve_url(str(url or "").strip())
        if not url:
            return {"error": "url is required"}
        page = self._store.get_page(url)
        source = "cache"
        if page is None and self._config.get("fetch_on_miss", True):
            page = self._live_fetch(url)
            source = "live"
        if page is None:
            return {"url": url, "error": "page not in index and live-fetch disabled/failed"}
        text = page.get("text") or ""
        truncated = bool(page.get("text_truncated"))
        full_chars = len(text)
        cap = int(max_chars or 0)
        if cap and full_chars > cap > 0:
            text = text[:cap] + "\n[…truncated]"
            truncated = True
        return {
            "url": url,
            "title": page.get("title") or "",
            "text": text,
            "chars": len(text),
            "full_chars": full_chars,
            "source": source,
            "truncated": truncated,
        }

    def list_keywords(self, limit: int = 200, contains: str = "") -> dict:
        rows = self._store.list_keywords(limit=int(limit or 200), contains=str(contains or ""))
        return {
            "count": len(rows),
            "keywords": [
                {"keyword": r.get("keyword"), "display": r.get("display"),
                 "page_count": int(r.get("page_count") or 0)}
                for r in rows
            ],
        }

    # --- helpers ---------------------------------------------------------- #
    def _resolve_url(self, url: str) -> str:
        """Resolve a possibly-relative URL against the instance ``base_url``.

        The index stores absolute URLs, so a relative path the agent passes
        (e.g. ``/ub/ueber-uns``) must be joined to ``base_url`` *before* the cache
        lookup (else it misses) and before the SSRF-guarded live fetch (which
        rejects scheme-less URLs). Absolute URLs pass through unchanged.
        - Claude Generated
        """
        if not url:
            return ""
        if "://" in url:
            return url
        base = str(self._config.get("base_url") or "").strip()
        if not base:
            return url  # nothing to resolve against; the guard will reject clearly
        from urllib.parse import urljoin

        return urljoin(base, url)

    def _live_fetch(self, url: str):
        """Live-fetch + extract + cache a page not yet in the index. Returns the
        stored page row dict or None on failure. - Claude Generated"""
        try:
            from src.utils.input_sources.url_fetch import fetch_guarded_response
            from bs4 import BeautifulSoup

            from .indexer import _extract_main_text, _index_pdf

            timeout = int(self._config.get("fetch_timeout", 20) or 20)
            resp = fetch_guarded_response(url, timeout=timeout)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            status = int(getattr(resp, "status_code", 0) or 0)
            content = resp.content or b""
            looks_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")
            base_url = str(self._config.get("base_url") or "").strip() or url
            if looks_pdf:
                title, text, _ = _index_pdf(url, content, timeout)
            else:
                soup = BeautifulSoup(content, "html.parser")
                title, text = _extract_main_text(soup)
            truncated = len(text) > 200_000
            if truncated:
                text = text[:200_000]
            if not text:
                return None
            self._store.upsert_page(
                url=url, base_url=base_url, title=title, text=text,
                http_status=status, content_type=content_type, text_truncated=truncated,
            )
            return self._store.get_page(url)
        except Exception as e:  # noqa: BLE001 — live-fetch failure must not crash the tool
            logger.warning(f"Live fetch failed for {url}: {e}")
            return None