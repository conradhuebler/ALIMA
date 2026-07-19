"""Lightweight UI string catalog (DE/EN) - Claude Generated.

Shared by the PyQt chat panel, the ``UnifiedMessageRenderer`` and the webapp
chrome. Flat dot-keys live in ``locales/<lang>.json`` at the repo root;
:func:`t` resolves active language → German → key literal and **never
raises** — a missing or broken catalog degrades to key literals, not to a
crash. Keys under the ``js.`` prefix are exported to the browser as
``window.__alimaI18n`` via :func:`catalog_for_js` (used by the shared
``alima_render.js`` in both the QWebEngine scaffold and the webapp template).

The chat panel's DE/EN toggle controls the *LLM answer* language and is a
separate axis — it does not touch this UI-chrome language.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"
DEFAULT_LANGUAGE = "de"

_language: str = DEFAULT_LANGUAGE
_catalogs: Dict[str, Dict[str, str]] = {}


def _catalog(lang: str) -> Dict[str, str]:
    """Load (and cache) one language catalog; failure-tolerant."""
    if lang in _catalogs:
        return _catalogs[lang]
    data: Dict[str, str] = {}
    path = LOCALES_DIR / f"{lang}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = {str(k): str(v) for k, v in raw.items()}
    except OSError:
        logger.warning("i18n: no catalog for language %r (%s)", lang, path)
    except (json.JSONDecodeError, ValueError):
        logger.error("i18n: broken catalog %s", path, exc_info=True)
    _catalogs[lang] = data
    return data


def set_language(lang: str) -> None:
    """Set the active UI language (falls back to the default on empty input)."""
    global _language
    _language = (lang or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE


def get_language() -> str:
    return _language


def t(key: str, **fmt) -> str:
    """Translate ``key``; fallback chain active → default → key literal."""
    for lang in (_language, DEFAULT_LANGUAGE):
        value = _catalog(lang).get(key)
        if value is not None:
            if not fmt:
                return value
            try:
                return value.format(**fmt)
            except (KeyError, IndexError, ValueError):
                # A broken placeholder must not take the UI down.
                return value
    return key


def catalog_for_js() -> Dict[str, str]:
    """Browser-facing subset (``js.*`` keys), default merged under active."""
    merged = dict(_catalog(DEFAULT_LANGUAGE))
    if _language != DEFAULT_LANGUAGE:
        merged.update(_catalog(_language))
    return {k: v for k, v in merged.items() if k.startswith("js.")}


def _reset_for_tests() -> None:
    """Test hook: drop caches and restore the default language."""
    global _language
    _language = DEFAULT_LANGUAGE
    _catalogs.clear()
