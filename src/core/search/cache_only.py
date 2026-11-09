"""Cache-only mode for network search sources - Claude Generated.

An instance of a network source (``catalog``, ``finc``) whose settings carry
``cache_only = True`` answers from local caches only and makes no network
request. Age limits are ignored on that path: a stale entry is still an answer,
and without that the mode would mostly come back empty.

A term without a cache entry is *not* a "no match". It is reported through the
result's ``errors`` with the marker message below, for two reasons:

* every cache layer already refuses to store an errored term as an empty hit,
  so a cache-only miss can never poison the mapping/DK caches;
* reporters can tell the marker apart from a real source failure and fold all
  misses of one run into a single summary line (``summary``) instead of one
  warning per term.

Qt-free, no network, no config I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.core.plugins.schema import BOOL, ConfigField

CACHE_ONLY_KEY = "cache_only"

# Stable prefix: reporters match on it (``is_cache_miss``), so the rest of the
# text may change without breaking them.
CACHE_MISS_PREFIX = "cache_only:"
CACHE_MISS_MESSAGE = (
    f"{CACHE_MISS_PREFIX} nicht im Cache, keine Netzabfrage (Modus „nur Cache“)"
)

# Appended to the generated agent tool description of a cache-only instance.
TOOL_DESCRIPTION_NOTE = (
    "\n\nCache-only mode (nur Cache) is active for this source: it makes NO network "
    "request, so expect no network errors or timeouts. Results come from the local "
    "cache only and may be older than usual. A term without a cache entry returns no "
    "hits and is listed under `cache_only.missing_terms`; that is not evidence that "
    "the catalog lacks it."
)


def cache_only_field() -> ConfigField:
    """The per-instance ``cache_only`` switch (BOOL, default off) - Claude Generated."""
    return ConfigField(
        key=CACHE_ONLY_KEY,
        label="Nur Cache (kein Netz)",
        kind=BOOL,
        default=False,
        help="Keine Netzabfragen an diese Quelle: Ergebnisse kommen nur aus dem "
        "lokalen Cache, auch aus älteren Einträgen. Begriffe ohne Cache-Eintrag "
        "bleiben ohne Treffer.",
    )


def is_cache_only(settings: Optional[Mapping[str, Any]]) -> bool:
    """Whether a settings dict switches cache-only on (tolerates str/bool) - Claude Generated."""
    if not settings:
        return False
    return bool(cache_only_field().coerce(settings.get(CACHE_ONLY_KEY)))


# Instance state ("Zustand"): An / Nur Cache / Aus. Stored as ``enabled`` plus,
# for search sources with a cache path, the ``cache_only`` setting. The settings
# tab and the institution presets (``src/utils/preset_apply.py``) share these
# helpers; they lived in ``src/ui/plugin_settings_tab.py`` until the preset code
# needed them without Qt. - Claude Generated
STATE_ON = "on"
STATE_CACHE_ONLY = "cache_only"
STATE_OFF = "off"
STATES = (STATE_ON, STATE_CACHE_ONLY, STATE_OFF)

# "Nur Cache" for a type: offered, shown disabled with a note, or not shown.
OFFER_CACHE_ONLY = "offered"
OFFER_BLOCKED = "blocked"
OFFER_NONE = "none"


def cache_only_offer(meta) -> str:
    """Whether the state "Nur Cache" is offered for a plugin type - Claude Generated.

    Offered when the type declares the ``cache_only`` field; shown disabled when
    the type carries a ``cache_only_note`` (a network source without a cache
    path); otherwise not shown (local sources, input sources, lookups).
    """
    if any(f.key == CACHE_ONLY_KEY for f in (getattr(meta, "config_fields", None) or [])):
        return OFFER_CACHE_ONLY
    if getattr(meta, "cache_only_note", ""):
        return OFFER_BLOCKED
    return OFFER_NONE


def instance_state(inst, offer: str) -> str:
    """The state an instance's stored values stand for - Claude Generated"""
    if not inst.enabled:
        return STATE_OFF
    if offer == OFFER_CACHE_ONLY and is_cache_only(inst.settings):
        return STATE_CACHE_ONLY
    return STATE_ON


def apply_state(inst, state: str, offer: str) -> None:
    """Write a chosen state back into ``enabled`` / ``settings["cache_only"]``.

    Only what the state decides is touched: "Aus" keeps a stored ``cache_only``
    value, and "An" writes ``cache_only = False`` only where the key already
    exists, so an untouched instance is saved unchanged. Types without the
    offer never get the key. - Claude Generated
    """
    inst.enabled = state != STATE_OFF
    if offer != OFFER_CACHE_ONLY or state == STATE_OFF:
        return
    if state == STATE_CACHE_ONLY:
        inst.settings[CACHE_ONLY_KEY] = True
    elif CACHE_ONLY_KEY in inst.settings:
        inst.settings[CACHE_ONLY_KEY] = False


def is_cache_miss(message: Any) -> bool:
    """Whether an error message is the cache-only miss marker - Claude Generated."""
    return isinstance(message, str) and message.startswith(CACHE_MISS_PREFIX)


def split_cache_misses(
    errors: Optional[Mapping[str, str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Split ``{"<source>:<term>": message}`` into real failures and misses.

    Returns ``(failures, misses_by_source)``; ``misses_by_source`` maps a source
    id to the terms it had no cache entry for, in input order. - Claude Generated
    """
    failures: Dict[str, str] = {}
    misses: Dict[str, List[str]] = {}
    for key, message in (errors or {}).items():
        if is_cache_miss(message):
            source, _, term = str(key).partition(":")
            misses.setdefault(source, []).append(term)
        else:
            failures[key] = message
    return failures, misses


def summary(source: str, missed: int, total: int) -> str:
    """One-line run summary of cache-only misses - Claude Generated."""
    return (
        f"{source}: {missed} von {total} Suchbegriffen nicht im Cache "
        "(Modus „nur Cache“, keine Netzabfrage)"
    )
