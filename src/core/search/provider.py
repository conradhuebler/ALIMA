"""Capability-based search-provider standard - Claude Generated.

The single standard for "a search source" in ALIMA. A provider declares which
:class:`SearchCapability` it supports and returns a typed :class:`ProviderResult`
instead of the ad-hoc nested dicts the legacy suggester contract used. This makes
the GND-keyword shape and the bibliographic-record / facet shape *explicit
variants of one standard* (finc no longer has to break the contract).

This module is **Qt-free** and lives in ``core/`` so both the orchestrator and the
MCP tool layer can import it without pulling in PyQt. The optional ``progress``
callback replaces the ``currentTerm`` Qt signal of ``BaseSuggester``.

Conversion helpers (:meth:`ProviderResult.from_gnd_keywords` / ``to_gnd_keywords``
and the finc record/facet variants) keep the legacy dict shapes round-trippable,
so existing callers keep working while the migration proceeds (facade discipline).

See ``docs/search_provider_plugins.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable


# WP2 raw cache: which request params are part of a source's cache key. Every
# source keys on ``search_type``; a provider declaring more (swb pages on
# ``max_pages``, finc keys on its facet set) says so via the class attribute
# ``raw_cache_param_keys``, which a copied plugin dir carries with it. - Claude Generated
_DEFAULT_RAW_CACHE_PARAM_KEYS: tuple = ("search_type",)


def raw_cache_params_for(
    source: str,
    *,
    search_type: str = "kw",
    max_pages: Optional[int] = 5,
    facets: Optional[Any] = None,
) -> Dict[str, Any]:
    """Canonical ``search_response_cache`` params for a source.

    The write seam and the aggregate/pipeline readers all build the cache-key
    params through this one function, so a non-default search (``title``, a custom
    ``max_pages``) is looked up under exactly the key it was stored with instead of
    silently missing. Keys with ``None`` values are dropped.

    ``source`` is a *cache-source label*: for GND sources it equals the provider id,
    but a provider may emit others (``catalog_titles``). Labels with no registered
    provider — and providers declaring nothing — use the default key set.
    - Claude Generated
    """
    # Lazy: registry imports this module, so a top-level import would cycle.
    from .registry import raw_cache_param_keys

    keys = raw_cache_param_keys(source)
    values: Dict[str, Any] = {"search_type": search_type, "max_pages": max_pages, "facets": facets}
    return {k: values[k] for k in keys if values.get(k) is not None}


class SearchCapability(Enum):
    """What a search source can do. A provider may declare several."""

    GND_KEYWORDS = "gnd_keywords"      # term -> GND keyword candidates (lobid, swb, catalog, gnd_local)
    TITLE_RECORDS = "title_records"    # query -> bibliographic records (finc, catalog titles)
    SUBJECT_FACETS = "subject_facets"  # term -> classification distribution (finc udk/rvk facets)
    CLASSIFICATION = "classification"  # title/keyword -> DK/RVK codes (catalog DK lookup)


@dataclass
class ResultItem:
    """One typed result. Which fields are populated depends on the capability:

    * ``GND_KEYWORDS``  → ``label``, ``gnd_ids``, ``count``, ``display_count``,
      ``classifications``.
    * ``TITLE_RECORDS`` → ``record`` (raw bibliographic dict), ``label`` (title).
    * ``SUBJECT_FACETS``→ ``code`` (facet value), ``count``, ``label`` (translated),
      ``extra["facet"]`` (facet field name).
    * ``CLASSIFICATION``→ ``code`` (DK/RVK), ``label``, ``count``.

    ``classifications`` is the canonical ``{system: codes}`` dict (WP-D1):
    dk/ddc/rvk are equal-rank system keys, only non-empty systems are carried.

    ``display_count`` is the *display-only* hit count (F-4): it never feeds ranking
    (see ``src/core/gnd_search_core.py`` count-landmine). ``None`` means "use
    ``count`` for display".
    """

    label: str = ""
    gnd_ids: Set[str] = field(default_factory=set)
    count: int = 0
    display_count: Optional[int] = None
    classifications: Dict[str, Set[str]] = field(default_factory=dict)
    record: Optional[Dict[str, Any]] = None
    code: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResult:
    """The typed result of one provider ``search`` call for one capability.

    ``per_term`` maps each query term to its result items. ``per_term_meta`` holds
    per-term metadata that is not a result item (e.g. finc ``result_count`` /
    raw facet blocks). ``errors`` maps a term to a *source-failure* message — an
    empty ``per_term[term]`` paired with an ``errors[term]`` means "source failed",
    NOT "no match" (mirrors ``BaseSuggester.last_errors``).
    """

    capability: SearchCapability
    per_term: Dict[str, List[ResultItem]] = field(default_factory=dict)
    per_term_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    # --- GND keyword shape -------------------------------------------------
    @classmethod
    def from_gnd_keywords(
        cls,
        per_term_dict: Dict[str, Dict[str, Dict[str, Any]]],
        errors: Optional[Dict[str, str]] = None,
    ) -> "ProviderResult":
        """Build from the canonical suggester ``{term: {keyword: {count, gnd_ids, classifications}}}`` shape (contract v2)."""
        per_term: Dict[str, List[ResultItem]] = {}
        for term, keywords in (per_term_dict or {}).items():
            items: List[ResultItem] = []
            for kw, data in (keywords or {}).items():
                data = data or {}
                dc = data.get("display_count")
                classifications = {
                    system: set(codes or [])
                    for system, codes in (data.get("classifications") or {}).items()
                    if codes
                }
                items.append(
                    ResultItem(
                        label=kw,
                        gnd_ids=set(data.get("gnd_ids", set()) or set()),
                        count=int(data.get("count", 0) or 0),
                        display_count=None if dc is None else int(dc),
                        classifications=classifications,
                    )
                )
            per_term[term] = items
        return cls(SearchCapability.GND_KEYWORDS, per_term=per_term, errors=dict(errors or {}))

    def to_gnd_keywords(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Render to the canonical nested ``{term: {keyword: {count, gnd_ids,
        classifications}}}`` shape (WP-D1). ``display_count`` is emitted only
        when set (additive, F-4)."""
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for term, items in self.per_term.items():
            kw_map: Dict[str, Dict[str, Any]] = {}
            for it in items:
                entry: Dict[str, Any] = {
                    "count": it.count,
                    "gnd_ids": set(it.gnd_ids),
                    "classifications": {
                        system: set(codes)
                        for system, codes in it.classifications.items()
                        if codes
                    },
                }
                if it.display_count is not None:
                    entry["display_count"] = it.display_count
                kw_map[it.label] = entry
            out[term] = kw_map
        return out

    # --- finc record / facet shape ----------------------------------------
    @classmethod
    def from_finc_records(
        cls,
        finc_dict: Dict[str, Dict[str, Any]],
        errors: Optional[Dict[str, str]] = None,
    ) -> "ProviderResult":
        """Build a ``TITLE_RECORDS`` result from the finc ``{term: {records,
        result_count, facets, errors}}`` shape. ``result_count`` / ``facets`` /
        per-term ``errors`` are kept in ``per_term_meta`` so the shape round-trips.
        """
        per_term: Dict[str, List[ResultItem]] = {}
        per_term_meta: Dict[str, Dict[str, Any]] = {}
        for term, entry in (finc_dict or {}).items():
            entry = entry or {}
            recs = entry.get("records", []) or []
            per_term[term] = [
                ResultItem(label=str((r or {}).get("title", "")), record=r) for r in recs
            ]
            per_term_meta[term] = {
                "result_count": int(entry.get("result_count", len(recs)) or 0),
                "facets": entry.get("facets", {}) or {},
                "errors": list(entry.get("errors", []) or []),
            }
        return cls(
            SearchCapability.TITLE_RECORDS,
            per_term=per_term,
            per_term_meta=per_term_meta,
            errors=dict(errors or {}),
        )

    def to_finc_records(self) -> Dict[str, Dict[str, Any]]:
        """Render a ``TITLE_RECORDS`` result back to the finc dict shape."""
        out: Dict[str, Dict[str, Any]] = {}
        for term, items in self.per_term.items():
            meta = self.per_term_meta.get(term, {})
            out[term] = {
                "records": [it.record for it in items if it.record is not None],
                "result_count": meta.get("result_count", len(items)),
                "facets": meta.get("facets", {}),
                "errors": meta.get("errors", []),
            }
        return out


@dataclass
class ProviderToolSpec:
    """Declarative MCP-tool descriptor a provider exposes (P3).

    Pure data (no MCP import) so the tool layer can *generate* the ``ToolDefinition``
    + handler from it instead of hand-writing both. ``result_shape`` selects the
    MCP serializer; the remaining flags capture the (historical) per-tool nuances so
    generation stays byte-faithful to the previous hand-written handlers:

    * ``source_label`` — the ``"source"`` field in the JSON response.
    * ``include_errors`` — whether the response carries an ``"errors"`` block.
    * ``cached`` — GND-keyword tools that use the mapping-first cache for the
      *default* options (``default_opts``) and the raw suggester otherwise.
    """

    name: str
    capability: "SearchCapability"
    description: str
    parameters: Dict[str, Any]
    provider_id: str = ""  # set by the registry when collecting specs
    result_shape: str = "gnd_keywords"  # gnd_keywords | title_records | finc
    source_label: str = ""
    include_errors: bool = True
    cached: bool = False
    add_gnd_urls: bool = False  # GND-keyword tools that enrich rows with gnd_urls
    unavailable_message: str = ""  # error returned when the source isn't initialised
    default_opts: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    """Structural contract every search source implements.

    ``id`` / ``label`` / ``capabilities`` are class attributes (read by the
    registry without instantiation). ``is_available`` gates the provider on its
    config; ``search`` runs one capability. ``progress`` is an optional per-term
    callback (the Qt-free replacement for ``currentTerm``).

    ``raw_cache_param_keys`` is optional (default ``("search_type",)``): the WP2
    raw-cache key params, declared here rather than in a central map so a copied
    plugin keeps its own key shape. See :func:`raw_cache_params_for`.
    """

    id: str
    label: str
    capabilities: Set[SearchCapability]
    raw_cache_param_keys: tuple

    def is_available(self, cfg: Any = None) -> bool: ...

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        **opts: Any,
    ) -> ProviderResult: ...
