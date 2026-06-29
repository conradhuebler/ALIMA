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
      ``ddc``, ``dk``.
    * ``TITLE_RECORDS`` → ``record`` (raw bibliographic dict), ``label`` (title).
    * ``SUBJECT_FACETS``→ ``code`` (facet value), ``count``, ``label`` (translated),
      ``extra["facet"]`` (facet field name).
    * ``CLASSIFICATION``→ ``code`` (DK/RVK), ``label``, ``count``.

    ``display_count`` is the *display-only* hit count (F-4): it never feeds ranking
    (see ``src/core/gnd_search_core.py`` count-landmine). ``None`` means "use
    ``count`` for display".
    """

    label: str = ""
    gnd_ids: Set[str] = field(default_factory=set)
    count: int = 0
    display_count: Optional[int] = None
    ddc: Set[str] = field(default_factory=set)
    dk: Set[str] = field(default_factory=set)
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
        """Build from the legacy ``{term: {keyword: {count, gndid, ddc, dk}}}`` shape."""
        per_term: Dict[str, List[ResultItem]] = {}
        for term, keywords in (per_term_dict or {}).items():
            items: List[ResultItem] = []
            for kw, data in (keywords or {}).items():
                data = data or {}
                dc = data.get("display_count")
                items.append(
                    ResultItem(
                        label=kw,
                        gnd_ids=set(data.get("gndid", set()) or set()),
                        count=int(data.get("count", 0) or 0),
                        display_count=None if dc is None else int(dc),
                        ddc=set(data.get("ddc", set()) or set()),
                        dk=set(data.get("dk", set()) or set()),
                    )
                )
            per_term[term] = items
        return cls(SearchCapability.GND_KEYWORDS, per_term=per_term, errors=dict(errors or {}))

    def to_gnd_keywords(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Render back to the legacy ``{term: {keyword: {count, gndid, ddc, dk}}}``
        shape. ``display_count`` is emitted only when set (additive, F-4)."""
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for term, items in self.per_term.items():
            kw_map: Dict[str, Dict[str, Any]] = {}
            for it in items:
                entry: Dict[str, Any] = {
                    "count": it.count,
                    "gndid": set(it.gnd_ids),
                    "ddc": set(it.ddc),
                    "dk": set(it.dk),
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
    """

    id: str
    label: str
    capabilities: Set[SearchCapability]

    def is_available(self, cfg: Any = None) -> bool: ...

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        **opts: Any,
    ) -> ProviderResult: ...
