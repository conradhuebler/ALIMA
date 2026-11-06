"""Local subject-indexing database as a lookup plugin - Claude Generated.

Reads one SQLite file in the plugin's own schema (``schema.py``, version 1):
local headings with their kind, K10plus PPN, finished local string and variant
forms, and local notation entries (DK, RVK) linked to the headings. A library
fills that file with its own importer; for the Freiberg Access database that is
``scripts/access_to_sql.py import-geo``.

Answers three questions: which local headings match a term (all kinds, optional
filter), which finished WinIBW lines a list of terms yields (found and not found
reported separately), and what the local data holds for a notation. The plugin
proposes, it never decides: a workflow step that uses it keeps the assigned
classification untouched.

Without a database file, or with a file in another schema, ``is_available()`` is
False: the tool generator then offers none of its tools, and every method
answers with an ``error`` entry instead of raising.

Registered as a built-in lookup (``src/utils/lookups/__init__.py`` imports the
package). Every installation therefore gets a ``geo_sachdb`` instance in the
lookup list (``plugin_migration.ensure_lookup_instances``); it stays inactive
without ``db_path``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc
from src.utils.lookups.registry import LookupToolSpec, register_lookup

from .schema import KINDS, normalize_notation
from .store import ClassificationRecord, GeoSachdbStore, HeadingRecord, fold, normalize_kinds

logger = logging.getLogger(__name__)

_KIND_PARAM = {
    "type": "array",
    "items": {"type": "string", "enum": list(KINDS)},
    "description": "Optional: only these kinds. Empty = all kinds.",
}


@register_lookup
class GeoSachdbLookup:
    """Lookup over a local subject database in schema v1. - Claude Generated"""

    id = "geo_sachdb"
    label = "Lokale Sacherschließungsdatenbank (SQLite)"

    def __init__(self, **config: Any):
        self._config = dict(config or {})
        try:
            self._max_results = max(1, int(self._config.get("max_results") or 10))
        except (TypeError, ValueError):
            self._max_results = 10
        self._store = GeoSachdbStore(str(self._config.get("db_path") or ""))

    @property
    def store(self) -> GeoSachdbStore:
        return self._store

    # ── plugin contract ────────────────────────────────────────────────────

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(key="db_path", label="SQLite-Datei", kind=TEXT, default="",
                        gates_availability=True,
                        help="Datei im Schema geo_sachdb v1 (für die Freiberger Access-Datenbank: "
                             "scripts/access_to_sql.py import-geo). Leer, nicht vorhanden oder "
                             "anderes Schema: Plugin nicht verfügbar, keine Werkzeuge."),
            ConfigField(key="max_results", label="Max. Treffer", kind=INT, default=10),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Lokale Sacherschließungsdatenbank als Nachschlagequelle: lokale "
            "Schlagwörter (Sachschlagwort, Geographikum, Person, Körperschaft, sonstiges) zu "
            "Begriffen, fertige WinIBW-Zeilen (5550 für Schlagwörter, 6700 für Notationen) und der "
            "Abgleich einer Notation mit den lokalen Einträgen.",
            input="Ein Suchbegriff, eine Liste von Begriffen oder eine Notation (z. B. 'DK 553.62').",
            output="Lokale Schlagwörter mit Art, PPN, fertiger Zeile und verknüpften Notationen; "
            "gefundene und nicht gefundene Begriffe getrennt; Herkunft der Datenbank.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [
            LookupToolSpec(
                name="geo_sachdb_search",
                description=(
                    "Search the library's local subject-indexing database for one term. Matches "
                    "local headings and their variant forms (umlauts and 'ae/oe/ue' spellings "
                    "meet), ranked: identical heading, identical variant, all words, substring. "
                    "Each hit has kind, K10plus PPN, the finished WinIBW line (5550) and the "
                    "linked local notations with their 6700 lines. Local proposals only."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Term to look up"},
                        "kinds": _KIND_PARAM,
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                method="search_terms",
                cacheable=False,  # local file read; caching only adds staleness
            ),
            LookupToolSpec(
                name="geo_sachdb_lines",
                description=(
                    "Finished WinIBW lines for a list of subject terms from the local database: "
                    "only identical headings or variant forms count. Returns per found term the "
                    "local headings with '5550 !PPN!' and their notations with '6700 …', and the "
                    "terms without a local heading separately (missing terms are normal: the "
                    "database covers what the subject needs locally)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "terms": {"type": "array", "items": {"type": "string"},
                                  "description": "Subject terms, e.g. the final keywords"},
                        "kinds": _KIND_PARAM,
                    },
                    "required": ["terms"],
                },
                method="lookup_terms",
                cacheable=False,
            ),
            LookupToolSpec(
                name="geo_sachdb_notation",
                description="Check a notation (DK or RVK) against the local entries: exact "
                "entries with label, PPN and 6700 line, the local headings that use it, and the "
                "nearest broader local DK notations when there is no exact entry.",
                parameters={
                    "type": "object",
                    "properties": {
                        "notation": {"type": "string", "description": "e.g. 'DK 553.62' or 'QP 925'"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["notation"],
                },
                method="match_notation",
                cacheable=False,
            ),
        ]

    # ── availability ───────────────────────────────────────────────────────

    def availability(self) -> Dict[str, Any]:
        return self._store.status()

    def is_available(self) -> bool:
        status = self._store.status()
        if not status["available"]:
            logger.info(f"geo_sachdb: nicht verfügbar: {status['reason']}")
        return bool(status["available"])

    def _unavailable(self) -> Optional[Dict[str, Any]]:
        status = self._store.status()
        if status["available"]:
            return None
        return {"error": f"geo_sachdb nicht verfügbar: {status['reason']}", "available": False}

    # ── shaping ────────────────────────────────────────────────────────────

    @staticmethod
    def _classification_dict(rec: ClassificationRecord, ambiguous: bool = False) -> Dict[str, Any]:
        item = {"system": rec.system, "notation": rec.notation, "label": rec.label,
                "ppn": rec.ppn or None, "line": rec.line}
        if ambiguous:
            item["ambiguous"] = True
        return item

    def _heading_dict(self, rec: HeadingRecord, matched: str = "", form: str = "",
                      score: float = 0.0) -> Dict[str, Any]:
        by_id = self._store.classifications()
        item: Dict[str, Any] = {
            "id": rec.id,
            "label": rec.label,
            "kind": rec.kind,
            "ppn": rec.ppn or None,
            "line": rec.line,
            "line_form": rec.line_form,
            "variants": rec.variants[:10],
            "classifications": [self._classification_dict(by_id[cid], amb)
                                for cid, amb in rec.links if cid in by_id],
        }
        if matched:
            item.update({"matched": matched, "matched_form": form, "score": score})
        return item

    # ── tools ──────────────────────────────────────────────────────────────

    def _ranked(self, query: str, min_score: float, kinds: Optional[set]) -> List[Any]:
        q = fold(query)
        if not q:
            return []
        words = set(q.split())
        scored = []
        for rec in self._store.headings():
            if kinds is not None and rec.kind not in kinds:
                continue
            best = (0.0, "", "")
            if rec.folded == q:
                best = (100.0, "ansetzung_exakt", rec.label)
            else:
                for vtext, vfold in zip(rec.variants, rec.folded_variants):
                    if vfold == q:
                        best = (90.0, "verweisung_exakt", vtext)
                        break
            if best[0] == 0.0 and words <= set(rec.folded.split()):
                best = (60.0, "ansetzung_woerter", rec.label)
            if best[0] == 0.0:
                for vtext, vfold in zip(rec.variants, rec.folded_variants):
                    if words <= set(vfold.split()):
                        best = (50.0, "verweisung_woerter", vtext)
                        break
            if best[0] == 0.0 and q in rec.folded:
                best = (30.0, "ansetzung_teil", rec.label)
            if best[0] == 0.0:
                for vtext, vfold in zip(rec.variants, rec.folded_variants):
                    if q in vfold:
                        best = (25.0, "verweisung_teil", vtext)
                        break
            if best[0] > 0 and best[0] >= float(min_score or 0):
                scored.append((best, rec))
        scored.sort(key=lambda item: (-item[0][0], len(item[1].label), item[1].id))
        return scored

    def search_terms(self, query: str, max_results: Optional[int] = None, min_score: float = 0.0,
                     kinds: Any = None) -> Dict[str, Any]:
        """Ranked local headings for ``query``. - Claude Generated

        Scores: identical heading 100, identical variant 90, every query word in
        the heading 60 (variant 50), query as substring of the heading 30
        (variant 25). ``min_score`` lets a deterministic caller keep only the
        identical matches; ``kinds`` restricts to some kinds (default: all).
        """
        blocked = self._unavailable()
        if blocked:
            return blocked
        try:
            wanted = normalize_kinds(kinds)
        except ValueError as exc:
            return {"error": str(exc), "results": [], "source": self._store.provenance()}
        limit = int(max_results or self._max_results)
        scored = self._ranked(query, min_score, wanted)
        return {
            "query": query,
            "total": len(scored),
            "results": [self._heading_dict(rec, m, f, s) for (s, m, f), rec in scored[:limit]],
            "source": self._store.provenance(),
        }

    def lookup_terms(self, terms: Any, kinds: Any = None, min_score: float = 90.0,
                     max_per_term: int = 3) -> Dict[str, Any]:
        """Finished lines for several terms; found and not found apart. - Claude Generated

        Only identical headings (100) and identical variants (90) count by
        default. ``lines`` lists every 5550 and 6700 line once, in the order the
        terms came in; a notation without a line rule (non-DK without PPN) is in
        ``classifications`` with ``line: null`` and not in ``lines``.
        """
        blocked = self._unavailable()
        if blocked:
            return blocked
        try:
            wanted = normalize_kinds(kinds)
        except ValueError as exc:
            return {"error": str(exc), "found": [], "not_found": [], "lines": [],
                    "source": self._store.provenance()}
        items = [terms] if isinstance(terms, str) else list(terms or [])
        found: List[Dict[str, Any]] = []
        not_found: List[str] = []
        lines: List[str] = []
        seen_terms = set()
        for raw in items:
            term = str(raw or "").strip()
            if not term or fold(term) in seen_terms:
                continue
            seen_terms.add(fold(term))
            scored = self._ranked(term, min_score, wanted)[: max(1, int(max_per_term or 3))]
            if not scored:
                not_found.append(term)
                continue
            headings = [self._heading_dict(rec, m, f, s) for (s, m, f), rec in scored]
            found.append({"term": term, "headings": headings})
            for head in headings:
                for line in [head["line"], *(c["line"] for c in head["classifications"])]:
                    if line and line not in lines:
                        lines.append(line)
        return {"found": found, "not_found": not_found, "lines": lines,
                "source": self._store.provenance()}

    def match_notation(self, notation: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """Local entries for a notation, headings using it, nearest broader DK. - Claude Generated"""
        blocked = self._unavailable()
        if blocked:
            return blocked
        from src.utils.lookups.local_scheme import digits_of

        limit = int(max_results or self._max_results)
        system, code = normalize_notation(notation)
        exact = self._store.by_notation(system, code)
        broader: List[ClassificationRecord] = []
        if not exact and system == "DK":
            digits = digits_of(code)
            above = [r for r in self._store.classifications().values()
                     if r.system == "DK" and r.digits and digits.startswith(r.digits) and r.digits != digits]
            above.sort(key=lambda r: (-len(r.digits), r.notation, r.id))
            broader = above[:3]
        linked: List[Dict[str, Any]] = []
        total_linked = 0
        ids = {r.id for r in exact}
        if ids:
            for rec in self._store.headings():
                if ids.intersection(cid for cid, _ in rec.links):
                    total_linked += 1
                    if len(linked) < limit:
                        linked.append({"id": rec.id, "label": rec.label, "kind": rec.kind, "line": rec.line})
        ppns = {r.ppn for r in exact if r.ppn}
        line = next((r.line for r in exact if r.ppn), None) or next((r.line for r in exact if r.line), None)
        return {
            "query": notation,
            "system": system,
            "normalized": code,
            "exact": [self._classification_dict(r) for r in exact],
            "line": line,
            "ambiguous": len(ppns) > 1,
            "broader": [self._classification_dict(r) for r in broader],
            "linked_terms": linked,
            "linked_terms_total": total_linked,
            "source": self._store.provenance(),
        }

