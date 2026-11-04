"""Local subject-indexing database (converted from Access) as a lookup plugin.

Answers three questions over a SQLite file written by ``scripts/access_to_sql.py``:
which local subject records match a term, what one record holds, and what the
local data says about a notation. Which table and column plays which role is
plugin configuration (``RoleMapping``), not code, because the meaning of the
source tables is confirmed by the library, not inferred by ALIMA.

The plugin proposes, it never decides: results carry the record's own wording,
identifiers and provenance, and a workflow step that uses them keeps the
assigned classification untouched.

Without a database file, or without a role mapping, ``is_available()`` is False:
the tool generator then offers none of its tools, and every method answers with
an ``error`` entry instead of raising.

Not registered as a built-in on import (``src/utils/lookups/__init__.py`` does
not import it). It is registered by the plugin loader as a Tier-2 code plugin, or
explicitly via ``register_lookup(GeoSachdbLookup)``.

Claude Generated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.plugins.schema import CHOICE, INT, TEXT, ConfigField, PluginDoc
from src.utils.lookups.registry import LookupToolSpec

from .store import TERM_FORMATS, TYPE_LABELS, GeoSachdbStore, RoleMapping, fold, normalize_code

logger = logging.getLogger(__name__)


class GeoSachdbLookup:
    """Lookup over a converted local subject database. - Claude Generated"""

    id = "geo_sachdb"
    label = "Lokale Sacherschließungsdatenbank (SQLite)"

    def __init__(self, **config: Any):
        self._config = dict(config or {})
        try:
            self._max_results = max(1, int(self._config.get("max_results") or 10))
        except (TypeError, ValueError):
            self._max_results = 10
        self._mapping = RoleMapping.from_settings(self._config)
        self._store = GeoSachdbStore(str(self._config.get("db_path") or ""), self._mapping)

    @property
    def store(self) -> GeoSachdbStore:
        return self._store

    # ── plugin contract ────────────────────────────────────────────────────

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(key="db_path", label="SQLite-Datei", kind=TEXT, default="",
                        gates_availability=True,
                        help="Mit scripts/access_to_sql.py erzeugte Datei. Leer oder nicht "
                             "vorhanden: Plugin nicht verfügbar, keine Werkzeuge."),
            ConfigField(key="term_table", label="Tabelle: Schlagwortsätze", kind=TEXT, default=""),
            ConfigField(key="term_id_column", label="Spalte: Satz-Schlüssel", kind=TEXT, default="",
                        help="Leer: SQLite-rowid."),
            ConfigField(key="term_column", label="Spalte: Ansetzung", kind=TEXT, default=""),
            ConfigField(key="term_format", label="Format der Ansetzung", kind=CHOICE,
                        default="plain", choices=list(TERM_FORMATS),
                        help="plain: nur der Text. swb_packed: |x|Ansetzung !ID! … ‡ |x|Verweisung …"),
            ConfigField(key="variant_column", label="Spalte: Verweisungsformen", kind=TEXT, default="",
                        help="Nur nötig, wenn die Verweisungen nicht in der Ansetzung stehen."),
            ConfigField(key="broader_column", label="Spalte: Oberbegriff", kind=TEXT, default=""),
            ConfigField(key="related_column", label="Spalte: verwandte Begriffe", kind=TEXT, default=""),
            ConfigField(key="note_columns", label="Spalten: Hinweise (kommagetrennt)", kind=TEXT, default=""),
            ConfigField(key="term_notation_columns", label="Spalten: Verweise auf Notationssätze (kommagetrennt)",
                        kind=TEXT, default="",
                        help="Werte werden gegen die Schlüsselspalte der Notationstabelle aufgelöst."),
            ConfigField(key="notation_table", label="Tabelle: Notationen", kind=TEXT, default=""),
            ConfigField(key="notation_id_column", label="Spalte: Notations-Schlüssel", kind=TEXT, default="",
                        help="Leer: SQLite-rowid."),
            ConfigField(key="notation_code_column", label="Spalte: Notation", kind=TEXT, default=""),
            ConfigField(key="notation_caption_column", label="Spalte: Erläuterung", kind=TEXT, default=""),
            ConfigField(key="notation_scheme", label="Notationssystem", kind=TEXT, default="DK",
                        help="Präfix, das beim Vergleich abgestreift wird (z. B. 'DK 553.62')."),
            ConfigField(key="max_results", label="Max. Treffer", kind=INT, default=10),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Lokale Sacherschließungsdatenbank (aus Access nach SQLite überführt) als "
            "Nachschlagequelle: findet lokale Schlagwortsätze zu einem Begriff, zeigt einen Satz "
            "vollständig und gleicht eine Notation mit den lokalen Notationssätzen ab. Welche "
            "Tabelle und Spalte welche Rolle hat, steht in der Plugin-Konfiguration.",
            input="Ein Suchbegriff, eine Satz-ID oder eine Notation (z. B. 'DK 553.62').",
            output="Lokale Sätze mit Ansetzung, Typ, Kennungen, Verweisungen, Oberbegriff und "
            "verknüpften Notationen samt Erläuterung, dazu die Herkunft der Datenbank.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [
            LookupToolSpec(
                name="geo_sachdb_search",
                description=(
                    "Search the library's local subject-indexing database (converted from its own "
                    "Access database) for a term. Matches preferred headings and their variant forms "
                    "(umlauts and 'ae/oe/ue' spellings meet). Returns local records with heading, "
                    "type, identifiers, broader term and the linked local notations with captions. "
                    "Local proposals only: they do not replace GND or catalogue evidence."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Term to look up"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                method="search_terms",
                cacheable=False,  # local file read; caching only adds staleness
            ),
            LookupToolSpec(
                name="geo_sachdb_record",
                description="Return one local subject record in full (all stored columns plus the "
                "parsed heading and linked notations). Use the 'id' from geo_sachdb_search.",
                parameters={
                    "type": "object",
                    "properties": {"record_id": {"type": "string", "description": "Record id"}},
                    "required": ["record_id"],
                },
                method="get_record",
                cacheable=False,
            ),
            LookupToolSpec(
                name="geo_sachdb_notation",
                description="Check a notation against the local notation records: exact local "
                "entries with caption, the local subject records linked to it, and the nearest "
                "broader local notations when there is no exact entry.",
                parameters={
                    "type": "object",
                    "properties": {
                        "notation": {"type": "string", "description": "e.g. 'DK 553.62' or '553.62'"},
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

    # ── helpers ────────────────────────────────────────────────────────────

    def _notation_dicts(self, refs: List[Any]) -> List[Dict[str, Any]]:
        by_id = self._store.notations_by_id()
        out: List[Dict[str, Any]] = []
        for ref in refs:
            records = by_id.get(ref) or []
            if not records:
                out.append({"ref": ref, "code": None, "resolved": False})
                continue
            for rec in records:
                out.append({
                    "ref": ref, "code": rec.code, "caption": rec.caption,
                    "resolved": True, "ambiguous": len(records) > 1,
                })
        return out

    def _term_dict(self, rec, matched: str = "", form: str = "", score: float = 0.0) -> Dict[str, Any]:
        head = rec.heading
        item: Dict[str, Any] = {
            "id": rec.id,
            "term": head.text,
            "type": head.type,
            "type_label": TYPE_LABELS.get(head.type, ""),
            "identifiers": list(head.identifiers),
            "variants": [v for _, v in head.variants][:10],
            "broader": rec.broader,
            "related": rec.related[:10],
            "notes": rec.notes,
            "notations": self._notation_dicts(rec.notation_refs),
        }
        if head.local_number:
            item["local_number"] = head.local_number
        if matched:
            item.update({"matched": matched, "matched_form": form, "score": score})
        return item

    # ── tools ──────────────────────────────────────────────────────────────

    def search_terms(self, query: str, max_results: Optional[int] = None, min_score: float = 0.0) -> Dict[str, Any]:
        """Ranked local subject records for ``query``. - Claude Generated

        Scores: identical heading 100, identical variant 90, every query word in
        the heading 60 (variant 50), query as substring of the heading 30
        (variant 25). ``min_score`` lets a deterministic caller keep only the
        identical matches.
        """
        blocked = self._unavailable()
        if blocked:
            return blocked
        if not self._mapping.has_terms:
            return {"error": "Keine Schlagworttabelle konfiguriert (term_table/term_column).",
                    "results": [], "source": self._store.provenance()}
        limit = int(max_results or self._max_results)
        q = fold(query)
        if not q:
            return {"query": query, "results": [], "source": self._store.provenance()}
        words = set(q.split())
        scored = []
        for rec in self._store.terms():
            best = (0.0, "", "")
            if rec.folded == q:
                best = (100.0, "ansetzung_exakt", rec.heading.text)
            else:
                for (_, vtext), vfold in zip(rec.heading.variants, rec.folded_variants):
                    if vfold == q:
                        best = (90.0, "verweisung_exakt", vtext)
                        break
            if best[0] == 0.0 and words <= set(rec.folded.split()):
                best = (60.0, "ansetzung_woerter", rec.heading.text)
            if best[0] == 0.0:
                for (_, vtext), vfold in zip(rec.heading.variants, rec.folded_variants):
                    if words <= set(vfold.split()):
                        best = (50.0, "verweisung_woerter", vtext)
                        break
            if best[0] == 0.0 and q in rec.folded:
                best = (30.0, "ansetzung_teil", rec.heading.text)
            if best[0] == 0.0:
                for (_, vtext), vfold in zip(rec.heading.variants, rec.folded_variants):
                    if q in vfold:
                        best = (25.0, "verweisung_teil", vtext)
                        break
            if best[0] > 0 and best[0] >= float(min_score or 0):
                scored.append((best, rec))
        scored.sort(key=lambda item: (-item[0][0], len(item[1].heading.text), str(item[1].id)))
        return {
            "query": query,
            "total": len(scored),
            "results": [self._term_dict(rec, m, f, s) for (s, m, f), rec in scored[:limit]],
            "source": self._store.provenance(),
        }

    def get_record(self, record_id: Any) -> Dict[str, Any]:
        """One subject record: stored columns + parsed view. - Claude Generated"""
        blocked = self._unavailable()
        if blocked:
            return blocked
        if not self._mapping.has_terms:
            return {"error": "Keine Schlagworttabelle konfiguriert (term_table/term_column)."}
        from .store import _norm_id

        key = _norm_id(record_id)
        parsed = next((r for r in self._store.terms() if r.id == key), None)
        raw = self._store.raw_term_row(key)
        if raw is None and parsed is None:
            return {"error": f"Kein Satz mit id {record_id!r}", "source": self._store.provenance()}
        return {
            "record": self._term_dict(parsed) if parsed else None,
            "stored": raw,
            "source": self._store.provenance(),
        }

    def match_notation(self, notation: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """Local entries for a notation, linked records, nearest broader entries. - Claude Generated"""
        blocked = self._unavailable()
        if blocked:
            return blocked
        if not self._mapping.has_notations:
            return {"error": "Keine Notationstabelle konfiguriert (notation_table/notation_code_column)."}
        from src.utils.lookups.local_scheme import digits_of

        limit = int(max_results or self._max_results)
        code, _ = normalize_code(notation, self._mapping.notation_scheme)
        compact = code.replace(" ", "")
        exact = [r for r in self._store.notations() if r.code.replace(" ", "") == compact]
        broader: List[Any] = []
        if not exact:
            digits = digits_of(code)
            above = [r for r in self._store.notations()
                     if r.digits and digits.startswith(r.digits) and r.digits != digits]
            above.sort(key=lambda r: (-len(r.digits), r.code))
            broader = above[:3]
        linked: List[Dict[str, Any]] = []
        total_linked = 0
        if exact and self._mapping.has_terms and self._mapping.term_notation_columns:
            ids = {r.id for r in exact if r.id is not None}
            for rec in self._store.terms():
                if ids.intersection(rec.notation_refs):
                    total_linked += 1
                    if len(linked) < limit:
                        linked.append({"id": rec.id, "term": rec.heading.text, "type": rec.heading.type})

        def as_dict(rec) -> Dict[str, Any]:
            return {"id": rec.id, "code": rec.code, "caption": rec.caption, "identifiers": rec.identifiers}

        return {
            "query": notation,
            "normalized": code,
            "exact": [as_dict(r) for r in exact],
            "broader": [as_dict(r) for r in broader],
            "linked_terms": linked,
            "linked_terms_total": total_linked,
            "source": self._store.provenance(),
        }
