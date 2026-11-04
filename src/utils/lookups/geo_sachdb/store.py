"""Read-only access to a ``geo_sachdb`` database (schema v1) - Claude Generated.

The file holds the tables defined in ``schema.py``; an importer writes it, this
module only reads it (every connection opens the file with ``mode=ro``). The
store loads all headings and notation entries of one instance once and searches
in memory; there is no full-text index.

It also builds the finished WinIBW lines (UB Freiberg practice, WP decision F3):

* heading: ``5550 !PPN!``; without PPN the heading's ``chain`` (``5550 <chain>``),
  without chain its ``label`` (``5550 <label>``). ``line_form`` says which one.
* notation: ``6700 !PPN!`` when the entry has a PPN, otherwise for DK
  ``6700 DK <notation>``. Other systems without PPN get no line (no rule).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import KINDS, check_schema, clean_label, compact, split_variants

logger = logging.getLogger(__name__)

HEADING_TAG = "5550"
CLASSIFICATION_TAG = "6700"

_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def fold(text: Any) -> str:
    """Search form: lower case, umlauts as ``ae/oe/ue/ss``, other diacritics
    dropped, punctuation to spaces.

    The Geo data mostly spells umlauts out (``Lagerstaette``) while GND headings
    use the letters; folding both sides the same way lets them meet.
    """
    raw = str(text or "").lower().replace("¬", "")
    raw = "".join(_TRANSLIT.get(ch, ch) for ch in raw)
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = re.sub(r"[^0-9a-z]+", " ", raw)
    return raw.strip()


def normalize_kinds(kinds: Any) -> Optional[set]:
    """``None``/empty → no filter; a string or list of kind names → the set.

    Unknown names raise ``ValueError`` so a misspelt filter does not silently
    return nothing.
    """
    if kinds is None or kinds == "" or kinds == []:
        return None
    items = [kinds] if isinstance(kinds, str) else list(kinds)
    by_fold = {fold(k): k for k in KINDS}
    out = set()
    for item in items:
        for part in str(item).split(","):
            if not part.strip():
                continue
            name = by_fold.get(fold(part))
            if name is None:
                raise ValueError(f"Unbekannte Art '{part.strip()}' (erlaubt: {', '.join(KINDS)})")
            out.add(name)
    return out or None


@dataclass
class ClassificationRecord:
    id: int
    system: str
    notation: str
    label: str
    ppn: str
    digits: str = ""

    @property
    def line(self) -> Optional[str]:
        if self.ppn:
            return f"{CLASSIFICATION_TAG} !{self.ppn}!"
        if self.system == "DK":
            return f"{CLASSIFICATION_TAG} DK {self.notation}"
        return None


@dataclass
class HeadingRecord:
    id: int
    label: str
    kind: str
    ppn: str
    chain: str
    variants: List[str]
    links: List[Tuple[int, bool]] = field(default_factory=list)  # (classification_id, ambiguous)
    folded: str = ""
    folded_variants: List[str] = field(default_factory=list)

    @property
    def line(self) -> str:
        return f"{HEADING_TAG} {self.line_value}"

    @property
    def line_form(self) -> str:
        if self.ppn:
            return "ppn"
        return "chain" if self.chain else "label"

    @property
    def line_value(self) -> str:
        if self.ppn:
            return f"!{self.ppn}!"
        return self.chain or self.label


class GeoSachdbStore:
    """Read-only view on one schema-v1 database. - Claude Generated"""

    def __init__(self, db_path: str):
        self.db_path = str(db_path or "").strip()
        self._headings: Optional[List[HeadingRecord]] = None
        self._classifications: Optional[Dict[int, ClassificationRecord]] = None
        self._status: Optional[Dict[str, Any]] = None

    # ── connection + status ───────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        uri = Path(self.db_path).expanduser().resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def status(self, refresh: bool = False) -> Dict[str, Any]:
        """``{"available": bool, "reason": str, "provenance": {...}}``.

        A positive result is kept; a negative one is re-checked on the next call,
        so a database that appears later is picked up without a restart.
        """
        if self._status is not None and not refresh:
            return self._status
        result = self._check()
        if result["available"]:
            self._status = result
        return result

    def _check(self) -> Dict[str, Any]:
        def no(reason: str) -> Dict[str, Any]:
            return {"available": False, "reason": reason, "provenance": {}}

        if not self.db_path:
            return no("Kein Datenbankpfad konfiguriert (Einstellung db_path).")
        path = Path(self.db_path).expanduser()
        if not path.is_file():
            return no(f"Datenbankdatei nicht gefunden: {path}")
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            return no(f"Datenbank nicht lesbar: {exc}")
        try:
            reason = check_schema(conn)
            if reason:
                return no(f"{path.name}: {reason}")
            if conn.execute("SELECT 1 FROM heading LIMIT 1").fetchone() is None and \
                    conn.execute("SELECT 1 FROM classification LIMIT 1").fetchone() is None:
                return no(f"{path.name}: Tabellen heading und classification sind leer")
            return {"available": True, "reason": "", "provenance": self._provenance(conn)}
        except sqlite3.Error as exc:
            return no(f"Datenbank nicht lesbar: {exc}")
        finally:
            conn.close()

    def _provenance(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        out: Dict[str, Any] = {"db_path": str(Path(self.db_path).expanduser())}
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        for key in ("schema_version", "source", "source_date", "source_sha256", "created_at", "importer"):
            if meta.get(key):
                out[key] = meta[key] if key != "source_sha256" else str(meta[key])[:16]
        out["rows"] = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("heading", "classification", "heading_classification")
        }
        return out

    def provenance(self) -> Dict[str, Any]:
        return dict(self.status().get("provenance") or {})

    # ── loading (once per instance) ───────────────────────────────────────

    def classifications(self) -> Dict[int, ClassificationRecord]:
        if self._classifications is not None:
            return self._classifications
        from src.utils.lookups.local_scheme import digits_of

        out: Dict[int, ClassificationRecord] = {}
        conn = self._connect()
        try:
            for cid, system, notation, label, ppn in conn.execute(
                "SELECT id, system, notation, label, ppn FROM classification ORDER BY id"
            ):
                if not str(notation or "").strip():
                    continue
                out[int(cid)] = ClassificationRecord(
                    id=int(cid), system=str(system or "").strip().upper(), notation=str(notation).strip(),
                    label=clean_label(label), ppn=str(ppn or "").strip(),
                    digits=digits_of(notation) if str(system).upper() == "DK" else "",
                )
        finally:
            conn.close()
        self._classifications = out
        return out

    def headings(self) -> List[HeadingRecord]:
        if self._headings is not None:
            return self._headings
        links: Dict[int, List[Tuple[int, bool]]] = {}
        out: List[HeadingRecord] = []
        conn = self._connect()
        try:
            for hid, cid, ambiguous in conn.execute(
                "SELECT heading_id, classification_id, ambiguous FROM heading_classification "
                "ORDER BY heading_id, position, classification_id"
            ):
                links.setdefault(int(hid), []).append((int(cid), bool(ambiguous)))
            for hid, label, kind, ppn, chain, variants in conn.execute(
                "SELECT id, label, kind, ppn, chain, variants FROM heading ORDER BY id"
            ):
                text = clean_label(label)
                if not text:
                    continue
                forms = split_variants(variants)
                out.append(HeadingRecord(
                    id=int(hid), label=text, kind=str(kind or "sonstiges"), ppn=str(ppn or "").strip(),
                    chain=str(chain or "").strip(), variants=forms, links=links.get(int(hid), []),
                    folded=fold(text), folded_variants=[fold(v) for v in forms],
                ))
        finally:
            conn.close()
        self._headings = out
        return out

    def by_notation(self, system: str, notation: str) -> List[ClassificationRecord]:
        key = compact(notation)
        return [c for c in self.classifications().values()
                if c.system == system and compact(c.notation) == key]
