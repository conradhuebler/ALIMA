"""Read-only access to a converted local subject-indexing database - Claude Generated.

The database is a SQLite file written by ``scripts/access_to_sql.py`` (one table
per Access table plus the ``alima_import_*`` provenance tables). This module never
writes to it: every query opens the file with ``mode=ro``.

What a table or column *means* is not decided here. The plugin instance carries a
role mapping (which table holds the subject records, which column the heading,
which columns refer to notation records, …); the store only reads what the mapping
names, after checking that those tables and columns exist.

Two heading formats are understood:

* ``plain``: the column holds the heading text, nothing else;
* ``swb_packed``: the packed form found in the Geo database's ``SWB`` and
  ``Geographica`` tables: ``|s|Heading !123456789! 74899 ‡ |s|Variant |s|Variant``.
  ``|x|`` (or the older ``x.``) is the heading-type indicator, ``!…!`` tokens are
  record identifiers, ``‡`` separates the preferred form from its variants. The
  reading of the indicators follows the RSWK heading types; what the ``!a…!``
  tokens and the trailing number stand for is not established, so both are kept
  verbatim (``identifiers``, ``local_number``) and never interpreted.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TERM_FORMATS = ("plain", "swb_packed")

#: RSWK heading-type indicators as used in the packed field.
TYPE_LABELS = {
    "s": "Sachschlagwort",
    "g": "Geografikum",
    "p": "Person",
    "k": "Körperschaft",
    "c": "Kongress",
    "t": "Werktitel",
    "f": "Formschlagwort",
    "z": "Zeitschlagwort",
}

_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
_SEGMENT = re.compile(r"\|([a-z])\|")
_OLD_TYPE = re.compile(r"^\s*([sgpkctfz])\.(?=\S)")
_OLD_SPLIT = re.compile(r"(?:^|\s)([sgpkctfz])\.(?=\S)")
_ID_TOKEN = re.compile(r"!(a?[0-9X]+)!")



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


def _clean(text: str) -> str:
    text = text.replace("¬", "")
    return re.sub(r"\s+", " ", text).strip(" ,;")


@dataclass
class ParsedHeading:
    """A heading field taken apart. ``raw`` is always kept."""

    raw: str
    text: str = ""
    type: str = ""
    identifiers: List[str] = field(default_factory=list)
    local_number: str = ""
    variants: List[Tuple[str, str]] = field(default_factory=list)


def _old_style_segments(text: str) -> List[Tuple[str, str]]:
    """``g.Nordchile g.Chile-Nord`` → ``[("g", "Nordchile"), ("g", "Chile-Nord")]``."""
    marks = list(_OLD_SPLIT.finditer(text))
    if not marks:
        return [("", text)] if text.strip() else []
    out = []
    if text[: marks[0].start()].strip():
        out.append(("", text[: marks[0].start()]))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((mark.group(1), text[mark.end(): end]))
    return out


def _segments(text: str) -> List[Tuple[str, str]]:
    parts = _SEGMENT.split(text or "")
    if len(parts) == 1:
        return _old_style_segments(parts[0])
    out: List[Tuple[str, str]] = []
    if parts[0].strip():
        out.extend(_old_style_segments(parts[0]))
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1]))
    return out


def parse_heading(raw: Any, term_format: str = "plain") -> ParsedHeading:
    """Split a heading field per ``term_format``. - Claude Generated"""
    text = "" if raw is None else str(raw)
    if term_format != "swb_packed":
        return ParsedHeading(raw=text, text=_clean(text))
    head, has_sep, tail = text.partition("‡")
    segments = _segments(head)
    parsed = ParsedHeading(raw=text)
    if not segments:
        return parsed
    first_type, first_text = segments[0]
    parsed.type = first_type
    parsed.identifiers = _ID_TOKEN.findall(first_text)
    body = _ID_TOKEN.sub(" ", first_text).rstrip()
    # A trailing number is split off only where the measured pattern has it:
    # right after the last ``!…!`` token (3-7 digits), or, without tokens, before
    # ``‡`` with 5-7 digits (a 4-digit year at the end of a heading stays).
    stripped = first_text.rstrip()
    if parsed.identifiers:
        after_last_id = stripped[stripped.rfind("!") + 1:]
        number = re.fullmatch(r"\s*(\d{3,7})", after_last_id)
    elif has_sep:
        number = re.search(r"\s(\d{5,7})$", stripped)
    else:
        number = None
    if number:
        parsed.local_number = number.group(1)
        body = body[: len(body) - len(number.group(1))]
    parsed.text = _clean(body)
    for vtype, vtext in segments[1:] + _segments(tail):
        cleaned = _clean(_ID_TOKEN.sub(" ", vtext))
        if cleaned:
            parsed.variants.append((vtype, cleaned))
    return parsed


def normalize_code(raw: Any, scheme: str = "") -> Tuple[str, List[str]]:
    """``"DK 624.137.2 !1293463876!"`` → ``("624.137.2", ["1293463876"])``."""
    text = str(raw or "")
    ids = _ID_TOKEN.findall(text)
    text = _ID_TOKEN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for prefix in {scheme.strip(), "DK", "UDK", "UDC"} - {""}:
        if text.upper().startswith(prefix.upper() + " "):
            text = text[len(prefix) + 1:].strip()
            break
    return text, ids


def _norm_id(value: Any) -> Any:
    """Join keys stored as Double (``2198.0``) with Integer references."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+(\.0+)?", stripped):
            return int(float(stripped))
        return stripped or None
    return value


def _split_list(value: Any) -> List[str]:
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


@dataclass
class RoleMapping:
    """Which table/column plays which role. Read from the plugin settings."""

    term_table: str = ""
    term_id_column: str = ""
    term_column: str = ""
    term_format: str = "plain"
    variant_column: str = ""
    broader_column: str = ""
    related_column: str = ""
    note_columns: List[str] = field(default_factory=list)
    term_notation_columns: List[str] = field(default_factory=list)
    notation_table: str = ""
    notation_id_column: str = ""
    notation_code_column: str = ""
    notation_caption_column: str = ""
    notation_scheme: str = "DK"

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "RoleMapping":
        s = settings or {}
        fmt = str(s.get("term_format") or "plain").strip()
        return cls(
            term_table=str(s.get("term_table") or "").strip(),
            term_id_column=str(s.get("term_id_column") or "").strip(),
            term_column=str(s.get("term_column") or "").strip(),
            term_format=fmt if fmt in TERM_FORMATS else "plain",
            variant_column=str(s.get("variant_column") or "").strip(),
            broader_column=str(s.get("broader_column") or "").strip(),
            related_column=str(s.get("related_column") or "").strip(),
            note_columns=_split_list(s.get("note_columns")),
            term_notation_columns=_split_list(s.get("term_notation_columns")),
            notation_table=str(s.get("notation_table") or "").strip(),
            notation_id_column=str(s.get("notation_id_column") or "").strip(),
            notation_code_column=str(s.get("notation_code_column") or "").strip(),
            notation_caption_column=str(s.get("notation_caption_column") or "").strip(),
            notation_scheme=str(s.get("notation_scheme") or "DK").strip(),
        )

    @property
    def has_terms(self) -> bool:
        return bool(self.term_table and self.term_column)

    @property
    def has_notations(self) -> bool:
        return bool(self.notation_table and self.notation_code_column)

    def required_columns(self) -> Dict[str, List[str]]:
        need: Dict[str, List[str]] = {}
        if self.has_terms:
            cols = [self.term_column, self.term_id_column, self.variant_column, self.broader_column,
                    self.related_column, *self.note_columns, *self.term_notation_columns]
            need[self.term_table] = [c for c in cols if c]
        if self.has_notations:
            cols = [self.notation_code_column, self.notation_id_column, self.notation_caption_column]
            need.setdefault(self.notation_table, []).extend(c for c in cols if c)
        return need


@dataclass
class TermRecord:
    id: Any
    heading: ParsedHeading
    broader: str = ""
    related: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    notation_refs: List[Any] = field(default_factory=list)
    folded: str = ""
    folded_variants: List[str] = field(default_factory=list)


@dataclass
class NotationRecord:
    id: Any
    code_raw: str
    code: str
    caption: str
    identifiers: List[str]
    digits: str


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class GeoSachdbStore:
    """Mapping-driven, read-only view on the converted database. - Claude Generated"""

    def __init__(self, db_path: str, mapping: RoleMapping):
        self.db_path = str(db_path or "").strip()
        self.mapping = mapping
        self._terms: Optional[List[TermRecord]] = None
        self._notations: Optional[List[NotationRecord]] = None
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
        if not (self.mapping.has_terms or self.mapping.has_notations):
            return no("Keine Rollen-Abbildung konfiguriert (term_table/term_column oder "
                      "notation_table/notation_code_column).")
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            return no(f"Datenbank nicht lesbar: {exc}")
        try:
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, columns in self.mapping.required_columns().items():
                if table not in existing:
                    return no(f"Tabelle '{table}' fehlt in {path.name}")
                have = {r[1] for r in conn.execute(f"PRAGMA table_info({_q(table)})")}
                missing = [c for c in columns if c not in have]
                if missing:
                    return no(f"Spalte(n) {missing} fehlen in Tabelle '{table}'")
                if conn.execute(f"SELECT 1 FROM {_q(table)} LIMIT 1").fetchone() is None:
                    return no(f"Tabelle '{table}' ist leer")
            return {"available": True, "reason": "", "provenance": self._provenance(conn, existing)}
        except sqlite3.Error as exc:
            return no(f"Datenbank nicht lesbar: {exc}")
        finally:
            conn.close()

    def _provenance(self, conn: sqlite3.Connection, existing: set) -> Dict[str, Any]:
        out: Dict[str, Any] = {"db_path": str(Path(self.db_path).expanduser())}
        if "alima_import_run" in existing:
            row = conn.execute(
                "SELECT source_path, source_sha256, source_mtime, reader, finished_at "
                "FROM alima_import_run ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                out.update({
                    "source_file": Path(str(row[0])).name,
                    "source_sha256": str(row[1])[:16],
                    "source_mtime": row[2],
                    "reader": row[3],
                    "imported_at": row[4],
                })
        if "alima_import_table" in existing:
            wanted = {self.mapping.term_table, self.mapping.notation_table} - {""}
            rows = conn.execute("SELECT target_table, rows_written FROM alima_import_table").fetchall()
            out["rows"] = {t: n for t, n in rows if t in wanted}
        return out

    def provenance(self) -> Dict[str, Any]:
        return dict(self.status().get("provenance") or {})

    # ── loading (once per instance) ───────────────────────────────────────

    def notations(self) -> List[NotationRecord]:
        if self._notations is not None:
            return self._notations
        self._notations = []
        m = self.mapping
        if not m.has_notations:
            return self._notations
        from src.utils.lookups.local_scheme import digits_of

        id_col = _q(m.notation_id_column) if m.notation_id_column else "rowid"
        cap_col = _q(m.notation_caption_column) if m.notation_caption_column else "NULL"
        conn = self._connect()
        try:
            for nid, code_raw, caption in conn.execute(
                f"SELECT {id_col}, {_q(m.notation_code_column)}, {cap_col} FROM {_q(m.notation_table)}"
            ):
                if code_raw is None or not str(code_raw).strip():
                    continue
                code, ids = normalize_code(code_raw, m.notation_scheme)
                self._notations.append(NotationRecord(
                    id=_norm_id(nid), code_raw=str(code_raw), code=code,
                    caption=_clean(str(caption or "")), identifiers=ids, digits=digits_of(code),
                ))
        finally:
            conn.close()
        return self._notations

    def notations_by_id(self) -> Dict[Any, List[NotationRecord]]:
        out: Dict[Any, List[NotationRecord]] = {}
        for rec in self.notations():
            if rec.id is not None:
                out.setdefault(rec.id, []).append(rec)
        return out

    def terms(self) -> List[TermRecord]:
        if self._terms is not None:
            return self._terms
        self._terms = []
        m = self.mapping
        if not m.has_terms:
            return self._terms
        extra = [c for c in (m.variant_column, m.broader_column, m.related_column) if c]
        cols = [m.term_id_column or "rowid", m.term_column, *extra, *m.note_columns, *m.term_notation_columns]
        select = ", ".join(c if c == "rowid" else _q(c) for c in cols)
        conn = self._connect()
        try:
            for row in conn.execute(f"SELECT {select} FROM {_q(m.term_table)}"):
                values = dict(zip(cols, row))
                heading = parse_heading(values.get(m.term_column), m.term_format)
                if not heading.text:
                    continue
                if m.variant_column and values.get(m.variant_column):
                    for part in re.split(r"[‡;\n]", str(values[m.variant_column])):
                        if _clean(part):
                            heading.variants.append(("", _clean(part)))
                broader = ""
                if m.broader_column and values.get(m.broader_column):
                    parsed = parse_heading(values[m.broader_column], m.term_format)
                    broader = parsed.text
                    if m.term_format == "swb_packed" and broader.lower() == "kein":
                        broader = ""
                related: List[str] = []
                if m.related_column and values.get(m.related_column):
                    rel = parse_heading(values[m.related_column], m.term_format)
                    related = [t for t in [rel.text, *(v for _, v in rel.variants)] if t]
                notes = [_clean(str(values[c])) for c in m.note_columns if values.get(c) not in (None, "")]
                refs = [_norm_id(values[c]) for c in m.term_notation_columns if values.get(c) is not None]
                self._terms.append(TermRecord(
                    id=_norm_id(values[cols[0]]), heading=heading, broader=broader, related=related,
                    notes=[n for n in notes if n], notation_refs=refs, folded=fold(heading.text),
                    folded_variants=[fold(v) for _, v in heading.variants],
                ))
        finally:
            conn.close()
        return self._terms

    def raw_term_row(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """All columns of one subject record, as stored."""
        m = self.mapping
        key = m.term_id_column or "rowid"
        key_sql = "rowid" if key == "rowid" else _q(key)
        conn = self._connect()
        try:
            cur = conn.execute(f"SELECT * FROM {_q(m.term_table)} WHERE {key_sql} = ?", (record_id,))
            row = cur.fetchone()
            if row is None and isinstance(record_id, int):
                cur = conn.execute(f"SELECT * FROM {_q(m.term_table)} WHERE {key_sql} = ?", (float(record_id),))
                row = cur.fetchone()
            if row is None:
                return None
            names = [d[0] for d in cur.description]
            return {n: (v.hex() if isinstance(v, bytes) else v) for n, v in zip(names, row)}
        finally:
            conn.close()
