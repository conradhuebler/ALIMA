"""Input schema of the ``geo_sachdb`` lookup plugin, version 1 - Claude Generated.

The plugin reads one SQLite file in this schema and nothing else. The schema is
independent of where the data comes from: every library fills it with its own
importer (``scripts/access_to_sql.py import-geo`` is the one for the Freiberg
Access database). This module is the single definition of the tables; it uses
only the standard library so that an importer can load it by file path without
importing ALIMA.

Tables:

``meta(key, value)``
    ``schema_version`` is required and must equal :data:`SCHEMA_VERSION`.
    Optional keys the plugin shows as provenance: ``source``, ``source_date``,
    ``source_sha256``, ``created_at``, ``importer``.

``heading(id, label, kind, ppn, chain, variants)``
    One local subject heading. ``label`` is the plain heading text (searched,
    shown to the model), ``kind`` one of :data:`KINDS`, ``ppn`` the K10plus
    authority PPN (digits, optionally a final ``X``), ``chain`` the finished
    local string as the library enters it, ``variants`` the variant forms as
    raw text (see :func:`split_variants`).

``classification(id, system, notation, label, ppn)``
    One local notation entry. ``system`` is ``DK`` or ``RVK``; ``notation`` is
    stored in the form :func:`normalize_notation` produces (``DK``: without the
    prefix, ``RVK``: ``QP 925``). The same notation may occur more than once.

``heading_classification(heading_id, classification_id, position, ambiguous)``
    The notations a heading uses, in order. ``ambiguous = 1`` marks a link that
    the source could not resolve to one entry (the importer links all candidates).
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1"

KINDS = ("Sachschlagwort", "Geographikum", "Person", "Körperschaft", "sonstiges")

SYSTEMS = ("DK", "RVK")

DDL = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE heading (
    id       INTEGER PRIMARY KEY,
    label    TEXT NOT NULL,
    kind     TEXT NOT NULL CHECK (kind IN ('Sachschlagwort', 'Geographikum', 'Person',
                                           'Körperschaft', 'sonstiges')),
    ppn      TEXT,
    chain    TEXT,
    variants TEXT
);
CREATE TABLE classification (
    id       INTEGER PRIMARY KEY,
    system   TEXT NOT NULL CHECK (system IN ('DK', 'RVK')),
    notation TEXT NOT NULL,
    label    TEXT,
    ppn      TEXT
);
CREATE TABLE heading_classification (
    heading_id        INTEGER NOT NULL REFERENCES heading(id),
    classification_id INTEGER NOT NULL REFERENCES classification(id),
    position          INTEGER NOT NULL,
    ambiguous         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (heading_id, classification_id)
);
CREATE INDEX ix_classification_notation ON classification (system, notation);
CREATE INDEX ix_heading_classification_cls ON heading_classification (classification_id);
"""

#: Tables and columns the plugin reads. Extra tables (e.g. an importer's
#: ``alima_import_*`` provenance) and extra columns are allowed.
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "meta": ("key", "value"),
    "heading": ("id", "label", "kind", "ppn", "chain", "variants"),
    "classification": ("id", "system", "notation", "label", "ppn"),
    "heading_classification": ("heading_id", "classification_id", "position", "ambiguous"),
}

_PPN = re.compile(r"[0-9]+[0-9X]")


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the four tables and write ``schema_version``. - Claude Generated"""
    conn.executescript(DDL)
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))


def check_schema(conn: sqlite3.Connection) -> str:
    """Empty string if ``conn`` holds schema version 1, else the reason. - Claude Generated"""
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing_tables = [t for t in REQUIRED_COLUMNS if t not in existing]
    if missing_tables:
        return (f"Tabelle(n) {missing_tables} fehlen: die Datei hat nicht das Schema "
                f"geo_sachdb v{SCHEMA_VERSION} (Importer: scripts/access_to_sql.py import-geo)")
    for table, columns in REQUIRED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        missing = [c for c in columns if c not in have]
        if missing:
            return f"Spalte(n) {missing} fehlen in Tabelle '{table}'"
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = str(row[0]).strip() if row and row[0] is not None else ""
    if version != SCHEMA_VERSION:
        return f"Schemaversion '{version or 'fehlt'}' statt '{SCHEMA_VERSION}'"
    return ""


def is_ppn(value: Any) -> bool:
    """A K10plus PPN: digits, the last character a digit or ``X``."""
    return bool(_PPN.fullmatch(str(value or "").strip()))


def normalize_notation(text: Any, default_system: str = "DK") -> Tuple[str, str]:
    """``(system, notation)`` in the stored form. - Claude Generated

    ``"DK 553.62"`` → ``("DK", "553.62")``, ``"UDK 553.62"`` → ``("DK", "553.62")``,
    ``"RVK QP 925"`` and ``"QP925"`` → ``("RVK", "QP 925")``, ``"553.62"`` →
    ``(default_system, "553.62")``. Whitespace is collapsed; nothing else is changed.
    An empty input gives ``("", "")``.
    """
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return "", ""
    head, _, rest = raw.partition(" ")
    if head.upper() in ("DK", "UDK", "UDC") and rest:
        return "DK", rest.strip()
    if head.upper() == "RVK" and rest:
        raw = rest.strip()
    rvk = re.fullmatch(r"([A-Z]{2}) ?([0-9].*)", raw)
    if rvk:
        return "RVK", f"{rvk.group(1)} {rvk.group(2).strip()}"
    return (default_system or "DK").upper(), raw


def compact(notation: str) -> str:
    """Comparison form: no spaces, upper case."""
    return re.sub(r"\s+", "", str(notation or "")).upper()


_INDICATOR = re.compile(r"\|[a-z](?:/[a-z])*\|")
_ID_TOKEN = re.compile(r"!a?[0-9X]*!")


def clean_label(text: Any) -> str:
    """Text for search and for the model: no ``|x|`` indicator, no ``!…!`` token,
    no ``¬`` non-sorting marks, single spaces. - Claude Generated"""
    out = _INDICATOR.sub(" ", str(text or ""))
    out = _ID_TOKEN.sub(" ", out).replace("¬", "")
    return re.sub(r"\s+", " ", out).strip(" ,;")


def split_variants(raw: Optional[str]) -> List[str]:
    """Variant forms out of ``heading.variants``. - Claude Generated

    Forms are separated by ``;``, a line break, ``‡`` or a leading ``|x|`` type
    indicator (``|s|Probegestein |s|Prüfstein``). Each form goes through
    :func:`clean_label`; empty forms are dropped, order is kept.
    """
    if not raw:
        return []
    out: List[str] = []
    for part in re.split(r"[;\n‡]", str(raw)):
        for piece in _INDICATOR.split(part):
            cleaned = clean_label(piece)
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out
