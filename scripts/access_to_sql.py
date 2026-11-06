#!/usr/bin/env python3
"""Access-Datenbank (.mdb/.accdb) nach SQLite: Bestandsaufnahme, 1:1-Umwandlung, Import.

Drei Modi:

``inspect <quelle.mdb> [--json BERICHT.json] [--samples N]``
    Gibt Tabellen, Spalten (Access-Typ, Länge, SQLite-Zieltyp), Schlüssel und
    Indizes sowie Zeilenzahlen aus. Schreibt nichts außer dem optionalen
    JSON-Bericht.

``convert <quelle.mdb> <ziel.db> [--tables T ...] [--exclude T ...]``
    Schreibt jede Benutzertabelle 1:1 in eine SQLite-Datei (bereinigte Namen,
    Access-Typen abgebildet, NULL bleibt NULL) plus drei Herkunftstabellen
    (``alima_import_run``, ``alima_import_table``, ``alima_import_column``).
    Nach dem Schreiben wird jede Tabelle zurückgelesen und über Zeilenzahl und
    SHA-256-Prüfsumme mit der Quelle verglichen; liefert mdbtools zusätzlich
    ``mdb-count``, geht auch diese Zahl in den Vergleich ein. Die Zieldatei
    wird erst dann (atomar) ersetzt, wenn alle Tabellen übereinstimmen; ein
    zweiter Lauf mit derselben Quelle erzeugt denselben Inhalt.

``import-geo <zwischenstand.sqlite> <ziel.db>``
    Importer für die Freiberger Sacherschließungsdatenbank Geowissenschaften:
    liest die Tabellen ``swb``, ``dk``, ``dk_bio``, ``dk_berg`` und
    ``dk_geographica`` aus der Ausgabe von ``convert`` und schreibt das Schema
    des Lookup-Plugins ``geo_sachdb`` (``src/utils/lookups/geo_sachdb/schema.py``).
    Die Regeln stehen bei :func:`import_geo`; jede wird gezählt und in
    ``alima_import_rule`` abgelegt.

Leser:

* ``mdbtools`` (Standard): ruft ``mdb-tables``, ``mdb-schema``, ``mdb-export``
  und, falls vorhanden, ``mdb-count`` per subprocess auf. CSV wird mit dem
  csv-Modul gelesen (Memo-Felder mit Zeilenumbrüchen stehen in Anführungszeichen),
  NULL wird über ein Steuerzeichen-Sentinel von der leeren Zeichenkette getrennt.
* ``access-parser`` (optional, zweiter Weg): reines Python-Paket ``access_parser``.
  Fehlt das jeweilige Werkzeug, bricht das Skript mit einer Meldung ab.

Die Umwandlungslogik (Schema-Parser, Typabbildung, Namensbereinigung, Schreiben,
Prüfsummen) arbeitet gegen das ``AccessSource``-Protokoll und ist ohne .mdb
testbar (``tests/test_access_to_sql.py``).

Claude Generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

TOOL_VERSION = "1"

# Memo-Felder können länger als das csv-Standardlimit (128 KiB) sein.
csv.field_size_limit(sys.maxsize)

#: mdb-export schreibt NULL als dieses Token (``-0``). Steuerzeichen kommen in
#: Katalogdaten praktisch nicht vor; ein Wert, der das Token als *Teil* enthält,
#: bricht die Umwandlung ab statt still verfälscht zu werden.
NULL_SENTINEL = "\x1eNULL\x1e"

#: Namen der Herkunftstabellen; Quelltabellen dürfen sie nicht belegen.
META_RUN = "alima_import_run"
META_TABLE = "alima_import_table"
META_COLUMN = "alima_import_column"
RESERVED_TARGETS = {META_RUN, META_TABLE, META_COLUMN}

#: Wörter, die als SQL-Bezeichner Ärger machen (Auswahl; Treffer bekommen "_").
_SQL_RESERVED = {
    "abort", "add", "all", "alter", "and", "as", "asc", "between", "by", "case",
    "check", "collate", "column", "commit", "create", "cross", "default", "delete",
    "desc", "distinct", "drop", "else", "end", "escape", "except", "exists", "from",
    "full", "group", "having", "in", "index", "inner", "insert", "intersect", "into",
    "is", "join", "key", "left", "like", "limit", "match", "natural", "not", "null",
    "of", "offset", "on", "or", "order", "outer", "primary", "references", "right",
    "rowid", "select", "set", "table", "then", "to", "transaction", "union", "unique",
    "update", "using", "values", "view", "when", "where", "with",
}

#: Access-Typname (wie ``mdb-schema … access`` ihn schreibt) → SQLite-Typ.
#: Currency/Numeric bleiben TEXT, damit die Dezimalstellen exakt bleiben.
ACCESS_TO_SQLITE: Dict[str, str] = {
    "boolean": "INTEGER",
    "byte": "INTEGER",
    "integer": "INTEGER",
    "long integer": "INTEGER",
    "single": "REAL",
    "double": "REAL",
    "currency": "TEXT",
    "numeric": "TEXT",
    "datetime": "TEXT",
    "text": "TEXT",
    "memo/hyperlink": "TEXT",
    "replication id": "TEXT",
    "binary": "BLOB",
    "ole": "BLOB",
}

#: access_parser-Typcodes → Access-Typname (gleiche Namen wie mdbtools).
ACCESS_PARSER_TYPES: Dict[int, str] = {
    1: "Boolean", 2: "Byte", 3: "Integer", 4: "Long Integer", 5: "Currency",
    6: "Single", 7: "Double", 8: "DateTime", 9: "Binary", 10: "Text", 11: "OLE",
    12: "Memo/Hyperlink", 15: "Replication ID", 16: "Numeric", 18: "Long Integer",
}


class ReaderUnavailable(RuntimeError):
    """Das gewählte Lesewerkzeug ist nicht installiert oder nicht nutzbar."""


class ConversionError(RuntimeError):
    """Daten ließen sich nicht verlustfrei umwandeln oder nicht verifizieren."""


# ─────────────────────────────────────────────────────────────────────────────
# Datenmodell
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ColumnSpec:
    name: str
    access_type: str
    size: Optional[int] = None


@dataclass
class IndexSpec:
    name: str
    columns: List[str]
    unique: bool = False
    primary: bool = False


@dataclass
class TableSpec:
    name: str
    columns: List[ColumnSpec] = field(default_factory=list)
    indexes: List[IndexSpec] = field(default_factory=list)

    @property
    def primary_key(self) -> List[str]:
        for idx in self.indexes:
            if idx.primary:
                return list(idx.columns)
        return []


class AccessSource(Protocol):
    """Was der Umwandler von einem Leser braucht. - Claude Generated"""

    label: str

    def describe(self) -> Dict[str, Any]: ...

    def schema(self) -> Dict[str, TableSpec]: ...

    def rows(self, table: str) -> Iterator[List[Any]]: ...

    def count(self, table: str) -> Optional[int]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Reine Hilfsfunktionen (ohne .mdb testbar)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}


def sanitize_identifier(name: str, used: Optional[set] = None) -> str:
    """Access-Namen in einen schlichten SQL-Bezeichner überführen.

    Kleinbuchstaben, Umlaute transliteriert (``ERLÄUTER`` → ``erlaeuter``), alles
    außer ``[a-z0-9_]`` wird ``_``, keine führende Ziffer, reservierte Wörter
    bekommen ein ``_`` angehängt. Mit ``used`` werden Kollisionen durch ``_2``,
    ``_3`` … aufgelöst und der vergebene Name eingetragen. - Claude Generated
    """
    text = "".join(_TRANSLIT.get(ch, ch) for ch in str(name or ""))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text) or "unnamed"
    if text[0].isdigit():
        text = "t_" + text
    if text in _SQL_RESERVED:
        text += "_"
    if used is None:
        return text
    candidate, n = text, 2
    while candidate in used:
        candidate = f"{text}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def sqlite_type_for(access_type: str) -> str:
    """SQLite-Typ für einen Access-Typnamen; Unbekanntes wird TEXT."""
    return ACCESS_TO_SQLITE.get(str(access_type or "").strip().lower(), "TEXT")


_CREATE_RE = re.compile(r"CREATE TABLE \[(?P<name>[^\]]+)\]\s*\((?P<body>.*?)\n\);", re.S)
_COL_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s+(?P<type>[^,(]+?)\s*(?:\((?P<size>\d+)\))?\s*,?\s*$")


def parse_access_schema(text: str) -> Dict[str, TableSpec]:
    """Spalten aus ``mdb-schema <datei> access`` lesen. - Claude Generated

    Erwartet Blöcke der Form::

        CREATE TABLE [Name]
         (
            [Spalte]   Text (200),
            [Zahl]     Long Integer
        );
    """
    tables: Dict[str, TableSpec] = {}
    for match in _CREATE_RE.finditer(text or ""):
        spec = TableSpec(name=match.group("name"))
        for line in match.group("body").splitlines():
            col = _COL_RE.match(line)
            if not col:
                continue
            size = col.group("size")
            spec.columns.append(ColumnSpec(
                name=col.group("name"),
                access_type=col.group("type").strip(),
                size=int(size) if size else None,
            ))
        tables[spec.name] = spec
    return tables


_IDX_RE = re.compile(
    r"CREATE (?P<unique>UNIQUE )?INDEX `(?P<name>[^`]+)` ON `(?P<table>[^`]+)` \((?P<cols>[^)]*)\)"
)
_TABLE_HEAD_RE = re.compile(r"CREATE TABLE `(?P<table>[^`]+)`")
_PK_RE = re.compile(r"PRIMARY KEY \((?P<cols>[^)]*)\)")


def parse_sqlite_keys(text: str) -> Dict[str, List[IndexSpec]]:
    """Primärschlüssel und Indizes aus ``mdb-schema --indexes <datei> sqlite``.

    Beziehungen (Fremdschlüssel) liest diese Funktion nicht. - Claude Generated
    """
    out: Dict[str, List[IndexSpec]] = {}
    current = None
    for line in (text or "").splitlines():
        head = _TABLE_HEAD_RE.search(line)
        if head:
            current = head.group("table")
            continue
        pk = _PK_RE.search(line)
        if pk and current:
            cols = re.findall(r"`([^`]+)`", pk.group("cols"))
            out.setdefault(current, []).append(IndexSpec(f"{current}_pkey", cols, unique=True, primary=True))
            continue
        idx = _IDX_RE.search(line)
        if idx:
            cols = re.findall(r"`([^`]+)`", idx.group("cols"))
            out.setdefault(idx.group("table"), []).append(
                IndexSpec(idx.group("name"), cols, unique=bool(idx.group("unique")))
            )
    return out


def coerce_value(value: Any, sqlite_type: str) -> Tuple[Any, bool]:
    """Einen Rohwert in den SQLite-Zieltyp bringen.

    Liefert ``(wert, ok)``. Lässt sich ein Wert nicht umwandeln, bleibt er als
    Text erhalten und ``ok`` ist False; der Aufrufer zählt das als Warnung,
    statt den Wert zu verwerfen. - Claude Generated
    """
    if value is None:
        return None, True
    try:
        if sqlite_type == "INTEGER":
            if isinstance(value, bool):
                return int(value), True
            if isinstance(value, (int, float)):
                if float(value) != int(value):
                    return value, False
                return int(value), True
            text = str(value).strip()
            if text.lower() in ("true", "false"):
                return int(text.lower() == "true"), True
            try:
                return int(text), True
            except ValueError:
                number = float(text)
                if number != int(number):
                    return str(value), False
                return int(number), True
        if sqlite_type == "REAL":
            return float(value), True
        if sqlite_type == "BLOB":
            if isinstance(value, (bytes, bytearray)):
                return bytes(value), True
            return bytes.fromhex(str(value).strip()), True
    except (TypeError, ValueError):
        return str(value), False
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex(), True
    return str(value), True


def _canonical(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"blob": bytes(value).hex()}
    if isinstance(value, float):
        return {"real": repr(value)}
    return value


class TableDigest:
    """Laufende SHA-256-Prüfsumme über Zeilen in Reihenfolge. - Claude Generated"""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.rows = 0

    def update(self, row: Sequence[Any]) -> None:
        line = json.dumps([_canonical(v) for v in row], ensure_ascii=False, separators=(",", ":"))
        self._hash.update(line.encode("utf-8"))
        self._hash.update(b"\n")
        self.rows += 1

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def iter_mdb_csv(stream: Iterable[str], expected_header: Optional[List[str]] = None) -> Iterator[List[Optional[str]]]:
    """CSV aus ``mdb-export -0 <SENTINEL>`` zeilenweise lesen. - Claude Generated

    ``stream`` muss mit ``newline=""`` geöffnet sein, damit Zeilenumbrüche in
    Memo-Feldern im Feld bleiben. NULL-Sentinel → ``None``; ``""`` bleibt die
    leere Zeichenkette. Weicht der Kopf von ``expected_header`` ab oder enthält
    ein Wert das Sentinel als Teil, wird abgebrochen.
    """
    reader = csv.reader(stream)
    header = next(reader, None)
    if header is None:
        return
    if expected_header is not None and header != list(expected_header):
        raise ConversionError(f"CSV-Kopf {header!r} passt nicht zum Schema {list(expected_header)!r}")
    width = len(header)
    for lineno, row in enumerate(reader, start=2):
        if len(row) != width:
            raise ConversionError(f"CSV-Datensatz {lineno}: {len(row)} statt {width} Felder")
        out: List[Optional[str]] = []
        for value in row:
            if value == NULL_SENTINEL:
                out.append(None)
            elif NULL_SENTINEL in value:
                raise ConversionError(f"CSV-Datensatz {lineno}: NULL-Sentinel im Wert")
            else:
                out.append(value)
        yield out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Leser
# ─────────────────────────────────────────────────────────────────────────────


class MdbToolsSource:
    """Leser über mdbtools (subprocess). - Claude Generated"""

    label = "mdbtools"
    REQUIRED = ("mdb-tables", "mdb-schema", "mdb-export")

    def __init__(self, path: Path, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.path = Path(path)
        self._run = runner
        missing = [tool for tool in self.REQUIRED if shutil.which(tool) is None]
        if missing:
            raise ReaderUnavailable(
                "mdbtools nicht gefunden (fehlt: " + ", ".join(missing) + "). "
                "Paket 'mdbtools' installieren oder --reader access-parser wählen."
            )
        self._has_count = shutil.which("mdb-count") is not None
        self._schema: Optional[Dict[str, TableSpec]] = None

    def _text(self, *args: str) -> str:
        proc = self._run(list(args), capture_output=True, check=False)
        if proc.returncode != 0:
            raise ConversionError(f"{args[0]} fehlgeschlagen ({proc.returncode}): "
                                  f"{proc.stderr.decode('utf-8', 'replace').strip()[:300]}")
        return proc.stdout.decode("utf-8")

    def describe(self) -> Dict[str, Any]:
        version = ""
        if shutil.which("mdb-ver"):
            try:
                version = self._text("mdb-ver", str(self.path)).strip()
            except ConversionError:
                version = ""
        tool = ""
        try:
            tool = self._text("mdb-export", "--version").strip()
        except ConversionError:
            pass
        return {"reader": self.label, "reader_version": tool, "jet_version": version}

    def schema(self) -> Dict[str, TableSpec]:
        if self._schema is not None:
            return self._schema
        names = [n for n in self._text("mdb-tables", "-1", str(self.path)).splitlines() if n.strip()]
        columns = parse_access_schema(self._text("mdb-schema", "--no-relations", str(self.path), "access"))
        keys = parse_sqlite_keys(self._text("mdb-schema", "--indexes", "--no-relations", str(self.path), "sqlite"))
        out: Dict[str, TableSpec] = {}
        for name in names:
            spec = columns.get(name) or TableSpec(name=name)
            spec.indexes = keys.get(name, [])
            out[name] = spec
        self._schema = out
        return out

    def rows(self, table: str) -> Iterator[List[Any]]:
        spec = self.schema()[table]
        cmd = ["mdb-export", "-0", NULL_SENTINEL, "-D", "%Y-%m-%d", "-T", "%Y-%m-%d %H:%M:%S",
               "-b", "hex", str(self.path), table]
        import tempfile

        errfile = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile)
        try:
            stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", newline="")
            expected = [c.name for c in spec.columns] if spec.columns else None
            yield from iter_mdb_csv(stream, expected)
        finally:
            if proc.stdout:
                proc.stdout.close()
            code = proc.wait()
            errfile.seek(0)
            stderr = errfile.read().decode("utf-8", "replace")
            errfile.close()
            if code != 0:
                raise ConversionError(f"mdb-export {table} fehlgeschlagen ({code}): {stderr.strip()[:300]}")

    def count(self, table: str) -> Optional[int]:
        if not self._has_count:
            return None
        try:
            return int(self._text("mdb-count", str(self.path), table).strip())
        except (ConversionError, ValueError):
            return None


class AccessParserSource:
    """Zweiter, optionaler Leser über das Python-Paket ``access_parser``.

    Datums- und Währungswerte formatiert dieses Paket anders als mdbtools; für
    die Prüfung von Text- und Zahlspalten taugt es als unabhängiger Vergleich.
    - Claude Generated
    """

    label = "access-parser"
    _SYSTEM = re.compile(r"^(MSys|f_[0-9A-F]{32}_Data$)")

    def __init__(self, path: Path):
        try:
            from access_parser import AccessParser  # type: ignore
        except ImportError as exc:
            raise ReaderUnavailable(
                "Python-Paket 'access_parser' nicht installiert (pip install access-parser) "
                "oder --reader mdbtools wählen."
            ) from exc
        import logging

        logging.getLogger("access_parser").setLevel(logging.CRITICAL)
        self.path = Path(path)
        self._db = AccessParser(str(self.path))
        self._schema: Optional[Dict[str, TableSpec]] = None

    def describe(self) -> Dict[str, Any]:
        try:
            from importlib.metadata import version

            tool = version("access_parser")
        except Exception:  # noqa: BLE001 - nur Beschreibung
            tool = ""
        return {"reader": self.label, "reader_version": tool, "jet_version": f"version={self._db.version}"}

    def schema(self) -> Dict[str, TableSpec]:
        if self._schema is not None:
            return self._schema
        out: Dict[str, TableSpec] = {}
        for name in sorted(self._db.catalog):
            if self._SYSTEM.match(name):
                continue
            table = self._db.get_table(name)
            if table is None:
                continue
            spec = TableSpec(name=name)
            for _, col in sorted(table.columns.items()):
                spec.columns.append(ColumnSpec(
                    name=col.col_name_str,
                    access_type=ACCESS_PARSER_TYPES.get(int(col.type), f"Unknown {col.type}"),
                    size=int(col.length) if col.length else None,
                ))
            if table.primary_keys:
                spec.indexes.append(IndexSpec(f"{name}_pkey", list(table.primary_keys), True, True))
            out[name] = spec
        self._schema = out
        return out

    def rows(self, table: str) -> Iterator[List[Any]]:
        spec = self.schema()[table]
        data = self._db.parse_table(table)
        names = [c.name for c in spec.columns]
        length = max((len(v) for v in data.values() if isinstance(v, list)), default=0)
        for i in range(length):
            yield [data[n][i] if isinstance(data.get(n), list) and i < len(data[n]) else None for n in names]

    def count(self, table: str) -> Optional[int]:
        return None


def open_source(path: Path, reader: str) -> AccessSource:
    if reader == "mdbtools":
        return MdbToolsSource(path)
    if reader == "access-parser":
        return AccessParserSource(path)
    raise ReaderUnavailable(f"Unbekannter Leser '{reader}'")


# ─────────────────────────────────────────────────────────────────────────────
# Planen und Schreiben
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ColumnPlan:
    source: str
    target: str
    access_type: str
    size: Optional[int]
    sqlite_type: str


@dataclass
class TablePlan:
    source: str
    target: str
    columns: List[ColumnPlan]
    indexes: List[IndexSpec]


def plan_tables(schema: Dict[str, TableSpec], tables: Optional[Sequence[str]] = None,
                exclude: Optional[Sequence[str]] = None) -> List[TablePlan]:
    """Zielnamen und -typen festlegen. Reihenfolge = Quellreihenfolge. - Claude Generated"""
    wanted = list(tables) if tables else list(schema)
    unknown = [t for t in wanted if t not in schema]
    if unknown:
        raise ConversionError(f"Tabellen nicht in der Quelle: {unknown}")
    skip = set(exclude or [])
    used = set(RESERVED_TARGETS)
    plans: List[TablePlan] = []
    for name in wanted:
        if name in skip:
            continue
        spec = schema[name]
        if not spec.columns:
            raise ConversionError(f"Tabelle '{name}': keine Spalten im Schema gefunden")
        target = sanitize_identifier(name, used)
        col_used: set = set()
        cols = [
            ColumnPlan(c.name, sanitize_identifier(c.name, col_used), c.access_type, c.size,
                       sqlite_type_for(c.access_type))
            for c in spec.columns
        ]
        plans.append(TablePlan(name, target, cols, list(spec.indexes)))
    return plans


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _create_meta(conn: sqlite3.Connection) -> None:
    conn.executescript(f"""
        CREATE TABLE {META_RUN} (
            id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime TEXT NOT NULL,
            reader TEXT NOT NULL,
            reader_version TEXT,
            jet_version TEXT,
            tool_version TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE TABLE {META_TABLE} (
            source_table TEXT PRIMARY KEY,
            target_table TEXT NOT NULL UNIQUE,
            rows_source INTEGER NOT NULL,
            rows_written INTEGER NOT NULL,
            rows_counted INTEGER,
            checksum_source TEXT NOT NULL,
            checksum_written TEXT NOT NULL,
            coercion_warnings INTEGER NOT NULL,
            index_notes TEXT
        );
        CREATE TABLE {META_COLUMN} (
            source_table TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            source_column TEXT NOT NULL,
            target_column TEXT NOT NULL,
            access_type TEXT NOT NULL,
            access_size INTEGER,
            sqlite_type TEXT NOT NULL,
            PRIMARY KEY (source_table, ordinal)
        );
    """)


@dataclass
class TableResult:
    source: str
    target: str
    rows_source: int
    rows_written: int
    rows_counted: Optional[int]
    checksum_source: str
    checksum_written: str
    coercion_warnings: int
    index_notes: List[str]

    @property
    def written_ok(self) -> bool:
        """Zurückgelesene Zeilen stimmen in Zahl und Prüfsumme mit der Quelle überein."""
        return self.rows_source == self.rows_written and self.checksum_source == self.checksum_written

    @property
    def count_ok(self) -> bool:
        """Die unabhängige Zählung des Lesers (mdb-count) fehlt oder stimmt überein."""
        return self.rows_counted is None or self.rows_counted == self.rows_source

    @property
    def verified(self) -> bool:
        return self.written_ok and self.count_ok


def _write_table(conn: sqlite3.Connection, source: AccessSource, plan: TablePlan,
                 progress: Callable[[str], None], batch: int = 2000) -> TableResult:
    cols = ", ".join(f"{_q(c.target)} {c.sqlite_type}" for c in plan.columns)
    conn.execute(f"CREATE TABLE {_q(plan.target)} ({cols})")
    placeholders = ", ".join("?" for _ in plan.columns)
    insert = f"INSERT INTO {_q(plan.target)} VALUES ({placeholders})"
    digest = TableDigest()
    warnings = 0
    pending: List[List[Any]] = []
    for raw in source.rows(plan.source):
        if len(raw) != len(plan.columns):
            raise ConversionError(f"{plan.source}: Zeile mit {len(raw)} statt {len(plan.columns)} Werten")
        row = []
        for value, col in zip(raw, plan.columns):
            coerced, ok = coerce_value(value, col.sqlite_type)
            warnings += 0 if ok else 1
            row.append(coerced)
        digest.update(row)
        pending.append(row)
        if len(pending) >= batch:
            conn.executemany(insert, pending)
            pending.clear()
            progress(f"  {plan.source}: {digest.rows} Zeilen …")
    if pending:
        conn.executemany(insert, pending)

    # Zurücklesen und gegenprüfen
    back = TableDigest()
    names = ", ".join(_q(c.target) for c in plan.columns)
    for row in conn.execute(f"SELECT {names} FROM {_q(plan.target)} ORDER BY rowid"):
        back.update(list(row))

    notes: List[str] = []
    by_source = {c.source: c.target for c in plan.columns}
    for idx in plan.indexes:
        cols = [by_source.get(c) for c in idx.columns]
        if not all(cols):
            notes.append(f"{idx.name}: Spalte unbekannt, übersprungen")
            continue
        name = sanitize_identifier(f"ix_{plan.target}_{'_'.join(cols)}")
        col_sql = ", ".join(_q(c) for c in cols)
        kind = "UNIQUE INDEX" if idx.unique else "INDEX"
        try:
            conn.execute(f"CREATE {kind} {_q(name)} ON {_q(plan.target)} ({col_sql})")
            notes.append(f"{name}: {'PK' if idx.primary else kind.lower()} ({', '.join(cols)})")
        except sqlite3.IntegrityError as exc:
            conn.execute(f"CREATE INDEX {_q(name)} ON {_q(plan.target)} ({col_sql})")
            notes.append(f"{name}: Eindeutigkeit verletzt ({exc}), als einfacher Index angelegt")

    return TableResult(plan.source, plan.target, digest.rows, back.rows, source.count(plan.source),
                       digest.hexdigest(), back.hexdigest(), warnings, notes)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def convert(source: AccessSource, source_path: Path, target: Path, *,
            tables: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None,
            allow_count_mismatch: bool = False,
            progress: Callable[[str], None] = lambda msg: None) -> List[TableResult]:
    """Quelle 1:1 nach ``target`` schreiben, prüfen, atomar ersetzen. - Claude Generated

    Bei einer Abweichung (Zeilenzahl oder Prüfsumme) bleibt eine vorhandene
    Zieldatei unberührt und es wird ``ConversionError`` geworfen.
    ``allow_count_mismatch`` lässt nur eine abweichende *unabhängige* Zählung
    (``mdb-count``) zu; sie steht dann in ``alima_import_table.rows_counted``.
    Zurückgelesene Zeilen und Prüfsumme müssen immer übereinstimmen.
    """
    source_path = Path(source_path)
    target = Path(target)
    plans = plan_tables(source.schema(), tables, exclude)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    info = source.describe()
    stat = source_path.stat()
    results: List[TableResult] = []
    conn = sqlite3.connect(str(tmp))
    try:
        _create_meta(conn)
        cur = conn.execute(
            f"INSERT INTO {META_RUN} (source_path, source_sha256, source_size, source_mtime, reader, "
            f"reader_version, jet_version, tool_version, started_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (str(source_path.resolve()), sha256_file(source_path), stat.st_size,
             datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
             info.get("reader", ""), info.get("reader_version", ""), info.get("jet_version", ""),
             TOOL_VERSION, _now()),
        )
        run_id = cur.lastrowid
        for n, plan in enumerate(plans, start=1):
            started = time.monotonic()
            progress(f"[{n}/{len(plans)}] {plan.source} → {plan.target}")
            result = _write_table(conn, source, plan, progress)
            results.append(result)
            for ordinal, col in enumerate(plan.columns):
                conn.execute(f"INSERT INTO {META_COLUMN} VALUES (?,?,?,?,?,?,?)",
                             (plan.source, ordinal, col.source, col.target, col.access_type, col.size, col.sqlite_type))
            conn.execute(f"INSERT INTO {META_TABLE} VALUES (?,?,?,?,?,?,?,?,?)",
                         (result.source, result.target, result.rows_source, result.rows_written,
                          result.rows_counted, result.checksum_source, result.checksum_written,
                          result.coercion_warnings, "; ".join(result.index_notes)))
            state = "ok" if result.written_ok else "ABWEICHUNG"
            counted = "" if result.count_ok else f", mdb-count meldet {result.rows_counted}"
            progress(f"  {plan.source}: {result.rows_written} Zeilen, Prüfsumme {state}{counted} "
                     f"({time.monotonic() - started:.1f} s)")
        conn.execute(f"UPDATE {META_RUN} SET finished_at = ? WHERE id = ?", (_now(), run_id))
        conn.commit()
    finally:
        conn.close()
    bad = [r for r in results if not (r.written_ok and (r.count_ok or allow_count_mismatch))]
    if bad:
        raise ConversionError(
            "Verifikation fehlgeschlagen für: " + ", ".join(
                f"{r.source} (Quelle {r.rows_source}, geschrieben {r.rows_written}, gezählt {r.rows_counted})"
                for r in bad
            ) + f"; Zwischendatei bleibt liegen: {tmp}"
        )
    os.replace(tmp, target)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# import-geo: Zwischenstand aus ``convert`` → Plugin-Schema geo_sachdb v1
# ─────────────────────────────────────────────────────────────────────────────

GEO_IMPORTER = "scripts/access_to_sql.py import-geo"
GEO_IMPORT_VERSION = "1"
GEO_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "src" / "utils" / "lookups" / "geo_sachdb" / "schema.py"

#: Platzhalter in ``DK.DK_ID`` ("keine DK eingetragen"); Verweise darauf entfallen.
GEO_PLACEHOLDER_DK_ID = 9999

#: Typkennzeichen am Anfang von ``SWB_AEQUIV`` → ``heading.kind``; alles andere
#: (``|c|``, ``|f|``, ``|t|``, ``|z|``, kein Kennzeichen …) wird ``sonstiges``.
GEO_KIND_BY_INDICATOR = {
    "s": "Sachschlagwort",
    "g": "Geographikum",
    "p": "Person",
    "k": "Körperschaft",
    "b": "Körperschaft",
}

#: Notationstabellen des Zwischenstands: (Tabelle, Notation, Erläuterung, DK_ID-Spalte).
#: Nur ``dk`` ist Ziel der ``SWB.DK_ID*``-Verweise. ``dk_alt`` wird nicht gelesen
#: (Entwicklerentscheidung F2), ebenso ``Geographica``, ``EAS``, ``SS_SG`` und die
#: Spalten ``NOTES``, ``SWB_NOTES``, ``OBER_BEGR``, ``VERWANDTE``.
GEO_CLASSIFICATION_TABLES: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("dk", "dk", "erl_uter", "dk_id"),
    ("dk_bio", "dk_mittler", "erlaeuteru", None),
    ("dk_berg", "dk_bergbau", "erlaeuteru", None),
    ("dk_geographica", "dk", "erl_uter", None),
)
GEO_HEADING_REFS = ("dk_id", "dk_id1", "dk_id2", "dk_id3")

_GEO_INDICATOR = re.compile(r"\|([a-z])(?:/[a-z])*\|")
_GEO_TOKEN = re.compile(r"!([^!\s]*)!")
_GEO_PPN = re.compile(r"[0-9]+[0-9X]")
_LINEBREAK = re.compile(r"[ \t]*(?:\r\n|\r|\n)[ \t]*")


def load_geo_schema():
    """Das Schema-Modul des Plugins per Dateipfad laden (ohne ALIMA-Import). - Claude Generated"""
    import importlib.util

    if not GEO_SCHEMA_FILE.is_file():
        raise ConversionError(f"Schema-Datei des Plugins nicht gefunden: {GEO_SCHEMA_FILE}")
    spec = importlib.util.spec_from_file_location("geo_sachdb_schema", GEO_SCHEMA_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_line(value: Any) -> Tuple[Optional[str], bool]:
    """Zeilenumbrüche (mit umgebenden Leerzeichen) zu einem Leerzeichen.

    Liefert ``(text, geändert)``; ``None`` und reine Leerzeichen werden ``None``.
    Sonst bleibt der Text unverändert (``¬``, ``{…}``-Platzhalter). - Claude Generated
    """
    if value is None:
        return None, False
    raw = str(value)
    text = _LINEBREAK.sub(" ", raw).strip()
    return (text or None), text != raw.strip()


def geo_ppn(text: str) -> Tuple[Optional[str], int]:
    """Erste ``!…!``-Nummer in Form einer PPN und Zahl weiterer PPN-Nummern.

    ``!a…!``-Kennungen, leere ``!!`` und andere Inhalte sind keine PPN. - Claude Generated
    """
    ppns = [t for t in _GEO_TOKEN.findall(text or "") if _GEO_PPN.fullmatch(t)]
    return (ppns[0] if ppns else None), max(0, len(ppns) - 1)


@dataclass
class GeoHeading:
    label: str
    kind: str
    ppn: Optional[str]
    chain: Optional[str]
    variants: Optional[str]
    indicator: str = ""
    further_ppns: int = 0


def parse_geo_heading(raw: Any, clean_label: Callable[[str], str]) -> GeoHeading:
    """``SWB_AEQUIV`` zerlegen. - Claude Generated

    ``|s|Testgestein !111111111! !a222222222! 12345 |s|Verweisung ‡ |s|Verweisung``:
    die bevorzugte Form reicht bis zum nächsten Typkennzeichen hinter der ersten
    PPN (``!…!`` mit Ziffern) oder, ohne PPN, bis zum zweiten Typkennzeichen;
    höchstens bis ``‡``. So bleibt eine Kette wie ``|g|Ort / |s|Thema !PPN!`` eine
    Form. Sie ist ``chain`` (unverändert bis auf Zeilenumbrüche); der Rest ab dort
    ist ``variants`` (Rohform, ein führendes ``‡`` entfällt). ``label`` ist
    ``chain`` ohne Kennzeichen, ``!…!``-Kennungen, ``¬`` und ohne die Zahl am Ende:
    direkt hinter der letzten Kennung 3-7 Ziffern; ohne Kennungen 5-7 Ziffern und
    nur, wenn der Wert ein ``‡`` enthält (eine Jahreszahl am Ende bleibt so
    stehen). Was die Zahl bedeutet, ist nicht geklärt.
    """
    text, _ = one_line(raw)
    text = text or ""
    head, has_sep, _tail = text.partition("‡")
    marks = list(_GEO_INDICATOR.finditer(head))
    indicator = ""
    if marks and not head[: marks[0].start()].strip():
        indicator = marks[0].group(1)
    first_ppn = next((m for m in _GEO_TOKEN.finditer(head) if _GEO_PPN.fullmatch(m.group(1))), None)
    after = first_ppn.end() if first_ppn else (marks[0].end() if indicator else 0)
    end = next((m.start() for m in marks if m.start() >= after), len(head))
    chain = head[:end].strip()
    variants = text[end:].strip()
    if variants.startswith("‡"):
        variants = variants[1:].strip()
    ppn, further = geo_ppn(chain)
    body = _GEO_TOKEN.sub(" ", _GEO_INDICATOR.sub(" ", chain, count=1) if indicator else chain).rstrip()
    stripped = chain.rstrip()
    if _GEO_TOKEN.search(chain):
        number = re.fullmatch(r"\s*(\d{3,7})", stripped[stripped.rfind("!") + 1:])
    elif has_sep:
        number = re.search(r"\s(\d{5,7})$", stripped)
    else:
        number = None
    if number:
        body = body[: len(body) - len(number.group(1))]
    return GeoHeading(
        label=clean_label(body), kind=GEO_KIND_BY_INDICATOR.get(indicator, "sonstiges"), ppn=ppn,
        chain=chain or None, variants=variants or None, indicator=indicator, further_ppns=further,
    )


def _ref_id(value: Any) -> Optional[int]:
    """DK_ID-Werte (Double in Access) als ganze Zahl; anderes → None."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


@dataclass
class GeoImportReport:
    rows: Dict[str, int]
    rules: Dict[str, int]
    with_ppn: int
    kinds: Dict[str, int]


#: Regeln, deren Zählung in ``alima_import_rule`` steht (Reihenfolge = Ausgabe).
GEO_RULES = {
    "dk_id_9999_entry": "Platzhalter-Eintrag DK_ID 9999 in dk nicht übernommen",
    "dk_ref_9999_hidden": "Verweise SWB.DK_ID* auf den Platzhalter 9999 ausgeblendet",
    "dk_id_duplicate": "DK_ID mit mehr als einem Eintrag in dk (Verknüpfung ambiguous=1)",
    "link_ambiguous": "Verknüpfungen mit ambiguous=1",
    "dk_ref_unresolved": "Verweise SWB.DK_ID* ohne Eintrag in dk (entfallen)",
    "dk_ref_repeated": "Verweis eines Satzes doppelt auf denselben Eintrag (einmal übernommen)",
    "classification_empty": "Notationszeilen ohne Notation (nicht übernommen)",
    "heading_without_label": "SWB-Sätze ohne Ansetzung (nicht übernommen)",
    "heading_without_indicator": "SWB-Sätze ohne Typkennzeichen am Anfang (kind = sonstiges)",
    "heading_further_ppn_ignored": "SWB-Sätze mit weiteren PPN-Nummern nach der ersten (ignoriert)",
    "newline_to_space": "Textfelder mit Zeilenumbruch, zu Leerzeichen gemacht",
}


def import_geo(staging: Path, target: Path, *,
               progress: Callable[[str], None] = lambda msg: None) -> GeoImportReport:
    """Zwischenstand (Ausgabe von ``convert``) ins Plugin-Schema v1 schreiben. - Claude Generated

    Liest ``swb`` und die Notationstabellen aus :data:`GEO_CLASSIFICATION_TABLES`
    (nur lesend), schreibt ``heading``, ``classification``,
    ``heading_classification``, ``meta`` sowie die Herkunft (``alima_import_run``
    aus dem Zwischenstand kopiert, ``alima_import_rule`` mit den Zählungen dieses
    Laufs). Die Zieldatei wird erst ersetzt, wenn sie die Schemaprüfung des
    Plugins besteht und die Zeilenzahlen stimmen.
    """
    schema = load_geo_schema()
    staging = Path(staging)
    target = Path(target)
    src = sqlite3.connect(Path(staging).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        existing = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        needed = ["swb", "dk"]
        missing = [t for t in needed if t not in existing]
        if missing:
            raise ConversionError(f"Tabellen {missing} fehlen im Zwischenstand {staging.name} "
                                  "(erzeugt mit: access_to_sql.py convert)")
        rules = {name: 0 for name in GEO_RULES}

        def text_of(value: Any) -> Optional[str]:
            text, changed = one_line(value)
            rules["newline_to_space"] += int(changed)
            return text

        classifications: List[Tuple[int, str, str, Optional[str], Optional[str]]] = []
        by_dk_id: Dict[int, List[int]] = {}
        for table, code_col, label_col, id_col in GEO_CLASSIFICATION_TABLES:
            if table not in existing:
                progress(f"  {table}: nicht im Zwischenstand, übersprungen")
                continue
            id_sql = _q(id_col) if id_col else "NULL"
            for dk_id, code_raw, label_raw in src.execute(
                f"SELECT {id_sql}, {_q(code_col)}, {_q(label_col)} FROM {_q(table)} ORDER BY rowid"
            ):
                ref = _ref_id(dk_id)
                if id_col and ref == GEO_PLACEHOLDER_DK_ID:
                    rules["dk_id_9999_entry"] += 1
                    continue
                code = text_of(code_raw) or ""
                label = text_of(label_raw)
                ppn, _ = geo_ppn(code)
                bare = re.sub(r"\s+", " ", _GEO_TOKEN.sub(" ", code)).strip().strip('"').strip()
                system, notation = schema.normalize_notation(bare, "DK")
                if not notation:
                    rules["classification_empty"] += 1
                    continue
                cid = len(classifications) + 1
                classifications.append((cid, system, notation, schema.clean_label(label) or None, ppn))
                if id_col and ref is not None:
                    by_dk_id.setdefault(ref, []).append(cid)
        rules["dk_id_duplicate"] = sum(1 for ids in by_dk_id.values() if len(ids) > 1)

        headings: List[Tuple[int, str, str, Optional[str], Optional[str], Optional[str]]] = []
        links: List[Tuple[int, int, int, int]] = []
        seen_ids: set = set()
        ref_cols = ", ".join(_q(c) for c in GEO_HEADING_REFS)
        for row in src.execute(f"SELECT lauf_nr, swb_aequiv, {ref_cols} FROM swb ORDER BY rowid"):
            hid = _ref_id(row[0])
            if hid is None or hid in seen_ids:
                raise ConversionError(f"swb.lauf_nr {row[0]!r} ist keine eindeutige ganze Zahl")
            seen_ids.add(hid)
            rules["newline_to_space"] += int(one_line(row[1])[1])
            parsed = parse_geo_heading(row[1], schema.clean_label)
            if not parsed.label:
                rules["heading_without_label"] += 1
                continue
            rules["heading_without_indicator"] += int(not parsed.indicator)
            rules["heading_further_ppn_ignored"] += int(parsed.further_ppns > 0)
            headings.append((hid, parsed.label, parsed.kind, parsed.ppn, parsed.chain, parsed.variants))
            linked: set = set()
            for position, value in enumerate(row[2:]):
                ref = _ref_id(value)
                if ref is None:
                    continue
                if ref == GEO_PLACEHOLDER_DK_ID:
                    rules["dk_ref_9999_hidden"] += 1
                    continue
                targets = by_dk_id.get(ref)
                if not targets:
                    rules["dk_ref_unresolved"] += 1
                    continue
                for cid in targets:
                    if cid in linked:
                        rules["dk_ref_repeated"] += 1
                        continue
                    linked.add(cid)
                    links.append((hid, cid, position, int(len(targets) > 1)))
        rules["link_ambiguous"] = sum(1 for link in links if link[3])

        run_rows: List[Tuple[Any, ...]] = []
        run_cols: List[str] = []
        if META_RUN in existing:
            cur = src.execute(f"SELECT * FROM {META_RUN} ORDER BY id")
            run_cols = [d[0] for d in cur.description]
            run_rows = cur.fetchall()
    finally:
        src.close()

    last_run = dict(zip(run_cols, run_rows[-1])) if run_rows else {}
    meta = {
        "source": Path(str(last_run.get("source_path") or staging)).name,
        "source_sha256": str(last_run.get("source_sha256") or sha256_file(staging)),
        "source_date": str(last_run.get("source_mtime") or ""),
        "created_at": _now(),
        "importer": f"{GEO_IMPORTER} {GEO_IMPORT_VERSION}",
    }

    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(str(tmp))
    try:
        schema.create_schema(conn)
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [(k, v) for k, v in meta.items() if v])
        conn.executemany("INSERT INTO classification (id, system, notation, label, ppn) VALUES (?,?,?,?,?)",
                         classifications)
        conn.executemany("INSERT INTO heading (id, label, kind, ppn, chain, variants) VALUES (?,?,?,?,?,?)",
                         headings)
        conn.executemany("INSERT INTO heading_classification (heading_id, classification_id, position, ambiguous) "
                         "VALUES (?,?,?,?)", links)
        if run_cols:
            cols = ", ".join(_q(c) for c in run_cols)
            conn.execute(f"CREATE TABLE {META_RUN} ({cols})")
            conn.executemany(f"INSERT INTO {META_RUN} VALUES ({', '.join('?' for _ in run_cols)})", run_rows)
        conn.execute("CREATE TABLE alima_import_rule (rule TEXT PRIMARY KEY, count INTEGER NOT NULL, note TEXT)")
        conn.executemany("INSERT INTO alima_import_rule VALUES (?,?,?)",
                         [(name, rules[name], note) for name, note in GEO_RULES.items()])
        conn.commit()
        reason = schema.check_schema(conn)
        written = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                   for t in ("heading", "classification", "heading_classification")}
        kinds = dict(conn.execute("SELECT kind, COUNT(*) FROM heading GROUP BY kind ORDER BY kind").fetchall())
        with_ppn = conn.execute("SELECT COUNT(*) FROM heading WHERE ppn IS NOT NULL").fetchone()[0]
    finally:
        conn.close()
    expected = {"heading": len(headings), "classification": len(classifications),
                "heading_classification": len(links)}
    if reason or written != expected:
        raise ConversionError(f"Prüfung fehlgeschlagen ({reason or written}); Zwischendatei bleibt liegen: {tmp}")
    os.replace(tmp, target)
    progress(f"  heading {written['heading']}, classification {written['classification']}, "
             f"heading_classification {written['heading_classification']}")
    return GeoImportReport(rows=written, rules=rules, with_ppn=with_ppn, kinds=kinds)


# ─────────────────────────────────────────────────────────────────────────────
# inspect
# ─────────────────────────────────────────────────────────────────────────────


def inspect(source: AccessSource, *, samples: int = 0, max_chars: int = 60) -> Dict[str, Any]:
    """Schema, Zeilenzahlen (csv-Zeilen + ggf. mdb-count) und Stichproben. - Claude Generated"""
    samples = max(0, min(int(samples or 0), 5))
    report: Dict[str, Any] = {"source": source.describe(), "tables": []}
    for plan in plan_tables(source.schema()):
        rows = 0
        shown: List[Dict[str, Any]] = []
        for raw in source.rows(plan.source):
            rows += 1
            if len(shown) < samples:
                shown.append({c.source: (None if v is None else str(v)[:max_chars])
                              for c, v in zip(plan.columns, raw)})
        report["tables"].append({
            "source": plan.source,
            "target": plan.target,
            "rows": rows,
            "rows_counted": source.count(plan.source),
            "columns": [asdict(c) for c in plan.columns],
            "indexes": [asdict(i) for i in plan.indexes],
            "samples": shown,
        })
    return report


def _print_inspect(report: Dict[str, Any]) -> None:
    src = report["source"]
    print(f"Leser: {src.get('reader')} {src.get('reader_version', '')}  Format: {src.get('jet_version', '')}")
    for table in report["tables"]:
        counted = table["rows_counted"]
        extra = "" if counted is None else f" (mdb-count {counted})"
        print(f"\n{table['source']} → {table['target']}: {table['rows']} Zeilen{extra}")
        for col in table["columns"]:
            size = f"({col['size']})" if col["size"] else ""
            print(f"  {col['source']:<24} {col['access_type'] + size:<22} → {col['target']} {col['sqlite_type']}")
        for idx in table["indexes"]:
            kind = "PK" if idx["primary"] else ("UNIQUE" if idx["unique"] else "INDEX")
            print(f"  [{kind}] {', '.join(idx['columns'])}")
        for sample in table["samples"]:
            print(f"  · {sample}")


def _main_import_geo(args: argparse.Namespace) -> int:
    """CLI-Teil von ``import-geo``: Zahlen ausgeben, keine Datensätze. - Claude Generated"""
    if args.target.resolve() == args.source.resolve():
        print("Ziel und Quelle sind dieselbe Datei", file=sys.stderr)
        return 1
    try:
        report = import_geo(args.source, args.target, progress=lambda msg: print(msg, file=sys.stderr))
    except (ConversionError, sqlite3.Error) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    rows = report.rows
    print(f"{args.target}: heading {rows['heading']} (mit PPN {report.with_ppn}), "
          f"classification {rows['classification']}, heading_classification {rows['heading_classification']}")
    print("kind: " + ", ".join(f"{k} {n}" for k, n in report.kinds.items()))
    for name, count in report.rules.items():
        print(f"  {name}: {count}")
    return 0


def main(argv: Optional[Sequence[str]] = None,
         source_factory: Callable[[Path, str], AccessSource] = open_source) -> int:
    parser = argparse.ArgumentParser(description="Access-Datenbank nach SQLite (inspect/convert/import-geo).")
    parser.add_argument("--reader", choices=["mdbtools", "access-parser"], default="mdbtools")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_inspect = sub.add_parser("inspect", help="Schema und Zeilenzahlen ausgeben, nichts schreiben")
    p_inspect.add_argument("source", type=Path)
    p_inspect.add_argument("--json", type=Path, help="Bericht zusätzlich als JSON schreiben")
    p_inspect.add_argument("--samples", type=int, default=0, help="Stichproben je Tabelle (höchstens 5)")
    p_convert = sub.add_parser("convert", help="nach SQLite schreiben und prüfen")
    p_convert.add_argument("source", type=Path)
    p_convert.add_argument("target", type=Path)
    p_convert.add_argument("--tables", nargs="+", help="nur diese Quelltabellen")
    p_convert.add_argument("--exclude", nargs="+", help="diese Quelltabellen auslassen")
    p_convert.add_argument("--allow-count-mismatch", action="store_true",
                           help="abweichendes mdb-count hinnehmen (wird protokolliert); "
                                "Prüfsumme und zurückgelesene Zeilen müssen weiter stimmen")
    p_geo = sub.add_parser("import-geo", help="Zwischenstand aus convert (SQLite) ins Schema des "
                                              "Plugins geo_sachdb (v1) schreiben")
    p_geo.add_argument("source", type=Path, help="SQLite-Datei, die convert geschrieben hat")
    p_geo.add_argument("target", type=Path)
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"Quelle nicht gefunden: {args.source}", file=sys.stderr)
        return 1
    if args.mode == "import-geo":
        return _main_import_geo(args)
    try:
        source = source_factory(args.source, args.reader)
    except ReaderUnavailable as exc:
        print(f"Leser nicht verfügbar: {exc}", file=sys.stderr)
        return 1

    try:
        if args.mode == "inspect":
            report = inspect(source, samples=args.samples)
            _print_inspect(report)
            if args.json:
                args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            return 0
        if args.target.resolve() == args.source.resolve():
            print("Ziel und Quelle sind dieselbe Datei", file=sys.stderr)
            return 1
        results = convert(source, args.source, args.target, tables=args.tables, exclude=args.exclude,
                          allow_count_mismatch=args.allow_count_mismatch,
                          progress=lambda msg: print(msg, file=sys.stderr))
    except ConversionError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    total = sum(r.rows_written for r in results)
    print(f"{len(results)} Tabellen, {total} Zeilen nach {args.target} geschrieben und geprüft.")
    for r in results:
        if not r.count_ok:
            print(f"Hinweis: {r.source}: {r.rows_source} Zeilen gelesen, mdb-count meldet {r.rows_counted}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
