#!/usr/bin/env python3
"""Access-Datenbank (.mdb/.accdb) nach SQLite: Bestandsaufnahme und 1:1-Umwandlung.

Zwei Modi:

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


def main(argv: Optional[Sequence[str]] = None,
         source_factory: Callable[[Path, str], AccessSource] = open_source) -> int:
    parser = argparse.ArgumentParser(description="Access-Datenbank nach SQLite (inspect/convert).")
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
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"Quelle nicht gefunden: {args.source}", file=sys.stderr)
        return 1
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
