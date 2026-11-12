"""Cache rows across installations: whitelist, validation, export read, merge.

Claude Generated. Used by the institution package (``src/utils/preset_cache.py``)
to ship parts of the search cache to colleagues: the DK cache
(``catalog_dk_cache``), the GND mappings (``search_mappings``) and, on request,
raw source answers (``search_response_cache``). Never the GND copy
(``gnd_entries`` lives in its own store, ``gnd_local.db``), never another table.

Three layers:

* :data:`CACHE_TABLES`: the columns of the three tables as the schema creates
  them (``_ukm_schema.SchemaMigrationMixin._init_database``), with their kind,
  primary key and NOT NULL columns. A test compares it with a freshly created
  SQLite database, so the whitelist cannot drift from the CREATE statements.
* :func:`check_columns` / :func:`check_row`: what a package may carry. Unknown
  table or column, a wrong type, a missing key, a row the export rule would not
  have written: rejected (``ValueError`` with the reason).
* :func:`iter_export_rows` / :func:`cache_inventory` (read) and
  :func:`merge_rows` (count or write). Both take a ``DatabaseManager`` so the
  package can read and count without the singleton; writes go through
  :meth:`CacheMergeMixin.merge_cache_rows` of :class:`UnifiedKnowledgeManager`.

Merge rule: per primary key the row with the newer ``last_updated`` wins; a
local row is never deleted; ``created_at`` and ``last_updated`` are taken from
the package. The SQL is plain ``SELECT``/``INSERT``/``UPDATE`` with ``?``
placeholders, no ``INSERT OR REPLACE``, no ``ON CONFLICT``, no
``ON DUPLICATE KEY``, no pragma, one transaction per block. Run against SQLite
only; the MariaDB/MySQL form is not tested against a server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

TEXT = "text"
INTEGER = "integer"
TIMESTAMP = "timestamp"
JSON_TEXT = "json"

#: Raw answers of these sources never leave an installation (KVK: answers of
#: other libraries' catalogues, fetched under their terms).
NEVER_EXPORTED_RAW_SOURCES = frozenset({"kvk"})

#: Rows per transaction when writing.
MERGE_BLOCK_SIZE = 500
#: Rows per SELECT when reading for an export.
EXPORT_PAGE_SIZE = 500

#: Status values of ``catalog_dk_cache`` that are not a hit.
_DK_SUCCESS = "success"

#: Keys of a raw answer that tell an empty answer (see :func:`raw_answer_problem`).
_RAW_COUNT_KEYS = ("totalItems", "total", "numFound")
_RAW_LIST_KEYS = ("member", "records", "docs")


@dataclass(frozen=True)
class CacheTableSpec:
    """Columns of one cache table as the schema creates them - Claude Generated"""

    name: str
    columns: Tuple[Tuple[str, str], ...]
    key: Tuple[str, ...]
    not_null: Tuple[str, ...]
    #: Columns written with a fixed value on every merge and never exported.
    reset: Tuple[Tuple[str, Any], ...] = ()

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _kind in self.columns)

    def kind(self, column: str) -> str:
        for name, kind in self.columns:
            if name == column:
                return kind
        raise KeyError(column)

    @property
    def reset_columns(self) -> Tuple[str, ...]:
        return tuple(name for name, _value in self.reset)

    @property
    def export_columns(self) -> Tuple[str, ...]:
        skip = set(self.reset_columns)
        return tuple(name for name in self.column_names if name not in skip)

    @property
    def required(self) -> Tuple[str, ...]:
        """Columns a package file must carry: key, NOT NULL, ``last_updated``."""
        out = list(self.key)
        for name in self.not_null + ("last_updated",):
            if name not in out:
                out.append(name)
        return tuple(out)


CACHE_TABLES: Dict[str, CacheTableSpec] = {
    "catalog_dk_cache": CacheTableSpec(
        "catalog_dk_cache",
        (("search_term", TEXT), ("normalized_term", TEXT), ("found_titles", JSON_TEXT),
         ("result_count", INTEGER), ("last_updated", TIMESTAMP), ("created_at", TIMESTAMP),
         ("search_status", TEXT), ("error_message", TEXT), ("retry_after", TIMESTAMP),
         ("consecutive_failures", INTEGER)),
        key=("search_term",),
        not_null=("search_term", "normalized_term"),
        # A failure state of the exporting machine says nothing about the target.
        reset=(("error_message", None), ("retry_after", None), ("consecutive_failures", 0)),
    ),
    "search_mappings": CacheTableSpec(
        "search_mappings",
        (("search_term", TEXT), ("normalized_term", TEXT), ("suggester_type", TEXT),
         ("found_gnd_ids", JSON_TEXT), ("found_classifications", JSON_TEXT),
         ("gnd_counts", JSON_TEXT), ("titles", JSON_TEXT), ("result_count", INTEGER),
         ("last_updated", TIMESTAMP), ("created_at", TIMESTAMP)),
        key=("search_term", "suggester_type"),
        not_null=("search_term", "normalized_term", "suggester_type"),
    ),
    "search_response_cache": CacheTableSpec(
        "search_response_cache",
        (("source", TEXT), ("query", TEXT), ("normalized_query", TEXT), ("params_hash", TEXT),
         ("params_json", JSON_TEXT), ("raw_json", JSON_TEXT), ("http_status", INTEGER),
         ("result_count", INTEGER), ("byte_size", INTEGER), ("last_updated", TIMESTAMP),
         ("created_at", TIMESTAMP)),
        key=("source", "normalized_query", "params_hash"),
        not_null=("source", "query", "normalized_query", "params_hash"),
    ),
}


def table_spec(table: str) -> CacheTableSpec:
    """The spec of a whitelisted table; ``ValueError`` for any other - Claude Generated"""
    spec = CACHE_TABLES.get(str(table or ""))
    if spec is None:
        raise ValueError(f"Tabelle '{table}' ist nicht erlaubt "
                         f"(erlaubt: {', '.join(CACHE_TABLES)})")
    return spec


# ── values ────────────────────────────────────────────────────────────────


def parse_timestamp(value: Any) -> datetime:
    """A stored time stamp as naive ``datetime``; ``ValueError`` if not one.

    Accepts the SQLite text form (``2026-11-10 14:02:00``, also with ``T`` and
    fractions), a ``datetime`` and a ``QDateTime`` (MariaDB through QtSql). An
    offset is converted to UTC and dropped. - Claude Generated
    """
    if hasattr(value, "toPyDateTime"):
        value = value.toPyDateTime()
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value.strip():
        try:
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"kein Zeitstempel: {value!r}") from exc
    else:
        raise ValueError(f"kein Zeitstempel: {value!r}")
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp


def format_timestamp(value: Any) -> str:
    """``YYYY-MM-DD HH:MM:SS`` (the form of SQLite's CURRENT_TIMESTAMP) - Claude Generated"""
    return parse_timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _from_db(kind: str, value: Any) -> Any:
    """A value as QtSql returned it, in the form a package carries.

    ``DatabaseManager.fetch_*`` returns ``""`` for SQL NULL in every column
    type (QSQLITE, checked), so ``""`` in an integer, time stamp or JSON column
    is read as NULL; no writer stores an empty string there. A text column keeps
    ``""``. - Claude Generated
    """
    if value is None or (value == "" and kind in (INTEGER, TIMESTAMP, JSON_TEXT)):
        return None
    if kind == TIMESTAMP:
        return format_timestamp(value)
    if kind == INTEGER:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return int(float(str(value)))
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    if not isinstance(value, str):
        return str(value)
    return value


def raw_answer_problem(raw: Any, result_count: Any) -> str:
    """Why a stored raw answer is not worth shipping, ``""`` when it is.

    A row in ``search_response_cache`` is written by a source after a fetch;
    failures are mostly not written (lobid raises before, swb skips after an
    error, finc skips a source that is down, the input tools skip payloads with
    ``"error"``). What remains to filter: a payload with a non-empty
    ``error``/``errors`` key, and an empty answer: ``totalItems``/``total``/
    ``numFound`` equal to 0, or ``member``/``records``/``docs`` empty with no
    count above 0, or ``subjects`` empty (swb). ``result_count`` is set by kvk
    and sru only (0 for every other source), so it alone does not mark an empty
    answer. Payloads of another shape are kept. - Claude Generated
    """
    if isinstance(result_count, int) and not isinstance(result_count, bool) and result_count > 0:
        return ""
    if raw in (None, "", [], {}):
        return "leere Antwort"
    if not isinstance(raw, dict):
        return ""
    if raw.get("error") or raw.get("errors"):
        return "Antwort mit Fehler"
    counts = [raw[k] for k in _RAW_COUNT_KEYS if isinstance(raw.get(k), int)
              and not isinstance(raw.get(k), bool)]
    if any(c > 0 for c in counts):
        return ""
    if counts and all(c == 0 for c in counts):
        return "leere Antwort"
    lists = [raw[k] for k in _RAW_LIST_KEYS if isinstance(raw.get(k), list)]
    if lists and not any(lists):
        return "leere Antwort"
    if "subjects" in raw and isinstance(raw["subjects"], dict) and not raw["subjects"]:
        return "leere Antwort"
    return ""


def row_problem(table: str, row: Dict[str, Any], parsed: Optional[Dict[str, Any]] = None) -> str:
    """Why a (checked) row may not travel, ``""`` when it may.

    The same rule as the export filter: ``catalog_dk_cache`` only ``success``
    with ``result_count > 0`` (a ``no_results`` or failure row would be read as
    a lasting answer in cache-only mode, which ignores the age); raw answers
    never from kvk, only with HTTP status 200 or none recorded, and not empty or
    failed (:func:`raw_answer_problem`). - Claude Generated
    """
    if table == "catalog_dk_cache":
        count = row.get("result_count")
        if row.get("search_status") != _DK_SUCCESS or not isinstance(count, int) or count <= 0:
            return "nur erfolgreiche Einträge mit Treffern"
    elif table == "search_response_cache":
        source = str(row.get("source") or "").strip().lower()
        if source in NEVER_EXPORTED_RAW_SOURCES:
            return f"Rohantworten der Quelle '{source}' werden nie übernommen"
        status = row.get("http_status")
        if status is not None and status != 200:
            return f"HTTP-Status {status}"
        raw = (parsed or {}).get("raw_json")
        if raw is None and row.get("raw_json") is not None:
            try:
                raw = json.loads(row["raw_json"])
            except (TypeError, ValueError):
                return "raw_json ist kein JSON"
        problem = raw_answer_problem(raw, row.get("result_count"))
        if problem:
            return problem
    return ""


def check_columns(spec: CacheTableSpec, columns: Any) -> Tuple[str, ...]:
    """The column list of a package file, checked against the schema - Claude Generated"""
    if not isinstance(columns, list) or not columns:
        raise ValueError("'columns' fehlt oder ist keine Liste")
    out: List[str] = []
    for col in columns:
        if not isinstance(col, str):
            raise ValueError(f"Spaltenname {col!r} ist kein Text")
        if col not in spec.column_names:
            raise ValueError(f"unbekannte Spalte '{col}' in {spec.name}")
        if col in out:
            raise ValueError(f"Spalte '{col}' doppelt")
        out.append(col)
    missing = [c for c in spec.required if c not in out]
    if missing:
        raise ValueError(f"Pflichtspalten fehlen: {', '.join(missing)}")
    return tuple(out)


def check_row(spec: CacheTableSpec, columns: Sequence[str], values: Any) -> Dict[str, Any]:
    """One row of a package file as a checked dict; ``ValueError`` otherwise.

    Kinds: text = ``str``, integer = ``int`` (not ``bool``), timestamp = text
    in the form :func:`parse_timestamp` reads (written back as
    ``YYYY-MM-DD HH:MM:SS``), json = text holding JSON. Key and NOT NULL columns
    and ``last_updated`` must be present; key columns must not be blank. Reset
    columns get their fixed value. Then :func:`row_problem`. - Claude Generated
    """
    if not isinstance(values, list):
        raise ValueError("Zeile ist keine JSON-Liste")
    if len(values) != len(columns):
        raise ValueError(f"Zeile hat {len(values)} Werte, erwartet {len(columns)}")
    row: Dict[str, Any] = {}
    parsed: Dict[str, Any] = {}
    for col, value in zip(columns, values):
        kind = spec.kind(col)
        if value is None:
            if col in spec.required:
                raise ValueError(f"'{col}' fehlt")
            row[col] = None
            continue
        if kind == TEXT:
            if not isinstance(value, str):
                raise ValueError(f"'{col}' ist kein Text")
            if col in spec.key and not value.strip():
                raise ValueError(f"'{col}' ist leer")
            row[col] = value
        elif kind == INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"'{col}' ist keine ganze Zahl")
            row[col] = value
        elif kind == TIMESTAMP:
            row[col] = format_timestamp(value)
        else:  # JSON_TEXT
            if not isinstance(value, str):
                raise ValueError(f"'{col}' ist kein Text")
            try:
                parsed[col] = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"'{col}' ist kein JSON") from exc
            row[col] = value
    for col, fixed in spec.reset:
        row[col] = fixed
    problem = row_problem(spec.name, row, parsed)
    if problem:
        raise ValueError(f"Eintrag {_key_text(spec, row)}: {problem}")
    return row


def _key_text(spec: CacheTableSpec, row: Dict[str, Any]) -> str:
    text = " / ".join(str(row.get(k)) for k in spec.key)
    return f"'{text[:80]}'"


# ── reading for an export ─────────────────────────────────────────────────


def _export_where(spec: CacheTableSpec, source: Optional[str]) -> Tuple[List[str], List[Any]]:
    """The SQL part of the export filter (the rest is :func:`row_problem`)."""
    if spec.name == "catalog_dk_cache":
        return ["search_status = ?", "result_count > 0"], [_DK_SUCCESS]
    if spec.name == "search_response_cache":
        if not source:
            raise ValueError("Rohantworten nur je Quelle")
        if source.strip().lower() in NEVER_EXPORTED_RAW_SOURCES:
            raise ValueError(f"Rohantworten der Quelle '{source}' werden nie exportiert")
        return ["source = ?", "(http_status IS NULL OR http_status = 200)"], [source]
    return [], []


def _after_key(spec: CacheTableSpec, last: Optional[Tuple[Any, ...]]) -> Tuple[str, List[Any]]:
    """Keyset condition ``key > last`` in plain SQL (both databases)."""
    if last is None:
        return "", []
    parts, params = [], []
    for n, col in enumerate(spec.key):
        eq = [f"{c} = ?" for c in spec.key[:n]]
        parts.append("(" + " AND ".join(eq + [f"{col} > ?"]) + ")")
        params.extend(list(last[:n]) + [last[n]])
    return "(" + " OR ".join(parts) + ")", params


def iter_export_rows(db, table: str, *, source: Optional[str] = None,
                     page_size: int = EXPORT_PAGE_SIZE,
                     skipped: Optional[Dict[str, int]] = None) -> Iterator[Dict[str, Any]]:
    """Rows of ``table`` that may travel, in key order, checked like an import.

    Pages by key (``WHERE key > last ORDER BY key LIMIT n``), so a table that
    changes meanwhile yields no row twice. A row that fails :func:`check_row`
    is left out and counted in ``skipped[reason]``. - Claude Generated
    """
    spec = table_spec(table)
    columns = spec.export_columns
    where, params = _export_where(spec, source)
    order = ", ".join(spec.key)
    last: Optional[Tuple[Any, ...]] = None
    while True:
        after, after_params = _after_key(spec, last)
        conditions = where + ([after] if after else [])
        sql = (f"SELECT {', '.join(columns)} FROM {spec.name}"
               + (" WHERE " + " AND ".join(conditions) if conditions else "")
               + f" ORDER BY {order} LIMIT {int(page_size)}")
        rows = db.fetch_all(sql, params + after_params)
        for raw in rows:
            last = tuple(raw.get(k) for k in spec.key)
            try:
                values = [_from_db(spec.kind(c), raw.get(c)) for c in columns]
                yield check_row(spec, columns, values)
            except (ValueError, TypeError) as exc:
                if skipped is not None:
                    reason = str(exc).split(": ", 1)[-1]
                    skipped[reason] = skipped.get(reason, 0) + 1
        if len(rows) < page_size:
            return


def _int(value: Any) -> int:
    try:
        return int(float(str(value))) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def cache_inventory(db) -> Dict[str, Any]:
    """Rows and approximate size per exportable part, for the export dialog.

    ``{"dk": (rows, bytes), "mappings": (rows, bytes), "raw": [(source, rows,
    bytes), …]}``. Sizes are ``LENGTH`` of the text columns (characters in
    SQLite, bytes in MariaDB), so approximate. Raw rows are counted by the SQL
    part of the filter; empty or failed answers drop out only during the export.
    kvk is never listed. - Claude Generated
    """
    dk = db.fetch_one(
        "SELECT COUNT(*) AS n, SUM(COALESCE(LENGTH(found_titles), 0)) AS b "
        "FROM catalog_dk_cache WHERE search_status = ? AND result_count > 0", [_DK_SUCCESS]) or {}
    mappings = db.fetch_one(
        "SELECT COUNT(*) AS n, SUM(COALESCE(LENGTH(found_gnd_ids), 0) "
        "+ COALESCE(LENGTH(found_classifications), 0) + COALESCE(LENGTH(gnd_counts), 0) "
        "+ COALESCE(LENGTH(titles), 0)) AS b FROM search_mappings") or {}
    raw_rows = db.fetch_all(
        "SELECT source, COUNT(*) AS n, SUM(COALESCE(byte_size, 0)) AS b "
        "FROM search_response_cache WHERE http_status IS NULL OR http_status = 200 "
        "GROUP BY source ORDER BY source")
    raw = [(str(r.get("source") or ""), _int(r.get("n")), _int(r.get("b"))) for r in raw_rows
           if str(r.get("source") or "").strip().lower() not in NEVER_EXPORTED_RAW_SOURCES
           and str(r.get("source") or "").strip()]
    return {"dk": (_int(dk.get("n")), _int(dk.get("b"))),
            "mappings": (_int(mappings.get("n")), _int(mappings.get("b"))),
            "raw": raw}


# ── merge ─────────────────────────────────────────────────────────────────


def _key_digest(spec: CacheTableSpec, row: Dict[str, Any]) -> bytes:
    blob = json.dumps([row[k] for k in spec.key], ensure_ascii=False)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).digest()


def _local_timestamp(value: Any) -> Optional[datetime]:
    try:
        return parse_timestamp(value)
    except ValueError:
        return None


def merge_rows(db, table: str, rows: Iterable[Dict[str, Any]], *, write: bool,
               block_size: int = MERGE_BLOCK_SIZE) -> Dict[str, int]:
    """Count (``write=False``) or write the merge of ``rows`` into ``table``.

    Per primary key: no local row → insert (``new``); the package row has the
    newer ``last_updated`` → update every column the file carries plus the reset
    columns (``updated``); otherwise the local row stays (``unchanged``, also
    when the local time stamp cannot be read). Nothing is deleted. For
    ``catalog_dk_cache``, ``local_failures`` counts the unchanged keys whose
    newer local row is not a hit. Writing runs one transaction per
    ``block_size`` rows; an exception rolls back the open block and is raised
    (earlier blocks stay). A key twice in ``rows`` raises ``ValueError``.
    ``rows`` are dicts from :func:`check_row`. - Claude Generated
    """
    spec = table_spec(table)
    counts = {"new": 0, "updated": 0, "unchanged": 0, "local_failures": 0}
    seen: set = set()
    where = " AND ".join(f"{k} = ?" for k in spec.key)
    probe_cols = "last_updated" + (", search_status" if spec.name == "catalog_dk_cache" else "")
    select_sql = f"SELECT {probe_cols} FROM {spec.name} WHERE {where}"
    in_block = 0
    tx_open = False
    try:
        for row in rows:
            digest = _key_digest(spec, row)
            if digest in seen:
                raise ValueError(f"Eintrag {_key_text(spec, row)} doppelt")
            seen.add(digest)
            if write and not tx_open:
                db.begin_transaction()
                tx_open = True
            key = [row[k] for k in spec.key]
            local = db.fetch_one(select_sql, key)
            if local is None:
                counts["new"] += 1
                if write:
                    cols = list(row)
                    db.execute_query(
                        f"INSERT INTO {spec.name} ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])
            else:
                local_ts = _local_timestamp(local.get("last_updated"))
                if local_ts is not None and parse_timestamp(row["last_updated"]) > local_ts:
                    counts["updated"] += 1
                    if write:
                        cols = [c for c in row if c not in spec.key]
                        db.execute_query(
                            f"UPDATE {spec.name} SET {', '.join(f'{c} = ?' for c in cols)} "
                            f"WHERE {where}", [row[c] for c in cols] + key)
                else:
                    counts["unchanged"] += 1
                    if spec.name == "catalog_dk_cache" and local.get("search_status") != _DK_SUCCESS:
                        counts["local_failures"] += 1
            if write:
                in_block += 1
                if in_block >= block_size:
                    db.commit_transaction()
                    tx_open = False
                    in_block = 0
        if tx_open:
            db.commit_transaction()
            tx_open = False
    except BaseException:
        if tx_open:
            try:
                db.rollback_transaction()
            except Exception:  # the original error is the one to report
                pass
        raise
    return counts


class CacheMergeMixin:
    """``merge_cache_rows``. Mixed into :class:`UnifiedKnowledgeManager` - Claude Generated"""

    def merge_cache_rows(self, table: str, rows: Iterable[Dict[str, Any]], *,
                         dry_run: bool = False,
                         block_size: int = MERGE_BLOCK_SIZE) -> Dict[str, int]:
        """Merge checked package rows into one of the three cache tables.

        See :func:`merge_rows`; ``dry_run`` only counts. - Claude Generated
        """
        return merge_rows(self.db_manager, table, rows, write=not dry_run, block_size=block_size)
