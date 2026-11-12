"""Cache parts of an institution package: write, read, plan, merge - Claude Generated.

A package can carry parts of the search cache, merged into the cache of the
target installation when the preset is applied:

* ``dk``: the DK cache ``catalog_dk_cache`` (what "Nur Cache" reads for
  Libero/finc), only rows with ``search_status = 'success'`` and
  ``result_count > 0``; the failure fields are not shipped and are reset on
  merge;
* ``mappings``: the GND mappings ``search_mappings``, all rows;
* ``raw:<source>``: the raw answers of one source in ``search_response_cache``,
  only on request, never ``kvk``, only answers with HTTP status 200 or none
  recorded and not empty or failed (``_ukm_cache_merge.raw_answer_problem``).

Never the GND copy (``gnd_entries``, ``gnd_local.db``).

File: ``files/cache/<table>[_<source>].jsonl.gz``, gzip, UTF-8, one JSON line
per row. The first line is the header ``{"table", "columns", "rows"}`` (raw
files also ``"source_filter"``), each further line a JSON list of the values in
the order of ``columns``. Written as two gzip members (header, rows) with
``mtime`` 0, read as one stream.

Preset block::

    "cache": {"isil": "DE-105", "created": "<ISO time>",
              "tables": [{"id": "dk_cache", "table": "catalog_dk_cache",
                          "source": "files/cache/catalog_dk_cache.jsonl.gz",
                          "sha256": "…", "rows": 102, "merge": "newer_wins"}, …]}

Guard: the cache is merged only when ``cache.isil`` equals the ISIL of the
target after the preset (``institution.isil`` of the preset, else the
installation's ``institution_isil``), ignoring case and white space; without an
ISIL on either side it is skipped. This guards against merging by mistake; it is
not a security boundary (the rows carry no ISIL, a copied file is not checked).

A wrong checksum, a missing or invalid file skips only that cache part; the rest
of the preset applies. The merge cannot be undone; ``revert`` leaves the cache
as it is. Qt-free at import; the database code is imported when used.
"""

from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR = "files/cache"
MERGE_NEWER_WINS = "newer_wins"

#: Bounds against decompression bombs, per file.
MAX_UNPACKED_BYTES = 300 * 1024 * 1024
MAX_ROWS_PER_FILE = 2_000_000
#: One JSON line (the raw cache stores answers up to 1 MB, escaped larger).
MAX_LINE_BYTES = 16 * 1024 * 1024

PART_DK = "dk"
PART_MAPPINGS = "mappings"
RAW_PREFIX = "raw:"
#: What GUI and CLI offer when the cache is wanted.
DEFAULT_PARTS: Tuple[str, ...] = (PART_DK, PART_MAPPINGS)

_PART_TABLES = {PART_DK: "catalog_dk_cache", PART_MAPPINGS: "search_mappings"}
_PART_IDS = {PART_DK: "dk_cache", PART_MAPPINGS: "mappings"}
_RAW_TABLE = "search_response_cache"
_HEADER_KEYS = {"table", "columns", "rows", "source_filter"}
_BLOCK_KEYS = {"isil", "created", "tables"}
_TABLE_KEYS = {"id", "table", "source", "sha256", "rows", "merge", "source_filter"}
_ENTRY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

STATUS_MERGE = "merge"
STATUS_SKIPPED = "skipped"
STATUS_MERGED = "merged"
STATUS_UNCHANGED = "unchanged"
STATUS_ERROR = "error"


class CacheFileError(ValueError):
    """A cache file of a package that may not be merged - Claude Generated"""


class CacheDbUnavailable(RuntimeError):
    """The target cache database cannot be used for this - Claude Generated"""


def normalize_isil(value: Any) -> str:
    """ISIL for comparison: no white space, case folded - Claude Generated"""
    return "".join(str(value or "").split()).casefold()


def parse_parts(value: Any) -> List[str]:
    """Cache parts from ``"dk,mappings,raw:lobid"`` or a list; ``ValueError``
    for an unknown part or ``raw:kvk`` - Claude Generated"""
    from src.core._ukm_cache_merge import NEVER_EXPORTED_RAW_SOURCES

    items = value.split(",") if isinstance(value, str) else list(value or [])
    out: List[str] = []
    for item in items:
        part = str(item or "").strip()
        if not part:
            continue
        if part.startswith(RAW_PREFIX):
            source = part[len(RAW_PREFIX):].strip()
            if not source:
                raise ValueError("raw: ohne Quelle (Form: raw:<quelle>)")
            if source.lower() in NEVER_EXPORTED_RAW_SOURCES:
                raise ValueError(f"Rohantworten der Quelle '{source}' werden nie exportiert")
            part = RAW_PREFIX + source
        elif part not in _PART_TABLES:
            raise ValueError(f"Unbekannter Cache-Teil '{part}' (erlaubt: dk, mappings, raw:<quelle>)")
        if part not in out:
            out.append(part)
    return out


def _part_target(part: str) -> Tuple[str, str, str, str]:
    """``(entry id, table, source filter, file name)`` of a part."""
    if part.startswith(RAW_PREFIX):
        source = part[len(RAW_PREFIX):]
        safe = _UNSAFE.sub("_", source)
        return f"raw_{safe}", _RAW_TABLE, source, f"{_RAW_TABLE}_{safe}.jsonl.gz"
    table = _PART_TABLES[part]
    return _PART_IDS[part], table, "", f"{table}.jsonl.gz"


def size_text(n: int) -> str:
    """Bytes as B/KB/MB with a decimal comma - Claude Generated"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB".replace(".", ",")
    return f"{n / (1024 * 1024):.1f} MB".replace(".", ",")


# ── the target database ───────────────────────────────────────────────────


def _db_identity(cfg: Any) -> Tuple[Any, ...]:
    kind = str(getattr(cfg, "db_type", "") or "sqlite").lower()
    if kind in ("sqlite", "sqlite3"):
        path = str(getattr(cfg, "sqlite_path", "") or "")
        return ("sqlite", os.path.realpath(os.path.expanduser(path)) if path else "")
    return ("server", str(getattr(cfg, "host", "") or "").strip().lower(),
            int(getattr(cfg, "port", 0) or 0), str(getattr(cfg, "database", "") or ""))


def _describe(cfg: Any) -> str:
    ident = _db_identity(cfg)
    if ident[0] == "sqlite":
        return f"SQLite {ident[1]}"
    return f"{str(getattr(cfg, 'db_type', '')).upper()} {ident[1]}:{ident[2]}/{ident[3]}"


def _running_manager() -> Any:
    """The knowledge manager of this process, ``None`` before it exists."""
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

    if UnifiedKnowledgeManager._instance is not None and UnifiedKnowledgeManager._initialized:
        return UnifiedKnowledgeManager._instance
    return None


def _check_running(ukm: Any, database_config: Any) -> None:
    """Raise :class:`CacheDbUnavailable` unless the running manager writes to
    the database the configuration names."""
    db = ukm.db_manager
    if getattr(db, "_fallback_triggered", False):
        raise CacheDbUnavailable(
            f"{str(getattr(database_config, 'db_type', '')).upper()} nicht nutzbar, die "
            "Anwendung läuft auf dem SQLite-Rückfall")
    if _db_identity(db.config) != _db_identity(database_config):
        raise CacheDbUnavailable(
            f"die laufende Anwendung nutzt {_describe(db.config)}, die Einstellung nennt "
            f"{_describe(database_config)} (nach einem Wechsel ohne Neustart); nach einem "
            "Neustart erneut anwenden")


@contextlib.contextmanager
def reading_db(database_config: Any, *, strict: bool = True,
               notes: Optional[List[str]] = None) -> Iterator[Any]:
    """A ``DatabaseManager`` to read the cache, or ``None`` (SQLite file absent).

    Uses the running knowledge manager when there is one (``strict``: only when
    it is on the configured database, else :class:`CacheDbUnavailable`);
    otherwise a temporary ``DatabaseManager`` of the configuration, closed in
    ``finally``. A missing SQLite file is not created. A fallback to SQLite is
    :class:`CacheDbUnavailable` with ``strict``, a line in ``notes`` without.
    - Claude Generated
    """
    ukm = _running_manager()
    if ukm is not None:
        if strict:
            _check_running(ukm, database_config)
        elif getattr(ukm.db_manager, "_fallback_triggered", False) and notes is not None:
            notes.append("Cache aus dem SQLite-Rückfall der laufenden Anwendung gelesen")
        yield ukm.db_manager
        return
    ident = _db_identity(database_config)
    if ident[0] == "sqlite" and not (ident[1] and os.path.isfile(ident[1])):
        yield None
        return
    import uuid

    from src.core.database_manager import DatabaseManager

    db = DatabaseManager(copy.deepcopy(database_config), f"preset_cache_{uuid.uuid4().hex}")
    try:
        db.get_connection()
        if getattr(db, "_fallback_triggered", False):
            message = (f"{str(getattr(database_config, 'db_type', '')).upper()} nicht nutzbar "
                       "(Rückfall auf SQLite)")
            if strict:
                raise CacheDbUnavailable(message)
            if notes is not None:
                notes.append(f"{message}; Cache aus der SQLite-Datei gelesen")
        yield db
    finally:
        db.close_connection()


def writing_manager(database_config: Any) -> Any:
    """The knowledge manager to merge with: the running one when it is on the
    configured database, else a new one (it becomes the process's manager).
    :class:`CacheDbUnavailable` otherwise - Claude Generated"""
    ukm = _running_manager()
    if ukm is not None:
        _check_running(ukm, database_config)
        return ukm
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

    ukm = UnifiedKnowledgeManager(database_config=copy.deepcopy(database_config))
    _check_running(ukm, database_config)
    return ukm


class _NoRows:
    """Stands in for an absent SQLite file when counting: nothing is local."""

    @staticmethod
    def fetch_one(_sql, _params=None):
        return None


# ── files ─────────────────────────────────────────────────────────────────


def _gzip_member(fileobj):
    """A gzip member without file name and time stamp (same rows, same bytes)."""
    return gzip.GzipFile(filename="", mode="wb", fileobj=fileobj, mtime=0)


def write_part_file(path: Path, table: str, columns: Sequence[str],
                    rows: Iterator[Dict[str, Any]], *, source_filter: str = "",
                    max_unpacked: Optional[int] = None,
                    max_rows: Optional[int] = None) -> Tuple[int, int]:
    """Write one cache file; ``(rows, unpacked bytes)``.

    The rows go to a temporary gzip member first, so the header can carry the
    count; then header member + rows member. ``ValueError`` past a bound (the
    reader would refuse the file). - Claude Generated
    """
    max_unpacked = MAX_UNPACKED_BYTES if max_unpacked is None else max_unpacked
    max_rows = MAX_ROWS_PER_FILE if max_rows is None else max_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, body_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".body")
    count = 0
    unpacked = 0
    try:
        with os.fdopen(fd, "wb") as raw, _gzip_member(raw) as gz:
            for row in rows:
                line = (json.dumps([row[c] for c in columns], ensure_ascii=False,
                                   separators=(",", ":")) + "\n").encode("utf-8")
                count += 1
                unpacked += len(line)
                if count > max_rows or unpacked > max_unpacked or len(line) > MAX_LINE_BYTES:
                    raise ValueError(f"{table}: zu groß für ein Paket (höchstens {max_rows} "
                                     f"Einträge, {size_text(max_unpacked)} entpackt, eine Zeile "
                                     f"höchstens {size_text(MAX_LINE_BYTES)})")
                gz.write(line)
        header: Dict[str, Any] = {"table": table, "columns": list(columns), "rows": count}
        if source_filter:
            header["source_filter"] = source_filter
        head = (json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with open(path, "wb") as out:
            with _gzip_member(out) as gz:
                gz.write(head)
            with open(body_name, "rb") as body:
                shutil.copyfileobj(body, out)
        return count, unpacked + len(head)
    finally:
        try:
            os.unlink(body_name)
        except OSError:
            pass


def _readline(gz, state: Dict[str, int], max_unpacked: int) -> bytes:
    line = gz.readline(MAX_LINE_BYTES + 1)
    if len(line) > MAX_LINE_BYTES:
        raise CacheFileError(f"Zeile länger als {size_text(MAX_LINE_BYTES)}")
    state["bytes"] += len(line)
    if state["bytes"] > max_unpacked:
        raise CacheFileError(f"entpackt größer als {size_text(max_unpacked)}")
    return line


def iter_part_file(path: Path, table: str, *, source_filter: str = "",
                   max_unpacked: Optional[int] = None,
                   max_rows: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream the checked rows of a cache file; :class:`CacheFileError` on the
    first problem (header, column, type, rule, row count, bounds, gzip).

    The header must name ``table`` (one of the three), its columns are checked
    against the schema (``_ukm_cache_merge.check_columns``), every row with
    ``check_row``; a raw file must carry only rows of ``source_filter``; the
    number of rows must equal the header. Reads line by line within
    :data:`MAX_LINE_BYTES` and ``max_unpacked`` in total (default
    :data:`MAX_UNPACKED_BYTES`, :data:`MAX_ROWS_PER_FILE`). - Claude Generated
    """
    from src.core._ukm_cache_merge import check_columns, check_row, table_spec

    max_unpacked = MAX_UNPACKED_BYTES if max_unpacked is None else max_unpacked
    max_rows = MAX_ROWS_PER_FILE if max_rows is None else max_rows

    try:
        spec = table_spec(table)
    except ValueError as exc:
        raise CacheFileError(str(exc)) from exc
    state = {"bytes": 0}
    try:
        with gzip.open(path, "rb") as gz:
            head = _readline(gz, state, max_unpacked)
            try:
                header = json.loads(head.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise CacheFileError("Kopfzeile ist kein JSON") from exc
            if not isinstance(header, dict):
                raise CacheFileError("Kopfzeile ist kein JSON-Objekt")
            unknown = [k for k in header if k not in _HEADER_KEYS]
            if unknown:
                raise CacheFileError(f"unbekannter Schlüssel in der Kopfzeile: {', '.join(map(str, unknown))}")
            if header.get("table") != spec.name:
                raise CacheFileError(f"Kopfzeile nennt die Tabelle '{header.get('table')}', "
                                     f"erwartet '{spec.name}'")
            try:
                columns = check_columns(spec, header.get("columns"))
            except ValueError as exc:
                raise CacheFileError(str(exc)) from exc
            declared = header.get("rows")
            if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
                raise CacheFileError("'rows' in der Kopfzeile fehlt oder ist ungültig")
            if declared > max_rows:
                raise CacheFileError(f"mehr als {max_rows} Einträge")
            if spec.name == _RAW_TABLE and str(header.get("source_filter") or "") != source_filter:
                raise CacheFileError(f"Kopfzeile nennt die Quelle '{header.get('source_filter')}', "
                                     f"erwartet '{source_filter}'")
            n = 0
            while True:
                line = _readline(gz, state, max_unpacked)
                if not line:
                    break
                n += 1
                if n > declared:
                    raise CacheFileError(f"mehr Zeilen als die Kopfzeile nennt ({declared})")
                try:
                    values = json.loads(line.decode("utf-8"))
                    row = check_row(spec, columns, values)
                except (UnicodeDecodeError, ValueError) as exc:
                    raise CacheFileError(f"Zeile {n + 1}: {exc}") from exc
                if spec.name == _RAW_TABLE and row.get("source") != source_filter:
                    raise CacheFileError(f"Zeile {n + 1}: Quelle '{row.get('source')}' statt "
                                         f"'{source_filter}'")
                yield row
            if n != declared:
                raise CacheFileError(f"{n} Zeilen, die Kopfzeile nennt {declared}")
    except (OSError, EOFError, zlib.error) as exc:
        raise CacheFileError(f"keine lesbare gzip-Datei ({exc})") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── export ────────────────────────────────────────────────────────────────


@dataclass
class CacheOffer:
    """One cache part the export dialog can offer - Claude Generated"""

    part: str
    label: str
    rows: int
    size: int
    default: bool


def cache_offers(database_config: Any) -> List[CacheOffer]:
    """The parts with rows in the cache in use, for the export dialog.

    ``dk`` and ``mappings`` on by default, raw answers off. Read only; a
    missing SQLite file is no offer. - Claude Generated
    """
    from src.core._ukm_cache_merge import cache_inventory

    with reading_db(database_config, strict=False) as db:
        if db is None:
            return []
        inv = cache_inventory(db)
    offers: List[CacheOffer] = []
    rows, size = inv["dk"]
    if rows:
        offers.append(CacheOffer(PART_DK, "DK-Cache (Libero/finc)", rows, size, True))
    rows, size = inv["mappings"]
    if rows:
        offers.append(CacheOffer(PART_MAPPINGS, "GND-Zuordnungen", rows, size, True))
    for source, rows, size in inv["raw"]:
        if rows:
            offers.append(CacheOffer(RAW_PREFIX + source, f"Rohantworten {source}", rows, size, False))
    return offers


def export_parts(database_config: Any, stage: Path, parts: Sequence[str], isil: str,
                 warnings: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Write the cache files into ``stage``; ``(cache block or {}, infos)``.

    ``ValueError`` for an empty ISIL or a part too large for a package. A part
    without rows is left out with a warning. - Claude Generated
    """
    from src.core._ukm_cache_merge import table_spec

    if not str(isil or "").strip():
        raise ValueError("Ohne ISIL (Einstellungen ▸ Datenquellen ▸ Einrichtung) wird kein "
                         "Cache exportiert: beim Anwenden würde er übersprungen")
    tables: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    notes: List[str] = []
    with reading_db(database_config, strict=False, notes=notes) as db:
        for part in parts:
            entry_id, table, source, name = _part_target(part)
            spec = table_spec(table)
            rel = f"{CACHE_DIR}/{name}"
            skipped: Dict[str, int] = {}
            if db is None:
                count, unpacked = 0, 0
            else:
                from src.core._ukm_cache_merge import iter_export_rows

                rows = iter_export_rows(db, table, source=source or None, skipped=skipped)
                count, unpacked = write_part_file(stage / rel, table, spec.export_columns, rows,
                                                  source_filter=source)
            for reason, n in sorted(skipped.items()):
                warnings.append(f"Cache {entry_id}: {n} Einträge nicht exportiert ({reason})")
            if not count:
                (stage / rel).unlink(missing_ok=True)
                warnings.append(f"Cache {entry_id}: keine Einträge, nicht exportiert")
                continue
            digest = sha256_file(stage / rel)
            item: Dict[str, Any] = {"id": entry_id, "table": table, "source": rel,
                                    "sha256": digest, "rows": count, "merge": MERGE_NEWER_WINS}
            if source:
                item["source_filter"] = source
            tables.append(item)
            infos.append(dict(item, size=(stage / rel).stat().st_size, unpacked=unpacked))
    warnings.extend(notes)
    if not tables:
        return {}, []
    warnings.append(f"Cache: enthält Daten aus dem Katalog und den Suchen der Einrichtung "
                    f"(ISIL {isil.strip()}); das Paket nur intern verteilen")
    block = {"isil": isil.strip(),
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "tables": tables}
    return block, infos


# ── plan ──────────────────────────────────────────────────────────────────


@dataclass
class CacheAction:
    """One cache part of a preset as it would be merged - Claude Generated"""

    id: str
    table: str
    source: str = ""
    sha256: str = ""
    rows: int = 0
    source_filter: str = ""
    status: str = STATUS_MERGE
    reason: str = ""
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    local_failures: int = 0

    @property
    def writes(self) -> bool:
        return self.status == STATUS_MERGE and (self.new + self.updated) > 0

    def skip(self, reason: str) -> None:
        self.status, self.reason = STATUS_SKIPPED, reason
        self.new = self.updated = self.unchanged = self.local_failures = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "table": self.table, "source": self.source,
                "sha256": self.sha256, "rows": self.rows, "source_filter": self.source_filter,
                "status": self.status, "reason": self.reason, "new": self.new,
                "updated": self.updated, "unchanged": self.unchanged,
                "local_failures": self.local_failures}

    def label(self) -> str:
        where = self.table + (f", Quelle {self.source_filter}" if self.source_filter else "")
        return f"{self.id} ({where})"


@dataclass
class CachePlan:
    """The cache block of a preset against the target - Claude Generated"""

    isil: str = ""
    created: str = ""
    target_isil: str = ""
    actions: List[CacheAction] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_writes(self) -> bool:
        return any(a.writes for a in self.actions)

    def as_dict(self) -> Dict[str, Any]:
        return {"isil": self.isil, "created": self.created, "target_isil": self.target_isil,
                "tables": [a.as_dict() for a in self.actions], "warnings": list(self.warnings)}

    def lines(self) -> List[str]:
        head = (f"Cache (Paket-ISIL {self.isil or '(fehlt)'}, Ziel-ISIL "
                f"{self.target_isil or '(fehlt)'}"
                + (f", erstellt {self.created}" if self.created else "") + "):")
        out = [head]
        for a in self.actions:
            if a.status == STATUS_SKIPPED:
                out.append(f"  {a.label()}: übersprungen: {a.reason}")
                continue
            line = (f"  {a.label()}: neu: {a.new}, aktualisiert: {a.updated}, "
                    f"unverändert (lokal neuer oder gleich): {a.unchanged}")
            if a.local_failures:
                line += (f"; davon {a.local_failures} lokal neuer, aber ohne Treffer "
                         "(Fehler oder keine Ergebnisse), sie bleiben")
            out.append(line)
        if not self.actions:
            out.append("  (keine Tabellen)")
        return out


def _entry_actions(block: Dict[str, Any], base_dir: Path, plan: CachePlan) -> None:
    from src.core._ukm_cache_merge import CACHE_TABLES, NEVER_EXPORTED_RAW_SOURCES

    entries = block.get("tables")
    if not isinstance(entries, list):
        plan.warnings.append("cache: 'tables' ist keine Liste; Cache übergangen")
        return
    seen: set = set()
    base = base_dir.resolve()
    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            plan.warnings.append(f"cache.tables[{n}]: ein Objekt erwartet; übergangen")
            continue
        eid = str(entry.get("id") or "").strip()
        action = CacheAction(eid or f"tables[{n}]", str(entry.get("table") or "").strip())
        plan.actions.append(action)
        for key in entry:
            if isinstance(key, str) and not key.startswith("_") and key not in _TABLE_KEYS:
                plan.warnings.append(f"Cache {action.id}: unbekannter Schlüssel '{key}' wird übergangen")
        if not _ENTRY_ID.match(eid):
            action.skip("'id' fehlt oder ist ungültig")
            continue
        if action.table not in CACHE_TABLES:
            action.skip(f"Tabelle '{action.table}' ist nicht erlaubt "
                        f"(erlaubt: {', '.join(CACHE_TABLES)})")
            continue
        action.source_filter = str(entry.get("source_filter") or "").strip()
        if action.table == _RAW_TABLE:
            if not action.source_filter:
                action.skip("Rohantworten ohne 'source_filter'")
                continue
            if action.source_filter.lower() in NEVER_EXPORTED_RAW_SOURCES:
                action.skip(f"Rohantworten der Quelle '{action.source_filter}' werden nie übernommen")
                continue
        elif action.source_filter:
            action.skip("'source_filter' nur bei search_response_cache")
            continue
        marker = (action.table, action.source_filter)
        if eid in seen or marker in seen:
            action.skip("doppelt im Preset")
            continue
        seen.update({eid, marker})
        if str(entry.get("merge") or "") != MERGE_NEWER_WINS:
            action.skip(f"Zusammenführung '{entry.get('merge')}' unbekannt (erwartet {MERGE_NEWER_WINS})")
            continue
        rows = entry.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            action.skip("'rows' fehlt oder ist ungültig")
            continue
        action.rows = rows
        raw_source = str(entry.get("source") or "").strip()
        rel = Path(raw_source)
        if not raw_source or rel.is_absolute() or raw_source.startswith("~") or rel.drive:
            action.skip("'source' muss ein relativer Pfad im Paket sein")
            continue
        path = (base_dir / rel).resolve()
        if path != base and base not in path.parents:
            action.skip("'source' darf den Paketordner nicht verlassen")
            continue
        action.source = str(path)
        expected = str(entry.get("sha256") or "").strip().lower()
        if not _SHA256.match(expected):
            action.skip("'sha256' fehlt oder ist keine SHA-256-Prüfsumme")
            continue
        action.sha256 = expected
        if not path.is_file():
            action.skip(f"Datei fehlt ({raw_source})")
            continue
        digest = sha256_file(path)
        if digest != expected:
            action.skip(f"Prüfsumme stimmt nicht (erwartet {expected[:12]}…, Datei {digest[:12]}…)")


def plan_cache(block: Any, base_dir: Path, target_isil: str, database_config: Any) -> CachePlan:
    """The cache part of a plan: checks and counts, writes nothing.

    Every reason not to merge skips the part (``status = skipped``) and is
    named; nothing here is an error of the whole preset. Counting reads the
    target cache (``reading_db``, strict). - Claude Generated
    """
    from src.core._ukm_cache_merge import merge_rows

    plan = CachePlan(target_isil=str(target_isil or "").strip())
    if not isinstance(block, dict):
        plan.warnings.append("cache: ein Objekt erwartet; Cache übergangen")
        return plan
    for key in block:
        if isinstance(key, str) and not key.startswith("_") and key not in _BLOCK_KEYS:
            plan.warnings.append(f"cache: unbekannter Schlüssel '{key}' wird übergangen")
    plan.isil = str(block.get("isil") or "").strip()
    plan.created = str(block.get("created") or "").strip()
    _entry_actions(block, base_dir, plan)
    pending = [a for a in plan.actions if a.status == STATUS_MERGE]
    isil_reason = ""
    if not normalize_isil(plan.isil):
        isil_reason = "ISIL fehlt im Paket"
    elif not normalize_isil(plan.target_isil):
        isil_reason = "ISIL fehlt im Ziel (Einstellungen ▸ Datenquellen ▸ Einrichtung oder institution.isil im Preset)"
    elif normalize_isil(plan.isil) != normalize_isil(plan.target_isil):
        isil_reason = f"ISIL passt nicht (Paket {plan.isil}, Ziel {plan.target_isil})"
    if isil_reason:
        for a in pending:
            a.skip(isil_reason)
    elif pending:
        try:
            with reading_db(database_config, strict=True) as db:
                for a in pending:
                    try:
                        counts = merge_rows(db if db is not None else _NoRows(), a.table,
                                            iter_part_file(Path(a.source), a.table,
                                                           source_filter=a.source_filter),
                                            write=False)
                    except CacheFileError as exc:
                        a.skip(f"Datei abgelehnt: {exc}")
                        continue
                    except ValueError as exc:  # a key twice
                        a.skip(f"Datei abgelehnt: {exc}")
                        continue
                    except RuntimeError as exc:  # QtSql: table or database not readable
                        a.skip(f"Cache-Datenbank nicht lesbar ({exc})")
                        continue
                    if counts["new"] + counts["unchanged"] + counts["updated"] != a.rows:
                        a.skip(f"'rows' im Preset ({a.rows}) passt nicht zur Datei")
                        continue
                    a.new, a.updated = counts["new"], counts["updated"]
                    a.unchanged, a.local_failures = counts["unchanged"], counts["local_failures"]
        except CacheDbUnavailable as exc:
            for a in pending:
                if a.status == STATUS_MERGE:
                    a.skip(f"Cache-Datenbank: {exc}")
        except Exception as exc:  # database not openable, a defect: only the cache is skipped
            from .error_visibility import log_caught

            log_caught(logger, exc, "Cache-Datenbank für die Preset-Vorschau")
            for a in pending:
                if a.status == STATUS_MERGE:
                    a.skip(f"Cache-Datenbank nicht nutzbar ({exc})")
    reasons = list(dict.fromkeys(a.reason for a in plan.actions if a.status == STATUS_SKIPPED))
    for reason in reasons:
        plan.warnings.append(f"Cache nicht eingespielt: {reason}")
    return plan


# ── apply ─────────────────────────────────────────────────────────────────


def merge_planned(cache_plan: CachePlan, database_config: Any) -> List[Dict[str, Any]]:
    """Merge the parts the plan counted; one ledger record per part.

    Never raises: a failure of one part is its record (``status: error``). The
    file is hashed again first; a changed file is not merged. Writes through
    ``UnifiedKnowledgeManager.merge_cache_rows`` (``writing_manager``).
    - Claude Generated
    """
    from .error_visibility import log_caught

    records: List[Dict[str, Any]] = []
    ukm = None
    unavailable = ""
    for a in cache_plan.actions:
        record = {"id": a.id, "table": a.table, "source_filter": a.source_filter,
                  "status": a.status if a.status == STATUS_SKIPPED else STATUS_UNCHANGED,
                  "reason": a.reason, "new": 0, "updated": 0, "unchanged": a.unchanged}
        records.append(record)
        if not a.writes:
            continue
        try:
            if ukm is None and not unavailable:
                ukm = writing_manager(database_config)
        except CacheDbUnavailable as exc:
            unavailable = f"Cache-Datenbank: {exc}"
        except Exception as exc:  # database not openable, a defect
            log_caught(logger, exc, "Cache-Datenbank für das Preset")
            unavailable = f"Cache-Datenbank nicht nutzbar ({exc})"
        if unavailable:
            record.update(status=STATUS_SKIPPED, reason=unavailable)
            continue
        try:
            if sha256_file(Path(a.source)) != a.sha256:
                record.update(status=STATUS_SKIPPED,
                              reason="Datei seit der Vorschau geändert (Prüfsumme)")
                continue
            counts = ukm.merge_cache_rows(a.table, iter_part_file(Path(a.source), a.table,
                                                                  source_filter=a.source_filter))
            record.update(status=STATUS_MERGED, new=counts["new"], updated=counts["updated"],
                          unchanged=counts["unchanged"])
        except Exception as exc:  # file, database or a defect: the preset still applies
            log_caught(logger, exc, f"Cache {a.id} einspielen")
            record.update(status=STATUS_ERROR, reason=str(exc) or type(exc).__name__)
    return records


def record_lines(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Ledger records of the cache as lines for the CLI and the GUI - Claude Generated"""
    out: List[str] = []
    for r in records or []:
        name = r.get("id") or r.get("table")
        status = r.get("status")
        if status in (STATUS_MERGED, STATUS_UNCHANGED):
            out.append(f"Cache {name}: neu: {r.get('new', 0)}, aktualisiert: {r.get('updated', 0)}, "
                       f"unverändert: {r.get('unchanged', 0)}")
        elif status == STATUS_ERROR:
            out.append(f"Cache {name}: Fehler: {r.get('reason')}")
        else:
            out.append(f"Cache {name}: übersprungen: {r.get('reason')}")
    return out
