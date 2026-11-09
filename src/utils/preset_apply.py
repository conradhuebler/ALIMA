"""Institution presets: find, read, plan, apply, revert - Claude Generated.

An institution package is a folder holding ``alima_presets.json`` and the files
it ships (for example the local subject database of ``geo_sachdb`` and the
shelf vocabulary of ``local_scheme``). This module finds the preset, reads it,
computes what it would change (:func:`build_plan`, a preview with ``before`` and
``after`` per change), applies that plan (:func:`apply_plan`) and undoes the
last application (:func:`revert_last`) from the ledger
``AlimaConfig.applied_presets``. The export side is ``preset_export.py``.

Rules:

* **Advisory.** Only values the preset names are set; every other value of the
  user stays as it is. A second application of the same preset is an empty
  plan and saves nothing.
* **No secrets.** Keys named ``api_key``/``token`` (also ``*_api_key``,
  ``*_token``) are dropped when the file is read, with one log warning; SECRET
  fields of a plugin are never set. ``secrets`` only declares what each user
  still has to enter.
* **Files fail closed.** A file is installed only when its ``sha256`` matches
  (and, for ``kind: geo_sachdb_db``, the read-only schema check of the plugin
  passes). A plan with an error is not applied at all. Installation writes a
  temporary file and replaces the target; a replaced file stays as
  ``<name>.bak`` (one generation).

The preset is found in this order: the environment variable ``ALIMA_PRESET``
(file, or folder with ``alima_presets.json``), ``<config_dir>/alima_presets.json``
(``config_dir`` = the directory of ``ConfigManager.config_file``), the project
directory. Relative ``files[].source`` paths are relative to the folder of the
preset file; ``files[].target`` is relative to ``<config_dir>/data/``.

Qt-free; no network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PRESET_FILENAME = "alima_presets.json"
ENV_VAR = "ALIMA_PRESET"
FORMAT_VERSION = 2

SOURCE_ENV = "env"
SOURCE_CONFIG_DIR = "config_dir"
SOURCE_PROJECT = "project"
SOURCE_GIVEN = "given"
SOURCE_LABELS = {
    SOURCE_ENV: f"Umgebungsvariable {ENV_VAR}",
    SOURCE_CONFIG_DIR: "Konfigurationsverzeichnis",
    SOURCE_PROJECT: "Projektverzeichnis",
    SOURCE_GIVEN: "ausgewählt",
}

#: ``SystemConfig`` fields a preset can set.
SYSTEM_KEYS = ("institution_isil", "institution_eln", "default_workflow")

KIND_GEO_DB = "geo_sachdb_db"
KIND_VOCABULARY = "vocabulary"
KIND_FILE = "file"
FILE_KINDS = (KIND_GEO_DB, KIND_VOCABULARY, KIND_FILE)

#: Placeholder in a plugin setting: ``@file:<id>`` → target path of that file.
FILE_REF_PREFIX = "@file:"

#: The workflow sentinel of the classic pipeline (``CLASSIC_STEM`` in
#: ``src/ui/settings/workflow_list.py``).
CLASSIC_WORKFLOW = "__classic__"

FILE_NEW = "new"
FILE_UNCHANGED = "unchanged"
FILE_REPLACED = "replaced"
_FILE_STATUS_LABELS = {FILE_NEW: "neu", FILE_UNCHANGED: "unverändert", FILE_REPLACED: "ersetzt"}

_SECRET_KEY = re.compile(r"(^|_)(api_key|token)$", re.IGNORECASE)
_FILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {
    "format", "preset", "institution_name", "llm", "catalog", "database",
    "institution", "workflow", "plugins", "files", "secrets",
}
_PLUGIN_KEYS = {"instance", "type", "category", "state", "label", "settings"}
_FILE_KEYS = {"id", "source", "sha256", "target", "kind"}

#: (path, mtime) of preset files whose dropped keys were already logged.
_WARNED: set = set()


class PresetError(Exception):
    """A preset cannot be read, or a plan cannot be applied or reverted."""


# ── locations ─────────────────────────────────────────────────────────────


def config_dir_of(config_manager: Any = None) -> Path:
    """Directory of the running configuration file - Claude Generated"""
    if config_manager is None:
        from .config_manager import ConfigManager

        config_manager = ConfigManager()
    return Path(config_manager.config_file).parent


def data_dir_of(config_manager: Any = None) -> Path:
    """``<config_dir>/data``: where preset files are installed - Claude Generated"""
    return config_dir_of(config_manager) / "data"


def resolve_preset_path(path: Any) -> Optional[Path]:
    """A preset file, or a folder holding ``alima_presets.json`` → the file.

    ``None`` when there is no such file. - Claude Generated
    """
    if path is None or str(path).strip() == "":
        return None
    p = Path(str(path)).expanduser()
    if p.is_dir():
        candidate = p / PRESET_FILENAME
        return candidate if candidate.is_file() else None
    return p if p.is_file() else None


def find_preset(config_dir: Any = None) -> Tuple[Optional[Path], str]:
    """``(path, source)`` of the preset in use, ``(None, "")`` without one.

    Order: ``ALIMA_PRESET``, ``<config_dir>/alima_presets.json``, the project
    directory; the first one found is used and named in the log. ``source`` is
    one of ``SOURCE_ENV``, ``SOURCE_CONFIG_DIR``, ``SOURCE_PROJECT``.
    - Claude Generated
    """
    env_value = os.environ.get(ENV_VAR, "").strip()
    if env_value:
        found = resolve_preset_path(env_value)
        if found is not None:
            return _found(found, SOURCE_ENV)
        logger.warning(
            "%s=%s: keine Datei %s gefunden; gesucht wird weiter im "
            "Konfigurationsverzeichnis und im Projektverzeichnis",
            ENV_VAR, env_value, PRESET_FILENAME,
        )
    try:
        cdir = Path(config_dir) if config_dir is not None else config_dir_of()
    except Exception as exc:  # ConfigManager not constructible here
        from .error_visibility import log_caught

        log_caught(logger, exc, "Konfigurationsverzeichnis für das Preset")
        cdir = None
    if cdir is not None and (cdir / PRESET_FILENAME).is_file():
        return _found(cdir / PRESET_FILENAME, SOURCE_CONFIG_DIR)
    try:
        from .path_utils import get_project_root

        candidate = get_project_root() / PRESET_FILENAME
    except Exception as exc:  # project root not determinable
        from .error_visibility import log_caught

        log_caught(logger, exc, "Projektverzeichnis für das Preset")
        candidate = None
    if candidate is not None and candidate.is_file():
        return _found(candidate, SOURCE_PROJECT)
    return None, ""


def _found(path: Path, source: str) -> Tuple[Path, str]:
    logger.info("Preset: %s (%s)", path, SOURCE_LABELS[source])
    return path, source


# ── reading ───────────────────────────────────────────────────────────────


@dataclass
class Preset:
    """One preset file as read: secret-shaped keys already removed - Claude Generated"""

    path: Path
    data: Dict[str, Any]
    source: str = SOURCE_GIVEN
    ignored_keys: List[str] = field(default_factory=list)

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def meta(self) -> Dict[str, Any]:
        meta = self.data.get("preset")
        return meta if isinstance(meta, dict) else {}

    @property
    def id(self) -> str:
        value = str(self.meta.get("id") or "").strip()
        if value:
            return value
        name = str(self.data.get("institution_name") or "").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
        return slug or self.path.parent.name or "preset"

    @property
    def version(self) -> str:
        return str(self.meta.get("version") or "").strip()

    @property
    def label(self) -> str:
        return (str(self.meta.get("label") or "").strip()
                or str(self.data.get("institution_name") or "").strip()
                or self.id)


def _strip_secret_keys(value: Any, where: str, found: List[str]) -> Any:
    """Copy of ``value`` without keys named like an API key or token."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            if isinstance(key, str) and _SECRET_KEY.search(key):
                if item not in (None, "", [], {}):
                    found.append(path)
                continue
            out[key] = _strip_secret_keys(item, path, found)
        return out
    if isinstance(value, list):
        return [_strip_secret_keys(item, f"{where}[{n}]", found)
                for n, item in enumerate(value, 1)]
    return value


def load_preset(path: Any, source: str = SOURCE_GIVEN) -> Preset:
    """Read a preset file or folder. Raises :class:`PresetError`.

    Keys named ``api_key``/``token`` (and ``*_api_key``/``*_token``) are dropped;
    when one of them carried a value, that is logged once as a warning.
    - Claude Generated
    """
    resolved = resolve_preset_path(path)
    if resolved is not None:
        resolved = resolved.resolve()
    if resolved is None:
        raise PresetError(
            f"Kein Preset gefunden: {path} (erwartet: die Datei {PRESET_FILENAME} "
            "oder ein Ordner, der sie enthält)"
        )
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PresetError(f"{resolved}: nicht lesbar ({exc})") from exc
    if not isinstance(data, dict):
        raise PresetError(f"{resolved}: kein JSON-Objekt")
    ignored: List[str] = []
    data = _strip_secret_keys(data, "", ignored)
    if ignored:
        try:
            marker = (str(resolved.resolve()), resolved.stat().st_mtime)
        except OSError:
            marker = (str(resolved), 0)
        if marker not in _WARNED:
            _WARNED.add(marker)
            logger.warning(
                "Im Preset gefundene Schlüssel werden nicht verwendet: %s (%s)",
                ", ".join(ignored), resolved,
            )
    return Preset(resolved, data, source, ignored)


# ── plan ──────────────────────────────────────────────────────────────────


@dataclass
class Change:
    """One value the preset sets; ``path`` addresses it in the config.

    Paths: ``["system_config", key]``, ``["instance", category, id]`` (the
    instance is created), ``["instance", category, id, "enabled"|"label"]``,
    ``["instance", category, id, "settings", key]``, ``["order", category]``
    (instance ids in list order). - Claude Generated
    """

    what: str
    path: List[Any]
    before: Any = None
    after: Any = None
    before_absent: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"what": self.what, "path": list(self.path), "before": self.before,
                "after": self.after, "before_absent": self.before_absent}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Change":
        return cls(str(data.get("what") or ""), list(data.get("path") or []),
                   data.get("before"), data.get("after"), bool(data.get("before_absent")))


@dataclass
class FileAction:
    """One file of the preset as it would be installed - Claude Generated"""

    id: str
    kind: str
    source: str
    target: str
    sha256: str
    size: int
    status: str

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "source": self.source, "target": self.target,
                "sha256": self.sha256, "size": self.size, "status": self.status}


@dataclass
class Plan:
    """What applying a preset would change - Claude Generated"""

    preset_id: str
    version: str = ""
    label: str = ""
    preset_path: str = ""
    source: str = SOURCE_GIVEN
    changes: List[Change] = field(default_factory=list)
    files: List[FileAction] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    secrets: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def is_empty(self) -> bool:
        """Nothing to write: no change and no file to install."""
        return not self.changes and all(f.status == FILE_UNCHANGED for f in self.files)

    def steps(self) -> List[Dict[str, Any]]:
        """The changes as replay steps (see :func:`replay_steps`)."""
        return [{"path": list(c.path), "value": c.after, "absent": False} for c in self.changes]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "preset_id": self.preset_id, "version": self.version, "label": self.label,
            "preset_path": self.preset_path, "source": self.source,
            "changes": [c.as_dict() for c in self.changes],
            "files": [f.as_dict() for f in self.files],
            "warnings": list(self.warnings), "errors": list(self.errors),
            "secrets": [dict(s) for s in self.secrets],
        }

    def lines(self) -> List[str]:
        """The plan as text for the CLI and the preview dialog."""
        head = f"Preset: {self.label}"
        if self.version:
            head += f" {self.version}"
        if self.preset_id and self.preset_id != self.label:
            head += f" ({self.preset_id})"
        out = [head]
        if self.preset_path:
            out.append(f"Datei: {self.preset_path} (Quelle: {SOURCE_LABELS.get(self.source, self.source)})")
        if self.errors:
            out.append(f"Fehler ({len(self.errors)}), es wird nichts angewendet:")
            out += [f"  ✗ {e}" for e in self.errors]
        if self.changes:
            out.append(f"Änderungen ({len(self.changes)}):")
            for ch in self.changes:
                before = "(nicht gesetzt)" if ch.before_absent else format_value(ch.path, ch.before)
                if _is_create(ch.path):
                    out.append(f"  + {ch.what}")
                else:
                    out.append(f"  {ch.what}: {before} → {format_value(ch.path, ch.after)}")
        if self.files:
            out.append(f"Dateien ({len(self.files)}):")
            for fa in self.files:
                extra = " (die vorherige bleibt als .bak)" if fa.status == FILE_REPLACED else ""
                out.append(f"  {_FILE_STATUS_LABELS.get(fa.status, fa.status)}: {fa.id} → "
                           f"{fa.target} ({_size(fa.size)}, sha256 {fa.sha256[:12]}…){extra}")
        if self.secrets:
            out.append("Noch einzutragen (nie im Preset):")
            for s in self.secrets:
                state = "gesetzt" if s.get("satisfied") else "fehlt"
                hint = f" ({s['hint']})" if s.get("hint") else ""
                out.append(f"  [{state}] {s['plugin']}.{s['key']}{hint}; im Plugin-Reiter "
                           f"oder als Umgebungsvariable {s['env_var']}")
        if self.warnings:
            out.append(f"Hinweise ({len(self.warnings)}):")
            out += [f"  ⚠ {w}" for w in self.warnings]
        if self.ok and self.is_empty:
            out.append("Nichts zu ändern: die Einrichtung entspricht dem Preset.")
        return out


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB".replace(".", ",")
    return f"{n / (1024 * 1024):.1f} MB".replace(".", ",")


def format_value(path: List[Any], value: Any) -> str:
    """A value as the plan shows it - Claude Generated"""
    if path and path[-1] == "enabled":
        return "an" if value else "aus"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if value is None or value == "":
        return "(leer)"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "(leer)"
    return str(value)


def _is_create(path: List[Any]) -> bool:
    return len(path) == 3 and path[0] == "instance"


def _ensure_categories() -> None:
    import src.core.search  # noqa: F401  (search providers)
    import src.utils.input_sources  # noqa: F401  (input sources)
    import src.utils.lookups  # noqa: F401  (lookups)


def _cm(config_manager: Any) -> Any:
    if config_manager is not None:
        return config_manager
    from .config_manager import ConfigManager

    return ConfigManager()


def build_plan(preset: Preset, config_manager: Any = None, *, config: Any = None) -> Plan:
    """Preview of what applying ``preset`` would change; nothing is written.

    ``config`` (default: the loaded configuration of ``config_manager``) is not
    modified. Hashes every shipped file once. - Claude Generated
    """
    cm = _cm(config_manager)
    base = config if config is not None else cm.load_config()
    sim = copy.deepcopy(base)
    plan = Plan(preset.id, preset.version, preset.label, str(preset.path), preset.source)
    if preset.ignored_keys:
        plan.warnings.append("Im Preset gefundene Schlüssel werden nicht verwendet: "
                             + ", ".join(preset.ignored_keys))
    try:
        fmt = int(preset.data.get("format") or FORMAT_VERSION)
    except (TypeError, ValueError):
        fmt = FORMAT_VERSION
    if fmt > FORMAT_VERSION:
        plan.warnings.append(f"Preset-Format {fmt}; diese Version kennt Format {FORMAT_VERSION}, "
                             "unbekannte Teile werden übergangen")
    for key in preset.data:
        if isinstance(key, str) and not key.startswith("_") and key not in _TOP_KEYS:
            plan.warnings.append(f"Unbekannter Schlüssel '{key}' wird übergangen")
    _ensure_categories()
    files = _plan_files(preset, data_dir_of(cm), plan)
    _plan_institution(preset, sim, plan)
    _plan_workflow(preset, sim, plan)
    _plan_plugins(preset, sim, plan, files)
    plan.secrets = secret_status(sim, preset.data.get("secrets"))
    return plan


def sha256_file(path: Any, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes - Claude Generated"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_file_kind(kind: str, path: Any) -> str:
    """Empty string when ``path`` is a valid file of ``kind``, else the reason.

    ``geo_sachdb_db``: opened read-only, ``geo_sachdb.schema.check_schema``.
    ``vocabulary``: YAML with an ``entries`` list (the form ``local_scheme``
    reads). ``file``: no check beyond the checksum. - Claude Generated
    """
    if kind == KIND_GEO_DB:
        from src.utils.lookups.geo_sachdb.schema import check_schema

        try:
            conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error as exc:
            return f"keine lesbare SQLite-Datei ({exc})"
        try:
            return check_schema(conn)
        except sqlite3.Error as exc:
            return f"keine lesbare SQLite-Datei ({exc})"
        finally:
            conn.close()
    if kind == KIND_VOCABULARY:
        import yaml

        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            return f"kein lesbares YAML ({exc})"
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            return "kein Vokabular in der Form von local_scheme (Liste 'entries' fehlt)"
    return ""


def _safe_target(data_dir: Path, raw: str) -> Tuple[Optional[Path], str]:
    text = str(raw or "").strip()
    rel = Path(text)
    if not text or rel.is_absolute() or text.startswith("~") or rel.drive:
        return None, f"Ziel '{text}' muss ein relativer Pfad im Datenordner sein"
    if any(part == ".." for part in rel.parts):
        return None, f"Ziel '{text}' darf den Datenordner nicht verlassen"
    return data_dir / rel, ""


def _plan_files(preset: Preset, data_dir: Path, plan: Plan) -> Dict[str, Optional[FileAction]]:
    """Check every shipped file; ``{id: FileAction or None (failed)}``."""
    out: Dict[str, Optional[FileAction]] = {}
    entries = preset.data.get("files")
    if entries is None:
        return out
    if not isinstance(entries, list):
        plan.errors.append("files: eine Liste erwartet")
        return out
    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            plan.errors.append(f"files[{n}]: ein Objekt erwartet")
            continue
        fid = str(entry.get("id") or "").strip()
        if not _FILE_ID.match(fid):
            plan.errors.append(f"files[{n}]: 'id' fehlt oder ist ungültig")
            continue
        if fid in out:
            plan.errors.append(f"Datei '{fid}': id doppelt")
            continue
        out[fid] = None
        where = f"Datei '{fid}'"
        for key in entry:
            if isinstance(key, str) and not key.startswith("_") and key not in _FILE_KEYS:
                plan.warnings.append(f"{where}: unbekannter Schlüssel '{key}' wird übergangen")
        kind = str(entry.get("kind") or KIND_FILE).strip()
        if kind not in FILE_KINDS:
            plan.errors.append(f"{where}: unbekannte Art '{kind}' (erlaubt: {', '.join(FILE_KINDS)})")
            continue
        source_raw = str(entry.get("source") or "").strip()
        if not source_raw:
            plan.errors.append(f"{where}: 'source' fehlt")
            continue
        src = Path(source_raw).expanduser()
        if not src.is_absolute():
            src = preset.base_dir / src
        target, reason = _safe_target(data_dir, entry.get("target") or Path(source_raw).name)
        if target is None:
            plan.errors.append(f"{where}: {reason}")
            continue
        expected = str(entry.get("sha256") or "").strip().lower()
        if not _SHA256.match(expected):
            plan.errors.append(f"{where}: 'sha256' fehlt oder ist keine SHA-256-Prüfsumme; "
                               "die Datei wird nicht installiert")
            continue
        if not src.is_file():
            plan.errors.append(f"{where}: Quelldatei fehlt ({src})")
            continue
        digest = sha256_file(src)
        if digest != expected:
            plan.errors.append(f"{where}: Prüfsumme stimmt nicht (erwartet {expected[:12]}…, "
                               f"Datei {digest[:12]}…); die Datei wird nicht installiert")
            continue
        reason = check_file_kind(kind, src)
        if reason:
            plan.errors.append(f"{where} ({kind}): {reason}; die Datei wird nicht installiert")
            continue
        if target.exists() and not target.is_file():
            plan.errors.append(f"{where}: Ziel {target} ist keine Datei")
            continue
        if not target.exists():
            status = FILE_NEW
        elif sha256_file(target) == digest:
            status = FILE_UNCHANGED
        else:
            status = FILE_REPLACED
        action = FileAction(fid, kind, str(src.resolve()), str(target), digest,
                            src.stat().st_size, status)
        plan.files.append(action)
        out[fid] = action
    return out


def _plan_institution(preset: Preset, sim: Any, plan: Plan) -> None:
    data = preset.data.get("institution")
    if data is None:
        return
    if not isinstance(data, dict):
        plan.warnings.append("institution: ein Objekt mit isil/eln erwartet; übergangen")
        return
    from . import institution

    for key, name, label in (("isil", institution.ISIL_KEY, "ISIL"),
                             ("eln", institution.ELN_KEY, "ELN")):
        if key in data:
            _record(sim, plan, f"Einrichtung: {label}", ["system_config", name],
                    institution.normalize(data[key]))
    for key in data:
        if isinstance(key, str) and not key.startswith("_") and key not in ("isil", "eln"):
            plan.warnings.append(f"institution: unbekannter Schlüssel '{key}' wird übergangen")


def _workflow_exists(name: str) -> bool:
    from src.core.agents.workflow_loader import find_workflow_file

    return find_workflow_file(name) is not None


def _plan_workflow(preset: Preset, sim: Any, plan: Plan) -> None:
    if "workflow" not in preset.data:
        return
    name = str(preset.data.get("workflow") or "").strip()
    if not name:
        plan.warnings.append("workflow: leer; der Standard-Workflow bleibt")
        return
    if name != CLASSIC_WORKFLOW and not _workflow_exists(name):
        plan.warnings.append(f"Workflow '{name}' nicht gefunden; der Standard-Workflow bleibt")
        return
    _record(sim, plan, "Standard-Workflow", ["system_config", "default_workflow"], name)


def _resolve_instance(sim: Any, entry: Dict[str, Any], iid: str, plan: Plan):
    """``(category, existing instance or None, type id)`` or ``None`` (skipped)."""
    from src.core.plugins import get_category, list_categories

    cat_hint = str(entry.get("category") or "").strip()
    type_hint = str(entry.get("type") or "").strip()
    matches = [p for p in sim.plugins
               if p.instance_id == iid and (not cat_hint or p.category == cat_hint)]
    if len(matches) > 1:
        plan.warnings.append(f"Instanz '{iid}': in mehreren Kategorien vorhanden; "
                             "'category' angeben. Eintrag übergangen")
        return None
    if matches:
        inst = matches[0]
        if type_hint and type_hint != inst.provider_id:
            plan.warnings.append(f"Instanz '{iid}' hat den Typ '{inst.provider_id}', nicht "
                                 f"'{type_hint}'. Eintrag übergangen")
            return None
        return inst.category, inst, inst.provider_id
    ptype = type_hint or iid
    cats = [c for c in list_categories()
            if (not cat_hint or c == cat_hint) and get_category(c).has_type(ptype)]
    if not cats:
        plan.warnings.append(f"Instanz '{iid}' gibt es nicht, und '{ptype}' ist kein "
                             "registrierter Typ. Eintrag übergangen")
        return None
    if len(cats) > 1:
        plan.warnings.append(f"Typ '{ptype}' gibt es in mehreren Kategorien ({', '.join(cats)}); "
                             "'category' angeben. Eintrag übergangen")
        return None
    return cats[0], None, ptype


def _plan_plugins(preset: Preset, sim: Any, plan: Plan,
                  files: Dict[str, Optional[FileAction]]) -> None:
    entries = preset.data.get("plugins")
    if entries is None:
        return
    if not isinstance(entries, list):
        plan.errors.append("plugins: eine Liste erwartet")
        return
    from src.core.plugins import get_category
    from src.core.plugins.schema import CHOICE, env_var_name
    from src.core.search.cache_only import (
        CACHE_ONLY_KEY, OFFER_CACHE_ONLY, OFFER_NONE, STATE_CACHE_ONLY, STATES,
        apply_state, cache_only_offer,
    )

    order: Dict[str, List[str]] = {}
    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            plan.warnings.append(f"plugins[{n}]: ein Objekt erwartet; übergangen")
            continue
        iid = str(entry.get("instance") or "").strip()
        if not iid:
            plan.warnings.append(f"plugins[{n}]: 'instance' fehlt; übergangen")
            continue
        for key in entry:
            if isinstance(key, str) and not key.startswith("_") and key not in _PLUGIN_KEYS:
                plan.warnings.append(f"Instanz '{iid}': unbekannter Schlüssel '{key}' wird übergangen")
        resolved = _resolve_instance(sim, entry, iid, plan)
        if resolved is None:
            continue
        category, inst, ptype = resolved
        try:
            meta = get_category(category).type_meta(ptype)
        except KeyError:
            meta = None
        if inst is None:
            label = str(entry.get("label") or "").strip() or (meta.label if meta else "") or iid
            _record(sim, plan, f"{label} [{iid}]: neu angelegt ({category}, Typ {ptype})",
                    ["instance", category, iid], {"provider_id": ptype, "label": label})
        elif "label" in entry:
            _record(sim, plan, f"{inst.display_label()} [{iid}]: Name",
                    ["instance", category, iid, "label"], str(entry.get("label") or ""))
        name = f"{_find(sim, category, iid).display_label()} [{iid}]"

        raw = entry.get("settings") or {}
        if not isinstance(raw, dict):
            plan.warnings.append(f"{name}: 'settings' ist kein Objekt; übergangen")
            raw = {}
        if raw and meta is None:
            plan.warnings.append(f"{name}: Typ '{ptype}' ist nicht geladen; Einstellungen nicht "
                                 "geprüft und nicht geschrieben")
            raw = {}
        fields = {f.key: f for f in (meta.config_fields if meta else [])}
        for key, value in raw.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            fld = fields.get(key)
            if fld is None:
                plan.warnings.append(f"{name}: unbekannte Einstellung '{key}', nicht geschrieben")
                continue
            if fld.secret:
                plan.warnings.append(f"{name}: '{key}' ist geheim und wird aus einem Preset nie "
                                     f"gesetzt (im Plugin-Reiter oder als {env_var_name(iid, key)})")
                continue
            if isinstance(value, str) and value.startswith(FILE_REF_PREFIX):
                fid = value[len(FILE_REF_PREFIX):].strip()
                if fid not in files:
                    plan.errors.append(f"{name}: '{key}' verweist auf {value}; diese Datei steht "
                                       "nicht unter files")
                    continue
                if files[fid] is None:
                    continue  # the file's own error is in the plan
                value = files[fid].target
            if fld.kind == CHOICE and fld.choices and str(value) not in fld.choices:
                plan.warnings.append(f"{name}: '{key}' = '{value}' ist nicht erlaubt "
                                     f"({', '.join(map(str, fld.choices))}); nicht geschrieben")
                continue
            _record(sim, plan, f"{name}: {fld.label or key}",
                    ["instance", category, iid, "settings", key], fld.coerce(value))

        if "state" in entry:
            state = str(entry.get("state") or "").strip()
            offer = cache_only_offer(meta) if meta is not None else OFFER_NONE
            if state not in STATES:
                plan.warnings.append(f"{name}: Zustand '{state}' unbekannt "
                                     f"({', '.join(STATES)}); nicht gesetzt")
            elif state == STATE_CACHE_ONLY and offer != OFFER_CACHE_ONLY:
                plan.warnings.append(f"{name}: 'Nur Cache' gibt es für den Typ '{ptype}' nicht; "
                                     "Zustand nicht gesetzt")
            else:
                current = _find(sim, category, iid)
                probe = copy.deepcopy(current)
                apply_state(probe, state, offer)
                _record(sim, plan, f"{name}: Zustand",
                        ["instance", category, iid, "enabled"], probe.enabled)
                if CACHE_ONLY_KEY in probe.settings:
                    _record(sim, plan, f"{name}: Nur Cache",
                            ["instance", category, iid, "settings", CACHE_ONLY_KEY],
                            probe.settings[CACHE_ONLY_KEY])
        ids = order.setdefault(category, [])
        if iid not in ids:
            ids.append(iid)

    for category, ids in order.items():
        current = [p.instance_id for p in sim.plugins if p.category == category]
        wanted = ids + [i for i in current if i not in ids]
        _record(sim, plan, f"Reihenfolge {category}", ["order", category], wanted)


def secret_status(config: Any, declared: Any) -> List[Dict[str, Any]]:
    """The declared per-user secrets and whether each is already there.

    A secret counts as there when its environment variable
    (``ALIMA_PLUGIN_<ID>_<KEY>``) is set or the instance stores a value.
    Nothing is written. - Claude Generated
    """
    from src.core.plugins.schema import env_var_name

    out: List[Dict[str, Any]] = []
    for item in declared if isinstance(declared, list) else []:
        if not isinstance(item, dict) or not item.get("plugin") or not item.get("key"):
            continue
        plugin, key = str(item["plugin"]), str(item["key"])
        env = env_var_name(plugin, key)
        inst = next((p for p in config.plugins if p.instance_id == plugin), None)
        stored = bool((inst.settings or {}).get(key)) if inst is not None else False
        out.append({"plugin": plugin, "key": key, "hint": str(item.get("hint") or ""),
                    "env_var": env, "satisfied": bool(os.environ.get(env)) or stored})
    return out


# ── reading and writing one addressed value ─────────────────────────────────


def _find(config: Any, category: str, iid: str):
    for inst in config.plugins:
        if inst.category == category and inst.instance_id == iid:
            return inst
    return None


def _get(config: Any, path: List[Any]) -> Tuple[bool, Any]:
    """``(present, value)`` at ``path`` - Claude Generated"""
    head = path[0] if path else ""
    if head == "system_config" and len(path) == 2:
        return True, getattr(config.system_config, path[1], None)
    if head == "order" and len(path) == 2:
        return True, [p.instance_id for p in config.plugins if p.category == path[1]]
    if head == "instance" and len(path) >= 3:
        inst = _find(config, path[1], path[2])
        if len(path) == 3:
            return inst is not None, ({"provider_id": inst.provider_id, "label": inst.label}
                                      if inst is not None else None)
        if inst is None:
            return False, None
        if len(path) == 4 and path[3] in ("enabled", "label"):
            return True, getattr(inst, path[3])
        if len(path) == 5 and path[3] == "settings":
            return path[4] in inst.settings, inst.settings.get(path[4])
    raise PresetError(f"unbekannter Pfad {path}")


def _reorder(plugins: List[Any], category: str, ids: List[str]) -> None:
    """Instances of ``category`` in the order of ``ids`` (others after, in
    their order), in the slots the category already holds in ``plugins``."""
    slots = [n for n, p in enumerate(plugins) if p.category == category]
    members = [plugins[n] for n in slots]
    rank = {iid: pos for pos, iid in enumerate(ids)}
    keyed = [((0, rank[p.instance_id]) if p.instance_id in rank else (1, n), n)
             for n, p in enumerate(members)]
    ordered = [members[n] for _key, n in sorted(keyed)]
    for slot, inst in zip(slots, ordered):
        plugins[slot] = inst


def _set(config: Any, path: List[Any], value: Any, *, absent: bool = False,
         strict: bool = True) -> None:
    """Write ``value`` at ``path`` (``absent``: remove it) - Claude Generated"""
    head = path[0] if path else ""
    if head == "system_config" and len(path) == 2:
        setattr(config.system_config, path[1], value)
        return
    if head == "order" and len(path) == 2:
        _reorder(config.plugins, path[1], list(value or []))
        return
    if head == "instance" and len(path) >= 3:
        category, iid = path[1], path[2]
        inst = _find(config, category, iid)
        if len(path) == 3:
            if absent:
                config.plugins[:] = [p for p in config.plugins
                                     if not (p.category == category and p.instance_id == iid)]
            elif inst is None:
                from .config_models import PluginInstanceConfig

                config.plugins.append(PluginInstanceConfig(
                    instance_id=iid, category=category,
                    provider_id=str((value or {}).get("provider_id") or iid),
                    label=str((value or {}).get("label") or "")))
            return
        if inst is None:
            if strict:
                raise PresetError(f"Instanz {category}/{iid} fehlt")
            return
        if len(path) == 4 and path[3] in ("enabled", "label"):
            setattr(inst, path[3], bool(value) if path[3] == "enabled" else str(value or ""))
            return
        if len(path) == 5 and path[3] == "settings":
            if absent:
                inst.settings.pop(path[4], None)
            else:
                inst.settings[path[4]] = value
            return
    raise PresetError(f"unbekannter Pfad {path}")


def _norm(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _same(a: Any, b: Any) -> bool:
    return _norm(a) == _norm(b)


def _record(sim: Any, plan: Plan, what: str, path: List[Any], after: Any) -> None:
    """Add a change unless ``path`` already holds ``after``; apply it to ``sim``."""
    present, before = _get(sim, path)
    if _is_create(path):
        if present:
            return
    elif present and _same(before, after):
        return
    plan.changes.append(Change(what, list(path), _norm(before) if present else None,
                               _norm(after), before_absent=not present))
    _set(sim, path, after)


def replay_steps(config: Any, steps: List[Dict[str, Any]]) -> None:
    """Write steps (``{path, value, absent}``) into ``config`` without checks.

    For a second copy of the configuration that should show the same result,
    e.g. the open settings dialog after a preset was applied. Steps for an
    instance the copy does not have are left out. - Claude Generated
    """
    for step in steps:
        _set(config, list(step["path"]), step.get("value"),
             absent=bool(step.get("absent")), strict=False)


# ── apply ─────────────────────────────────────────────────────────────────


def _install_file(action: FileAction) -> Dict[str, Any]:
    """Copy one file into place: temporary file, checksum, then replace.

    A file already at the target is copied to ``<name>.bak`` first (one
    generation). Returns the ledger record. - Claude Generated
    """
    src, dest = Path(action.source), Path(action.target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            for block in iter(lambda: inp.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != action.sha256:
            raise PresetError(f"Datei '{action.id}': Quelle hat sich seit der Vorschau geändert "
                              "(Prüfsumme); nicht installiert")
        backup = ""
        if dest.exists():
            bak = dest.with_name(dest.name + ".bak")
            shutil.copy2(dest, bak)
            backup = str(bak)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"id": action.id, "kind": action.kind, "target": str(dest), "sha256": action.sha256,
            "status": FILE_REPLACED if backup else FILE_NEW, "backup": backup}


def _undo_files(records: List[Dict[str, Any]]) -> List[str]:
    """Undo installed files (newest first); returns lines for a report."""
    lines: List[str] = []
    for rec in reversed(records):
        target = Path(rec.get("target") or "")
        try:
            if not target.is_file():
                lines.append(f"{rec.get('id')}: Datei fehlt bereits ({target})")
                continue
            if sha256_file(target) != rec.get("sha256"):
                lines.append(f"{rec.get('id')}: seit dem Anwenden geändert, bleibt ({target})")
                continue
            if rec.get("status") == FILE_NEW:
                target.unlink()
                lines.append(f"{rec.get('id')}: gelöscht ({target})")
                continue
            bak = Path(rec.get("backup") or str(target) + ".bak")
            if bak.is_file():
                os.replace(bak, target)
                lines.append(f"{rec.get('id')}: vorherige Datei aus {bak.name} zurück ({target})")
            else:
                lines.append(f"{rec.get('id')}: keine Sicherung {bak.name} mehr, Datei bleibt ({target})")
        except OSError as exc:
            lines.append(f"{rec.get('id')}: nicht zurückgesetzt ({exc})")
    return lines


def apply_plan(plan: Plan, config_manager: Any = None) -> Optional[Dict[str, Any]]:
    """Apply a plan from :func:`build_plan`; returns the ledger entry.

    ``None`` when the plan is empty (nothing is saved). Raises
    :class:`PresetError` for a plan with errors (nothing is applied) and when
    the configuration no longer holds a value the plan was built on (preview
    again). Files are installed before the configuration is saved; a failure
    of either undoes the installed files. - Claude Generated
    """
    if plan.errors:
        raise PresetError("Das Preset hat Fehler, es wird nichts angewendet:\n"
                          + "\n".join(f"- {e}" for e in plan.errors))
    if plan.is_empty:
        return None
    cm = _cm(config_manager)
    work = copy.deepcopy(cm.load_config())
    for ch in plan.changes:
        present, value = _get(work, ch.path)
        expected_present = not ch.before_absent
        if present != expected_present or (present and not _is_create(ch.path)
                                           and not _same(value, ch.before)):
            raise PresetError(f"Die Konfiguration hat sich seit der Vorschau geändert "
                              f"({ch.what}); bitte die Vorschau neu erstellen")
        _set(work, ch.path, ch.after)
    installed: List[Dict[str, Any]] = []
    try:
        for action in plan.files:
            if action.status != FILE_UNCHANGED:
                installed.append(_install_file(action))
    except BaseException:
        _undo_files(installed)
        raise
    entry = {
        "id": plan.preset_id, "version": plan.version, "label": plan.label,
        "source": plan.preset_path,
        "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "changes": [c.as_dict() for c in plan.changes],
        "files": installed,
    }
    work.applied_presets = list(getattr(work, "applied_presets", None) or []) + [entry]
    if not cm.save_config(work):
        _undo_files(installed)
        raise PresetError("Die Konfiguration konnte nicht gespeichert werden; "
                          "installierte Dateien wurden zurückgesetzt")
    logger.info("Preset '%s' %s angewendet: %d Änderungen, %d Dateien", plan.preset_id,
                plan.version, len(plan.changes), len(installed))
    return entry


# ── revert ────────────────────────────────────────────────────────────────


@dataclass
class RevertResult:
    """What :func:`revert_last` did - Claude Generated"""

    entry: Dict[str, Any]
    restored: List[str] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def lines(self) -> List[str]:
        head = f"Zurückgesetzt: {self.entry.get('label') or self.entry.get('id')}"
        if self.entry.get("version"):
            head += f" {self.entry['version']}"
        out = [head]
        out += [f"  ↺ {line}" for line in self.restored]
        out += [f"  = {line}" for line in self.kept]
        if self.files:
            out.append("Dateien:")
            out += [f"  {line}" for line in self.files]
        return out


def last_applied(config: Any) -> Optional[Dict[str, Any]]:
    """The newest ledger entry, ``None`` without one - Claude Generated"""
    ledger = list(getattr(config, "applied_presets", None) or [])
    return ledger[-1] if ledger else None


def revert_last(config_manager: Any = None) -> RevertResult:
    """Undo the last applied preset from the ledger.

    Each key the preset set goes back to its previous value, but only while it
    still holds the value the preset wrote; a key changed since stays. An
    instance the preset created is removed unless it was changed since. Files:
    a new one is deleted, a replaced one comes back from ``<name>.bak``; a file
    changed since stays. Raises :class:`PresetError` without a ledger entry.
    - Claude Generated
    """
    cm = _cm(config_manager)
    work = copy.deepcopy(cm.load_config())
    ledger = list(getattr(work, "applied_presets", None) or [])
    if not ledger:
        raise PresetError("Es ist kein Preset angewendet.")
    entry = ledger[-1]
    result = RevertResult(entry)
    changes = [Change.from_dict(d) for d in entry.get("changes") or [] if isinstance(d, dict)]
    preset_keys: Dict[Tuple[str, str], set] = {}
    for ch in changes:
        if len(ch.path) == 5 and ch.path[0] == "instance":
            preset_keys.setdefault((ch.path[1], ch.path[2]), set()).add(ch.path[4])
    changed_since: set = set()
    for ch in reversed(changes):
        try:
            present, value = _get(work, ch.path)
        except PresetError:
            result.kept.append(f"{ch.what}: Pfad unbekannt, übergangen")
            continue
        if _is_create(ch.path):
            key = (ch.path[1], ch.path[2])
            if not present:
                continue
            inst = _find(work, *key)
            extra = set(inst.settings) - preset_keys.get(key, set())
            if key in changed_since or extra or inst.label != (ch.after or {}).get("label", inst.label):
                result.kept.append(f"{ch.what}: seit dem Anwenden geändert, die Instanz bleibt")
                continue
            _set(work, ch.path, None, absent=True)
            result.steps.append({"path": list(ch.path), "value": None, "absent": True})
            result.restored.append(f"{ch.what}: entfernt")
            continue
        if not present or not _same(value, ch.after):
            if ch.path[0] == "instance":
                changed_since.add((ch.path[1], ch.path[2]))
            shown = format_value(ch.path, value) if present else "(nicht gesetzt)"
            result.kept.append(f"{ch.what}: seit dem Anwenden geändert, bleibt {shown}")
            continue
        _set(work, ch.path, ch.before, absent=ch.before_absent)
        result.steps.append({"path": list(ch.path), "value": ch.before, "absent": ch.before_absent})
        before = "(nicht gesetzt)" if ch.before_absent else format_value(ch.path, ch.before)
        result.restored.append(f"{ch.what}: wieder {before}")
    work.applied_presets = ledger[:-1]
    if not cm.save_config(work):
        raise PresetError("Die Konfiguration konnte nicht gespeichert werden; nichts zurückgesetzt")
    result.files = _undo_files([r for r in entry.get("files") or [] if isinstance(r, dict)])
    logger.info("Preset '%s' %s zurückgesetzt: %d Werte, %d behalten", entry.get("id"),
                entry.get("version", ""), len(result.restored), len(result.kept))
    return result


# ── first start ───────────────────────────────────────────────────────────


def describe_preset(preset: Preset) -> List[str]:
    """Short lines on what the preset sets (institution, workflow, plugins,
    files), without hashing anything; for the summary page of the setup
    wizard. - Claude Generated"""
    data = preset.data
    out: List[str] = []
    inst = data.get("institution")
    if isinstance(inst, dict) and (inst.get("isil") or inst.get("eln")):
        out.append(f"Einrichtung: ISIL {inst.get('isil') or '(leer)'}, ELN {inst.get('eln') or '(leer)'}")
    if data.get("workflow"):
        out.append(f"Standard-Workflow: {data['workflow']}")
    plugins = [p for p in data.get("plugins") or [] if isinstance(p, dict) and p.get("instance")]
    if plugins:
        labels = {"on": "an", "cache_only": "nur Cache", "off": "aus"}
        parts = [f"{p['instance']} ({labels.get(p.get('state'), 'Einstellungen')})" for p in plugins]
        out.append("Plugins: " + ", ".join(parts))
    files = [f for f in data.get("files") or [] if isinstance(f, dict) and f.get("id")]
    if files:
        out.append("Dateien: " + ", ".join(str(f["id"]) for f in files))
    secrets = [s for s in data.get("secrets") or [] if isinstance(s, dict) and s.get("plugin")]
    if secrets:
        out.append("Noch einzutragen: " + ", ".join(f"{s['plugin']}.{s.get('key')}" for s in secrets))
    return out


def apply_after_setup(preset_path: Any, config_manager: Any = None,
                      source: str = SOURCE_GIVEN) -> List[str]:
    """Apply institution, workflow, plugins and files of a preset after the
    setup wizard; returns lines for the log or the terminal. Never raises: a
    broken preset must not stop the wizard. - Claude Generated"""
    if not preset_path:
        return []
    try:
        preset = load_preset(preset_path, source)
        plan = build_plan(preset, config_manager)
        if plan.errors:
            lines = [f"Preset {plan.label} nicht angewendet:"] + [f"  {e}" for e in plan.errors]
            logger.warning("\n".join(lines))
            return lines
        if plan.is_empty:
            return [f"Preset {plan.label}: nichts zu ändern"]
        apply_plan(plan, config_manager)
        lines = [f"Preset {plan.label} {plan.version}".rstrip() + " angewendet:"]
        lines += [line for line in plan.lines()[1:] if not line.startswith("Datei: ")]
        logger.info("\n".join(lines))
        return lines
    except Exception as exc:  # unreadable file, disk full, a defect
        from .error_visibility import log_caught

        log_caught(logger, exc, "Preset nach dem Einrichtungsassistenten")
        return [f"Preset nicht angewendet: {exc}"]
