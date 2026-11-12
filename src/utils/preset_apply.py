"""Institution presets: find, read, plan, apply, revert - Claude Generated.

An institution package is a folder holding ``alima_presets.json`` and the files
it ships (for example the local subject database of ``geo_sachdb`` and the
shelf vocabulary of ``local_scheme``). This module finds the preset, reads it,
computes what it would change (:func:`build_plan`, a preview with ``before`` and
``after`` per change), applies that plan (:func:`apply_plan`) and undoes the
last application (:func:`revert_last`) from the ledger
``AlimaConfig.applied_presets``. The export side is ``preset_export.py``; the
cache parts a package can carry (block ``cache``) are ``preset_cache.py``.

Rules:

* **Advisory.** Only values the preset names are set; every other value of the
  user stays as it is. A second application of the same preset is an empty
  plan and saves nothing.
* **No secrets, one exception.** Keys named ``api_key``/``token`` (also
  ``*_api_key``, ``*_token``) are dropped when the file is read, with one log
  warning; SECRET fields of a plugin are never set. ``secrets`` only declares
  what each user still has to enter. The one exception is ``providers[*].api_key``:
  the key of an LLM provider entry, written into a package only on explicit
  request (``preset_export``). It is never shown in the plan, a log line or an
  exception text.
* **LLM access.** ``providers`` adds or replaces ``UnifiedProvider`` entries by
  name (case-insensitive, at their place); a key already configured stays when
  the package carries none. ``default_models`` sets the default provider/model
  per scope (``general``, ``pipeline``, ``vision``, ``decision``) and can be left
  out per application (``Plan.apply_default_models``).
* **Files fail closed.** A file is installed only when its ``sha256`` matches
  (and, for ``kind: geo_sachdb_db``, the read-only schema check of the plugin
  passes). A plan with an error is not applied at all. Installation writes a
  temporary file and replaces the target; a replaced file stays as
  ``<name>.bak`` (one generation).
* **The cache is separate.** Parts of the search cache under ``cache`` are
  merged only when the ISIL of the package equals the ISIL of the target; a
  wrong checksum, a missing or invalid cache file skips only that part, never
  the preset. The merge cannot be undone; :func:`revert_last` leaves it.

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
    "format", "preset", "institution_name", "catalog", "database",
    "institution", "workflow", "plugins", "files", "secrets", "cache",
    "providers", "default_models",
}
#: Top-level keys of earlier formats that are no longer read; the plan names
#: each once as a hint (not as an unknown key). - Claude Generated
_RETIRED_TOP_KEYS = {
    "llm": "Veralteter Block 'llm' wird ignoriert; Provider stehen unter 'providers', "
           "Standardmodelle unter 'default_models'",
}
_PLUGIN_KEYS = {"instance", "type", "category", "state", "label", "settings"}
_FILE_KEYS = {"id", "source", "sha256", "target", "kind"}

# ── LLM providers (block ``providers``) and default models (``default_models``)

#: Plan paths: ``["provider", name]`` (the access fields of one provider
#: entry, never its key), ``["provider", name, "api_key"]`` (its key),
#: ``["default_model", scope]`` (``{"provider", "model"}``).
PROVIDER_PATH = "provider"
DEFAULT_MODEL_PATH = "default_model"
PROVIDER_KEY_FIELD = "api_key"
#: ``UnifiedProvider`` fields a provider entry carries for every type. Not in a
#: package: ``id`` (per installation), ``available_models`` (filled at run
#: time), ``api_key`` (only on request, see ``preset_export``).
PROVIDER_FIELDS = ("name", "provider_type", "enabled", "base_url", "preferred_model",
                   "description")
#: Fields only the type ``ollama`` reads (``UnifiedProvider``: "Ollama specific").
OLLAMA_FIELDS = ("host", "port", "use_ssl", "connection_type")
OLLAMA_CONNECTION_TYPES = ("native_client", "openai_compatible")
_PROVIDER_DEFAULTS = {"enabled": True, "base_url": "", "preferred_model": "", "description": "",
                      "host": "", "port": 11434, "use_ssl": False,
                      "connection_type": "native_client"}
#: Scope → (provider field, model field) of ``UnifiedProviderConfig``.
DEFAULT_MODEL_FIELDS = {
    "general": ("preferred_provider", "preferred_model"),
    "pipeline": ("pipeline_default_provider", "pipeline_default_model"),
    "vision": ("vision_default_provider", "vision_default_model"),
    "decision": ("decision_default_provider", "decision_default_model"),
}
DEFAULT_MODEL_LABELS = {"general": "Allgemein (Agentik, Chat)", "pipeline": "Pipeline",
                        "vision": "Bild", "decision": "Entscheidung"}
_PROVIDER_ENTRY_PATH = re.compile(r"^providers\[\d+\]$")
#: Characters never accepted in a provider name or model (path separators,
#: shell metacharacters, control characters).
_FORBIDDEN_CHARS = re.compile(r"[\x00-\x1f\x7f/\\`$;|&<>]")
_FORBIDDEN_MODEL_CHARS = re.compile(r"[\x00-\x20\x7f\\`$;|&<>]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_HOST = re.compile(r"^[A-Za-z0-9._\-\[\]:]{0,253}$")

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
    """Copy of ``value`` without keys named like an API key or token.

    One key is kept: ``api_key`` directly in an entry of the top-level list
    ``providers`` (``providers[n].api_key``). Every other secret-shaped key,
    also deeper inside a provider entry, is dropped. - Claude Generated"""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            if key == PROVIDER_KEY_FIELD and _PROVIDER_ENTRY_PATH.match(where):
                out[key] = item
                continue
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
    when one of them carried a value, that is logged once as a warning (names
    only, never a value). Kept: ``providers[n].api_key``. - Claude Generated
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

    def __repr__(self) -> str:
        """Like the dataclass repr, with a provider key as ``***`` (a repr can
        end up in a log or a traceback) - Claude Generated"""
        before, after = self.before, self.after
        if len(self.path) == 3 and self.path[0] == "provider" and self.path[2] == "api_key":
            before, after = ("***" if before else before), ("***" if after else after)
        return (f"Change(what={self.what!r}, path={self.path!r}, before={before!r}, "
                f"after={after!r}, before_absent={self.before_absent!r})")


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


PROVIDER_NEW = "new"
PROVIDER_REPLACED = "replaced"
PROVIDER_UNCHANGED = "unchanged"
PROVIDER_REJECTED = "rejected"
_PROVIDER_STATUS_LABELS = {PROVIDER_NEW: "neu", PROVIDER_REPLACED: "ersetzt",
                           PROVIDER_UNCHANGED: "unverändert", PROVIDER_REJECTED: "abgelehnt"}
#: What happens to the key of one provider entry.
KEY_SET = "set"          # the package carries a key that differs: it is written
KEY_SAME = "same"        # the package carries the key already configured
KEY_KEPT = "kept"        # no key in the package; the configured one stays
KEY_NONE = ""            # no key in the package, none configured
_KEY_LABELS = {KEY_SET: "Schlüssel wird gesetzt", KEY_SAME: "Schlüssel unverändert",
               KEY_KEPT: "vorhandener Schlüssel bleibt"}


@dataclass
class ProviderAction:
    """One entry of the block ``providers`` as the plan sees it; never holds
    the key itself - Claude Generated"""

    name: str
    provider_type: str = ""
    status: str = PROVIDER_NEW
    endpoint: str = ""
    key: str = KEY_NONE
    #: replaced: ``"field: before → after"`` per changed field; rejected: the reason.
    details: List[str] = field(default_factory=list)

    def line(self) -> str:
        label = _PROVIDER_STATUS_LABELS.get(self.status, self.status)
        if self.status == PROVIDER_REJECTED:
            return f"✗ {label}: {self.name}: {'; '.join(self.details)}"
        text = f"{label}: {self.name} ({self.provider_type}, {self.endpoint})"
        if self.details:
            text += ": " + ", ".join(self.details)
        if self.key in _KEY_LABELS:
            text += f"; {_KEY_LABELS[self.key]}"
        return text

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "provider_type": self.provider_type, "status": self.status,
                "endpoint": self.endpoint, "key": self.key, "details": list(self.details)}


def _is_key_path(path: List[Any]) -> bool:
    return len(path) == 3 and path[0] == PROVIDER_PATH and path[2] == PROVIDER_KEY_FIELD


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
    #: ``preset_cache.CachePlan`` of the block ``cache``; ``None`` without one.
    cache: Any = None
    #: The entries of the block ``providers`` (new, replaced, unchanged, rejected).
    providers: List[ProviderAction] = field(default_factory=list)
    #: Whether the changes of ``default_models`` are applied (default: yes,
    #: they overwrite). ``build_plan(default_models=False)``, the CLI switch
    #: ``--no-default-models`` and the check box of the preview set it off.
    apply_default_models: bool = True

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def default_model_changes(self) -> List[Change]:
        """The changes of ``default_models``, applied or not - Claude Generated"""
        return [c for c in self.changes if c.path and c.path[0] == DEFAULT_MODEL_PATH]

    @property
    def applied_changes(self) -> List[Change]:
        """The changes :func:`apply_plan` writes: all, without those of
        ``default_models`` when ``apply_default_models`` is off - Claude Generated"""
        return [c for c in self.changes
                if self.apply_default_models or not (c.path and c.path[0] == DEFAULT_MODEL_PATH)]

    @property
    def has_provider_keys(self) -> bool:
        """A key from the package would be written into the configuration."""
        return any(_is_key_path(c.path) for c in self.applied_changes)

    @property
    def is_empty(self) -> bool:
        """Nothing to write: no change, no file to install, no cache row to merge."""
        return (not self.applied_changes and all(f.status == FILE_UNCHANGED for f in self.files)
                and not (self.cache is not None and self.cache.has_writes))

    def steps(self) -> List[Dict[str, Any]]:
        """The applied changes as replay steps (see :func:`replay_steps`)."""
        return [{"path": list(c.path), "value": c.after, "absent": False}
                for c in self.applied_changes]

    def as_dict(self) -> Dict[str, Any]:
        """The plan as data; provider keys appear as ``"***"``."""
        changes = []
        for c in self.changes:
            d = c.as_dict()
            if _is_key_path(c.path):
                d["before"] = "***" if d["before"] else d["before"]
                d["after"] = "***" if d["after"] else d["after"]
            changes.append(d)
        return {
            "preset_id": self.preset_id, "version": self.version, "label": self.label,
            "preset_path": self.preset_path, "source": self.source,
            "changes": changes,
            "files": [f.as_dict() for f in self.files],
            "warnings": list(self.warnings), "errors": list(self.errors),
            "secrets": [dict(s) for s in self.secrets],
            "cache": self.cache.as_dict() if self.cache is not None else None,
            "providers": [p.as_dict() for p in self.providers],
            "apply_default_models": self.apply_default_models,
        }

    def lines(self) -> List[str]:
        """The plan as text for the CLI and the preview dialog. Never contains
        a provider key (only whether one is set)."""
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
        general = [c for c in self.changes
                   if not (c.path and c.path[0] in (PROVIDER_PATH, DEFAULT_MODEL_PATH))]
        if general:
            out.append(f"Änderungen ({len(general)}):")
            for ch in general:
                before = "(nicht gesetzt)" if ch.before_absent else format_value(ch.path, ch.before)
                if _is_create(ch.path):
                    out.append(f"  + {ch.what}")
                else:
                    out.append(f"  {ch.what}: {before} → {format_value(ch.path, ch.after)}")
        if self.providers:
            out.append(f"Provider ({len(self.providers)}):")
            out += [f"  {p.line()}" for p in self.providers]
        defaults = self.default_model_changes
        if defaults:
            if self.apply_default_models:
                out.append(f"Standardmodelle ({len(defaults)}, werden überschrieben; abwählbar):")
            else:
                out.append(f"Standardmodelle ({len(defaults)}, abgewählt: bleiben, wie sie sind):")
            for ch in defaults:
                before = "(nicht gesetzt)" if ch.before_absent else format_value(ch.path, ch.before)
                out.append(f"  {ch.what}: {before} → {format_value(ch.path, ch.after)}")
        if self.files:
            out.append(f"Dateien ({len(self.files)}):")
            for fa in self.files:
                extra = " (die vorherige bleibt als .bak)" if fa.status == FILE_REPLACED else ""
                out.append(f"  {_FILE_STATUS_LABELS.get(fa.status, fa.status)}: {fa.id} → "
                           f"{fa.target} ({_size(fa.size)}, sha256 {fa.sha256[:12]}…){extra}")
        if self.cache is not None:
            out += self.cache.lines()
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
    """A value as the plan shows it; a provider key only as set or empty - Claude Generated"""
    if _is_key_path(path):
        return "(gesetzt)" if value else "(leer)"
    if path and path[0] == PROVIDER_PATH and len(path) == 2:
        return provider_endpoint(value) if isinstance(value, dict) else "(nicht vorhanden)"
    if path and path[0] == DEFAULT_MODEL_PATH:
        value = value if isinstance(value, dict) else {}
        provider, model = value.get("provider") or "", value.get("model") or ""
        if not provider:
            return "(leer)"
        return f"{provider} / {model}" if model else f"{provider} / (Modell des Providers)"
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


def build_plan(preset: Preset, config_manager: Any = None, *, config: Any = None,
               default_models: bool = True, llm: bool = True) -> Plan:
    """Preview of what applying ``preset`` would change; nothing is written.

    ``config`` (default: the loaded configuration of ``config_manager``) is not
    modified. Hashes every shipped file once. ``default_models=False``: the
    changes of the block ``default_models`` are listed but not applied
    (``Plan.apply_default_models``). ``llm=False`` leaves the blocks
    ``providers`` and ``default_models`` out entirely (the setup wizard, which
    asks for its provider and models itself). - Claude Generated
    """
    cm = _cm(config_manager)
    base = config if config is not None else cm.load_config()
    sim = copy.deepcopy(base)
    plan = Plan(preset.id, preset.version, preset.label, str(preset.path), preset.source)
    plan.apply_default_models = bool(default_models)
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
        if isinstance(key, str) and key in _RETIRED_TOP_KEYS:
            plan.warnings.append(_RETIRED_TOP_KEYS[key])
        elif isinstance(key, str) and not key.startswith("_") and key not in _TOP_KEYS:
            plan.warnings.append(f"Unbekannter Schlüssel '{key}' wird übergangen")
    _ensure_categories()
    files = _plan_files(preset, data_dir_of(cm), plan)
    _plan_institution(preset, sim, plan)
    _plan_workflow(preset, sim, plan)
    _plan_plugins(preset, sim, plan, files)
    if llm:
        _plan_providers(preset, sim, plan)
        _plan_default_models(preset, sim, plan)
    plan.secrets = secret_status(sim, preset.data.get("secrets"))
    if "cache" in preset.data:
        _plan_cache(preset, sim, base, plan)
    return plan


def _plan_cache(preset: Preset, sim: Any, base: Any, plan: Plan) -> None:
    """The block ``cache``: checked and counted against the cache database of
    the configuration; the target ISIL is the one after the preset (``sim``).
    Reads only. - Claude Generated"""
    from .preset_cache import plan_cache

    target_isil = str(getattr(sim.system_config, "institution_isil", "") or "")
    plan.cache = plan_cache(preset.data.get("cache"), preset.base_dir, target_isil,
                            getattr(base, "database_config", None))
    plan.warnings.extend(plan.cache.warnings)


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


def provider_access(provider: Any) -> Dict[str, Any]:
    """The access fields of a ``UnifiedProvider`` as a package carries them,
    without its key: ``PROVIDER_FIELDS``, plus ``OLLAMA_FIELDS`` for the type
    ``ollama`` - Claude Generated"""
    names = PROVIDER_FIELDS + (OLLAMA_FIELDS if provider.provider_type == "ollama" else ())
    return {name: getattr(provider, name) for name in names}


def provider_endpoint(access: Dict[str, Any]) -> str:
    """The address a provider entry connects to, for the preview - Claude Generated"""
    url = str(access.get("base_url") or "")
    if url:
        return url
    if access.get("provider_type") == "ollama" and access.get("host"):
        from .ollama_endpoint import base_url

        return base_url(str(access["host"]), int(access.get("port") or 11434),
                        bool(access.get("use_ssl")))
    if access.get("provider_type") == "ollama":
        return "http://localhost:11434"
    return "Standardadresse des Anbieters"


def _check_model(value: Any, what: str) -> str:
    if not isinstance(value, str):
        return f"{what}: Text erwartet"
    if len(value) > 200 or _FORBIDDEN_MODEL_CHARS.search(value):
        return f"{what}: ungültiger Modellname (Leerraum, Steuer- oder Sonderzeichen)"
    return ""


def read_provider_entry(entry: Any) -> Tuple[Optional[Dict[str, Any]], str, str, List[str]]:
    """Check one entry of the block ``providers`` (input from outside).

    Returns ``(access, key, reason, notes)``: ``access`` are the fields of
    :func:`provider_access` with defaults filled in (``None`` when the entry is
    rejected), ``key`` the ``api_key`` of the entry (``""`` without one),
    ``reason`` why it is rejected, ``notes`` hints on fields that are skipped.
    Neither ``reason`` nor ``notes`` ever contain the key. Rules: known type
    (``config_models.PROVIDER_TYPES``), a name without path separators, shell or
    control characters, ``base_url`` only http/https without user or password,
    a host without scheme or path, a port 1 to 65535. - Claude Generated
    """
    from .config_models import PROVIDER_TYPES

    if not isinstance(entry, dict):
        return None, "", "ein Objekt erwartet", []
    notes: List[str] = []
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "", "'name' fehlt", notes
    if name != name.strip() or len(name) > 100 or _FORBIDDEN_CHARS.search(name):
        return None, "", ("'name' ist ungültig (höchstens 100 Zeichen, ohne / \\ ` $ ; | & < > "
                          "und Steuerzeichen)"), notes
    ptype = entry.get("provider_type")
    if ptype not in PROVIDER_TYPES:
        return None, "", (f"Typ '{ptype}' ist unbekannt (erlaubt: "
                          f"{', '.join(PROVIDER_TYPES)})"), notes
    allowed = set(PROVIDER_FIELDS) | {PROVIDER_KEY_FIELD}
    if ptype == "ollama":
        allowed |= set(OLLAMA_FIELDS)
    for key in entry:
        if not isinstance(key, str) or key.startswith("_") or key in allowed:
            continue
        if key in OLLAMA_FIELDS:
            notes.append(f"Provider {name}: '{key}' gilt nur für den Typ ollama; übergangen")
        else:
            notes.append(f"Provider {name}: unbekannter Schlüssel '{key}' wird übergangen")
    access: Dict[str, Any] = {"name": name, "provider_type": ptype}
    for key in PROVIDER_FIELDS[2:] + (OLLAMA_FIELDS if ptype == "ollama" else ()):
        value = entry.get(key)
        access[key] = _PROVIDER_DEFAULTS[key] if value is None else value
    for key in ("enabled", "use_ssl"):
        if key in access and not isinstance(access[key], bool):
            return None, "", f"'{key}': true oder false erwartet", notes
    for key in ("base_url", "description", "host", "connection_type"):
        if key in access and not isinstance(access[key], str):
            return None, "", f"'{key}': Text erwartet", notes
    reason = _check_model(access["preferred_model"], "'preferred_model'")
    if reason:
        return None, "", reason, notes
    if len(access["description"]) > 500 or _CONTROL_CHARS.search(access["description"]):
        return None, "", "'description': höchstens 500 Zeichen, ohne Steuerzeichen", notes
    url = access["base_url"].strip()
    access["base_url"] = url
    if url:
        from urllib.parse import urlparse

        from .net_guard import require_http_url

        try:
            parsed = urlparse(url)
            has_userinfo = bool(parsed.username or parsed.password)
        except ValueError:
            return None, "", "'base_url' ist keine lesbare Adresse", notes
        if has_userinfo:
            return None, "", "'base_url' enthält Zugangsdaten (Benutzer/Passwort); nicht erlaubt", notes
        if _CONTROL_CHARS.search(url) or " " in url:
            return None, "", "'base_url' enthält Leer- oder Steuerzeichen", notes
        try:
            require_http_url(url, what="base_url")
        except ValueError:
            return None, "", "'base_url': nur eine http- oder https-Adresse mit Rechnername", notes
    if ptype == "ollama":
        host = access["host"].strip()
        access["host"] = host
        if "://" in host or "/" in host or not _HOST.match(host):
            return None, "", "'host': nur der Rechnername oder die IP-Adresse (ohne Schema und Pfad)", notes
        port = access["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            return None, "", "'port': eine Zahl von 1 bis 65535 erwartet", notes
        if access["connection_type"] not in OLLAMA_CONNECTION_TYPES:
            return None, "", (f"'connection_type': erlaubt sind "
                              f"{', '.join(OLLAMA_CONNECTION_TYPES)}"), notes
    key = entry.get(PROVIDER_KEY_FIELD)
    if key is None:
        key = ""
    if not isinstance(key, str):
        return None, "", "'api_key': Text erwartet", notes
    key = key.strip()
    if key and (_CONTROL_CHARS.search(key) or any(ch.isspace() for ch in key)):
        return None, "", "'api_key' enthält Leer- oder Steuerzeichen", notes
    return access, key, "", notes


def _find_provider(config: Any, name: Any):
    """The provider of that name, case-insensitive (``get_provider_by_name``)."""
    return config.unified_config.get_provider_by_name(str(name))


def _plan_providers(preset: Preset, sim: Any, plan: Plan) -> None:
    """The block ``providers``: add, replace (same name, at its place) or keep
    each entry; a key only when the package carries one. A rejected entry is a
    hint and is left out; the rest of the preset stays applicable (as with
    ``plugins``). - Claude Generated"""
    entries = preset.data.get("providers")
    if entries is None:
        return
    if not isinstance(entries, list):
        plan.warnings.append("providers: eine Liste erwartet; übergangen")
        return
    seen: set = set()
    for n, entry in enumerate(entries, 1):
        access, key, reason, notes = read_provider_entry(entry)
        plan.warnings.extend(notes)
        raw_name = entry.get("name") if isinstance(entry, dict) else None
        shown = f"providers[{n}]"
        if (isinstance(raw_name, str) and raw_name.strip() and len(raw_name) <= 100
                and not _CONTROL_CHARS.search(raw_name)):
            shown += f" ({raw_name})"
        if access is None:
            plan.providers.append(ProviderAction(shown, status=PROVIDER_REJECTED, details=[reason]))
            plan.warnings.append(f"Provider {shown} übergangen: {reason}")
            continue
        name = access["name"]
        if name.lower() in seen:
            reason = f"Name '{name}' doppelt (Groß-/Kleinschreibung zählt nicht)"
            plan.providers.append(ProviderAction(shown, status=PROVIDER_REJECTED, details=[reason]))
            plan.warnings.append(f"Provider {shown} übergangen: {reason}")
            continue
        seen.add(name.lower())
        current = _find_provider(sim, name)
        action = ProviderAction(name, access["provider_type"], endpoint=provider_endpoint(access))
        if current is None:
            action.status = PROVIDER_NEW
        else:
            before = provider_access(current)
            if _same(before, access):
                action.status = PROVIDER_UNCHANGED
            else:
                action.status = PROVIDER_REPLACED
                for field_name in PROVIDER_FIELDS + OLLAMA_FIELDS:
                    if field_name in before or field_name in access:
                        old, new = before.get(field_name), access.get(field_name)
                        if not _same(old, new):
                            action.details.append(f"{field_name}: {format_value([field_name], old)} "
                                                  f"→ {format_value([field_name], new)}")
        _record(sim, plan, f"Provider {name}", [PROVIDER_PATH, name], access)
        configured_key = str(getattr(_find_provider(sim, name), "api_key", "") or "")
        if key:
            action.key = KEY_SAME if key == configured_key else KEY_SET
            _record(sim, plan, f"Provider {name}: Schlüssel",
                    [PROVIDER_PATH, name, PROVIDER_KEY_FIELD], key)
        elif configured_key:
            action.key = KEY_KEPT
        elif access["provider_type"] in ("gemini", "anthropic"):
            plan.warnings.append(f"Provider {name}: ohne Schlüssel nicht nutzbar; den Schlüssel "
                                 "im Anbieter-Dialog eintragen")
        plan.providers.append(action)


def _plan_default_models(preset: Preset, sim: Any, plan: Plan) -> None:
    """The block ``default_models``: per scope ``{"provider", "model"}``. The
    provider must exist after the block ``providers``; without ``model`` the
    preferred model of the provider is used (``decision`` needs a model). The
    changes overwrite unless ``plan.apply_default_models`` is off.
    - Claude Generated"""
    block = preset.data.get("default_models")
    if block is None:
        return
    if not isinstance(block, dict):
        plan.warnings.append("default_models: ein Objekt erwartet; übergangen")
        return
    for scope in block:
        if isinstance(scope, str) and not scope.startswith("_") and scope not in DEFAULT_MODEL_FIELDS:
            plan.warnings.append(f"default_models: unbekannter Bereich '{scope}' wird übergangen "
                                 f"(bekannt: {', '.join(DEFAULT_MODEL_FIELDS)})")
    unified = sim.unified_config
    for scope, label in DEFAULT_MODEL_LABELS.items():
        if scope not in block:
            continue
        item = block[scope]
        where = f"Standardmodell {label}"
        if not isinstance(item, dict):
            plan.warnings.append(f"{where}: ein Objekt mit provider/model erwartet; übergangen")
            continue
        for key in item:
            if isinstance(key, str) and not key.startswith("_") and key not in ("provider", "model"):
                plan.warnings.append(f"{where}: unbekannter Schlüssel '{key}' wird übergangen")
        provider_name = item.get("provider")
        if not isinstance(provider_name, str) or not provider_name.strip():
            plan.warnings.append(f"{where}: 'provider' fehlt; übergangen")
            continue
        provider = unified.get_provider_by_name(provider_name.strip())
        if provider is None:
            plan.warnings.append(f"{where}: Provider '{provider_name}' gibt es nach dem Preset "
                                 "nicht; übergangen")
            continue
        model = item.get("model") or ""
        reason = _check_model(model, f"{where}: 'model'")
        if reason:
            plan.warnings.append(f"{reason}; übergangen")
            continue
        if not model and scope == "decision":
            plan.warnings.append(f"{where}: 'model' fehlt (ein Entscheidungsmodell hat keinen "
                                 "Rückfall); übergangen")
            continue
        if not model:
            model = unified.resolve_provider_model(provider.name)[1]
        if not provider.enabled or not provider.is_usable:
            plan.warnings.append(f"{where}: Provider {provider.name} ist "
                                 f"{'ausgeschaltet' if not provider.enabled else 'ohne Schlüssel nicht nutzbar'}; "
                                 "bis dahin gilt der nächste Standard")
        _record(sim, plan, where, [DEFAULT_MODEL_PATH, scope],
                {"provider": provider.name, "model": model})


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
    if head == PROVIDER_PATH and len(path) in (2, 3):
        provider = _find_provider(config, path[1])
        if len(path) == 2:
            return provider is not None, (provider_access(provider) if provider is not None else None)
        if path[2] == PROVIDER_KEY_FIELD:
            return provider is not None, (provider.api_key if provider is not None else None)
    if head == DEFAULT_MODEL_PATH and len(path) == 2 and path[1] in DEFAULT_MODEL_FIELDS:
        prov_field, model_field = DEFAULT_MODEL_FIELDS[path[1]]
        unified = config.unified_config
        return True, {"provider": getattr(unified, prov_field, "") or "",
                      "model": getattr(unified, model_field, "") or ""}
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
    if head == PROVIDER_PATH and len(path) in (2, 3):
        _set_provider(config, path, value, absent=absent, strict=strict)
        return
    if head == DEFAULT_MODEL_PATH and len(path) == 2 and path[1] in DEFAULT_MODEL_FIELDS:
        prov_field, model_field = DEFAULT_MODEL_FIELDS[path[1]]
        value = value if isinstance(value, dict) and not absent else {}
        setattr(config.unified_config, prov_field, str(value.get("provider") or ""))
        setattr(config.unified_config, model_field, str(value.get("model") or ""))
        return
    raise PresetError(f"unbekannter Pfad {path}")


def _drop_legacy_key_mirror(unified: Any, provider: Any) -> None:
    """``gemini_api_key``/``anthropic_api_key`` mirror the key of the provider
    named ``gemini``/``anthropic`` (``sync_legacy_from_providers``), and a
    mirrored key recreates that provider on load
    (``ConfigManager._create_providers_from_api_keys``). When a preset step
    removes such a provider or clears its key, the mirror goes too, or the next
    load would bring it back. - Claude Generated"""
    name = str(getattr(provider, "name", "") or "").lower()
    if name not in ("gemini", "anthropic"):
        return
    attr = f"{name}_api_key"
    if getattr(unified, attr, "") and getattr(unified, attr, "") == getattr(provider, "api_key", ""):
        setattr(unified, attr, "")


def _set_provider(config: Any, path: List[Any], value: Any, *, absent: bool,
                  strict: bool) -> None:
    """Write a provider step: ``["provider", name]`` adds the entry or sets
    its access fields in place (``id``, ``available_models`` and the key stay);
    ``absent`` removes it. ``["provider", name, "api_key"]`` sets the key
    (``absent``: empty). - Claude Generated"""
    unified = config.unified_config
    provider = _find_provider(config, path[1])
    if len(path) == 2:
        if absent:
            if provider is not None:
                _drop_legacy_key_mirror(unified, provider)
                unified.providers[:] = [p for p in unified.providers if p is not provider]
            return
        access = {k: v for k, v in dict(value or {}).items()
                  if k in PROVIDER_FIELDS or k in OLLAMA_FIELDS}
        if provider is None:
            from .config_models import UnifiedProvider

            unified.providers.append(UnifiedProvider(**access))
        else:
            for key, item in access.items():
                setattr(provider, key, item)
        return
    if path[2] != PROVIDER_KEY_FIELD:
        raise PresetError(f"unbekannter Pfad {path}")
    if provider is None:
        if strict:
            raise PresetError(f"Provider {path[1]} fehlt")
        return
    new_key = "" if absent else str(value or "")
    if not new_key:
        _drop_legacy_key_mirror(unified, provider)
    provider.api_key = new_key


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


def _touches_llm(changes: List[Change]) -> bool:
    return any(c.path and c.path[0] in (PROVIDER_PATH, DEFAULT_MODEL_PATH) for c in changes)


def _save(cm: Any, work: Any, changes: List[Change]) -> bool:
    """Save through ``ConfigManager.save_config``. With provider or default
    model steps the unified section is written as it is in ``work``
    (``preserve_unified=False``, as ``save_unified_config`` does): the
    preserving save never writes an empty provider list or an empty general
    default, so removing the only provider or restoring an empty default would
    not reach the file. - Claude Generated"""
    if _touches_llm(changes):
        return cm.save_config(work, preserve_unified=False)
    return cm.save_config(work)


def apply_plan(plan: Plan, config_manager: Any = None, *,
               default_models: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """Apply a plan from :func:`build_plan`; returns the ledger entry.

    ``None`` when the plan is empty (nothing is saved). Raises
    :class:`PresetError` for a plan with errors (nothing is applied) and when
    the configuration no longer holds a value the plan was built on (preview
    again). Files are installed before the configuration is saved; a failure
    of either undoes the installed files. The cache parts are merged after the
    files and before the save, into the cache database of the saved
    configuration (``preset_cache.merge_planned``, never raises); their counts
    go into the ledger entry. A merged cache stays when the save fails.
    ``default_models`` (``None``: as the plan says) sets
    ``plan.apply_default_models`` before anything is applied, so
    ``plan.steps()`` afterwards names what was written. Provider keys of the
    package go into the configuration file and into the ledger entry (which
    keeps the previous key for :func:`revert_last`); log lines name counts only.
    - Claude Generated
    """
    if plan.errors:
        raise PresetError("Das Preset hat Fehler, es wird nichts angewendet:\n"
                          + "\n".join(f"- {e}" for e in plan.errors))
    if default_models is not None:
        plan.apply_default_models = bool(default_models)
    if plan.is_empty:
        return None
    changes = plan.applied_changes
    cm = _cm(config_manager)
    work = copy.deepcopy(cm.load_config())
    for ch in changes:
        present, value = _get(work, ch.path)
        expected_present = not ch.before_absent
        if present != expected_present or (present and not _is_create(ch.path)
                                           and not _same(value, ch.before)):
            raise PresetError(f"Die Konfiguration hat sich seit der Vorschau geändert "
                              f"({ch.what}); bitte die Vorschau neu erstellen")
        _set(work, ch.path, ch.after)
    if plan.cache is not None and plan.cache.has_writes:
        from .preset_cache import normalize_isil

        if normalize_isil(work.system_config.institution_isil) != normalize_isil(plan.cache.target_isil):
            raise PresetError("Die ISIL der Einrichtung hat sich seit der Vorschau geändert; "
                              "bitte die Vorschau neu erstellen")
    installed: List[Dict[str, Any]] = []
    try:
        for action in plan.files:
            if action.status != FILE_UNCHANGED:
                installed.append(_install_file(action))
    except BaseException:
        _undo_files(installed)
        raise
    cache_records: List[Dict[str, Any]] = []
    if plan.cache is not None and plan.cache.actions:
        from .preset_cache import merge_planned

        cache_records = merge_planned(plan.cache, getattr(work, "database_config", None))
    entry = {
        "id": plan.preset_id, "version": plan.version, "label": plan.label,
        "source": plan.preset_path,
        "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "changes": [c.as_dict() for c in changes],
        "files": installed,
    }
    if cache_records:
        entry["cache"] = cache_records
    work.applied_presets = list(getattr(work, "applied_presets", None) or []) + [entry]
    if not _save(cm, work, changes):
        _undo_files(installed)
        merged = any(r.get("new") or r.get("updated") for r in cache_records)
        raise PresetError("Die Konfiguration konnte nicht gespeichert werden; "
                          "installierte Dateien wurden zurückgesetzt"
                          + ("; der eingespielte Cache bleibt" if merged else ""))
    logger.info("Preset '%s' %s angewendet: %d Änderungen (%d Provider, %d Schlüssel, "
                "%d Standardmodelle), %d Dateien, Cache %s", plan.preset_id, plan.version,
                len(changes), sum(1 for c in changes if c.path[0] == PROVIDER_PATH
                                  and len(c.path) == 2),
                sum(1 for c in changes if _is_key_path(c.path)),
                sum(1 for c in changes if c.path[0] == DEFAULT_MODEL_PATH), len(installed),
                ", ".join(f"{r['id']} {r['status']} +{r['new']}/~{r['updated']}"
                          for r in cache_records) or "-")
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
        if any(r.get("new") or r.get("updated") for r in self.entry.get("cache") or []
               if isinstance(r, dict)):
            out.append("Cache: bleibt wie eingespielt (das Zusammenführen ist nicht "
                       "rückgängig zu machen)")
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
    changed since stays. The cache is not touched (its merge is final).
    Providers: a replaced entry gets its previous access fields and key back
    (each only while unchanged since); an entry the preset added is removed
    unless it was changed since or has a key the preset did not write.
    Raises :class:`PresetError` without a ledger entry. - Claude Generated
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
    providers_changed_since: set = set()
    for ch in reversed(changes):
        try:
            present, value = _get(work, ch.path)
        except PresetError:
            result.kept.append(f"{ch.what}: Pfad unbekannt, übergangen")
            continue
        if ch.path[0] == PROVIDER_PATH and len(ch.path) == 2 and ch.before_absent:
            # Added by the preset. Its key step (later in the list) ran first.
            if not present:
                continue
            provider = _find_provider(work, ch.path[1])
            if (str(ch.path[1]).lower() in providers_changed_since
                    or not _same(value, ch.after) or provider.api_key):
                result.kept.append(f"{ch.what}: seit dem Anwenden geändert, der Provider bleibt")
                continue
            _set(work, ch.path, None, absent=True)
            result.steps.append({"path": list(ch.path), "value": None, "absent": True})
            result.restored.append(f"{ch.what}: entfernt")
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
            elif ch.path[0] == PROVIDER_PATH:
                providers_changed_since.add(str(ch.path[1]).lower())
            shown = format_value(ch.path, value) if present else "(nicht gesetzt)"
            result.kept.append(f"{ch.what}: seit dem Anwenden geändert, bleibt {shown}")
            continue
        _set(work, ch.path, ch.before, absent=ch.before_absent)
        result.steps.append({"path": list(ch.path), "value": ch.before, "absent": ch.before_absent})
        before = "(nicht gesetzt)" if ch.before_absent else format_value(ch.path, ch.before)
        result.restored.append(f"{ch.what}: wieder {before}")
    work.applied_presets = ledger[:-1]
    if not _save(cm, work, changes):
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
    cache = data.get("cache")
    if isinstance(cache, dict) and isinstance(cache.get("tables"), list):
        ids = [str(t.get("id")) for t in cache["tables"] if isinstance(t, dict) and t.get("id")]
        if ids:
            out.append(f"Cache (ISIL {cache.get('isil') or '(fehlt)'}): " + ", ".join(ids))
    names = [str(p["name"]) for p in data.get("providers") or []
             if isinstance(p, dict) and isinstance(p.get("name"), str) and p["name"].strip()]
    if names:
        out.append("Provider im Preset (der Assistent legt sie nicht an, nur die Vorbelegung; "
                   "anlegen: „Preset anwenden…“ oder python3 src/alima_cli.py preset apply): "
                   + ", ".join(names))
    return out


def apply_after_setup(preset_path: Any, config_manager: Any = None,
                      source: str = SOURCE_GIVEN) -> List[str]:
    """Apply institution, workflow, plugins, files and cache of a preset after
    the setup wizard; returns lines for the log or the terminal. Never raises: a
    broken preset must not stop the wizard. The wizard saves its database
    setting before calling this, so the cache goes into that database. The
    blocks ``providers`` and ``default_models`` are left out (``llm=False``):
    the wizard has just set up its provider and models from the user's
    answers, and the preset only pre-filled them. - Claude Generated"""
    if not preset_path:
        return []
    try:
        preset = load_preset(preset_path, source)
        plan = build_plan(preset, config_manager, llm=False)
        if plan.errors:
            lines = [f"Preset {plan.label} nicht angewendet:"] + [f"  {e}" for e in plan.errors]
            logger.warning("\n".join(lines))
            return lines
        if plan.is_empty:
            return [f"Preset {plan.label}: nichts zu ändern"]
        entry = apply_plan(plan, config_manager)
        lines = [f"Preset {plan.label} {plan.version}".rstrip() + " angewendet:"]
        lines += [line for line in plan.lines()[1:] if not line.startswith("Datei: ")]
        if (entry or {}).get("cache"):
            from .preset_cache import record_lines

            lines += ["Ergebnis:"] + [f"  {line}" for line in record_lines(entry["cache"])]
        logger.info("\n".join(lines))
        return lines
    except Exception as exc:  # unreadable file, disk full, a defect
        from .error_visibility import log_caught

        log_caught(logger, exc, "Preset nach dem Einrichtungsassistenten")
        return [f"Preset nicht angewendet: {exc}"]
