"""Export the running setup as an institution package - Claude Generated.

:func:`export_preset` writes one folder: ``alima_presets.json`` (format 2) and
the files the chosen plugin instances read (the database of ``geo_sachdb``,
the vocabulary of ``local_scheme``), each with its SHA-256. Paths in the
preset are relative; a file setting becomes ``@file:<id>``. Written are: the
chosen instances in list order with state, name and the settings they store
(the vocabulary of ``local_scheme`` also when only its default is in use),
ISIL/ELN, the standard workflow, the LLM address and models and a server
database as wizard pre-fill.

Never written: SECRET fields and keys named like an API key or token (a used
secret is declared under ``secrets`` instead), a database password, the path of
a local SQLite database, any other setting holding a local path, settings the
plugin type does not declare. Each omission is reported in
``ExportResult.warnings``. The export reads the saved configuration unless a
``config`` is passed. Qt-free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .preset_apply import (
    FILE_REF_PREFIX,
    FORMAT_VERSION,
    KIND_GEO_DB,
    KIND_VOCABULARY,
    PRESET_FILENAME,
    PresetError,
    _SECRET_KEY,
    _cm,
    _ensure_categories,
    sha256_file,
)

logger = logging.getLogger(__name__)

#: Settings that name a file the package ships: (type, key) → file kind.
FILE_SETTINGS: Dict[Tuple[str, str], str] = {
    ("geo_sachdb", "db_path"): KIND_GEO_DB,
    ("local_scheme", "vocabulary"): KIND_VOCABULARY,
}

#: Folder inside the package that holds the shipped files.
FILES_DIR = "files"

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass
class ExportResult:
    """What :func:`export_preset` wrote - Claude Generated"""

    path: Path
    preset_file: Path
    instances: List[str] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        out = [f"Paket: {self.path}",
               f"Instanzen ({len(self.instances)}): {', '.join(self.instances) or '(keine)'}"]
        for f in self.files:
            out.append(f"Datei: {f['id']} → {f['source']} (sha256 {f['sha256'][:12]}…)")
        out += [f"⚠ {w}" for w in self.warnings]
        return out


def exportable_instances(config: Any) -> List[Tuple[str, str, str, bool]]:
    """``(category, instance_id, label, enabled)`` in list order - Claude Generated"""
    return [(p.category, p.instance_id, p.display_label(), bool(p.enabled))
            for p in getattr(config, "plugins", []) or []]


def _looks_like_local_path(value: str) -> bool:
    text = value.strip()
    return bool(text) and (text.startswith("~") or os.path.isabs(text)
                           or bool(_WINDOWS_PATH.match(text)))


def _resolve_file_setting(provider_id: str, value: Any) -> Optional[Path]:
    """The file a file setting points to, ``None`` when there is none."""
    if provider_id == "local_scheme":
        from src.utils.lookups.local_scheme import LocalSchemeLookup

        return LocalSchemeLookup(vocabulary=value or "").vocabulary_path()
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_file() else None


def _export_instance(inst: Any, stage: Path, files: List[Dict[str, Any]],
                     secrets: List[Dict[str, str]], warnings: List[str]) -> Dict[str, Any]:
    from src.core.plugins import get_category
    from src.core.search.cache_only import (
        CACHE_ONLY_KEY, OFFER_CACHE_ONLY, OFFER_NONE, cache_only_offer, instance_state,
    )

    iid = inst.instance_id
    try:
        meta = get_category(inst.category).type_meta(inst.provider_id)
    except KeyError:
        meta = None
        warnings.append(f"{iid}: Typ '{inst.provider_id}' ist nicht geladen; exportiert wird "
                        "nur der Zustand")
    offer = cache_only_offer(meta) if meta is not None else OFFER_NONE
    entry: Dict[str, Any] = {"instance": iid, "category": inst.category,
                             "type": inst.provider_id, "state": instance_state(inst, offer)}
    if inst.label:
        entry["label"] = inst.label
    if meta is None:
        return entry
    fields = {f.key: f for f in meta.config_fields}
    keys = list(inst.settings)
    for (ptype, key) in FILE_SETTINGS:
        if ptype == inst.provider_id and key in fields and key not in keys:
            keys.append(key)  # the file in use by default is shipped too
    settings: Dict[str, Any] = {}
    for key in keys:
        fld = fields.get(key)
        if fld is None:
            warnings.append(f"{iid}: Einstellung '{key}' gehört nicht zum Typ "
                            f"'{inst.provider_id}'; nicht exportiert")
            continue
        value = inst.settings.get(key, fld.default)
        if fld.secret or _SECRET_KEY.search(key):
            from src.core.plugins.schema import env_var_name

            if value or os.environ.get(env_var_name(iid, key)):
                secrets.append({"plugin": iid, "key": key,
                                "hint": fld.help or fld.label or key})
            continue
        if key == CACHE_ONLY_KEY and offer == OFFER_CACHE_ONLY:
            continue  # carried by "state"
        kind = FILE_SETTINGS.get((inst.provider_id, key))
        if kind is not None:
            src = _resolve_file_setting(inst.provider_id, value)
            if src is None:
                if key in inst.settings and str(value or "").strip():
                    warnings.append(f"{iid}: Datei zu '{key}' nicht gefunden; Einstellung "
                                    "nicht exportiert")
                continue
            fid = f"{iid}_{key}"
            rel = Path(FILES_DIR) / iid / src.name
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, stage / rel)
            files.append({"id": fid, "kind": kind, "source": rel.as_posix(),
                          "target": f"{iid}/{src.name}", "sha256": sha256_file(stage / rel)})
            settings[key] = FILE_REF_PREFIX + fid
            continue
        if isinstance(value, str) and _looks_like_local_path(value):
            warnings.append(f"{iid}: '{key}' enthält einen lokalen Pfad; nicht exportiert")
            continue
        settings[key] = value
    if settings:
        entry["settings"] = settings
    return entry


def _llm_block(config: Any) -> Dict[str, Any]:
    """Provider type, name, address and models of the general default
    provider, without its key (wizard pre-fill) - Claude Generated"""
    from .config_models import ProviderScope

    unified = config.unified_config
    providers = list(getattr(unified, "providers", None) or [])
    preferred = getattr(unified, "preferred_provider", "")
    provider = (next((p for p in providers if p.name == preferred and getattr(p, "enabled", True)), None)
                or next((p for p in providers if getattr(p, "enabled", True)), None))
    block: Dict[str, Any] = {}
    if provider is None:
        return block
    block["provider_type"] = provider.provider_type
    block["provider_name"] = provider.name
    if getattr(provider, "base_url", ""):
        block["base_url"] = provider.base_url
    if getattr(provider, "preferred_model", ""):
        block["default_model"] = provider.preferred_model
    task_models: Dict[str, str] = {}
    for name, scope in (("pipeline", ProviderScope.PIPELINE), ("vision", ProviderScope.VISION)):
        try:
            _prov, model = unified.resolve_default_provider_model(scope=scope)
        except Exception as exc:  # resolution reads the provider list only
            from .error_visibility import log_caught

            log_caught(logger, exc, f"Modell für {name} beim Preset-Export")
            model = ""
        if model:
            task_models[name] = model
    if task_models:
        block["task_models"] = task_models
    return block


def _database_block(config: Any, warnings: List[str]) -> Dict[str, Any]:
    """A server database (MariaDB/MySQL) without password; nothing for SQLite,
    whose path is local to the workstation - Claude Generated"""
    db = config.database_config
    db_type = str(getattr(db, "db_type", "sqlite") or "sqlite").lower()
    if db_type not in ("mysql", "mariadb"):
        return {}
    block: Dict[str, Any] = {"type": db_type}
    for key in ("host", "database", "username", "charset"):
        if getattr(db, key, ""):
            block[key] = getattr(db, key)
    if getattr(db, "port", 3306) != 3306:
        block["port"] = db.port
    if getattr(db, "password", ""):
        warnings.append("Datenbank: das Passwort wird nicht exportiert")
    return block


def export_preset(config_manager: Any, dest_dir: Any, instance_ids: Optional[List[str]] = None,
                  *, preset_id: str = "", version: str = "", label: str = "",
                  config: Any = None) -> ExportResult:
    """Write the package folder ``dest_dir`` (new or empty).

    ``instance_ids``: the instances to include, in list order; ``None`` = all.
    Raises :class:`PresetError` for a non-empty folder or an unknown instance
    id. - Claude Generated
    """
    cm = _cm(config_manager)
    config = config if config is not None else cm.load_config()
    _ensure_categories()
    dest = Path(dest_dir).expanduser()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise PresetError(f"{dest}: der Ordner ist nicht leer; bitte einen neuen oder leeren Ordner wählen")
    plugins = list(getattr(config, "plugins", []) or [])
    if instance_ids is None:
        selected = plugins
    else:
        wanted = list(dict.fromkeys(str(i) for i in instance_ids))
        unknown = [i for i in wanted if i not in {p.instance_id for p in plugins}]
        if unknown:
            raise PresetError(f"Unbekannte Instanz(en): {', '.join(unknown)}")
        selected = [p for p in plugins if p.instance_id in set(wanted)]

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=str(dest.parent)))
    try:
        os.chmod(stage, 0o755)  # mkdtemp creates 0700; the package is meant to be shared
        files: List[Dict[str, Any]] = []
        secrets: List[Dict[str, str]] = []
        warnings: List[str] = []
        entries = [_export_instance(inst, stage, files, secrets, warnings) for inst in selected]
        pid = (preset_id or "").strip() or dest.name
        data: Dict[str, Any] = {
            "_comment": "ALIMA-Einrichtungspaket. Anwenden: alima preset apply <Ordner> "
                        "(Vorschau mit --dry-run). Enthält keine Schlüssel und keine Tokens.",
            "format": FORMAT_VERSION,
            "preset": {"id": pid, "version": (version or "").strip() or "1",
                       "label": (label or "").strip() or pid},
        }
        system = config.system_config
        isil = str(getattr(system, "institution_isil", "") or "").strip()
        eln = str(getattr(system, "institution_eln", "") or "").strip()
        if isil or eln:
            data["institution"] = {"isil": isil, "eln": eln}
        workflow = str(getattr(system, "default_workflow", "") or "").strip()
        if workflow:
            data["workflow"] = workflow
        llm = _llm_block(config)
        if llm:
            data["llm"] = llm
        database = _database_block(config, warnings)
        if database:
            data["database"] = database
        if entries:
            data["plugins"] = entries
        if files:
            data["files"] = files
        if secrets:
            data["secrets"] = secrets
        (stage / PRESET_FILENAME).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if dest.exists():
            dest.rmdir()  # checked empty above
        os.replace(stage, dest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    logger.info("Preset exportiert: %s (%d Instanzen, %d Dateien)", dest, len(entries), len(files))
    return ExportResult(dest, dest / PRESET_FILENAME, [e["instance"] for e in entries],
                        files, warnings)
