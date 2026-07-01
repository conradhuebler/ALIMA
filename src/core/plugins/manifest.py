"""Plugin manifest (``plugin.toml``) parsing + validation - Claude Generated.

A directory plugin describes itself in a ``plugin.toml`` manifest. Two shapes:

* **declarative** (Tier-1, safe): ``kind`` references a *built-in type* of the
  named ``category`` and ``settings`` supplies its values. No code is loaded.
* **code** (Tier-2, consent-gated): ``entry.module`` / ``entry.class`` point at a
  Python class implementing the category's contract. Loaded only after the
  security gate (see :mod:`.security` + :mod:`.loader`).

Validation here is *structural only* — it never imports plugin code. Whether a
declarative ``kind`` actually exists is checked by the owning category adapter;
whether code is safe/approved is the loader's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore
    _TOML_BINARY = True
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    import tomli as _toml  # type: ignore
    _TOML_BINARY = True

SUPPORTED_API_VERSIONS = {"1"}
_DECLARATIVE = "declarative"
_CODE = "code"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ManifestError(ValueError):
    """Raised when a manifest is missing/malformed/unsupported."""


@dataclass
class PluginManifest:
    """Validated contents of a ``plugin.toml``."""

    id: str
    label: str
    category: str
    type: str  # "declarative" | "code"
    api_version: str = "1"
    capabilities: List[str] = field(default_factory=list)
    usage_hint: str = ""
    # Natural-language self-description (design requirement: a plugin explains
    # itself in prose + documents its input/output, it is not left implicit in
    # code). For a declarative plugin these may be empty → the referenced built-in
    # type's ``doc()`` is used as the fallback.
    description: str = ""
    input: str = ""
    output: str = ""
    # declarative
    kind: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    # code
    entry_module: str = ""
    entry_class: str = ""
    # set by the loader when read from disk
    source_dir: Optional[Path] = None

    @property
    def is_declarative(self) -> bool:
        return self.type == _DECLARATIVE

    @property
    def is_code(self) -> bool:
        return self.type == _CODE


def parse_manifest(data: Dict[str, Any], *, source_dir: Optional[Path] = None) -> PluginManifest:
    """Build a :class:`PluginManifest` from a parsed-TOML dict. Validates shape."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a table/mapping")
    plugin = data.get("plugin", data)  # allow both top-level and [plugin] table

    def _req(key: str) -> str:
        val = plugin.get(key)
        if not val or not isinstance(val, str):
            raise ManifestError(f"manifest field '{key}' is required and must be a string")
        return val

    pid = _req("id")
    if not _ID_RE.match(pid):
        raise ManifestError(
            f"plugin id '{pid}' invalid — use lowercase letters, digits, '_' or '-'"
        )
    category = _req("category")
    ptype = _req("type")
    if ptype not in (_DECLARATIVE, _CODE):
        raise ManifestError(f"plugin type '{ptype}' must be '{_DECLARATIVE}' or '{_CODE}'")

    api_version = str(plugin.get("api_version", "1"))
    if api_version not in SUPPORTED_API_VERSIONS:
        raise ManifestError(
            f"api_version '{api_version}' unsupported (supported: {sorted(SUPPORTED_API_VERSIONS)})"
        )

    label = plugin.get("label") or pid
    caps = list(plugin.get("capabilities", []) or [])
    usage_hint = str(plugin.get("usage_hint", "") or "")
    doc = data.get("doc", plugin.get("doc", {})) or {}

    manifest = PluginManifest(
        id=pid,
        label=str(label),
        category=category,
        type=ptype,
        api_version=api_version,
        capabilities=caps,
        usage_hint=usage_hint,
        description=str(plugin.get("description", doc.get("description", "")) or ""),
        input=str(doc.get("input", plugin.get("input", "")) or ""),
        output=str(doc.get("output", plugin.get("output", "")) or ""),
        source_dir=Path(source_dir) if source_dir else None,
    )

    if ptype == _DECLARATIVE:
        kind = plugin.get("kind")
        if not kind or not isinstance(kind, str):
            raise ManifestError("declarative plugin requires a string 'kind' (built-in type id)")
        manifest.kind = kind
        settings = data.get("settings", plugin.get("settings", {})) or {}
        if not isinstance(settings, dict):
            raise ManifestError("'settings' must be a table/mapping")
        manifest.settings = dict(settings)
    else:  # code
        entry = data.get("entry", plugin.get("entry", {})) or {}
        module = entry.get("module")
        cls = entry.get("class")
        if not module or not cls:
            raise ManifestError("code plugin requires [entry] with 'module' and 'class'")
        if ".." in str(module) or Path(str(module)).is_absolute():
            raise ManifestError("entry.module must be a relative path inside the plugin dir")
        manifest.entry_module = str(module)
        manifest.entry_class = str(cls)

    return manifest


def load_manifest_file(path: Path) -> PluginManifest:
    """Read + validate a ``plugin.toml`` file. Raises :class:`ManifestError`."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        with open(path, "rb") as fh:
            data = _toml.load(fh)
    except Exception as exc:  # tomllib raises TOMLDecodeError
        raise ManifestError(f"could not parse {path.name}: {exc}") from exc
    return parse_manifest(data, source_dir=path.parent)
