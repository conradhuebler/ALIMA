"""Directory plugin loader - Claude Generated.

Scans ``<plugins_root>/<name>/plugin.toml`` and turns each manifest into either:

* a **declarative** plugin → a pre-seeded :class:`PluginInstanceConfig` of an
  existing built-in *type* (no code executed), or
* a **code** plugin → a *new type* registered into the owning category's registry
  after the security gate (AST scan + trust-on-first-use hash + operator
  approval), then optionally a seed instance.

The loader is UI-agnostic: the approval decision is delegated to an injected
``approve_cb`` (the Qt dialog in the GUI, a prompt in the CLI, or ``None`` = deny
in headless contexts). It never persists config itself — it mutates the passed
``approved_plugins`` ledger and returns discovered instances for the caller to
merge + save.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .category import get_category
from .manifest import ManifestError, PluginManifest, load_manifest_file
from .schema import coerce_settings
from .security import ScanFinding, hash_dir, max_severity, scan_dir

logger = logging.getLogger(__name__)

# approve_cb(manifest, findings, digest) -> bool
ApproveCallback = Callable[[PluginManifest, List[ScanFinding], str], bool]

# Code plugins already imported+registered in THIS process (id -> loaded hash).
# A code plugin's class can not be cleanly re-imported into the same registry id,
# so once loaded we skip re-import on subsequent scans (unless the hash changed,
# which needs a process restart to pick up). - Claude Generated
_LOADED_CODE_PLUGINS: Dict[str, str] = {}


@dataclass
class LoadedPlugin:
    """Outcome of processing one plugin directory."""

    manifest: PluginManifest
    status: str  # "loaded" | "blocked" | "denied" | "error"
    detail: str = ""
    findings: List[ScanFinding] = field(default_factory=list)
    type_id: str = ""  # code plugins: the registered type id


@dataclass
class LoadResult:
    plugins: List[LoadedPlugin] = field(default_factory=list)
    instances: List["Any"] = field(default_factory=list)  # PluginInstanceConfig

    @property
    def loaded(self) -> List[LoadedPlugin]:
        return [p for p in self.plugins if p.status == "loaded"]


def _import_class_from_file(py_file: Path, class_name: str) -> type:
    """Import ``class_name`` from an arbitrary ``.py`` file (in-process, unsandboxed)."""
    spec = importlib.util.spec_from_file_location(f"alima_plugin_{py_file.stem}", py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec for {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ImportError(f"{py_file.name} has no class '{class_name}'")
    return getattr(module, class_name)


def _instance_from_declarative(manifest: PluginManifest) -> "Any":
    """Build a PluginInstanceConfig from a declarative manifest (no code)."""
    from src.utils.config_models import PluginInstanceConfig  # lazy: avoid cycle

    category = get_category(manifest.category)
    fields = category.config_fields(manifest.kind)
    settings = coerce_settings(fields, manifest.settings)
    return PluginInstanceConfig(
        instance_id=manifest.id,
        category=manifest.category,
        provider_id=manifest.kind,
        label=manifest.label,
        enabled=True,
        is_primary=False,
        usage_hint=manifest.usage_hint,
        settings=settings,
    )


def discover(
    plugins_root: Path,
    *,
    approved_plugins: Optional[Dict[str, str]] = None,
    approve_cb: Optional[ApproveCallback] = None,
    enable_code_plugins: bool = False,
) -> LoadResult:
    """Scan ``plugins_root`` and load every valid plugin.

    ``approved_plugins`` (id -> approved hash) is read *and updated in place* on
    fresh approvals; the caller persists it. Missing dir / no plugins → empty
    result (never raises).
    """
    result = LoadResult()
    root = Path(plugins_root)
    if not root.is_dir():
        return result
    approved = approved_plugins if approved_plugins is not None else {}

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = sub / "plugin.toml"
        if not manifest_path.is_file():
            continue
        try:
            manifest = load_manifest_file(manifest_path)
        except ManifestError as exc:
            logger.warning("Plugin '%s' has an invalid manifest: %s", sub.name, exc)
            result.plugins.append(
                LoadedPlugin(manifest=_stub_manifest(sub.name), status="error", detail=str(exc))
            )
            continue

        try:
            category = get_category(manifest.category)
        except KeyError as exc:
            result.plugins.append(LoadedPlugin(manifest=manifest, status="error", detail=str(exc)))
            continue

        if manifest.is_declarative:
            if not category.has_type(manifest.kind):
                result.plugins.append(
                    LoadedPlugin(
                        manifest=manifest,
                        status="error",
                        detail=f"unknown built-in type '{manifest.kind}' for category '{manifest.category}'",
                    )
                )
                continue
            result.instances.append(_instance_from_declarative(manifest))
            result.plugins.append(LoadedPlugin(manifest=manifest, status="loaded"))
            continue

        # --- code plugin (Tier 2) ---------------------------------------------
        findings = scan_dir(sub)
        if not enable_code_plugins:
            result.plugins.append(
                LoadedPlugin(
                    manifest=manifest,
                    status="blocked",
                    detail="code plugins disabled (enable_code_plugins=False)",
                    findings=findings,
                )
            )
            continue

        digest = hash_dir(sub)
        if _LOADED_CODE_PLUGINS.get(manifest.id) == digest:
            # Already imported+registered in this process at this exact hash.
            result.plugins.append(
                LoadedPlugin(manifest=manifest, status="loaded", detail="already loaded", findings=findings)
            )
            continue
        if approved.get(manifest.id) != digest:
            decision = bool(approve_cb(manifest, findings, digest)) if approve_cb else False
            if not decision:
                result.plugins.append(
                    LoadedPlugin(
                        manifest=manifest,
                        status="denied",
                        detail=f"awaiting approval (severity: {max_severity(findings)})",
                        findings=findings,
                    )
                )
                continue
            approved[manifest.id] = digest  # trust-on-first-use; caller persists

        try:
            cls = _import_class_from_file(sub / manifest.entry_module, manifest.entry_class)
            type_id = category.register_code_type(cls)
            _LOADED_CODE_PLUGINS[manifest.id] = digest
        except Exception as exc:  # import or registration failure
            logger.exception("Code plugin '%s' failed to load", manifest.id)
            result.plugins.append(
                LoadedPlugin(manifest=manifest, status="error", detail=str(exc), findings=findings)
            )
            continue

        result.plugins.append(
            LoadedPlugin(manifest=manifest, status="loaded", findings=findings, type_id=type_id)
        )

    return result


def _stub_manifest(name: str) -> PluginManifest:
    return PluginManifest(id=name, label=name, category="?", type="?")


def _reset_for_tests() -> None:
    """Clear the process-level loaded-code-plugin cache (tests only)."""
    _LOADED_CODE_PLUGINS.clear()
