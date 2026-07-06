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

import importlib
import logging
import sys
import types
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


def _plugin_package_name(plugin_id: str) -> str:
    """The synthetic ``sys.modules`` package name for one plugin id - Claude Generated"""
    return "alima_plugin_" + plugin_id.replace("-", "_")


def _static_class_id(entry_file: Path, class_name: str) -> Optional[str]:
    """AST-read the literal ``id = "..."`` from the entry class, without executing
    any plugin code. ``None`` when the file/class/attribute can't be read
    statically (dynamic ids fall back to the post-import check). - Claude Generated
    """
    try:
        import ast

        tree = ast.parse(entry_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "id" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    return stmt.value.value
    return None


def _import_entry_class(plugin_dir: Path, manifest: PluginManifest) -> type:
    """Load a plugin dir as a synthetic package, return the entry class - Claude Generated.

    Package-style loading: a stub package ``alima_plugin_<id>`` with
    ``__path__ = [plugin_dir]`` is placed in ``sys.modules``, then the entry
    module is imported as a submodule — so a multi-file plugin can use
    single-level relative imports (``from .suggester import X``). The plugin's
    ``__init__.py`` is *never* executed (it is built-in-mode glue only).

    Hardening (still in-process and unsandboxed, see module docstring):
    the entry file must not be a symlink and must resolve to a file directly
    inside the plugin dir; the loaded class's ``id`` must equal the manifest id;
    a failed import removes every partially-registered ``alima_plugin_<id>*``
    module from ``sys.modules``.
    """
    root = Path(plugin_dir).resolve()
    entry = Path(plugin_dir) / manifest.entry_module
    if entry.is_symlink():
        raise ImportError(f"entry module '{manifest.entry_module}' is a symlink — not allowed")
    resolved = entry.resolve()
    if not resolved.is_file() or resolved.parent != root:
        raise ImportError(
            f"entry module '{manifest.entry_module}' not found inside the plugin directory"
        )
    # Pre-import consistency: a blueprint copy where only plugin.toml was
    # renamed would otherwise run @register_provider with the OLD class id and
    # blow up on the registry collision — detect the mismatch statically and
    # refuse BEFORE executing any plugin code. - Claude Generated
    static_id = _static_class_id(resolved, manifest.entry_class)
    if static_id is not None and static_id != manifest.id:
        raise ImportError(
            f"{manifest.entry_module}: class '{manifest.entry_class}' still has "
            f"id = \"{static_id}\", but plugin.toml says id = \"{manifest.id}\" — "
            f"set the class attribute to id = \"{manifest.id}\" (both must be "
            "identical; nothing was imported)"
        )
    pkg_name = _plugin_package_name(manifest.id)
    module_name = f"{pkg_name}.{resolved.stem}"
    module = sys.modules.get(module_name)
    if module is None:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root)]  # anchors `from .x import y` inside the dir
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        try:
            module = importlib.import_module(module_name)
        except BaseException:
            for name in [
                m for m in sys.modules if m == pkg_name or m.startswith(pkg_name + ".")
            ]:
                sys.modules.pop(name, None)
            raise
    cls = getattr(module, manifest.entry_class, None)
    if cls is None:
        raise ImportError(f"{manifest.entry_module} has no class '{manifest.entry_class}'")
    if not isinstance(cls, type):
        raise ImportError(
            f"'{manifest.entry_class}' in {manifest.entry_module} is not a class"
        )
    cls_id = getattr(cls, "id", None)
    if cls_id != manifest.id:
        raise ImportError(
            f"plugin class id '{cls_id}' does not match manifest id '{manifest.id}' — "
            "rename the class `id` attribute to match plugin.toml"
        )
    return cls


def _seed_instance(manifest: PluginManifest, type_id: str) -> "Any":
    """Build the PluginInstanceConfig for a discovered plugin - Claude Generated.

    Declarative: ``type_id`` is the built-in ``kind``. Code: ``type_id`` is the
    freshly registered type — without this seed a loaded code plugin would be
    invisible (no list entry, no MCP tool), since everything downstream is
    instance-driven.
    """
    from src.utils.config_models import PluginInstanceConfig  # lazy: avoid cycle

    category = get_category(manifest.category)
    try:
        fields = category.config_fields(type_id)
    except Exception:
        fields = []
    settings = coerce_settings(fields, manifest.settings)
    return PluginInstanceConfig(
        instance_id=manifest.id,
        category=manifest.category,
        provider_id=type_id,
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
            result.instances.append(_seed_instance(manifest, manifest.kind))
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
            # Re-seed the instance so a config that lost it heals (dedup is the
            # caller's job). - Claude Generated
            result.instances.append(_seed_instance(manifest, manifest.id))
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
            cls = _import_entry_class(sub, manifest)
            type_id = category.register_code_type(cls)
            _LOADED_CODE_PLUGINS[manifest.id] = digest
        except Exception as exc:  # import or registration failure
            logger.exception("Code plugin '%s' failed to load", manifest.id)
            detail = str(exc)
            if isinstance(exc, ValueError) and "already registered" in detail:
                detail += (
                    " — the plugin id collides with an existing type; rename the "
                    "plugin id (in plugin.toml AND the class `id` attribute)"
                )
            result.plugins.append(
                LoadedPlugin(manifest=manifest, status="error", detail=detail, findings=findings)
            )
            continue

        # Seed the instance for the new type — this is what makes the plugin
        # visible (settings list) and usable (MCP tools are per-instance).
        # - Claude Generated
        result.instances.append(_seed_instance(manifest, type_id))
        result.plugins.append(
            LoadedPlugin(manifest=manifest, status="loaded", findings=findings, type_id=type_id)
        )

    return result


def _stub_manifest(name: str) -> PluginManifest:
    return PluginManifest(id=name, label=name, category="?", type="?")


def _reset_for_tests() -> None:
    """Clear the loaded-code-plugin cache + synthetic plugin modules (tests only)."""
    _LOADED_CODE_PLUGINS.clear()
    for name in [m for m in sys.modules if m.startswith("alima_plugin_")]:
        sys.modules.pop(name, None)
