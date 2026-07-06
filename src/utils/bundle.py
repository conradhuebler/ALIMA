"""Institutional bundle deploy: plugins + advisory config profile - Claude Generated.

A *bundle* packages an institution's ALIMA setup so it can be installed on many
workstations with one command:

    <bundle>/
      bundle.toml     # [bundle] id/version/label/institution/alima_min_version
                      # [secrets] required = [{plugin=…, key=…, hint=…}]
      plugins/        # Tier-1 declarative (endpoints) + Tier-2 code plugin dirs
      profile.json    # advisory config overlay — WHITELISTED, non-secret keys only
      approvals.json  # {plugin_id: sha256} — admin-vouched (optional; verified on install)

Design (locked with the operator):

* **Native installer** — ``alima bundle {build,install,list,remove}`` drives these
  pure, Qt-free functions (a GUI action can reuse them later).
* **Advisory** — the profile *seeds* config; the user may change everything after.
  ``remove`` restores exactly the keys the bundle set (recorded in the ledger).
* **Per-user secrets** — a bundle never carries tokens/keys. It *declares* which
  per-user secrets are needed; the installer reports them (GUI Plugins tab or the
  ``ALIMA_PLUGIN_<ID>_<KEY>`` env override). Plugins with a gating secret stay
  "unavailable" until filled — no bundle code needed.

The profile whitelist is the safety boundary: ``unified_config`` (LLM providers +
API keys), ``database_config`` and anything secret-shaped are rejected, so a
bundle can never clobber a user's own credentials.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # 3.11+ stdlib; tomli fallback mirrors src/core/plugins/manifest.py
    import tomllib as _toml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# --- profile.json whitelist (the user-config protection boundary) ----------
PROFILE_ALLOWED_SECTIONS = {"search_provider_config", "system_config"}
SYSTEM_ALLOWED_KEYS = {
    "url_fetch_allowlist",
    "url_fetch_max_bytes",
    "enable_code_plugins",
    "enable_response_cache",
    "aggregate_from_raw",
    "enable_dk_splitting",
    "dk_split_threshold",
}
_SECRET_SUBSTR = ("api_key", "apikey", "token", "secret", "password")


class BundleError(Exception):
    """A bundle is malformed or its profile touches forbidden config."""


@dataclass
class BundleManifest:
    id: str
    version: str = "0"
    label: str = ""
    institution: str = ""
    alima_min_version: str = ""
    required_secrets: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class InstallReport:
    bundle_id: str
    version: str
    label: str = ""
    plugins: List[Tuple[str, str, str]] = field(default_factory=list)  # (id, status, severity)
    profile_keys: List[str] = field(default_factory=list)
    required_secrets: List[Dict[str, Any]] = field(default_factory=list)  # + env_var/satisfied
    integrity_mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# manifest / file readers
# ---------------------------------------------------------------------------
def _read_manifest(bundle_dir: Path) -> BundleManifest:
    f = bundle_dir / "bundle.toml"
    if not f.is_file():
        raise BundleError(f"no bundle.toml in {bundle_dir}")
    with open(f, "rb") as fh:
        data = _toml.load(fh)
    b = data.get("bundle", data) or {}
    bid = b.get("id")
    if not isinstance(bid, str) or not _ID_RE.match(bid):
        raise BundleError(f"bundle.toml: invalid or missing id '{bid}'")
    required = []
    sec = data.get("secrets", {}) or {}
    for item in sec.get("required", []) or []:
        if isinstance(item, dict) and item.get("plugin") and item.get("key"):
            required.append({
                "plugin": str(item["plugin"]),
                "key": str(item["key"]),
                "hint": str(item.get("hint", "")),
            })
    return BundleManifest(
        id=bid,
        version=str(b.get("version", "0")),
        label=str(b.get("label", bid)),
        institution=str(b.get("institution", "")),
        alima_min_version=str(b.get("alima_min_version", "")),
        required_secrets=required,
    )


def _plugin_id(plugin_dir: Path) -> Optional[str]:
    f = plugin_dir / "plugin.toml"
    if not f.is_file():
        return None
    with open(f, "rb") as fh:
        data = _toml.load(fh)
    return (data.get("plugin", data) or {}).get("id")


def _plugin_is_code(plugin_dir: Path) -> bool:
    f = plugin_dir / "plugin.toml"
    with open(f, "rb") as fh:
        data = _toml.load(fh)
    return str((data.get("plugin", data) or {}).get("type", "")) == "code"


def _bundle_plugin_dirs(bundle_dir: Path) -> List[Path]:
    root = bundle_dir / "plugins"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "plugin.toml").is_file())


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise BundleError(f"{path.name}: invalid JSON ({exc})")


# ---------------------------------------------------------------------------
# build (admin side)
# ---------------------------------------------------------------------------
def build_bundle(src_dir: str | Path, out_path: Optional[str | Path] = None) -> Path:
    """Validate a bundle dir, write ``approvals.json`` (SHA-256 per plugin), and
    optionally zip it. Returns the built artifact path (dir or .zip). - Claude Generated"""
    from src.core.plugins.security import hash_dir

    src = Path(src_dir).resolve()
    manifest = _read_manifest(src)  # validates
    approvals: Dict[str, str] = {}
    for pdir in _bundle_plugin_dirs(src):
        pid = _plugin_id(pdir)
        if not pid:
            raise BundleError(f"plugin dir '{pdir.name}' has no id in plugin.toml")
        if pid in approvals:
            raise BundleError(f"duplicate plugin id '{pid}' in bundle")
        approvals[pid] = hash_dir(pdir)
    (src / "approvals.json").write_text(
        json.dumps(approvals, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Built bundle '%s' v%s (%d plugins)", manifest.id, manifest.version, len(approvals))

    if out_path is None:
        return src
    out = Path(out_path)
    if out.suffix != ".zip":
        raise BundleError("out_path must end in .zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                zf.write(p, p.relative_to(src).as_posix())  # bundle.toml at zip root
    return out


# ---------------------------------------------------------------------------
# install (workstation side)
# ---------------------------------------------------------------------------
def _resolve_source(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Return (bundle_dir, tmp_holder). Extract a .zip to a temp dir if needed."""
    if path.is_dir():
        return path, None
    if path.suffix == ".zip" and path.is_file():
        tmp = tempfile.TemporaryDirectory(prefix="alima_bundle_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp.name)
        root = Path(tmp.name)
        if (root / "bundle.toml").is_file():
            return root, tmp
        subs = [d for d in root.iterdir() if d.is_dir() and (d / "bundle.toml").is_file()]
        if len(subs) == 1:
            return subs[0], tmp
        tmp.cleanup()
        raise BundleError("zip does not contain a bundle.toml at its root")
    raise BundleError(f"not a bundle directory or .zip: {path}")


def _apply_profile(config: Any, profile: dict) -> Dict[str, Any]:
    """Overlay whitelisted profile keys; return a restore map for clean removal.

    Fail-closed: any section/key outside the whitelist (or secret-shaped) aborts
    the install so a bundle can never overwrite user credentials. - Claude Generated"""
    for section in profile:
        if section not in PROFILE_ALLOWED_SECTIONS:
            raise BundleError(
                f"profile.json: section '{section}' is not allowed "
                f"(allowed: {sorted(PROFILE_ALLOWED_SECTIONS)})"
            )
    restore: Dict[str, Any] = {}

    spc = profile.get("search_provider_config")
    if isinstance(spc, dict):
        # ``search_provider_config.providers`` is a *derived mirror* of the per-
        # instance enabled flags (recomputed from instances on every save), so we
        # must toggle the matching instances — the authoritative state — not the
        # gate itself (which would be clobbered on save). We snapshot each touched
        # instance's prior enabled flag for a precise ``remove``. - Claude Generated
        providers = spc.get("providers", {}) or {}
        prev_instances: Dict[str, bool] = {}
        for pid, enabled in providers.items():
            for inst in config.plugins:
                if inst.category == "search_provider" and inst.provider_id == pid:
                    prev_instances.setdefault(inst.instance_id, inst.enabled)
                    inst.enabled = bool(enabled)
            config.search_provider_config.set_enabled(pid, bool(enabled))  # mirror (re-derived)
        restore["search_provider_instances"] = prev_instances

    sysc = profile.get("system_config")
    if isinstance(sysc, dict):
        prev = {}
        for k, v in sysc.items():
            if k not in SYSTEM_ALLOWED_KEYS or any(s in k.lower() for s in _SECRET_SUBSTR):
                raise BundleError(f"profile.json: system_config key '{k}' is not allowed")
            prev[k] = getattr(config.system_config, k, None)
            setattr(config.system_config, k, v)
        restore["system_config"] = prev

    return restore


def install_bundle(path: str | Path, *, config_manager: Any = None) -> InstallReport:
    """Install a bundle (dir or .zip): copy plugins, pre-approve, overlay profile,
    record provenance. Advisory — the user may change everything afterwards."""
    from src.core.plugins.security import hash_dir, max_severity, scan_dir
    from src.utils.config_manager import ConfigManager
    from src.utils.plugin_discovery import discover_plugins

    src, tmp = _resolve_source(Path(path).resolve())
    try:
        manifest = _read_manifest(src)
        cm = config_manager or ConfigManager()
        config = cm.load_config()

        report = InstallReport(manifest.id, manifest.version, manifest.label)

        # 1) copy plugin dirs into the user plugins dir
        dest_root = cm.plugins_dir
        dest_root.mkdir(parents=True, exist_ok=True)
        bundle_dirs = _bundle_plugin_dirs(src)
        plugin_ids: List[str] = []
        plugin_dirnames: List[str] = []
        has_code = False
        for pdir in bundle_dirs:
            pid = _plugin_id(pdir)
            plugin_ids.append(pid)
            plugin_dirnames.append(pdir.name)
            has_code = has_code or _plugin_is_code(pdir)
            dst = dest_root / pdir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(pdir, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        # 2) approve exactly what landed on disk; verify against admin approvals
        expected = _read_json(src / "approvals.json")
        for pid, dirname in zip(plugin_ids, plugin_dirnames):
            digest = hash_dir(dest_root / dirname)
            if pid in expected and expected[pid] != digest:
                report.integrity_mismatches.append(pid)

        findings_by_id: Dict[str, list] = {}

        def approve_cb(m, findings, digest):  # institutional trust: auto-approve
            findings_by_id[m.id] = findings
            return True

        result = discover_plugins(
            config,
            plugins_dir=dest_root,
            approve_cb=approve_cb,
            enable_code_plugins=True,
        )
        loaded = {p.type_id or p.manifest.id: p for p in (result.plugins if result else [])}
        for pid in plugin_ids:
            p = loaded.get(pid)
            status = p.status if p else "unknown"
            sev = max_severity(findings_by_id.get(pid, [])) if pid in findings_by_id else "none"
            report.plugins.append((pid, status, sev))

        # 3) advisory profile overlay (whitelisted)
        profile = _read_json(src / "profile.json")
        restore = _apply_profile(config, profile)
        report.profile_keys = sorted(profile.keys())

        if has_code and not getattr(config.system_config, "enable_code_plugins", False):
            report.warnings.append(
                "bundle ships code plugins but profile.json does not set "
                "system_config.enable_code_plugins=true — they load now but not on restart"
            )

        # 4) instance ids this bundle owns (seeded by discover). The seeded
        # instance_id equals the plugin's declared id for both declarative and
        # code plugins (provider_id is the *kind* for declarative), so match on
        # instance_id, not provider_id. - Claude Generated
        pid_set = set(plugin_ids)
        instance_ids = [
            i.instance_id for i in config.plugins
            if i.category == "search_provider" and i.instance_id in pid_set
        ]

        # 5) required per-user secrets (declared, never shipped)
        report.required_secrets = _secret_status(config, manifest.required_secrets)

        # 6) provenance ledger for precise remove
        config.installed_bundles[manifest.id] = {
            "version": manifest.version,
            "label": manifest.label,
            "institution": manifest.institution,
            "plugin_ids": plugin_ids,
            "plugin_dirs": plugin_dirnames,
            "instance_ids": instance_ids,
            "profile_keys": report.profile_keys,
            "profile_restore": restore,
            "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cm.save_config(config)
        return report
    finally:
        if tmp is not None:
            tmp.cleanup()


def _secret_status(config: Any, required: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    from src.core.plugins.schema import env_var_name

    by_instance = {i.instance_id: i for i in config.plugins}
    out = []
    for req in required:
        plugin, key = req["plugin"], req["key"]
        env = env_var_name(plugin, key)
        inst = by_instance.get(plugin)
        in_settings = bool((inst.settings or {}).get(key)) if inst else False
        out.append({
            "plugin": plugin,
            "key": key,
            "hint": req.get("hint", ""),
            "env_var": env,
            "satisfied": bool(os.environ.get(env)) or in_settings,
        })
    return out


# ---------------------------------------------------------------------------
# list / remove (workstation side)
# ---------------------------------------------------------------------------
def list_bundles(*, config_manager: Any = None) -> List[Dict[str, Any]]:
    from src.utils.config_manager import ConfigManager

    cm = config_manager or ConfigManager()
    config = cm.load_config()
    out = []
    for bid, rec in sorted(config.installed_bundles.items()):
        out.append({
            "id": bid,
            "version": rec.get("version", "?"),
            "label": rec.get("label", bid),
            "plugins": rec.get("plugin_ids", []),
            "installed_at": rec.get("installed_at", ""),
        })
    return out


def remove_bundle(bundle_id: str, *, config_manager: Any = None) -> None:
    """Reverse an install via its ledger: drop instances/approvals/plugin dirs and
    restore the profile keys the bundle set. Advisory — leaves other config alone."""
    from src.utils.config_manager import ConfigManager

    cm = config_manager or ConfigManager()
    config = cm.load_config()
    rec = config.installed_bundles.get(bundle_id)
    if rec is None:
        raise BundleError(f"no installed bundle '{bundle_id}'")

    instance_ids = set(rec.get("instance_ids", []))
    plugin_ids = set(rec.get("plugin_ids", []))
    config.plugins = [p for p in config.plugins if p.instance_id not in instance_ids]
    for pid in plugin_ids:
        config.approved_plugins.pop(pid, None)

    # restore whitelisted profile keys to their pre-install values
    restore = rec.get("profile_restore", {}) or {}
    inst_prev = restore.get("search_provider_instances")
    if isinstance(inst_prev, dict):
        by_id = {p.instance_id: p for p in config.plugins}
        for iid, prev in inst_prev.items():
            if iid in by_id:
                by_id[iid].enabled = bool(prev)
    sysc = restore.get("system_config")
    if isinstance(sysc, dict):
        for k, prev in sysc.items():
            setattr(config.system_config, k, prev)

    del config.installed_bundles[bundle_id]
    cm.save_config(config)

    # delete the copied plugin dirs last (config already consistent)
    dest_root = cm.plugins_dir
    for dirname in rec.get("plugin_dirs", []):
        shutil.rmtree(dest_root / dirname, ignore_errors=True)
