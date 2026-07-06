#!/usr/bin/env python3
"""Proof-of-concept: run ALIMA entirely on *own* (external) plugins - Claude Generated.

This helper turns every built-in search provider into a user plugin under
``~/.config/alima/plugins/`` and flips the built-ins off, so the whole app is
driven by copyable, operator-owned plugin dirs — the thing the plugin system was
built to make possible.

What it does (``deploy``):

1. **generate** — copy each built-in blueprint
   ``src/core/search/providers/<name>/`` → ``<plugins_dir>/poc_<name>/`` and
   rename *only* the plugin ``id`` (``id = "<name>"`` in ``plugin.toml`` **and**
   the provider class). Tool names (``search_lobid`` …), ``source_label`` and the
   class name are left untouched, so the classic pipeline, the deterministic
   agentic functions (which call tools by name) and the raw-cache provenance all
   keep resolving.
2. **approve** — enable ``system_config.enable_code_plugins`` and pre-approve each
   copy headlessly (SHA-256 via the loader's trust-on-first-use ledger), so no GUI
   dialog is needed. The ``poc_*`` instances are auto-seeded *enabled*.
3. **disable built-ins** — set every built-in search instance ``enabled = False``
   (mirrored into ``search_provider_config``), leaving only the ``poc_*`` plugins live.

``--revert`` undoes all three: re-enables built-ins, drops the ``poc_*`` instances
and approvals, deletes the copied dirs.

Usage::

    python examples/plugins_poc/deploy_poc.py            # deploy + disable built-ins
    python examples/plugins_poc/deploy_poc.py --revert   # restore built-ins
    python examples/plugins_poc/deploy_poc.py --keep-builtins   # deploy alongside
    python examples/plugins_poc/deploy_poc.py --generate-into /tmp/x  # copies only

The de-hardcode that makes the *classic* pipeline follow the enabled plugins lives
in ``src/utils/pipeline_utils.py`` (``execute_gnd_search`` empty-intersection
fallback); this script only touches config + the plugins dir.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Make ``src`` importable when run as a script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 6 built-in search providers, in a stable order.
PROVIDER_NAMES = ["lobid", "swb", "catalog", "finc", "sru", "gnd_local"]
BUILTIN_DIR = _REPO_ROOT / "src" / "core" / "search" / "providers"
DEFAULT_PREFIX = "poc_"
_SEARCH_CATEGORY = "search_provider"


def _rewrite_id(text: str, old_id: str, new_id: str) -> str:
    """Rename the plugin id token only (``id = "<old>"`` → ``id = "<new>"``).

    Matches the ``[plugin] id`` key in ``plugin.toml`` and the class-level ``id``
    attribute in ``provider.py``. Deliberately *not* a blanket replace: tool names
    (``search_lobid``) and ``source_label`` stay as they are. - Claude Generated"""
    pattern = re.compile(r'id\s*=\s*"' + re.escape(old_id) + r'"')
    return pattern.sub(f'id = "{new_id}"', text)


def generate(dst_dir: Path, prefix: str = DEFAULT_PREFIX) -> List[Tuple[str, Path]]:
    """Copy all 6 built-in provider dirs into ``dst_dir`` with renamed ids.

    Returns ``[(new_id, path), …]``. Overwrites an existing ``poc_*`` copy so the
    generator is idempotent. - Claude Generated"""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: List[Tuple[str, Path]] = []
    for name in PROVIDER_NAMES:
        src = BUILTIN_DIR / name
        if not (src / "plugin.toml").is_file():
            raise FileNotFoundError(f"built-in blueprint missing: {src}")
        new_id = f"{prefix}{name}"
        dst = dst_dir / new_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for fname in ("plugin.toml", "provider.py"):
            f = dst / fname
            if f.is_file():
                f.write_text(
                    _rewrite_id(f.read_text(encoding="utf-8"), name, new_id),
                    encoding="utf-8",
                )
        out.append((new_id, dst))
    return out


def _approve_all(*_args) -> bool:
    """Headless approval callback: trust every discovered POC plugin."""
    return True


def deploy(prefix: str = DEFAULT_PREFIX, disable_builtins: bool = True) -> None:
    """Generate the POC plugins into the real plugins dir, approve, disable built-ins."""
    from src.utils.config_manager import ConfigManager
    from src.utils.plugin_discovery import discover_plugins

    cm = ConfigManager()
    config = cm.load_config()
    config.system_config.enable_code_plugins = True

    plugins_dir = cm.plugins_dir
    generated = generate(plugins_dir, prefix)
    print(f"Generated {len(generated)} POC plugins into {plugins_dir}")

    result = discover_plugins(
        config,
        plugins_dir=plugins_dir,
        approve_cb=_approve_all,
        enable_code_plugins=True,
        persist=lambda: cm.save_config(config),
    )
    if result is not None:
        for p in result.plugins:
            print(f"  {p.type_id or p.manifest.id:16s} → {p.status}"
                  f"{'' if p.status == 'loaded' else '  (' + (p.detail or '') + ')'}")

    if disable_builtins:
        disabled = []
        for inst in config.plugins:
            if inst.category == _SEARCH_CATEGORY and inst.provider_id in PROVIDER_NAMES:
                inst.enabled = False
                disabled.append(inst.provider_id)
        for pid in PROVIDER_NAMES:
            config.search_provider_config.set_enabled(pid, False)
        print(f"Disabled built-in providers: {sorted(set(disabled)) or PROVIDER_NAMES}")

    cm.save_config(config)
    live = [i.instance_id for i in config.enabled_instances_for(_SEARCH_CATEGORY)]
    print(f"Enabled search instances now: {live}")
    print("Done. Verify in the GUI (Plugins tab), then run a search / agentic pipeline.")


def revert(prefix: str = DEFAULT_PREFIX) -> None:
    """Restore built-ins and remove the POC plugins/approvals/dirs."""
    from src.utils.config_manager import ConfigManager

    cm = ConfigManager()
    config = cm.load_config()

    for inst in config.plugins:
        if inst.category == _SEARCH_CATEGORY and inst.provider_id in PROVIDER_NAMES:
            inst.enabled = True
    for pid in PROVIDER_NAMES:
        config.search_provider_config.set_enabled(pid, True)

    config.plugins = [p for p in config.plugins if not str(p.instance_id).startswith(prefix)]
    for key in [k for k in config.approved_plugins if str(k).startswith(prefix)]:
        del config.approved_plugins[key]

    cm.save_config(config)

    plugins_dir = cm.plugins_dir
    removed = 0
    for name in PROVIDER_NAMES:
        d = plugins_dir / f"{prefix}{name}"
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    print(f"Reverted: built-ins re-enabled, {removed} POC dirs removed.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deploy/revert the own-plugins-only POC.")
    ap.add_argument("--revert", action="store_true", help="Restore built-ins, remove POC plugins.")
    ap.add_argument("--keep-builtins", action="store_true",
                    help="Deploy the POC plugins but leave built-ins enabled (side-by-side).")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help='Id prefix (default "poc_").')
    ap.add_argument("--generate-into", metavar="DIR",
                    help="Only copy+rename the blueprints into DIR (no config changes).")
    args = ap.parse_args(argv)

    if args.generate_into:
        made = generate(Path(args.generate_into), args.prefix)
        for new_id, path in made:
            print(f"  {new_id:16s} → {path}")
        return 0
    if args.revert:
        revert(args.prefix)
        return 0
    deploy(prefix=args.prefix, disable_builtins=not args.keep_builtins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
