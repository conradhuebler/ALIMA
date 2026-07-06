"""CLI handler for `alima bundle {build,install,list,remove}` - Claude Generated.

Thin frontend over :mod:`src.utils.bundle`; all logic lives there so a future GUI
action can reuse it. Institutional bundles ship plugins + an advisory config
profile; secrets are declared, never carried. See `docs/institutional_bundles.md`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils import bundle as bundle_mod


def _handle_build(args) -> int:
    out = bundle_mod.build_bundle(args.source, args.output)
    print(f"✅ Built bundle → {out}")
    return 0


def _handle_install(args) -> int:
    rep = bundle_mod.install_bundle(args.path)
    print(f"✅ Installed bundle '{rep.bundle_id}' v{rep.version}"
          + (f" — {rep.label}" if rep.label else ""))

    if rep.plugins:
        print("  Plugins:")
        for pid, status, sev in rep.plugins:
            extra = "" if sev in ("none", "") else f"  [scan: {sev}]"
            mark = "✓" if status == "loaded" else "✗"
            print(f"    {mark} {pid}  ({status}){extra}")
    if rep.profile_keys:
        print(f"  Profil angewendet (beratend): {', '.join(rep.profile_keys)}")
    if rep.integrity_mismatches:
        print(f"  ⚠️ Integritäts-Abweichung (Hash ≠ approvals.json): "
              f"{', '.join(rep.integrity_mismatches)}")
    unmet = [s for s in rep.required_secrets if not s["satisfied"]]
    if rep.required_secrets:
        print("  Per-User-Secrets:")
        for s in rep.required_secrets:
            state = "gesetzt" if s["satisfied"] else "FEHLT"
            hint = f" — {s['hint']}" if s["hint"] else ""
            print(f"    [{state}] {s['plugin']}.{s['key']}  (env: {s['env_var']}){hint}")
    if unmet:
        print(f"  → {len(unmet)} Secret(s) noch offen: im GUI-Plugin-Tab eintragen "
              "oder als Umgebungsvariable setzen.")
    for w in rep.warnings:
        print(f"  ⚠️ {w}")
    return 0


def _handle_export(args) -> int:
    out = bundle_mod.export_bundle(
        args.output,
        bundle_id=args.id,
        version=args.version,
        label=args.label,
        institution=args.institution,
        instance_ids=getattr(args, "plugins", None),
    )
    print(f"✅ Exported current setup → {out}")
    print("   (Secrets wurden entfernt und nur deklariert — vor Verteilung prüfen.)")
    return 0


def _handle_list(args) -> int:
    bundles = bundle_mod.list_bundles()
    if not bundles:
        print("Keine Bundles installiert.")
        return 0
    print(f"Installierte Bundles ({len(bundles)}):")
    for b in bundles:
        plugins = ", ".join(b["plugins"]) or "—"
        print(f"  {b['id']}  v{b['version']}  [{b['label']}]")
        print(f"      Plugins: {plugins}   installiert: {b['installed_at']}")
    return 0


def _handle_remove(args) -> int:
    bundle_mod.remove_bundle(args.id)
    print(f"✅ Bundle '{args.id}' entfernt (Built-in-Zustand wiederhergestellt).")
    return 0


def handle_bundle(args, logger: logging.Logger) -> int:
    """Dispatch `alima bundle <action>`; returns a process exit code."""
    action = getattr(args, "bundle_action", None)
    handlers = {
        "build": _handle_build,
        "install": _handle_install,
        "export": _handle_export,
        "list": _handle_list,
        "remove": _handle_remove,
    }
    fn = handlers.get(action)
    if fn is None:
        print("Usage: alima bundle {build|install|list|remove} …")
        return 2
    try:
        return fn(args)
    except bundle_mod.BundleError as exc:
        print(f"❌ Bundle-Fehler: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"❌ Nicht gefunden: {exc}")
        return 1
    except Exception as exc:  # keep the CLI from dumping a traceback
        logger.error("bundle %s failed: %s", action, exc, exc_info=True)
        print(f"❌ Unerwarteter Fehler: {exc}")
        return 1
