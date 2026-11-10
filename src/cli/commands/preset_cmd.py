"""CLI handler for ``alima preset {apply,revert,export,show}`` - Claude Generated.

Thin frontend over :mod:`src.utils.preset_apply` and
:mod:`src.utils.preset_export`; the GUI group "Einrichtung (Preset)" calls the
same functions. See ``docs/institutional_presets.md``.
"""

from __future__ import annotations

import logging

from src.utils import preset_apply as pa


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{question} [j/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("j", "ja", "y", "yes")


def _handle_apply(args) -> int:
    if args.path:
        preset = pa.load_preset(args.path, pa.SOURCE_GIVEN)
    else:
        path, source = pa.find_preset()
        if path is None:
            print("❌ Kein Preset gefunden (ALIMA_PRESET, Konfigurationsverzeichnis, "
                  "Projektverzeichnis). Pfad angeben: alima preset apply <ordner|datei>")
            return 1
        preset = pa.load_preset(path, source)
    plan = pa.build_plan(preset)
    print("\n".join(plan.lines()))
    if plan.errors:
        return 1
    if args.dry_run or plan.is_empty:
        return 0
    if not _confirm("Preset anwenden?", args.yes):
        print("Abgebrochen, nichts geändert.")
        return 1
    entry = pa.apply_plan(plan)
    files = len((entry or {}).get("files") or [])
    print(f"✅ Angewendet: {len(plan.changes)} Änderungen, {files} Dateien installiert. "
          "Rückgängig: alima preset revert")
    return 0


def _handle_revert(args) -> int:
    from src.utils.config_manager import ConfigManager

    entry = pa.last_applied(ConfigManager().load_config())
    if entry is None:
        print("Es ist kein Preset angewendet.")
        return 1
    print(f"Zuletzt angewendet: {entry.get('label') or entry.get('id')} {entry.get('version', '')} "
          f"am {entry.get('applied_at', '?')}, {len(entry.get('changes') or [])} Änderungen, "
          f"{len(entry.get('files') or [])} Dateien")
    if not _confirm("Zurücksetzen?", args.yes):
        print("Abgebrochen, nichts geändert.")
        return 1
    result = pa.revert_last()
    print("\n".join(result.lines()))
    return 0


def _handle_export(args) -> int:
    from src.utils.preset_export import export_preset

    result = export_preset(None, args.folder, getattr(args, "plugins", None),
                           preset_id=args.id or "", version=args.version or "",
                           label=args.label or "",
                           include_database_password=bool(
                               getattr(args, "with_database_password", False)))
    print("\n".join(result.lines()))
    print("✅ Exportiert (gespeicherter Stand). Keine Schlüssel und Tokens im Paket; "
          "Endpunkte und Dateien vor dem Verteilen prüfen.")
    return 0


def _handle_show(args) -> int:
    from src.utils.config_manager import ConfigManager

    path, source = pa.find_preset()
    if path is None:
        print("Kein Preset gefunden (ALIMA_PRESET, "
              f"{pa.config_dir_of() / pa.PRESET_FILENAME}, Projektverzeichnis).")
    else:
        try:
            preset = pa.load_preset(path, source)
            name = f"{preset.label} {preset.version}".strip()
        except pa.PresetError as exc:
            name = f"nicht lesbar: {exc}"
        print(f"Gefunden: {name}\n  {path} ({pa.SOURCE_LABELS[source]})")
    entry = pa.last_applied(ConfigManager().load_config())
    if entry is None:
        print("Zuletzt angewendet: keines")
    else:
        print(f"Zuletzt angewendet: {entry.get('label') or entry.get('id')} "
              f"{entry.get('version', '')} am {entry.get('applied_at', '?')} "
              f"({len(entry.get('changes') or [])} Änderungen, "
              f"{len(entry.get('files') or [])} Dateien)")
    return 0


def handle_preset(args, logger: logging.Logger) -> int:
    """Dispatch ``alima preset <action>``; returns a process exit code."""
    action = getattr(args, "preset_action", None)
    handlers = {
        "apply": _handle_apply,
        "revert": _handle_revert,
        "export": _handle_export,
        "show": _handle_show,
    }
    fn = handlers.get(action)
    if fn is None:
        print("Usage: alima preset {apply|revert|export|show} …")
        return 2
    try:
        return fn(args)
    except pa.PresetError as exc:
        print(f"❌ Preset: {exc}")
        return 1
    except Exception as exc:  # keep the CLI from dumping a traceback
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, f"preset {action}", expected_level="error")
        print(f"❌ Unerwarteter Fehler: {exc}")
        return 1
