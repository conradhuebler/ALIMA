"""``alima rules`` — manage the personal indexing rules. Claude Generated.

The rules themselves live in ``~/.config/alima/rules.yaml`` and are described in
``src/core/user_rules.py``. This command is the headless counterpart to the GUI
rule dialog and to the chat's ``propose_rule``: list, add, enable, disable,
remove, and exchange rules with another machine.

Export keeps the ``origin`` block verbatim; import keeps it too and only adds
where the rule came from. A rule is worth sharing precisely because someone
formulated it for a reason, and that reason travels with it.
"""

from __future__ import annotations

import logging
from typing import Any, List

from src.core.user_rules import RuleStore, UserRule

logger = logging.getLogger(__name__)


def _store(args) -> RuleStore:
    return RuleStore(getattr(args, "rules_file", None))


def _print_rule(rule: UserRule, *, verbose: bool = False) -> None:
    mark = "●" if rule.enabled else "○"
    print(f"{mark} {rule.id}  {rule.text}")
    if rule.applies_when:
        print(f"    Bedingung: {rule.applies_when}")
    print(f"    Gilt für:  {rule.scope_label()}")
    print(f"    Herkunft:  {rule.origin_label()}")
    if verbose:
        for key, value in sorted(rule.origin.items()):
            print(f"      {key}: {value}")


def handle_rules(args, logger: logging.Logger) -> int:
    """Dispatch ``alima rules <action>``. Returns a process exit code."""
    action = getattr(args, "rules_action", None) or "list"
    handlers = {
        "list": _list,
        "show": _show,
        "add": _add,
        "enable": lambda a, l: _set_enabled(a, l, True),
        "disable": lambda a, l: _set_enabled(a, l, False),
        "remove": _remove,
        "export": _export,
        "import": _import,
    }
    handler = handlers.get(action)
    if handler is None:
        print(f"Unbekannte Aktion: {action}")
        return 2
    return handler(args, logger)


# ----------------------------------------------------------------------


def _list(args, logger: logging.Logger) -> int:
    store = _store(args)
    rules: List[UserRule] = store.load()
    if getattr(args, "enabled_only", False):
        rules = [r for r in rules if r.enabled]
    if not rules:
        print(f"Keine Regeln in {store.path}")
        return 0
    active = sum(1 for r in rules if r.enabled)
    print(f"{len(rules)} Regel(n) in {store.path} — {active} aktiv\n")
    for rule in rules:
        _print_rule(rule)
        print()
    return 0


def _show(args, logger: logging.Logger) -> int:
    store = _store(args)
    rule = store.get(args.rule_id)
    if rule is None:
        print(f"Keine Regel mit der Id '{args.rule_id}'.")
        return 1
    _print_rule(rule, verbose=True)
    return 0


def _add(args, logger: logging.Logger) -> int:
    store = _store(args)
    rule = store.add(
        args.text,
        applies_when=getattr(args, "when", "") or "",
        workflows=getattr(args, "workflows", None),
        steps=getattr(args, "steps", None),
        enabled=not getattr(args, "inactive", False),
        origin={
            "source": "cli",
            "note": getattr(args, "note", "") or "",
            "author": getattr(args, "author", "") or "",
        },
    )
    if rule is None:
        print("Regel konnte nicht gespeichert werden (leerer Text oder Schreibfehler).")
        return 1
    print(f"Gespeichert in {store.path}:\n")
    _print_rule(rule)
    return 0


def _set_enabled(args, logger: logging.Logger, enabled: bool) -> int:
    store = _store(args)
    if not store.set_enabled(args.rule_id, enabled):
        print(f"Keine Regel mit der Id '{args.rule_id}'.")
        return 1
    print(f"{args.rule_id}: {'aktiv' if enabled else 'stumm'}")
    return 0


def _remove(args, logger: logging.Logger) -> int:
    store = _store(args)
    rule = store.get(args.rule_id)
    if rule is None:
        print(f"Keine Regel mit der Id '{args.rule_id}'.")
        return 1
    if not getattr(args, "yes", False):
        _print_rule(rule, verbose=True)
        answer = input("\nDiese Regel samt Herkunft löschen? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "j", "ja"):
            # Declining is the answer to a question, not a failure. Exit 0 so a
            # script around it does not treat it as an error. - Claude Generated
            print("Abgebrochen.")
            return 0
    if not store.remove(args.rule_id):
        print("Löschen fehlgeschlagen.")
        return 1
    print(f"{args.rule_id} gelöscht.")
    return 0


def _export(args, logger: logging.Logger) -> int:
    store = _store(args)
    target = getattr(args, "out", None) or "alima_rules.yaml"
    ok, count = store.export(
        target,
        ids=getattr(args, "ids", None),
        enabled_only=bool(getattr(args, "enabled_only", False)),
    )
    if not ok:
        print(f"Export nach {target} fehlgeschlagen.")
        return 1
    print(f"{count} Regel(n) nach {target} geschrieben (Herkunft unverändert).")
    return 0


def _import(args, logger: logging.Logger) -> int:
    store = _store(args)
    activate = bool(getattr(args, "activate", False))
    ok, added = store.import_file(args.source, activate=activate)
    if not ok:
        print(f"Import aus {args.source} fehlgeschlagen.")
        return 1
    if not added:
        print(f"{args.source} enthielt keine Regeln.")
        return 0
    state = "aktiv" if activate else "inaktiv (mit `alima rules enable <id>` scharf schalten)"
    print(f"{len(added)} Regel(n) übernommen, {state}:\n")
    for rule in added:
        _print_rule(rule)
        print()
    return 0
