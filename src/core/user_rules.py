"""Persönliche Zusatzregeln — operator-authored rules injected into prompts.

A rule is one sentence the operator wants every matching run to follow, e.g.
"Formschlagwörter gehören nicht in core_keywords". Rules live in a YAML file in
the user's config directory (``~/.config/alima/rules.yaml``), not in the
repository: they are personal or institution-specific, they differ per machine,
and they are formulated in the chat about a concrete result rather than in a
commit.

Two things this module deliberately does NOT do:

* **It does not evaluate conditions.** ``applies_when`` is prose ("bei
  Überblickswerken") and is rendered into the rule line for the model to judge.
  A condition like "is this a survey work" is not something an expression over
  the SharedContext could decide, and a half-working evaluator would be worse
  than none.
* **It does not substitute placeholders.** At the generic injection points the
  rendered block is appended *after* the prompt's own ``{name}`` rendering. The
  reflection gate is the exception: there the block goes in as the value of
  ``{user_rules_gate}``, which is safe because ``prompt_resolver._render``
  substitutes in a single pass and never re-scans what it inserted. Either way
  braces inside a rule text stay literal (``ReflectionBraceSafetyTest``).

``scope`` stays structural (workflow + step globs) because it decides which
prompts a rule reaches at all, and therefore what it costs in tokens.

Claude Generated
"""

from __future__ import annotations

import fnmatch
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from src.utils.error_visibility import log_caught

logger = logging.getLogger(__name__)

#: Schema version of ``rules.yaml`` and of the export format.
RULES_FILE_VERSION = 1

#: Heading of the block appended to a system prompt. Also the string the
#: operator greps for in a step log to confirm a rule arrived.
RULES_BLOCK_HEADING = "## Persönliche Zusatzregeln"

#: Placed under the heading so a user rule cannot quietly void the JSON contract
#: the agentic steps depend on.
RULES_BLOCK_PREAMBLE = (
    "Diese Regeln ergänzen die Vorgaben oben. "
    "Sie ändern nicht das geforderte Antwortformat."
)

#: Pseudo step ids for the prompts that are not workflow steps.
STEP_PLANNER = "planner"
STEP_REFLECTION = "reflection"
STEP_CHAT = "chat"


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class UserRule:
    """One rule, as stored in ``rules.yaml``. - Claude Generated"""

    id: str = ""
    text: str = ""
    #: Prose precondition, judged by the model. Never evaluated in code.
    applies_when: str = ""
    #: Glob patterns; empty list means "no restriction" (same as ``["*"]``).
    workflows: List[str] = field(default_factory=lambda: ["*"])
    steps: List[str] = field(default_factory=lambda: ["*"])
    enabled: bool = True
    #: Free-form provenance; survives export/import unchanged.
    origin: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "applies_when": self.applies_when,
            "scope": {"workflows": list(self.workflows), "steps": list(self.steps)},
            "enabled": bool(self.enabled),
            "origin": dict(self.origin),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UserRule":
        scope = raw.get("scope") or {}
        if not isinstance(scope, dict):
            scope = {}
        return cls(
            id=str(raw.get("id") or ""),
            text=str(raw.get("text") or ""),
            applies_when=str(raw.get("applies_when") or ""),
            workflows=_as_patterns(scope.get("workflows")),
            steps=_as_patterns(scope.get("steps")),
            enabled=bool(raw.get("enabled", True)),
            origin=dict(raw.get("origin") or {}),
        )

    # -- display -------------------------------------------------------

    def scope_label(self) -> str:
        """Human-readable scope, for the GUI table and the CLI listing."""
        return f"{'|'.join(self.workflows)} × {'|'.join(self.steps)}"

    def origin_label(self) -> str:
        """Short provenance line: source, date, author."""
        parts = [str(self.origin.get("source") or "?")]
        created = str(self.origin.get("created") or "")
        if created:
            parts.append(created[:10])
        author = str(self.origin.get("author") or "")
        if author:
            parts.append(author)
        return ", ".join(parts)


def _as_patterns(value: Any) -> List[str]:
    """Normalise a scope entry to a non-empty list of glob patterns."""
    if value is None:
        return ["*"]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ["*"]
    out = [str(v).strip() for v in value if str(v).strip()]
    return out or ["*"]


# ----------------------------------------------------------------------
# Selection + rendering
# ----------------------------------------------------------------------


def _matches(patterns: Sequence[str], value: str) -> bool:
    """True when ``value`` matches any glob in ``patterns``.

    An empty ``value`` (unknown workflow or step) matches only ``*`` — a rule
    scoped to a named step must not leak into a prompt whose step we cannot
    identify.
    """
    for pat in patterns:
        if pat == "*":
            return True
        if value and fnmatch.fnmatchcase(value, pat):
            return True
    return False


def select_rules(
    rules: Iterable[UserRule], *, workflow: str = "", step: str = ""
) -> List[UserRule]:
    """The enabled rules that apply to this workflow/step. - Claude Generated"""
    return [
        r
        for r in rules
        if r.enabled
        and r.text.strip()
        and _matches(r.workflows, workflow or "")
        and _matches(r.steps, step or "")
    ]


def render_rule_line(rule: UserRule) -> str:
    """One bullet: the rule, prefixed by its prose condition when it has one."""
    text = rule.text.strip()
    cond = rule.applies_when.strip()
    if cond:
        return f"- Nur {cond}: {text}"
    return f"- {text}"


def render_rules_block(rules: Sequence[UserRule]) -> str:
    """The block appended to a system prompt. Empty selection ⇒ ``""``.

    The empty return is what keeps every prompt byte-identical to before when
    no rule is active; it is pinned by a test. - Claude Generated
    """
    if not rules:
        return ""
    lines = [RULES_BLOCK_HEADING, RULES_BLOCK_PREAMBLE]
    lines.extend(render_rule_line(r) for r in rules)
    return "\n".join(lines)


def append_rules_block(system_prompt: str, block: str) -> str:
    """Append a rendered block to a finished system prompt.

    Appends rather than substituting a slot: the block must never pass through
    the caller's ``{name}`` rendering, or braces in a rule text would be eaten
    (agentic) or crash the run (classic ``str.format``). - Claude Generated
    """
    if not block:
        return system_prompt
    base = (system_prompt or "").rstrip()
    return f"{base}\n\n{block}\n" if base else f"{block}\n"


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------


#: Env override for the rules file. Set by the test bootstrap so a suite never
#: reads the operator's real rules — those would silently change every prompt a
#: test builds, and the failure would look like a broken workflow rather than
#: leaked machine state. Also useful for running against a second rule set.
#: - Claude Generated
RULES_PATH_ENV = "ALIMA_RULES_FILE"

#: Serialises the read-modify-write of every mutation. The store is reached
#: from at least two threads in the GUI — the rule dialog on the UI thread and
#: ``propose_rule`` inside the chat worker — and each mutation loads the whole
#: file, changes one entry and writes it back. The write itself is atomic; the
#: sequence is not, so without this a rule stored while the dialog is open is
#: overwritten by the dialog's next save. - Claude Generated
_STORE_LOCK = threading.RLock()


def default_rules_path() -> Path:
    """``rules.yaml`` next to ``config.json``, unless overridden by env."""
    override = os.environ.get(RULES_PATH_ENV)
    if override:
        return Path(override).expanduser()
    try:
        from src.utils.config_manager import ConfigManager

        return Path(ConfigManager().rules_file)
    except Exception as exc:
        log_caught(logger, exc, "user_rules: config dir lookup")
        return Path("~/.config/alima/rules.yaml").expanduser()


class RuleStore:
    """Reads and writes ``rules.yaml``.

    Reading never raises: a missing file is an empty rule set, and a malformed
    one is reported (ERROR for bug-shaped causes) and then treated as empty. A
    broken rules file must not be able to stop a pipeline run.
    - Claude Generated
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path).expanduser() if path else default_rules_path()

    # -- read ----------------------------------------------------------

    def load(self) -> List[UserRule]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:
            log_caught(logger, exc, f"user_rules: reading {self.path}")
            return []
        return _rules_from_payload(raw, source=str(self.path))

    # -- write ---------------------------------------------------------

    def save(self, rules: Sequence[UserRule]) -> bool:
        payload = {
            "version": RULES_FILE_VERSION,
            "rules": [r.to_dict() for r in rules],
        }
        try:
            _atomic_write_yaml(self.path, payload)
            return True
        except Exception as exc:
            log_caught(logger, exc, f"user_rules: writing {self.path}", expected_level="error")
            return False

    # -- mutations -----------------------------------------------------

    def add(
        self,
        text: str,
        *,
        applies_when: str = "",
        workflows: Optional[Sequence[str]] = None,
        steps: Optional[Sequence[str]] = None,
        enabled: bool = True,
        origin: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserRule]:
        """Append a new rule and persist. Returns the stored rule, or None."""
        with _STORE_LOCK:
            rules = self.load()
            rule = UserRule(
                id=next_rule_id(rules),
                text=(text or "").strip(),
                applies_when=(applies_when or "").strip(),
                workflows=_as_patterns(list(workflows) if workflows else None),
                steps=_as_patterns(list(steps) if steps else None),
                enabled=bool(enabled),
                origin=_with_created(dict(origin or {})),
            )
            if not rule.text:
                return None
            rules.append(rule)
            return rule if self.save(rules) else None

    def update(self, rule: UserRule) -> bool:
        """Replace the stored rule with the same id."""
        with _STORE_LOCK:
            rules = self.load()
            for idx, existing in enumerate(rules):
                if existing.id == rule.id:
                    rules[idx] = rule
                    return self.save(rules)
            return False

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        with _STORE_LOCK:
            rules = self.load()
            hit = False
            for rule in rules:
                if rule.id == rule_id:
                    rule.enabled = bool(enabled)
                    hit = True
            return self.save(rules) if hit else False

    def remove(self, rule_id: str) -> bool:
        with _STORE_LOCK:
            rules = self.load()
            kept = [r for r in rules if r.id != rule_id]
            if len(kept) == len(rules):
                return False
            return self.save(kept)

    def find_duplicate(self, text: str) -> Optional[UserRule]:
        """An existing rule with the same text, ignoring case and whitespace.

        The first real session stored the same sentence twice within six
        minutes, because nothing looked. - Claude Generated
        """
        needle = " ".join((text or "").split()).casefold()
        if not needle:
            return None
        for rule in self.load():
            if " ".join(rule.text.split()).casefold() == needle:
                return rule
        return None

    def get(self, rule_id: str) -> Optional[UserRule]:
        for rule in self.load():
            if rule.id == rule_id:
                return rule
        return None

    # -- exchange ------------------------------------------------------

    def export(
        self,
        target: Path | str,
        *,
        ids: Optional[Sequence[str]] = None,
        enabled_only: bool = False,
    ) -> Tuple[bool, int]:
        """Write a shareable file. Provenance is copied verbatim.

        Returns ``(ok, count)``. Nothing in ``origin`` is rewritten or dropped:
        who formulated a rule, when and why is the point of sharing it.
        """
        rules = self.load()
        if ids:
            wanted = set(ids)
            rules = [r for r in rules if r.id in wanted]
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        payload = {
            "version": RULES_FILE_VERSION,
            "exported": datetime.now().isoformat(timespec="seconds"),
            "rules": [r.to_dict() for r in rules],
        }
        try:
            _atomic_write_yaml(Path(target).expanduser(), payload)
            return True, len(rules)
        except Exception as exc:
            log_caught(logger, exc, f"user_rules: exporting to {target}", expected_level="error")
            return False, 0

    def import_file(
        self, source: Path | str, *, activate: bool = False
    ) -> Tuple[bool, List[UserRule]]:
        """Merge a rules file into the store.

        The imported ``origin`` is kept as it is and only *extended* by
        ``imported_from``/``imported_at``, so a rule stays traceable to the
        person who wrote it. Rules land disabled unless ``activate`` is set: a
        foreign file may carry a dozen rules and none of them should change a
        run before the operator has looked at them. - Claude Generated
        """
        src = Path(source).expanduser()
        try:
            with open(src, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:
            log_caught(logger, exc, f"user_rules: reading import {src}", expected_level="error")
            return False, []

        incoming = _rules_from_payload(raw, source=str(src))
        if not incoming:
            return True, []

        with _STORE_LOCK:
            existing = self.load()
            taken = {r.id for r in existing}
            added: List[UserRule] = []
            stamp = datetime.now().isoformat(timespec="seconds")
            for rule in incoming:
                origin = dict(rule.origin)
                origin["imported_from"] = src.name
                origin["imported_at"] = stamp
                if rule.id in taken or not rule.id:
                    if rule.id:
                        origin["original_id"] = rule.id
                    rule.id = next_rule_id(existing + added)
                rule.origin = origin
                rule.enabled = bool(activate)
                taken.add(rule.id)
                added.append(rule)

            if not self.save(existing + added):
                return False, []
            return True, added


def _rules_from_payload(raw: Any, *, source: str) -> List[UserRule]:
    """Parse a ``{version, rules: [...]}`` payload; skip unusable entries."""
    if not isinstance(raw, dict):
        logger.warning(f"user_rules: {source} is not a mapping — ignored")
        return []
    entries = raw.get("rules")
    if entries is None:
        return []
    if not isinstance(entries, list):
        logger.warning(f"user_rules: 'rules' in {source} is not a list — ignored")
        return []
    out: List[UserRule] = []
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning(f"user_rules: skipping non-mapping rule entry in {source}")
            continue
        try:
            rule = UserRule.from_dict(entry)
        except Exception as exc:
            log_caught(logger, exc, f"user_rules: parsing a rule in {source}")
            continue
        if rule.text.strip():
            out.append(rule)
    return out


def _with_created(origin: Dict[str, Any]) -> Dict[str, Any]:
    origin.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    return origin


def next_rule_id(existing: Sequence[UserRule]) -> str:
    """``r-JJJJMMTT-NN``, unique against ``existing``. - Claude Generated"""
    day = datetime.now().strftime("%Y%m%d")
    taken = {r.id for r in existing}
    for n in range(1, 1000):
        candidate = f"r-{day}-{n:02d}"
        if candidate not in taken:
            return candidate
    return f"r-{day}-{datetime.now().strftime('%H%M%S')}"


def _atomic_write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """Write YAML via a temp file in the same directory, then ``os.replace``.

    Same shape as ``ConfigManager._atomic_write_json``: a crash before the
    replace leaves the previous rules file intact. - Claude Generated
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=100)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------
# Scope vocabulary
# ----------------------------------------------------------------------

#: What the three non-workflow prompts are for, in the operator's language.
#: ``reflection`` matters most here: it is the only place where a rule that
#: asks for something *at the end* of a run can still act.
_PSEUDO_STEPS = (
    (STEP_PLANNER, "Planer — entscheidet, welcher Schritt als nächstes läuft"),
    (STEP_REFLECTION, "Reflexion — letzter LLM-Turn; hier entsteht eine Ausgabe am Laufende"),
    (STEP_CHAT, "Chat — das Gespräch selbst"),
)

#: Used when the workflow file cannot be read. Deliberately short: a wrong list
#: is worse than a short one, because a rule scoped to a step id that does not
#: exist silently never fires.
_FALLBACK_STEPS = (("*", "überall"),) + _PSEUDO_STEPS


def available_scope_steps(workflow_name: str = "") -> List[Tuple[str, str]]:
    """``(step_id, description)`` a rule can be scoped to, ``*`` first.

    Read from the workflow YAML rather than hardcoded, so the list cannot drift
    away from the steps that actually run. Shared by the rule dialog and by the
    ``propose_rule`` tool schema — the model needs the real ids, otherwise it
    defaults everything to ``*``. - Claude Generated
    """
    steps: List[Tuple[str, str]] = [("*", "überall — nur, wenn die Regel wirklich für jeden Schritt gilt")]
    try:
        from src.core.agents.workflow_loader import find_workflow_file, load_workflow

        name = workflow_name
        if not name:
            from src.utils.config_manager import ConfigManager

            cfg = ConfigManager().load_config()
            name = getattr(cfg.system_config, "default_workflow", "") or "alima_v51"
        path = find_workflow_file(name)
        if path:
            workflow = load_workflow(path)
            for step in workflow.steps:
                if not getattr(step, "enabled", True):
                    continue
                label = (getattr(step, "description", "") or "").strip().replace("\n", " ")
                steps.append((step.id, label[:120]))
    except Exception as exc:
        log_caught(logger, exc, "user_rules: reading workflow steps for the scope list")
        return list(_FALLBACK_STEPS)
    steps.extend(_PSEUDO_STEPS)
    return steps


# ----------------------------------------------------------------------
# Convenience for the injection points
# ----------------------------------------------------------------------


def rules_block_for(
    *, workflow: str = "", step: str = "", store: Optional[RuleStore] = None
) -> Tuple[str, List[UserRule]]:
    """Rendered block + the rules it came from, for one prompt.

    Returns ``("", [])`` when nothing applies, which is the byte-identical
    case every injection point relies on. - Claude Generated
    """
    try:
        rules = (store or RuleStore()).load()
    except Exception as exc:
        log_caught(logger, exc, "user_rules: loading for prompt injection")
        return "", []
    selected = select_rules(rules, workflow=workflow, step=step)
    return render_rules_block(selected), selected
