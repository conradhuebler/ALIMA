"""Rule chat-tools — persönliche Zusatzregeln aus dem Gespräch. Claude Generated.

A rule is one standing instruction the operator wants every matching run to
follow. It is usually formulated in passing while discussing a result ("mach
das künftig immer so"), which is exactly where it is lost today. These tools
let the agent notice it, offer it, and — after the user confirms in the same
inline bubble the mutation tools use — store it in ``~/.config/alima/rules.yaml``.

Confirmation is not re-implemented here: ``_MutationToolBase`` already carries
the audit + gateway handoff (Qt bubble in the GUI, y/N on stdin in the CLI).
Writes that lose something (``propose_rule``, ``delete_rule``) go through it;
``set_rule_enabled`` does not, because it is reversible and visible in the rule
dialog, and a confirmation there would be pure friction.

The store itself is Qt-free (``src/core/user_rules.py``) — these classes only
bind it to a chat session.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.core.user_rules import RuleStore, UserRule, available_scope_steps
from src.utils.error_visibility import log_caught
from src.ui.chat_tools.mutations import _MutationToolBase

logger = logging.getLogger(__name__)

#: Used as the audit id when no knowledge manager is available (headless). The
#: gateway only needs a key to wait on; without one ``_ask_user`` would fail
#: safe to "rejected" and the user would never be asked at all.
#:
#: Kept **positive** and far above real row ids: the id goes into the accept
#: URL of the confirmation bubble, and a negative number there yields a URL the
#: parser rejects. - Claude Generated
_SYNTHETIC_AUDIT_BASE = 900_000_000
_synthetic_counter = [0]


def _synthetic_audit_id() -> int:
    _synthetic_counter[0] += 1
    return _SYNTHETIC_AUDIT_BASE + _synthetic_counter[0]


def _rule_payload(rule: UserRule) -> Dict[str, Any]:
    """The rule as the LLM and the confirmation bubble should see it."""
    return {
        "id": rule.id,
        "text": rule.text,
        "applies_when": rule.applies_when,
        "scope": rule.scope_label(),
        "enabled": rule.enabled,
        "origin": rule.origin_label(),
    }


class _RuleToolBase(_MutationToolBase):
    """Rule tools reuse the mutation scaffolding but need no pipeline.

    One can formulate a rule without a loaded result, so ``pipeline_manager``
    stays optional here. - Claude Generated
    """

    def __init__(
        self,
        *,
        gateway: Any = None,
        chat_config: Any = None,
        session_id: str = "",
        kb_manager: Any = None,
        store: Optional[RuleStore] = None,
        pipeline_manager: Any = None,
    ) -> None:
        super().__init__(
            pipeline_manager=pipeline_manager,
            kb_manager=kb_manager,
            gateway=gateway,
            chat_config=chat_config,
            session_id=session_id,
        )
        self._store = store

    @property
    def store(self) -> RuleStore:
        if self._store is None:
            self._store = RuleStore()
        return self._store

    def _ask_user(self, audit_id: Optional[int], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Always ask — ``autonomous_pipeline`` does not cover rules.

        The base class treats autonomous mode as "skip the y/N", which is right
        for a keyword replacement: that changes the run in front of the user,
        and the run is what they started. A rule is a different kind of thing —
        it changes **every future run**, silently, until someone notices. On the
        operator's own machine autonomous mode was on, and six rules landed in
        fifteen minutes without a single question, one of them a duplicate and
        one not a rule at all. So this override drops the shortcut and keeps the
        rest of the base behaviour, fail-safe-to-rejected included.
        - Claude Generated
        """
        if self.gateway is None or audit_id is None:
            return {"accepted": False, "reject_reason": "no_gateway"}
        return self.gateway.request_decision(audit_id, self.name, payload)

    def _confirm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the user, via the same gateway the mutation tools use."""
        audit_id = self._record_pending(payload) or _synthetic_audit_id()
        decision = self._ask_user(audit_id, payload)
        self._record_outcome(
            audit_id,
            bool(decision.get("accepted")),
            str(decision.get("reject_reason") or ""),
        )
        return decision

    @staticmethod
    def _rejected(decision: Dict[str, Any], what: str) -> str:
        reason = str(decision.get("reject_reason") or "")
        if reason in ("no_gateway", "non_interactive"):
            # Webapp (AutoRejectGateway) and non-interactive CLI have no channel
            # to ask on. Say so, instead of reporting a refusal the user never
            # made. - Claude Generated
            message = (
                "Hier ist keine Bestätigung möglich (Webapp / nicht-interaktiver "
                f"Lauf). {what} bitte in der GUI oder mit `alima rules` ablegen."
            )
        elif reason == "timeout":
            message = "Keine Antwort des Nutzers — nichts gespeichert."
        else:
            message = "Vom Nutzer abgelehnt — nichts gespeichert."
        return json.dumps(
            {"status": "rejected", "reason": reason or "rejected", "message": message},
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------
# List
# ----------------------------------------------------------------------


class ListRulesTool(_RuleToolBase):
    name = "list_rules"
    description = (
        "List the operator's personal indexing rules (persönliche Zusatzregeln) "
        "that are stored on this machine, with id, text, condition, scope, "
        "whether they are active, and where they came from. Use it before "
        "proposing a new rule so you do not duplicate an existing one, and "
        "whenever the user asks which rules currently apply."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled_only": {
                "type": "boolean",
                "description": "Only list active rules (default: false, list all).",
            },
        },
    }

    def execute(self, session: Any, **kwargs: Any) -> str:
        enabled_only = bool(kwargs.get("enabled_only", False))
        rules = self.store.load()
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return json.dumps(
            {
                "count": len(rules),
                "rules": [_rule_payload(r) for r in rules],
                "file": str(self.store.path),
            },
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------
# Propose (confirmed)
# ----------------------------------------------------------------------


class ProposeRuleTool(_RuleToolBase):
    name = "propose_rule"
    operation = "rule_save"
    description = (
        "Offer to store a standing personal rule the user has just formulated "
        "('mach das künftig immer so', 'bei Lehrbüchern nie …'). The tool asks "
        "the user to confirm and only stores the rule if they accept; once "
        "stored it is appended to the prompts of every matching run. "
        "Write `text` as one self-contained instruction in the user's own "
        "wording. Put a condition into `applies_when` as prose (e.g. 'bei "
        "Überblickswerken') — it is judged by the model, not evaluated.\n"
        "DECIDE THE SCOPE: `steps` says at which point of the run the rule is "
        "read. Work out where the rule actually acts and name those steps; the "
        "`steps` parameter lists them with what each one does. '*' means the "
        "rule is put into EVERY prompt — costs tokens in steps that cannot act "
        "on it, so use it only for a rule that genuinely applies everywhere. A "
        "rule about the finished output (a catalogue entry, an export format) "
        "belongs to 'reflection', the last turn of the run. Say in your reply "
        "which scope you chose and why, so the user can correct it.\n"
        "Never call this without the user having said something that should "
        "hold beyond the current case."
    )
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # The step ids come from the workflow on disk, not from a hardcoded
        # list: a rule scoped to a step that does not exist never fires, and a
        # model with no list defaults everything to '*'. Built per chat turn, so
        # an edited workflow is reflected without a restart. - Claude Generated
        self.parameters_schema = _with_step_choices(type(self).parameters_schema)

    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The rule as one clear instruction, in German.",
            },
            "applies_when": {
                "type": "string",
                "description": "Prose condition, e.g. 'bei Überblickswerken'. Optional.",
            },
            "workflows": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workflow name globs, e.g. ['alima_v51*']. Default: all.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Step ids the rule is read at. Globs allowed ('selection*').",
            },
            "reason": {
                "type": "string",
                "description": "Why this rule — shown to the user and stored as provenance.",
            },
        },
        "required": ["text"],
    }

    def execute(self, session: Any, **kwargs: Any) -> str:
        text = str(kwargs.get("text") or "").strip()
        if not text:
            return json.dumps(
                {"status": "error", "message": "Kein Regeltext angegeben."},
                ensure_ascii=False,
            )
        applies_when = str(kwargs.get("applies_when") or "").strip()
        workflows = _as_list(kwargs.get("workflows"))
        steps = _as_list(kwargs.get("steps"))
        reason = str(kwargs.get("reason") or "").strip()

        existing = self.store.find_duplicate(text)
        if existing is not None:
            return json.dumps(
                {
                    "status": "exists",
                    "rule": _rule_payload(existing),
                    "message": (
                        "Diese Regel ist bereits abgelegt. Sag dem Nutzer, dass sie "
                        "schon gilt, statt sie erneut vorzuschlagen."
                    ),
                },
                ensure_ascii=False,
            )

        payload = {
            "text": text,
            "applies_when": applies_when,
            "scope": f"{'|'.join(workflows or ['*'])} × {'|'.join(steps or ['*'])}",
            "reason": reason,
        }
        decision = self._confirm(payload)
        if not decision.get("accepted"):
            return self._rejected(decision, "Die Regel")

        rule = self.store.add(
            text,
            applies_when=applies_when,
            workflows=workflows,
            steps=steps,
            enabled=True,
            origin={
                "source": "chat",
                "session_id": str(self.session_id or ""),
                "note": reason,
                "author": _author(self.chat_config),
            },
        )
        if rule is None:
            return json.dumps(
                {"status": "error", "message": "Regel konnte nicht gespeichert werden."},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "saved",
                "rule": _rule_payload(rule),
                "message": (
                    "Regel gespeichert und ab dem nächsten Lauf aktiv. "
                    "Nenne dem Nutzer Wortlaut und Geltungsbereich."
                ),
            },
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------
# Enable / disable (reversible, unconfirmed)
# ----------------------------------------------------------------------


class SetRuleEnabledTool(_RuleToolBase):
    name = "set_rule_enabled"
    operation = "rule_toggle"
    description = (
        "Activate or silence a stored personal rule by its id. Reversible and "
        "visible in the rule dialog, so it needs no confirmation. Use "
        "`list_rules` to find the id."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string", "description": "Rule id, e.g. 'r-20260903-01'."},
            "enabled": {"type": "boolean", "description": "True = active, False = silenced."},
        },
        "required": ["rule_id", "enabled"],
    }

    def execute(self, session: Any, **kwargs: Any) -> str:
        rule_id = str(kwargs.get("rule_id") or "").strip()
        enabled = bool(kwargs.get("enabled"))
        if not self.store.set_enabled(rule_id, enabled):
            return json.dumps(
                {"status": "error", "message": f"Keine Regel mit der Id '{rule_id}'."},
                ensure_ascii=False,
            )
        return json.dumps(
            {"status": "ok", "rule_id": rule_id, "enabled": enabled},
            ensure_ascii=False,
        )


class SetRuleScopeTool(_RuleToolBase):
    name = "set_rule_scope"
    operation = "rule_scope"
    description = (
        "Change at which steps an existing rule is read, without touching its "
        "wording. Use it when a rule turns out to be too broad ('*' but it only "
        "matters for the final output) or too narrow. Reversible and shown in "
        "the rule dialog, so it needs no confirmation. Find the id with "
        "`list_rules`, then say in your reply what you changed."
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.parameters_schema = _with_step_choices(type(self).parameters_schema)

    parameters_schema = {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string", "description": "Rule id, e.g. 'r-20260903-01'."},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Step ids the rule is read at. Globs allowed ('selection*').",
            },
            "workflows": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workflow name globs, e.g. ['alima_v51*']. Omit to keep.",
            },
        },
        "required": ["rule_id"],
    }

    def execute(self, session: Any, **kwargs: Any) -> str:
        rule_id = str(kwargs.get("rule_id") or "").strip()
        rule = self.store.get(rule_id)
        if rule is None:
            return json.dumps(
                {"status": "error", "message": f"Keine Regel mit der Id '{rule_id}'."},
                ensure_ascii=False,
            )
        steps = _as_list(kwargs.get("steps"))
        workflows = _as_list(kwargs.get("workflows"))
        if not steps and not workflows:
            return json.dumps(
                {"status": "error", "message": "Weder steps noch workflows angegeben."},
                ensure_ascii=False,
            )
        before = rule.scope_label()
        if steps:
            rule.steps = steps
        if workflows:
            rule.workflows = workflows
        if not self.store.update(rule):
            return json.dumps(
                {"status": "error", "message": "Geltungsbereich konnte nicht gespeichert werden."},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "rule": _rule_payload(rule),
                "changed_from": before,
                "message": "Geltungsbereich geändert; gilt ab dem nächsten Lauf.",
            },
            ensure_ascii=False,
        )


# ----------------------------------------------------------------------
# Delete (confirmed)
# ----------------------------------------------------------------------


class DeleteRuleTool(_RuleToolBase):
    name = "delete_rule"
    operation = "rule_delete"
    description = (
        "Delete a stored personal rule permanently, including its provenance. "
        "Asks the user to confirm. Prefer `set_rule_enabled` with false when "
        "the rule should merely stop applying."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string", "description": "Rule id, e.g. 'r-20260903-01'."},
            "reason": {"type": "string", "description": "Why it should go. Shown to the user."},
        },
        "required": ["rule_id"],
    }

    def execute(self, session: Any, **kwargs: Any) -> str:
        rule_id = str(kwargs.get("rule_id") or "").strip()
        rule = self.store.get(rule_id)
        if rule is None:
            return json.dumps(
                {"status": "error", "message": f"Keine Regel mit der Id '{rule_id}'."},
                ensure_ascii=False,
            )
        payload = dict(_rule_payload(rule))
        payload["reason"] = str(kwargs.get("reason") or "").strip()
        payload["action"] = "delete"
        decision = self._confirm(payload)
        if not decision.get("accepted"):
            return self._rejected(decision, "Das Löschen")
        if not self.store.remove(rule_id):
            return json.dumps(
                {"status": "error", "message": "Regel konnte nicht gelöscht werden."},
                ensure_ascii=False,
            )
        return json.dumps({"status": "deleted", "rule_id": rule_id}, ensure_ascii=False)


# ----------------------------------------------------------------------


def _with_step_choices(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Copy ``schema`` with the live step ids spelled out under ``steps``.

    The model cannot guess a workflow's step ids, and a guessed one silently
    never matches — so the choices, with what each step does, go into the
    parameter description. - Claude Generated
    """
    import copy

    out = copy.deepcopy(schema)
    try:
        choices = available_scope_steps()
    except Exception as exc:
        log_caught(logger, exc, "rules tool: building the step choice list")
        return out
    listing = "; ".join(f"'{sid}' = {desc}" if desc else f"'{sid}'" for sid, desc in choices)
    steps = out.get("properties", {}).get("steps")
    if isinstance(steps, dict):
        steps["description"] = (
            f"{steps.get('description', '')} Verfügbar: {listing}."
        ).strip()
    return out


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _author(chat_config: Any) -> str:
    """Best-effort author name for provenance; empty is fine."""
    return str(getattr(chat_config, "rule_author", "") or "")


def rule_tools(
    *,
    gateway: Any = None,
    chat_config: Any = None,
    session_id: str = "",
    kb_manager: Any = None,
    store: Optional[RuleStore] = None,
) -> Sequence[_RuleToolBase]:
    """The rule toolset for one chat session. - Claude Generated"""
    kwargs = dict(
        gateway=gateway,
        chat_config=chat_config,
        session_id=session_id,
        kb_manager=kb_manager,
        store=store,
    )
    return [
        ListRulesTool(**kwargs),
        ProposeRuleTool(**kwargs),
        SetRuleEnabledTool(**kwargs),
        SetRuleScopeTool(**kwargs),
        DeleteRuleTool(**kwargs),
    ]
