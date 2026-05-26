"""Mutation chat-tools — P-ε. Claude Generated.

Write-tools that propose changes to the active KeywordAnalysisState. Each
tool follows this flow:

1. Validate the requested change against the current state.
2. Record a PENDING audit row in `chat_mutations`.
3. If `chat_config.autonomous_pipeline` is True → apply immediately.
   Otherwise block on `ProposalGateway.request_decision()` until the user
   clicks accept/reject in the inline chat bubble.
4. Apply the mutation via `KeywordAnalysisState.apply_*` (which emits a
   state-bus event so the Review tab refreshes).
5. Update the audit row with the outcome.
6. Return a JSON status string for the LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.ui.chat_tools.base import BaseChatTool

logger = logging.getLogger(__name__)


def _kas(pipeline_manager):
    """Return the active KeywordAnalysisState (or None)."""
    if pipeline_manager is None:
        return None
    return getattr(pipeline_manager, "current_analysis_state", None)


def _canonical_keyword(kw: str) -> str:
    """Strip the GND-ID tag for equality checks."""
    return kw.split(" (GND-ID:")[0].strip()


class _MutationToolBase(BaseChatTool):
    """Common scaffolding for mutation tools."""

    operation: str = ""

    def __init__(
        self,
        *,
        pipeline_manager: Any,
        kb_manager: Any,
        gateway: Any,
        chat_config: Any,
        session_id: str,
    ) -> None:
        self.pipeline_manager = pipeline_manager
        self.kb_manager = kb_manager
        self.gateway = gateway
        self.chat_config = chat_config
        self.session_id = session_id

    # -- audit helpers ------------------------------------------------

    def _record_pending(self, payload: Dict[str, Any]) -> Optional[int]:
        if self.kb_manager is None:
            return None
        try:
            return self.kb_manager.record_mutation_pending(
                self.session_id, self.name, self.operation, payload
            )
        except Exception:
            logger.exception("record_mutation_pending failed")
            return None

    def _record_outcome(
        self, audit_id: Optional[int], accepted: bool, reject_reason: str = ""
    ) -> None:
        if self.kb_manager is None or audit_id is None:
            return
        try:
            self.kb_manager.record_mutation_outcome(
                audit_id, accepted, reject_reason
            )
        except Exception:
            logger.exception("record_mutation_outcome failed")

    # -- decision ------------------------------------------------------

    def _ask_user(self, audit_id: Optional[int], payload: Dict[str, Any]) -> Dict[str, Any]:
        if bool(getattr(self.chat_config, "autonomous_pipeline", False)):
            return {"accepted": True, "reject_reason": ""}
        if self.gateway is None or audit_id is None:
            # No gateway available → fail safe to rejected so we never
            # silently apply an unconfirmed mutation.
            return {"accepted": False, "reject_reason": "no_gateway"}
        return self.gateway.request_decision(audit_id, self.name, payload)

    def _sync_shared_context(self) -> None:
        """P-ζ: re-derive pm.last_shared_context from current_analysis_state
        so chat tools reading SharedContext don't see stale data after a
        successful mutation.
        """
        pm = self.pipeline_manager
        if pm is None or getattr(pm, "current_analysis_state", None) is None:
            return
        try:
            from src.core.agents.shared_context import SharedContext
            pm.last_shared_context = SharedContext.from_keyword_analysis_state(
                pm.current_analysis_state
            )
        except Exception:
            logger.exception("_sync_shared_context failed")


# ----------------------------------------------------------------------
# Keyword replacement
# ----------------------------------------------------------------------


class ProposeKeywordReplacementTool(_MutationToolBase):
    name = "propose_keyword_replacement"
    operation = "keyword_replacement"
    description = (
        "Propose replacing one keyword in the current pipeline's initial_keywords "
        "with a different keyword. Requires user confirmation unless autonomous "
        "mode is active. Provide a short `reason` so the user can decide."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "old": {"type": "string", "description": "Keyword to replace (canonical form, without GND-ID tag)."},
            "new": {"type": "string", "description": "Replacement keyword."},
            "reason": {"type": "string", "description": "Short justification for the user."},
            "gnd_id": {"type": "string", "description": "Optional GND-ID for the new keyword."},
        },
        "required": ["old", "new", "reason"],
    }

    def available_for(self, session: Any) -> bool:
        kas = _kas(self.pipeline_manager)
        return bool(kas and getattr(kas, "initial_keywords", None))

    def execute(self, session: Any, **kwargs: Any) -> str:
        old = (kwargs.get("old") or "").strip()
        new = (kwargs.get("new") or "").strip()
        reason = (kwargs.get("reason") or "").strip()
        gnd_id = (kwargs.get("gnd_id") or "").strip() or None

        kas = _kas(self.pipeline_manager)
        if kas is None:
            return json.dumps({"status": "invalid", "reason": "no_active_pipeline_state"})
        if not old or not new:
            return json.dumps({"status": "invalid", "reason": "old_and_new_required"})

        existing = [_canonical_keyword(k) for k in kas.initial_keywords]
        if old not in existing:
            return json.dumps({
                "status": "invalid",
                "reason": "old_keyword_not_in_initial_keywords",
                "available": existing,
            })

        payload = {"old": old, "new": new, "reason": reason, "gnd_id": gnd_id}
        audit_id = self._record_pending(payload)
        decision = self._ask_user(audit_id, payload)
        accepted = bool(decision.get("accepted"))

        applied = False
        if accepted:
            try:
                applied = kas.apply_keyword_replacement(old, new, gnd_id)
            except Exception as e:
                logger.exception("apply_keyword_replacement raised")
                self._record_outcome(audit_id, False, f"apply_error: {e}")
                return json.dumps({
                    "status": "apply_error",
                    "audit_id": audit_id,
                    "error": str(e),
                })
            if applied:
                self._sync_shared_context()

        self._record_outcome(
            audit_id, accepted, decision.get("reject_reason", "")
        )

        return json.dumps({
            "audit_id": audit_id,
            "status": "applied" if (accepted and applied) else
                      ("rejected" if not accepted else "no_op"),
            "old": old,
            "new": new,
            "gnd_id": gnd_id,
            "reject_reason": decision.get("reject_reason", ""),
        })


# ----------------------------------------------------------------------
# DK code change
# ----------------------------------------------------------------------


class ProposeDkChangeTool(_MutationToolBase):
    name = "propose_dk_change"
    operation = "dk_change"
    description = (
        "Propose adding or removing a DK classification code on the current "
        "pipeline's dk_classifications. Requires user confirmation unless "
        "autonomous mode is active. Provide a short `reason`."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "DK code (e.g. '004.42')."},
            "action": {
                "type": "string",
                "enum": ["add", "remove"],
                "description": "'add' to insert, 'remove' to drop.",
            },
            "reason": {"type": "string", "description": "Short justification for the user."},
        },
        "required": ["code", "action", "reason"],
    }

    def available_for(self, session: Any) -> bool:
        # Available whenever we have an active KAS — add works on empty,
        # remove needs codes; finer check inside execute.
        return _kas(self.pipeline_manager) is not None

    def execute(self, session: Any, **kwargs: Any) -> str:
        code = (kwargs.get("code") or "").strip()
        action = (kwargs.get("action") or "").strip().lower()
        reason = (kwargs.get("reason") or "").strip()

        kas = _kas(self.pipeline_manager)
        if kas is None:
            return json.dumps({"status": "invalid", "reason": "no_active_pipeline_state"})
        if not code or action not in ("add", "remove"):
            return json.dumps({
                "status": "invalid",
                "reason": "code_and_action_required",
            })

        current_codes: List[str] = list(kas.dk_classifications)
        if action == "add" and code in current_codes:
            return json.dumps({
                "status": "invalid",
                "reason": "code_already_present",
                "code": code,
            })
        if action == "remove" and code not in current_codes:
            return json.dumps({
                "status": "invalid",
                "reason": "code_not_present",
                "code": code,
                "available": current_codes,
            })

        payload = {"code": code, "action": action, "reason": reason}
        audit_id = self._record_pending(payload)
        decision = self._ask_user(audit_id, payload)
        accepted = bool(decision.get("accepted"))

        applied = False
        if accepted:
            try:
                applied = kas.apply_classification_update(code, action)
            except Exception as e:
                logger.exception("apply_classification_update raised")
                self._record_outcome(audit_id, False, f"apply_error: {e}")
                return json.dumps({
                    "status": "apply_error",
                    "audit_id": audit_id,
                    "error": str(e),
                })
            if applied:
                self._sync_shared_context()

        self._record_outcome(
            audit_id, accepted, decision.get("reject_reason", "")
        )

        return json.dumps({
            "audit_id": audit_id,
            "status": "applied" if (accepted and applied) else
                      ("rejected" if not accepted else "no_op"),
            "code": code,
            "action": action,
            "reject_reason": decision.get("reject_reason", ""),
        })


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def mutation_tools(
    *,
    pipeline_manager: Any,
    kb_manager: Any,
    gateway: Any,
    chat_config: Any,
    session_id: str,
) -> List[BaseChatTool]:
    """Build the mutation tool list for a chat session."""
    common = dict(
        pipeline_manager=pipeline_manager,
        kb_manager=kb_manager,
        gateway=gateway,
        chat_config=chat_config,
        session_id=session_id,
    )
    return [
        ProposeKeywordReplacementTool(**common),
        ProposeDkChangeTool(**common),
    ]
