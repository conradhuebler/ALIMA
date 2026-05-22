"""Generic chat tools — SharedContext inspection. WP10 P-δ.2. Claude Generated.

Session-scoped, workflow-agnostic. Operate on
``session.last_shared_context`` (an instance of
``src.core.agents.shared_context.SharedContext``) and
``session.messages`` (the chat history).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.ui.chat_tools.base import BaseChatTool


def _shared_context(session: Any):
    """Return session.last_shared_context or None if missing."""
    return getattr(session, "last_shared_context", None)


class ListAvailableDataTool(BaseChatTool):
    name = "list_available_data"
    description = (
        "List which pipeline state slots are populated for the current "
        "chat session. Returns slot names + counts so the agent can decide "
        "which deeper tool to call next."
    )
    parameters_schema = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        return True  # Always available so agent can check pipeline state

    def execute(self, session: Any, **_: Any) -> str:
        ctx = _shared_context(session)
        if ctx is None:
            return json.dumps({"slots_populated": [], "counts": {}})
        data = ctx.to_dict()
        populated: List[str] = []
        counts: Dict[str, int] = {}
        for key, value in data.items():
            if not value:
                continue
            populated.append(key)
            if isinstance(value, (list, dict, str)):
                counts[key] = len(value)
        return json.dumps(
            {"slots_populated": populated, "counts": counts},
            ensure_ascii=False,
        )


class GetExtraTool(BaseChatTool):
    name = "get_extra"
    description = (
        "Fetch a free-form value from SharedContext.extra by key "
        "(workflow-defined scratch storage)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Key in shared_context.extra"},
        },
        "required": ["key"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _shared_context(session)
        return bool(ctx and ctx.extra)

    def execute(self, session: Any, key: str = "", **_: Any) -> str:
        ctx = _shared_context(session)
        if ctx is None or not key:
            return json.dumps({"error": "Missing context or empty key"})
        value = ctx.extra.get(key)
        if value is None:
            return json.dumps(
                {"key": key, "value": None, "available_keys": sorted(ctx.extra.keys())}
            )
        return json.dumps({"key": key, "value": value}, ensure_ascii=False, default=str)


class GetStepResultTool(BaseChatTool):
    name = "get_step_result"
    description = (
        "Fetch the raw result of a completed pipeline step by id "
        "(e.g. 'extraction', 'selection', 'classification')."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "description": "Pipeline step id"},
        },
        "required": ["step_id"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _shared_context(session)
        return bool(ctx and ctx.step_results)

    def execute(self, session: Any, step_id: str = "", **_: Any) -> str:
        ctx = _shared_context(session)
        if ctx is None or not step_id:
            return json.dumps({"error": "Missing context or empty step_id"})
        if step_id not in ctx.step_results:
            return json.dumps(
                {
                    "error": f"Unknown step: {step_id}",
                    "available_steps": sorted(ctx.step_results.keys()),
                }
            )
        return json.dumps(
            {"step_id": step_id, "result": ctx.step_results[step_id]},
            ensure_ascii=False,
            default=str,
        )


class GetMessagesHistoryTool(BaseChatTool):
    name = "get_messages_history"
    description = (
        "Return chat history from the current session. Supports pagination: "
        "use `last_n` for recent turns, `offset` to skip from the start."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "last_n": {
                "type": "integer",
                "description": "How many recent turns to return (default 5).",
                "minimum": 1,
                "maximum": 200,
            },
            "offset": {
                "type": "integer",
                "description": "Skip this many turns from the start of history. Use to paginate older messages.",
                "default": 0,
                "minimum": 0,
            },
        },
    }

    def available_for(self, session: Any) -> bool:
        return bool(getattr(session, "messages", None))

    def execute(self, session: Any, last_n: int = 5, offset: int = 0, **_: Any) -> str:
        messages = getattr(session, "messages", []) or []
        total = len(messages)
        if last_n < 1:
            last_n = 1
        if offset < 0:
            offset = 0
        start = offset
        end = min(offset + last_n, total)
        slice_ = messages[start:end]
        return json.dumps(
            {
                "count": len(slice_),
                "total": total,
                "offset": offset,
                "has_more": end < total,
                "messages": slice_,
            },
            ensure_ascii=False,
        )


def generic_tools() -> List[BaseChatTool]:
    """Instantiate one of each generic tool. Used by build_chat_toolset."""
    return [
        ListAvailableDataTool(),
        GetExtraTool(),
        GetStepResultTool(),
        GetMessagesHistoryTool(),
    ]
