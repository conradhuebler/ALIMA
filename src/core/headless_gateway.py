"""Headless proposal gateways — Qt-free permission gates for the CLI/HTTP
agent frontends.

Claude Generated (P-ι).

The GUI uses ``src/ui/chat_tools/proposal_gateway.ProposalGateway`` — a
``QSemaphore`` bridge that blocks the worker thread until the user clicks an
inline accept/reject bubble. Headless frontends have no such bubble, so they
inject one of the gateways here. Both expose the same duck-typed contract the
mutation/pipeline tools call (see ``_MutationToolBase._ask_user`` in
``src/ui/chat_tools/mutations.py``)::

    request_decision(audit_id, tool_name, payload, timeout_ms=...) -> {
        "accepted": bool, "reject_reason": str
    }

Note: when ``ChatConfig.autonomous_pipeline`` is True, ``_ask_user`` short-
circuits to accept *before* touching the gateway, so neither class is invoked
in autonomous mode.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional, TextIO

logger = logging.getLogger(__name__)


def _summarize_payload(payload: Dict[str, Any], max_len: int = 200) -> str:
    """Compact one-line rendering of a proposal payload for a prompt."""
    try:
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        s = str(payload)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


class StdinProposalGateway:
    """Interactive y/n confirmation on a terminal (CLI default).

    Prints the proposal to ``stream`` (stderr by default, so it never pollutes
    a ``--output``-bound stdout JSON) and reads a single line from ``stdin``.
    Accepts ``y``/``yes``/``j``/``ja``; everything else rejects. A non-tty or
    EOF rejects with ``reject_reason="non_interactive"`` so a piped invocation
    never silently applies a mutation.
    """

    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        stdin: Optional[TextIO] = None,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._stdin = stdin if stdin is not None else sys.stdin

    def request_decision(
        self,
        audit_id: Optional[int],
        tool_name: str,
        payload: Dict[str, Any],
        timeout_ms: int = 120_000,
    ) -> Dict[str, Any]:
        stdin = self._stdin
        # Non-interactive (pipe, EOF, closed): fail safe to reject.
        if stdin is None or not hasattr(stdin, "readline"):
            return {"accepted": False, "reject_reason": "non_interactive"}
        try:
            is_tty = stdin.isatty()
        except Exception:
            is_tty = False
        if not is_tty:
            return {"accepted": False, "reject_reason": "non_interactive"}

        self._stream.write(
            f"\n⚠️  Bestätigung erforderlich — {tool_name}\n"
            f"    {_summarize_payload(payload)}\n"
            f"    Ausführen? [y/N] "
        )
        self._stream.flush()
        try:
            line = stdin.readline()
        except Exception:
            return {"accepted": False, "reject_reason": "non_interactive"}
        if not line:  # EOF
            return {"accepted": False, "reject_reason": "non_interactive"}

        answer = line.strip().lower()
        if answer in ("y", "yes", "j", "ja"):
            return {"accepted": True, "reject_reason": ""}
        return {"accepted": False, "reject_reason": "user_declined"}


class AutoRejectGateway:
    """Non-interactive gateway that always rejects.

    Used by the HTTP endpoint when ``autonomous`` is not set: there is no
    channel to prompt the caller mid-stream, so any confirmation-gated
    operation is refused rather than silently applied.
    """

    def request_decision(
        self,
        audit_id: Optional[int],
        tool_name: str,
        payload: Dict[str, Any],
        timeout_ms: int = 120_000,
    ) -> Dict[str, Any]:
        logger.info("AutoRejectGateway rejected %s (non-autonomous headless)", tool_name)
        return {"accepted": False, "reject_reason": "non_interactive"}
