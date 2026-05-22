"""ProposalGateway — cross-thread sync for mutation confirmations (P-ε).

Mutation chat-tools (`propose_keyword_replacement`, `propose_dk_change`)
execute inside ``ChatAgentWorker``'s QThread, but the inline confirmation
bubble is rendered + clicked on the main (UI) thread. This gateway
bridges them: the tool blocks on a ``QSemaphore`` while the UI emits a
signal, renders the bubble, and finally calls ``resolve_decision`` from
the click handler to release the semaphore with the user's answer.

Claude Generated.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import QObject, QSemaphore, pyqtSignal


class ProposalGateway(QObject):
    """Qt-safe blocking handoff between worker-thread tools and the UI."""

    # Emitted on the worker thread; Qt marshals to the UI thread via
    # the connection's default (auto) delivery. PipelineChatPanel
    # connects this to its `_render_proposal_bubble` slot.
    proposal_requested = pyqtSignal(int, str, dict)  # audit_id, tool_name, payload

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._waiters: Dict[int, Tuple[QSemaphore, Dict[str, Any]]] = {}

    def request_decision(
        self,
        audit_id: int,
        tool_name: str,
        payload: Dict[str, Any],
        timeout_ms: int = 120_000,
    ) -> Dict[str, Any]:
        """Block the calling thread until the user accepts/rejects.

        Returns ``{"accepted": bool, "reject_reason": str}``. On timeout,
        returns ``{"accepted": False, "reject_reason": "timeout"}``.
        """
        sem = QSemaphore(0)
        result: Dict[str, Any] = {}
        self._waiters[audit_id] = (sem, result)
        try:
            self.proposal_requested.emit(audit_id, tool_name, payload)
            acquired = sem.tryAcquire(1, timeout_ms)
        finally:
            self._waiters.pop(audit_id, None)
        if not acquired:
            return {"accepted": False, "reject_reason": "timeout"}
        return result

    def resolve_decision(
        self,
        audit_id: int,
        accepted: bool,
        reject_reason: str = "",
    ) -> None:
        """Release a pending waiter with the user's decision."""
        entry = self._waiters.get(audit_id)
        if entry is None:
            return
        sem, result = entry
        result["accepted"] = bool(accepted)
        result["reject_reason"] = reject_reason or ""
        sem.release()

    def pending_audit_ids(self) -> list[int]:
        """Audit ids currently waiting for a user decision."""
        return list(self._waiters.keys())
