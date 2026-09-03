"""ProposalBar — one-shot confirmation strip between chat log and input.

Claude Generated.

Confirmations used to be an inline HTML block in the chat log with
``Akzeptieren``/``Ablehnen`` anchors. Two things were wrong with that: the
anchors stayed live after the decision, so the operator could keep clicking a
question that had already been answered, and a decision surface that scrolls
away with the log is easy to miss while tokens are still arriving.

This bar takes over the decision. It sits directly above the input field, shows
one proposal at a time, and disappears the moment it is answered — the log keeps
the record of what was proposed and what was decided.

The blocking handoff is unchanged: the tool waits on ``ProposalGateway``, the
gateway emits ``proposal_requested`` on the UI thread, the host panel forwards it
here, and a button press resolves the semaphore.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Headline per tool. Unknown tools fall back to a generic line rather than
#: showing nothing — a confirmation must never be anonymous.
_TITLES = {
    "propose_rule": "📌 Als dauerhafte Regel ablegen?",
    "delete_rule": "🗑️ Regel endgültig löschen?",
    "propose_keyword_replacement": "🔁 Keyword ersetzen?",
    "propose_dk_change": "🏷️ DK-Klassifikation ändern?",
    "run_pipeline": "▶️ Pipeline starten?",
    "rerun_step": "🔁 Schritt erneut ausführen?",
}

_STYLE = """
QFrame#proposalBar {
    background-color: #3d2a00;
    border: 1px solid #ffb86c;
    border-radius: 4px;
}
QLabel#proposalTitle { color: #ffb86c; font-weight: bold; }
QLabel#proposalBody  { color: #f8f8f2; }
QLabel#proposalMeta  { color: #b0b0b0; font-size: 9pt; }
QPushButton#accept {
    background-color: #2e7d32; color: #fff; border: none;
    border-radius: 3px; padding: 4px 14px; font-weight: bold;
}
QPushButton#accept:hover { background-color: #43a047; }
QPushButton#reject {
    background-color: #6d2020; color: #fff; border: none;
    border-radius: 3px; padding: 4px 14px; font-weight: bold;
}
QPushButton#reject:hover { background-color: #922b2b; }
"""


class ProposalBar(QFrame):
    """One pending confirmation, or hidden."""

    #: ``(audit_id, accepted)`` — emitted exactly once per proposal.
    decided = pyqtSignal(int, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("proposalBar")
        self.setStyleSheet(_STYLE)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._audit_id: Optional[int] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(3)

        self.title_label = QLabel("")
        self.title_label.setObjectName("proposalTitle")
        outer.addWidget(self.title_label)

        self.body_label = QLabel("")
        self.body_label.setObjectName("proposalBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        outer.addWidget(self.body_label)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("proposalMeta")
        self.meta_label.setWordWrap(True)
        outer.addWidget(self.meta_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch()
        self.reject_btn = QPushButton("✗ Ablehnen")
        self.reject_btn.setObjectName("reject")
        self.accept_btn = QPushButton("✓ Annehmen")
        self.accept_btn.setObjectName("accept")
        self.accept_btn.setDefault(True)
        buttons.addWidget(self.reject_btn)
        buttons.addWidget(self.accept_btn)
        outer.addLayout(buttons)

        self.accept_btn.clicked.connect(lambda: self._decide(True))
        self.reject_btn.clicked.connect(lambda: self._decide(False))

        self.setVisible(False)

    # -- public API ----------------------------------------------------

    def show_proposal(self, audit_id: int, tool_name: str, payload: Dict[str, Any]) -> None:
        """Display a proposal and wait for a click."""
        self._audit_id = int(audit_id)
        self.title_label.setText(_TITLES.get(tool_name, f"⚠️ Bestätigung: {tool_name}"))
        self.body_label.setText(_describe(tool_name, payload))
        self.meta_label.setText(_meta(tool_name, payload))
        self.meta_label.setVisible(bool(self.meta_label.text()))
        self.setVisible(True)
        self.accept_btn.setFocus()

    def dismiss(self) -> None:
        """Hide without deciding (e.g. the run was cancelled)."""
        self._audit_id = None
        self.setVisible(False)

    @property
    def pending_audit_id(self) -> Optional[int]:
        return self._audit_id

    # -- internals -----------------------------------------------------

    def _decide(self, accepted: bool) -> None:
        """Emit once, then hide — the same question is never answered twice."""
        audit_id = self._audit_id
        if audit_id is None:
            return
        self._audit_id = None
        self.setVisible(False)
        self.decided.emit(audit_id, bool(accepted))


def _describe(tool_name: str, payload: Dict[str, Any]) -> str:
    """The change itself, in one or two lines."""
    if tool_name in ("propose_rule", "delete_rule"):
        text = str(payload.get("text", "")).strip()
        when = str(payload.get("applies_when", "") or "").strip()
        return f"Nur {when}: {text}" if when else text
    if tool_name == "propose_keyword_replacement":
        old = str(payload.get("old", ""))
        new = str(payload.get("new", ""))
        gnd = str(payload.get("gnd_id", "") or "")
        return f"{old} → {new}" + (f"  (GND-ID: {gnd})" if gnd else "")
    if tool_name == "propose_dk_change":
        verb = "hinzufügen" if str(payload.get("action", "")) == "add" else "entfernen"
        return f"{payload.get('code', '')} ({verb})"
    return ", ".join(f"{k}: {v}" for k, v in payload.items() if k != "reason")


def _meta(tool_name: str, payload: Dict[str, Any]) -> str:
    """Scope and reason — the context the decision needs."""
    parts = []
    scope = str(payload.get("scope", "") or "").strip()
    if scope and tool_name in ("propose_rule", "delete_rule"):
        parts.append(f"Gilt für: {scope}")
    reason = str(payload.get("reason", "") or "").strip()
    if reason:
        parts.append(f"Begründung: {reason}")
    return "   ·   ".join(parts)
