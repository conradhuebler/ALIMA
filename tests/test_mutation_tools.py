"""P-ε — Mutation Tools tests. Claude Generated.

Covers:

1. ``chat_mutations`` schema + record_mutation_{pending,outcome}.
2. ``ProposeKeywordReplacementTool`` accept/reject/autonomous flow.
3. ``ProposeDkChangeTool`` accept/reject + invalid action handling.
4. ``ProposalGateway`` cross-thread sync via a real QThread.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QApplication
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.data_models import KeywordAnalysisState
    from src.ui.chat_tools.mutations import (
        ProposeKeywordReplacementTool,
        ProposeDkChangeTool,
    )
    from src.ui.chat_tools.proposal_gateway import ProposalGateway
    PYQT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    PYQT_IMPORT_ERROR = exc


# Singleton QApplication for Qt-bound tests.
_qapp = None


def _ensure_qapp():
    global _qapp
    if _qapp is None and PYQT_IMPORT_ERROR is None:
        _qapp = QApplication.instance() or QApplication([])
    return _qapp


def _make_sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


# ---------------------------------------------------------------------------
# Test class 1: chat_mutations schema + record methods
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestChatMutationsSchema(unittest.TestCase):

    def setUp(self):
        _ensure_qapp()
        UnifiedKnowledgeManager.reset()
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.km = UnifiedKnowledgeManager(
            database_config=_make_sqlite_config(self.temp_db.name)
        )

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass

    def test_chat_mutations_table_exists(self):
        rows = self.km.db_manager.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        names = {r["name"] for r in rows}
        self.assertIn("chat_mutations", names)

    def test_chat_mutations_indexes_exist(self):
        rows = self.km.db_manager.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        names = {r["name"] for r in rows}
        self.assertIn("idx_chat_mutations_session", names)
        self.assertIn("idx_chat_mutations_created", names)

    def test_record_pending_then_outcome_accepted(self):
        audit_id = self.km.record_mutation_pending(
            "sess1", "propose_keyword_replacement",
            "keyword_replacement",
            {"old": "Cd", "new": "Cadmium"},
        )
        self.assertIsInstance(audit_id, int)

        self.km.record_mutation_outcome(audit_id, True)
        row = self.km.db_manager.fetch_one(
            "SELECT accepted, reject_reason, applied_at FROM chat_mutations WHERE id = ?",
            [audit_id],
        )
        self.assertEqual(int(row["accepted"]), 1)
        self.assertIsNotNone(row["applied_at"])

    def test_record_outcome_rejected_stores_reason(self):
        audit_id = self.km.record_mutation_pending(
            "sess1", "propose_dk_change", "dk_change",
            {"code": "004.42", "action": "add"},
        )
        self.km.record_mutation_outcome(audit_id, False, "user_declined")
        row = self.km.db_manager.fetch_one(
            "SELECT accepted, reject_reason FROM chat_mutations WHERE id = ?",
            [audit_id],
        )
        self.assertEqual(int(row["accepted"]), 0)
        self.assertEqual(row["reject_reason"], "user_declined")

    def test_pending_row_has_null_accepted(self):
        audit_id = self.km.record_mutation_pending(
            "sess1", "propose_keyword_replacement",
            "keyword_replacement",
            {"old": "x", "new": "y"},
        )
        # PyQt6 QSqlQuery returns NULL BOOLEAN as either None or "" depending on
        # dialect; both are falsy and distinct from the typed True/False stored
        # by record_mutation_outcome.
        row = self.km.db_manager.fetch_one(
            "SELECT accepted FROM chat_mutations WHERE id = ?",
            [audit_id],
        )
        self.assertIn(row["accepted"], (None, ""))


# ---------------------------------------------------------------------------
# Tool test scaffolding
# ---------------------------------------------------------------------------


def _make_kas(initial_keywords=None, dk_classifications=None):
    kas = KeywordAnalysisState(
        original_abstract="abstract",
        initial_keywords=list(initial_keywords or []),
        search_suggesters_used=["lobid"],
    )
    if dk_classifications:
        kas.dk_classifications = list(dk_classifications)
    return kas


def _make_tool_deps(kas, autonomous=False, gateway_decision=None):
    """Build pipeline_manager mock, kb mock, gateway mock, chat_config mock."""
    pipeline_manager = SimpleNamespace(current_analysis_state=kas)

    kb = MagicMock()
    kb.record_mutation_pending.return_value = 42
    kb.record_mutation_outcome.return_value = None

    gateway = MagicMock()
    gateway.request_decision.return_value = (
        gateway_decision or {"accepted": True, "reject_reason": ""}
    )

    chat_config = SimpleNamespace(autonomous_pipeline=autonomous)
    return pipeline_manager, kb, gateway, chat_config


# ---------------------------------------------------------------------------
# Test class 2: ProposeKeywordReplacementTool
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestProposeKeywordReplacementTool(unittest.TestCase):

    def _build(self, kas, **kwargs):
        pm, kb, gw, cc = _make_tool_deps(kas, **kwargs)
        tool = ProposeKeywordReplacementTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="sess_test",
        )
        return tool, pm, kb, gw

    def test_accept_applies_mutation(self):
        kas = _make_kas(initial_keywords=["Cadmium", "Soil"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            old="Cadmium", new="Schwermetall", reason="more general",
        ))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["audit_id"], 42)
        self.assertIn("Schwermetall", kas.initial_keywords[0])
        kb.record_mutation_pending.assert_called_once()
        kb.record_mutation_outcome.assert_called_once_with(42, True, "")
        gw.request_decision.assert_called_once()

    def test_reject_skips_mutation(self):
        kas = _make_kas(initial_keywords=["Cadmium"])
        tool, _pm, kb, gw = self._build(
            kas, gateway_decision={"accepted": False, "reject_reason": "no"},
        )
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            old="Cadmium", new="Schwermetall", reason="r",
        ))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(kas.initial_keywords, ["Cadmium"])
        kb.record_mutation_outcome.assert_called_once_with(42, False, "no")

    def test_autonomous_skips_gateway(self):
        kas = _make_kas(initial_keywords=["Cadmium"])
        tool, _pm, kb, gw = self._build(kas, autonomous=True)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            old="Cadmium", new="Schwermetall", reason="r",
        ))
        self.assertEqual(result["status"], "applied")
        gw.request_decision.assert_not_called()
        kb.record_mutation_outcome.assert_called_once_with(42, True, "")

    def test_invalid_old_keyword_short_circuits(self):
        kas = _make_kas(initial_keywords=["Cadmium"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            old="Unicorn", new="Schwermetall", reason="r",
        ))
        self.assertEqual(result["status"], "invalid")
        kb.record_mutation_pending.assert_not_called()
        gw.request_decision.assert_not_called()


# ---------------------------------------------------------------------------
# Test class 3: ProposeDkChangeTool
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestProposeDkChangeTool(unittest.TestCase):

    def _build(self, kas, **kwargs):
        pm, kb, gw, cc = _make_tool_deps(kas, **kwargs)
        tool = ProposeDkChangeTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="sess_test",
        )
        return tool, pm, kb, gw

    def test_add_accept_applies(self):
        kas = _make_kas(dk_classifications=["004.42"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            code="005.13", action="add", reason="r",
        ))
        self.assertEqual(result["status"], "applied")
        self.assertIn("005.13", kas.dk_classifications)

    def test_remove_accept_applies(self):
        kas = _make_kas(dk_classifications=["004.42", "005.13"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            code="004.42", action="remove", reason="r",
        ))
        self.assertEqual(result["status"], "applied")
        self.assertNotIn("004.42", kas.dk_classifications)

    def test_add_when_already_present_is_invalid(self):
        kas = _make_kas(dk_classifications=["004.42"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            code="004.42", action="add", reason="r",
        ))
        self.assertEqual(result["status"], "invalid")
        gw.request_decision.assert_not_called()

    def test_remove_when_absent_is_invalid(self):
        kas = _make_kas(dk_classifications=["004.42"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            code="999.99", action="remove", reason="r",
        ))
        self.assertEqual(result["status"], "invalid")

    def test_unknown_action_is_invalid(self):
        kas = _make_kas(dk_classifications=["004.42"])
        tool, _pm, kb, gw = self._build(kas)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            code="004.42", action="toggle", reason="r",
        ))
        self.assertEqual(result["status"], "invalid")


# ---------------------------------------------------------------------------
# Test class 4: ProposalGateway cross-thread sync
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestProposalGateway(unittest.TestCase):

    def setUp(self):
        _ensure_qapp()
        self.gateway = ProposalGateway()

    def test_timeout_returns_rejected(self):
        result = self.gateway.request_decision(
            audit_id=1, tool_name="x", payload={}, timeout_ms=50
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reject_reason"], "timeout")

    def test_resolve_in_other_thread_releases_waiter(self):
        # Worker thread calls request_decision; main thread resolves.
        result_box: dict = {}

        def worker():
            result_box["res"] = self.gateway.request_decision(
                audit_id=99, tool_name="t", payload={}, timeout_ms=5000
            )

        t = threading.Thread(target=worker)
        t.start()
        # Give the worker time to register the waiter.
        for _ in range(50):
            if 99 in self.gateway.pending_audit_ids():
                break
            QApplication.processEvents()
            t.join(timeout=0.01)
        else:
            t.join(timeout=2.0)
            self.fail("worker never registered audit_id 99")

        self.gateway.resolve_decision(99, accepted=True, reject_reason="")
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "worker did not return after resolve_decision")
        self.assertEqual(result_box["res"]["accepted"], True)

    def test_resolve_unknown_id_is_noop(self):
        # Should not raise.
        self.gateway.resolve_decision(999_999, accepted=True)


if __name__ == "__main__":
    unittest.main()
