"""Tests for the extracted RepetitionWarningBar (F-5 split). Claude Generated.

The repetition-warning panel was moved verbatim out of ``pipeline_chat_panel.py``
into ``repetition_warning_bar.py`` as a self-contained ``QFrame``. It has no
QWebEngineView, so it constructs under the shared offscreen QApplication — we
drive its colour state machine and signals directly. We also stub-test the panel
delegators (the panel keeps the public ``show/hide_repetition_warning`` API and
re-wires the bar's signals).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ui.repetition_warning_bar import RepetitionWarningBar
from src.ui.pipeline_chat_panel import PipelineChatPanel


class TestRepetitionWarningBarStateMachine(unittest.TestCase):

    def _bar(self) -> RepetitionWarningBar:
        return RepetitionWarningBar()

    def test_show_then_resolve_then_hide_transitions(self):
        bar = self._bar()
        try:
            self.assertEqual(bar._warning_style_state, "hidden")
            bar.show_warning("ngram", "details", [{"label": "A", "params": {"x": 1}}])
            self.assertEqual(bar._warning_style_state, "orange")
            self.assertEqual(bar.warning_details_label.text(), "details")
            self.assertEqual(bar.warning_title_label.text(), "Phrasenwiederholung erkannt")
            # one suggestion button was built
            self.assertEqual(bar.suggestions_button_layout.count(), 1)

            # resolve → green
            bar.hide_warning(resolved=True)
            self.assertEqual(bar._warning_style_state, "green")
            self.assertEqual(bar.warning_icon_label.text(), "✅")

            # dismiss → hidden
            bar.hide_warning()
            self.assertEqual(bar._warning_style_state, "hidden")
        finally:
            bar.deleteLater()

    def test_repeat_same_detection_type_is_deduped(self):
        bar = self._bar()
        try:
            bar.show_warning("char_pattern", "d1", [{"label": "A"}])
            self.assertEqual(bar.suggestions_button_layout.count(), 1)
            # Same type again while orange → already_showing branch, no rebuild,
            # details still update.
            bar.show_warning("char_pattern", "d2", [{"label": "A"}, {"label": "B"}])
            self.assertEqual(bar._warning_style_state, "orange")
            self.assertEqual(bar.warning_details_label.text(), "d2")
            self.assertEqual(bar.suggestions_button_layout.count(), 1)
        finally:
            bar.deleteLater()

    def test_grace_period_starts_countdown(self):
        bar = self._bar()
        try:
            bar.show_warning("ngram", "d", [], grace_period=True, grace_seconds=2.0)
            self.assertTrue(bar.grace_timer.isActive())
            self.assertIn("2.0s", bar.countdown_label.text())
        finally:
            bar.grace_timer.stop()
            bar.deleteLater()

    def test_abort_button_emits_and_hides(self):
        bar = self._bar()
        try:
            fired = []
            bar.abort_requested.connect(lambda: fired.append(True))
            bar.show_warning("ngram", "d", [{"label": "A"}])
            bar._on_abort_requested()
            self.assertEqual(fired, [True])
            self.assertEqual(bar._warning_style_state, "hidden")
        finally:
            bar.deleteLater()

    def test_suggestion_click_emits_params_and_hides(self):
        bar = self._bar()
        try:
            payloads = []
            bar.retry_requested.connect(payloads.append)
            bar.show_warning("ngram", "d", [{"label": "A", "params": {"temperature": 0.9}}])
            bar._on_suggestion_clicked({"temperature": 0.9})
            self.assertEqual(payloads, [{"temperature": 0.9}])
            self.assertEqual(bar._warning_style_state, "hidden")
        finally:
            bar.deleteLater()


class TestPanelDelegation(unittest.TestCase):
    """The panel forwards to the bar and re-wires its signals (stubbed)."""

    def test_show_repetition_warning_delegates(self):
        stub = SimpleNamespace(repetition_bar=MagicMock())
        PipelineChatPanel.show_repetition_warning(
            stub, "ngram", "d", [{"label": "A"}], grace_period=True, grace_seconds=1.5
        )
        stub.repetition_bar.show_warning.assert_called_once_with(
            "ngram", "d", [{"label": "A"}], True, 1.5
        )

    def test_hide_repetition_warning_delegates(self):
        stub = SimpleNamespace(repetition_bar=MagicMock())
        PipelineChatPanel.hide_repetition_warning(stub, resolved=True)
        stub.repetition_bar.hide_warning.assert_called_once_with(True)

    def test_retry_reemits_signal_and_logs_with_step_id(self):
        stub = SimpleNamespace(
            retry_with_variations=MagicMock(),
            add_pipeline_message=MagicMock(),
            current_step_id="search",
        )
        PipelineChatPanel._on_repetition_retry(stub, {"temperature": 0.9})
        stub.retry_with_variations.emit.assert_called_once_with({"temperature": 0.9})
        stub.add_pipeline_message.assert_called_once()
        args = stub.add_pipeline_message.call_args[0]
        self.assertIn("Retry mit Parametern", args[0])
        self.assertEqual(args[2], "search")

    def test_abort_reemits_signal(self):
        stub = SimpleNamespace(abort_generation_requested=MagicMock())
        PipelineChatPanel._on_repetition_abort(stub)
        stub.abort_generation_requested.emit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
