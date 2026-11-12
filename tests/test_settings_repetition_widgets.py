"""Settings dialog ▸ Wiederholungserkennung: widgets load and write the config - Claude Generated

Uses the real dialog over a stubbed config (``_DialogHarness`` from
``test_settings_dialog_state``). ``segment_chars``/``smoothing_segments`` have
no widget and must survive a save.
"""

from __future__ import annotations

import unittest

from tests.test_settings_dialog_state import _DialogHarness


class RepetitionWidgetsTest(_DialogHarness, unittest.TestCase):
    def configure(self, cfg):
        rep = cfg.repetition_config
        rep.warn_threshold = 0.2
        rep.abort_threshold = 0.1
        rep.thinking_warn_threshold = 0.15
        rep.thinking_abort_threshold = 0.05
        rep.repetition_action = "retry"
        rep.classic_repetition_action = "abort"
        rep.max_retries = 2
        rep.grace_period_seconds = 3.5
        rep.segment_chars = 600
        rep.smoothing_segments = 5

    def test_loaded_values_are_shown(self):
        dlg = self._dialog()
        self.assertAlmostEqual(dlg.repetition_warn_threshold.value(), 0.2)
        self.assertAlmostEqual(dlg.repetition_abort_threshold.value(), 0.1)
        self.assertAlmostEqual(dlg.repetition_thinking_warn_threshold.value(), 0.15)
        self.assertEqual(dlg.repetition_action_combo.currentData(), "retry")
        self.assertEqual(dlg.classic_repetition_action_combo.currentData(), "abort")
        self.assertEqual(dlg.repetition_max_retries.value(), 2)

    def test_edited_values_are_written(self):
        dlg = self._dialog()
        dlg.repetition_warn_threshold.setValue(0.13)
        dlg.repetition_abort_threshold.setValue(0.3)  # above warn → clamped
        dlg.repetition_thinking_abort_threshold.setValue(0.02)
        dlg.repetition_action_combo.setCurrentIndex(dlg.repetition_action_combo.findData("agent"))
        dlg.classic_repetition_action_combo.setCurrentIndex(
            dlg.classic_repetition_action_combo.findData("retry")
        )
        dlg.repetition_max_retries.setValue(0)
        dlg.repetition_auto_abort.setChecked(False)
        rep = dlg._get_config_from_ui().repetition_config
        self.assertAlmostEqual(rep.warn_threshold, 0.13)
        self.assertAlmostEqual(rep.abort_threshold, 0.13)
        self.assertAlmostEqual(rep.thinking_abort_threshold, 0.02)
        self.assertEqual(rep.repetition_action, "agent")
        self.assertEqual(rep.classic_repetition_action, "retry")
        self.assertEqual(rep.max_retries, 0)
        self.assertFalse(rep.auto_abort)
        # no widget → unchanged
        self.assertEqual(rep.segment_chars, 600)
        self.assertEqual(rep.smoothing_segments, 5)


if __name__ == "__main__":
    unittest.main()
