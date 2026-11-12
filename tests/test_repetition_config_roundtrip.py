"""``repetition_config`` round trip, and migration of the former detector keys.

History: until Sep 2026 the section was written to disk and never read back,
and ``AlimaManager`` read it only in ``__init__`` (the GUI builds the manager
once). Both halves are pinned here.

Nov 2026: the n-gram/window/character detector was replaced by the
information-rate measurement. Saved configs still carry the old keys; they must
neither fail the load nor be reported as typos on every start, and
``enabled``/``auto_abort``/``grace_period_seconds`` carry over.
- Claude Generated
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager


OLD_SECTION = {
    "enabled": False,
    "auto_abort": False,
    "show_suggestions": True,
    "grace_period_seconds": 7.5,
    "ngram_size": 6,
    "ngram_threshold": 3,
    "window_size": 111,
    "window_similarity_threshold": 0.9,
    "min_windows": 4,
    "char_repeat_threshold": 20,
    "min_text_length": 250,
    "check_interval": 200,
}


class RepetitionConfigRoundTripTest(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def _load(self, payload: dict):
        """Write a config and return its manager - Claude Generated"""
        payload = {"unified_config": {"providers": [], "gemini_api_key": "K"}, **payload}
        self.tmp.write_text(json.dumps(payload))
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        return cm

    def test_values_on_disk_reach_the_config(self):
        cm = self._load({
            "repetition_config": {
                "enabled": False,
                "auto_abort": False,
                "grace_period_seconds": 7.5,
                "warn_threshold": 0.2,
                "abort_threshold": 0.1,
                "thinking_warn_threshold": 0.15,
                "thinking_abort_threshold": 0.05,
                "repetition_action": "retry",
                "classic_repetition_action": "abort",
                "max_retries": 2,
            }
        })
        rep = cm.load_config().repetition_config
        self.assertFalse(rep.enabled)
        self.assertFalse(rep.auto_abort)
        self.assertEqual(rep.grace_period_seconds, 7.5)
        self.assertEqual(rep.warn_threshold, 0.2)
        self.assertEqual(rep.abort_threshold, 0.1)
        self.assertEqual(rep.thinking_warn_threshold, 0.15)
        self.assertEqual(rep.thinking_abort_threshold, 0.05)
        self.assertEqual(rep.repetition_action, "retry")
        self.assertEqual(rep.classic_repetition_action, "abort")
        self.assertEqual(rep.max_retries, 2)

    def test_absent_section_keeps_the_defaults(self):
        from src.utils.config_models import RepetitionDetectionConfig

        rep = self._load({}).load_config().repetition_config
        self.assertEqual(rep, RepetitionDetectionConfig())
        self.assertEqual(rep.repetition_action, "agent")
        self.assertEqual(rep.classic_repetition_action, "retry")

    def test_a_removed_field_does_not_reset_the_section(self):
        """`build_config_section` discipline: an unknown key is dropped, not fatal."""
        rep = self._load({
            "repetition_config": {"warn_threshold": 0.3, "was_removed_in_v9": True}
        }).load_config().repetition_config
        self.assertEqual(rep.warn_threshold, 0.3)

    def test_old_detector_keys_migrate_quietly(self):
        """Gate: old keys never reach the typo warning, three values kept.

        Checked at the seam (what ``build_config_section`` receives) rather
        than through log capture, which other tests' logging setup can mute.
        """
        from unittest import mock

        from src.utils import config_manager as cm_mod
        from src.utils.config_models import RepetitionDetectionConfig

        real = cm_mod.build_config_section
        seen = {}

        def spy(cls, data, *, context):
            if context == "repetition_config":
                seen["data"] = dict(data or {})
            return real(cls, data, context=context)

        cm = self._load({"repetition_config": dict(OLD_SECTION)})
        with mock.patch.object(cm_mod, "build_config_section", side_effect=spy):
            rep = cm.load_config().repetition_config
        self.assertEqual(
            sorted(seen["data"]), ["auto_abort", "enabled", "grace_period_seconds"],
            "retired keys must be dropped before build_config_section names them as typos",
        )
        self.assertFalse(rep.enabled)
        self.assertFalse(rep.auto_abort)
        self.assertEqual(rep.grace_period_seconds, 7.5)
        # the new measurement starts from its defaults
        defaults = RepetitionDetectionConfig()
        self.assertEqual(rep.warn_threshold, defaults.warn_threshold)
        self.assertEqual(rep.abort_threshold, defaults.abort_threshold)

    def test_old_keys_disappear_on_save(self):
        cm = self._load({"repetition_config": dict(OLD_SECTION)})
        cm.save_config(cm.load_config())
        saved = json.loads(self.tmp.read_text())["repetition_config"]
        for key in ("ngram_threshold", "char_repeat_threshold", "window_size", "show_suggestions"):
            self.assertNotIn(key, saved)
        self.assertIn("warn_threshold", saved)
        self.assertFalse(saved["enabled"])

    def test_invalid_values_are_normalised_not_fatal(self):
        rep = self._load({
            "repetition_config": {
                "repetition_action": "explode", "classic_repetition_action": "agent",
                "max_retries": 99, "warn_threshold": 0.05, "abort_threshold": 0.2,
            }
        }).load_config().repetition_config
        self.assertEqual(rep.repetition_action, "agent")
        self.assertEqual(rep.classic_repetition_action, "retry")
        self.assertEqual(rep.max_retries, 3)
        self.assertLessEqual(rep.abort_threshold, rep.warn_threshold)

    def test_save_reload_is_idempotent(self):
        cm = self._load({"repetition_config": {"warn_threshold": 0.25}})
        cm.save_config(cm.load_config())
        ConfigManager.reset()
        cm2 = ConfigManager()
        cm2.config_file = self.tmp
        self.assertEqual(cm2.load_config().repetition_config.warn_threshold, 0.25)


class DetectorPicksUpChangedConfigTest(unittest.TestCase):
    """The second half: a changed config must reach the *running* manager."""

    def test_reload_rebuilds_the_detector(self):
        from unittest.mock import MagicMock

        from src.core.alima_manager import AlimaManager
        from src.utils.config_models import AlimaConfig, RepetitionDetectionConfig

        cfg = AlimaConfig()
        cfg.repetition_config.warn_threshold = 0.3
        cm = MagicMock()
        cm.load_config.return_value = cfg

        mgr = AlimaManager(MagicMock(), MagicMock(), cm, MagicMock())
        self.assertEqual(mgr.repetition_config.warn_threshold, 0.3)

        # A settings save loads a new object (not the same one mutated).
        cfg.repetition_config = RepetitionDetectionConfig(warn_threshold=0.2)
        mgr.reload_repetition_config()
        self.assertEqual(mgr.repetition_config.warn_threshold, 0.2)


if __name__ == "__main__":
    unittest.main()
