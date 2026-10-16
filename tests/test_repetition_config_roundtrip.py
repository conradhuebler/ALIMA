"""``repetition_config`` was written to disk and never read back.

``ConfigManager._parse_config`` built ``AlimaConfig(...)`` without the section —
the string did not occur in ``config_manager.py`` at all — so every value the
settings dialog wrote was serialised by ``asdict`` and then discarded on the next
load. On top of that ``AlimaManager._init_repetition_detector`` runs only in
``__init__`` and the GUI builds the manager once, so a changed value did not
reach the running detector either. The six controls in Settings ▸ System were
inert in both directions.

The detector itself works (``alima_manager.analyze_abstract`` feeds it every
chunk and the warning bar reacts) — only the config path was broken.
- Claude Generated
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager


class RepetitionConfigRoundTripTest(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def _load(self, payload: dict):
        """Write a config and return its manager.

        The provider block is kept for realism, not because it is required —
        ``_parse_config`` used to raise "No LLM providers configured" without
        it and its own ``except`` handed back a default ``AlimaConfig()``,
        silently discarding whatever else this fixture set; that fallback was
        removed (see ``tests/test_provider_reset_does_not_persist.py``), so an
        empty providers list no longer exercises it. - Claude Generated
        """
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
                "ngram_threshold": 3,
                "min_text_length": 250,
                "char_repeat_threshold": 20,
                "grace_period_seconds": 7.5,
                "window_size": 111,
            }
        })
        rep = cm.load_config().repetition_config
        self.assertFalse(rep.enabled)
        self.assertFalse(rep.auto_abort)
        self.assertEqual(rep.ngram_threshold, 3)
        self.assertEqual(rep.min_text_length, 250)
        self.assertEqual(rep.char_repeat_threshold, 20)
        self.assertEqual(rep.grace_period_seconds, 7.5)
        self.assertEqual(rep.window_size, 111)

    def test_absent_section_keeps_the_defaults(self):
        rep = self._load({}).load_config().repetition_config
        self.assertTrue(rep.enabled)
        self.assertEqual(rep.ngram_threshold, 8)

    def test_a_removed_field_does_not_reset_the_section(self):
        """`build_config_section` discipline: an unknown key is dropped, not fatal."""
        rep = self._load({
            "repetition_config": {"ngram_threshold": 4, "was_removed_in_v9": True}
        }).load_config().repetition_config
        self.assertEqual(rep.ngram_threshold, 4)

    def test_save_reload_is_idempotent(self):
        cm = self._load({"repetition_config": {"ngram_threshold": 5}})
        cm.save_config(cm.load_config())
        ConfigManager.reset()
        cm2 = ConfigManager()
        cm2.config_file = self.tmp
        self.assertEqual(cm2.load_config().repetition_config.ngram_threshold, 5)


class DetectorPicksUpChangedConfigTest(unittest.TestCase):
    """The second half: a changed config must reach the *running* detector."""

    def test_reload_rebuilds_the_detector(self):
        from unittest.mock import MagicMock

        from src.core.alima_manager import AlimaManager
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.repetition_config.ngram_threshold = 9
        cm = MagicMock()
        cm.load_config.return_value = cfg

        mgr = AlimaManager(MagicMock(), MagicMock(), cm, MagicMock())
        self.assertEqual(mgr.repetition_detector.config.ngram_threshold, 9)

        cfg.repetition_config.ngram_threshold = 2
        mgr.reload_repetition_config()
        self.assertEqual(mgr.repetition_detector.config.ngram_threshold, 2)


if __name__ == "__main__":
    unittest.main()
