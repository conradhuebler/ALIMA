"""A removed config field must not reset the whole configuration.

Claude Generated. Every section used to be built with ``Cls(**data)``, which
raises ``TypeError`` on a key the dataclass does not have. That exception is
caught far up in ``_load_config_from_file``, whose answer is a complete
``AlimaConfig()`` — so one leftover key from an older version silently drops
providers, database path and every other setting.

That is not hypothetical: ``ChatConfig.default_provider``/``default_model`` were
removed (they outranked the settings and no surface could show them), and every
existing config.json still carries them.
"""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager, build_config_section
from src.utils.config_models import ChatConfig, DatabaseConfig


class BuildConfigSectionTest(unittest.TestCase):
    def test_known_keys_are_applied(self):
        cfg = build_config_section(
            ChatConfig, {"max_iterations": 7, "temperature": 0.1}, context="chat_config"
        )
        self.assertEqual(cfg.max_iterations, 7)
        self.assertEqual(cfg.temperature, 0.1)

    def test_an_unknown_key_is_dropped_instead_of_raising(self):
        cfg = build_config_section(
            ChatConfig,
            {"max_iterations": 7, "default_provider": "LLMachine"},
            context="chat_config",
        )
        self.assertEqual(cfg.max_iterations, 7)
        self.assertFalse(hasattr(cfg, "default_provider"))

    def test_the_dropped_key_is_named_in_the_log(self):
        # Silently dropping would hide a typo in a hand-edited config.
        logging.disable(logging.NOTSET)
        with self.assertLogs("src.utils.config_manager", level="WARNING") as caught:
            build_config_section(ChatConfig, {"deafult_provider": "x"}, context="chat_config")
        self.assertIn("deafult_provider", "\n".join(caught.output))

    def test_an_empty_section_yields_defaults(self):
        self.assertEqual(build_config_section(ChatConfig, {}, context="chat_config"),
                         ChatConfig())
        self.assertEqual(build_config_section(ChatConfig, None, context="chat_config"),
                         ChatConfig())

    def test_the_strict_construction_really_would_raise(self):
        # The mutation this test file guards against.
        with self.assertRaises(TypeError):
            ChatConfig(**{"default_provider": "LLMachine"})


class LoadWithRemovedKeysTest(unittest.TestCase):
    """End to end: an old config.json keeps its settings."""

    def _load(self, payload: dict):
        manager = ConfigManager()
        original = manager.config_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manager.config_file = path
            try:
                return manager._load_config_from_file()
            finally:
                manager.config_file = original

    def test_a_leftover_chat_default_does_not_wipe_the_rest(self):
        config = self._load({
            "config_version": "2.0",
            "chat_config": {
                "max_iterations": 11,
                "default_provider": "LLMachine",          # removed field
                "default_model": "north-mini-code-1.0:latest",  # removed field
            },
            "database_config": {"db_type": "sqlite", "sqlite_path": "/tmp/alima-test.db"},
            "unified_config": {
                "agentic_default_provider": "LLMachine",
                "agentic_default_model": "gemma4:31b-cloud",
                "providers": [
                    {"name": "LLMachine", "provider_type": "openai_compatible",
                     "enabled": True},
                ],
            },
        })
        # The section with the removed keys still parsed…
        self.assertEqual(config.chat_config.max_iterations, 11)
        # …and, the point of the test, nothing else was reset to defaults.
        self.assertEqual(config.database_config.sqlite_path, "/tmp/alima-test.db")
        self.assertEqual(config.unified_config.agentic_default_model, "gemma4:31b-cloud")
        self.assertEqual([p.name for p in config.unified_config.providers], ["LLMachine"])


if __name__ == "__main__":
    unittest.main()
