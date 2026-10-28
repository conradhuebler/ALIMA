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
        # agentic_default ist der allgemeine Standard geworden
        self.assertEqual(config.unified_config.preferred_model, "gemma4:31b-cloud")
        self.assertEqual([p.name for p in config.unified_config.providers], ["LLMachine"])


class AgenticDefaultFoldsIntoGeneralTest(unittest.TestCase):
    """Agentik, Chat und Allgemein sind eine Einstellung: ein früher gesetzter
    agentic_default wird der allgemeine Standard (er war der, mit dem Chat und
    Agentik tatsächlich liefen). Claude Generated."""

    _load = LoadWithRemovedKeysTest._load

    def _unified(self, **kw):
        return self._load({
            "config_version": "2.0",
            "database_config": {"db_type": "sqlite", "sqlite_path": "/tmp/alima-test.db"},
            "unified_config": {"providers": [], **kw},
        }).unified_config

    def test_agentic_default_becomes_the_general_default(self):
        u = self._unified(agentic_default_provider="A", agentic_default_model="a1")
        self.assertEqual((u.preferred_provider, u.preferred_model), ("A", "a1"))

    def test_it_wins_over_a_differing_general_default(self):
        u = self._unified(agentic_default_provider="A", agentic_default_model="a1",
                          preferred_provider="B", preferred_model="b1")
        self.assertEqual((u.preferred_provider, u.preferred_model), ("A", "a1"))

    def test_an_existing_general_default_is_kept_when_agentic_is_unset(self):
        u = self._unified(preferred_provider="B", preferred_model="b1")
        self.assertEqual((u.preferred_provider, u.preferred_model), ("B", "b1"))

    def test_the_legacy_keys_are_not_written_back(self):
        import json
        import tempfile
        from pathlib import Path

        from src.utils.config_manager import ConfigManager

        cfg = self._load({
            "config_version": "2.0",
            "database_config": {"db_type": "sqlite", "sqlite_path": "/tmp/alima-test.db"},
            "unified_config": {"agentic_default_provider": "A", "agentic_default_model": "a1",
                               "providers": []},
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"unified_config": {
                "agentic_default_provider": "A", "agentic_default_model": "a1"}}))
            manager = ConfigManager()
            original = manager.config_file
            manager.config_file = path
            try:
                manager.save_config(cfg)
                written = json.loads(path.read_text())
            finally:
                manager.config_file = original
        self.assertNotIn("agentic_default_provider", written.get("unified_config", {}))
        self.assertNotIn("agentic_default_model", written.get("unified_config", {}))


if __name__ == "__main__":
    unittest.main()


class RemovedSettingsDoNotResetTheConfigTest(unittest.TestCase):
    """The Sep-2026 removals, on a config written before them.

    Removed: SystemConfig.debug/data_dir/temp_dir (no readers — the log level
    beside them was wired up instead), the whole PromptConfig section (three
    fields, zero readers repo-wide), ChatConfig.system_prompt_override,
    DatabaseConfig.auto_create_tables, PluginInstanceConfig.is_primary, and
    AlimaConfig.created_at/updated_at. Every existing config.json carries some of
    them. - Claude Generated
    """

    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def test_an_old_config_keeps_everything_that_still_exists(self):
        self.tmp.write_text(json.dumps({
            "unified_config": {"providers": [], "gemini_api_key": "KEEP-ME"},
            "system_config": {
                "debug": True, "log_level": "DEBUG",
                "data_dir": "/old/data", "temp_dir": "/old/tmp",
                "autosave_dir": "/keep/results", "contact_email": "me@x.org",
            },
            "prompt_config": {"prompts_file": "old.json", "prompt_cache_enabled": False},
            "chat_config": {"system_prompt_override": "gone", "max_iterations": 12},
            "database_config": {"auto_create_tables": False, "sqlite_path": "/keep/db.sqlite"},
            "plugins": [{"instance_id": "lobid", "category": "search_provider",
                         "provider_id": "lobid", "enabled": True, "is_primary": True,
                         "settings": {}}],
            "created_at": "2025-01-01", "updated_at": "2025-06-01",
        }))
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        cfg = cm.load_config()

        # Nothing was reset by the unknown keys.
        self.assertEqual(cfg.system_config.autosave_dir, "/keep/results")
        self.assertEqual(cfg.system_config.contact_email, "me@x.org")
        self.assertEqual(cfg.database_config.sqlite_path, "/keep/db.sqlite")
        self.assertEqual(cfg.chat_config.max_iterations, 12)
        self.assertEqual(cfg.unified_config.gemini_api_key, "KEEP-ME")
        self.assertTrue(any(p.provider_id == "lobid" for p in cfg.plugins))

        # The removed names are gone from the model.
        for obj, name in [
            (cfg.system_config, "debug"),
            (cfg.system_config, "data_dir"),
            (cfg.system_config, "temp_dir"),
            (cfg.chat_config, "system_prompt_override"),
            (cfg.database_config, "auto_create_tables"),
            (cfg, "prompt_config"),
            (cfg, "created_at"),
        ]:
            self.assertFalse(hasattr(obj, name), f"{name} sollte entfernt sein")

    def test_a_legacy_log_level_string_still_maps(self):
        """The combo used to store DEBUG/INFO/...; those values must keep working."""
        from src.utils.logging_utils import log_level_to_verbosity

        self.assertEqual(log_level_to_verbosity("DEBUG"), 2)
        self.assertEqual(log_level_to_verbosity("INFO"), 1)
        self.assertEqual(log_level_to_verbosity("CRITICAL"), 0)
        self.assertEqual(log_level_to_verbosity("debug"), 2)
        self.assertEqual(log_level_to_verbosity("bogus"), 1)


if __name__ == "__main__":
    unittest.main()
