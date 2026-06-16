"""Regression: provider key/model persistence at the right places.

Claude Generated (provider-system cleanup, Phase 2).

Covers the config-layer guarantees added while de-tangling the provider system:

* legacy ``gemini_api_key`` / ``gemini_preferred_model`` migrate onto the
  authoritative ``UnifiedProvider`` object on load (no silent drop);
* editing the provider object and saving keeps the legacy mirror in sync and
  survives a reload;
* ``save_config`` writes atomically with 0600 permissions and leaves no temp
  files behind;
* a provider-only default has its empty model filled and persisted.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.config_models import ProviderScope, UnifiedProvider


class _ConfigRoundTripBase(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self._tmp = tempfile.mkdtemp()
        self.config_file = Path(self._tmp) / "config.json"
        self.cm = ConfigManager()
        self.cm.config_file = self.config_file

    def tearDown(self):
        ConfigManager.reset()

    def _seed(self, unified: dict):
        self.config_file.write_text(json.dumps({"unified_config": unified}))

    def _reload(self):
        """Force a fresh load from disk via a new singleton."""
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.config_file
        return cm, cm.load_config()


class TestLegacyKeyMigration(_ConfigRoundTripBase):
    def test_legacy_gemini_key_and_model_migrate_to_provider(self):
        self._seed({
            "providers": [],
            "gemini_api_key": "LEGACY_GEM_KEY",
            "gemini_preferred_model": "gemini-1.5-pro",
        })
        cfg = self.cm.load_config()
        gem = cfg.unified_config.get_provider_by_name("gemini")
        self.assertIsNotNone(gem, "gemini provider must be materialized from the legacy key")
        self.assertEqual(gem.api_key, "LEGACY_GEM_KEY")
        # The preferred model must NOT be dropped during migration.
        self.assertEqual(gem.preferred_model, "gemini-1.5-pro")

    def test_provider_edit_syncs_legacy_mirror_and_survives_reload(self):
        self._seed({
            "providers": [],
            "gemini_api_key": "OLD",
            "gemini_preferred_model": "gemini-1.5-pro",
        })
        cfg = self.cm.load_config()
        cfg.unified_config.get_provider_by_name("gemini").api_key = "NEW"
        self.assertTrue(self.cm.save_config(cfg))

        on_disk = json.loads(self.config_file.read_text())["unified_config"]
        self.assertEqual(on_disk["gemini_api_key"], "NEW",
                         "legacy mirror must sync from the authoritative provider object")
        persisted = {p["name"]: p for p in on_disk["providers"]}
        self.assertEqual(persisted["gemini"]["api_key"], "NEW")

        _, cfg2 = self._reload()
        self.assertEqual(cfg2.unified_config.get_provider_by_name("gemini").api_key, "NEW")
        self.assertEqual(cfg2.unified_config.gemini_api_key, "NEW")


class TestAtomicSave(_ConfigRoundTripBase):
    def test_save_is_atomic_0600_and_leaves_no_temp_files(self):
        cfg = self.cm.load_config()
        cfg.unified_config.providers = [
            UnifiedProvider(name="ollama_local", provider_type="ollama",
                            enabled=True, preferred_model="cogito:32b")
        ]
        self.assertTrue(self.cm.save_config(cfg))

        self.assertTrue(self.config_file.exists())
        mode = stat.S_IMODE(os.stat(self.config_file).st_mode)
        self.assertEqual(mode, 0o600, f"config must be 0600, got {oct(mode)}")
        leftovers = [p.name for p in Path(self._tmp).iterdir() if p.name != "config.json"]
        self.assertEqual(leftovers, [], f"atomic write left temp files: {leftovers}")


class TestDefaultModelPersistence(_ConfigRoundTripBase):
    def test_empty_default_model_is_filled_and_persisted(self):
        self._seed({
            "providers": [{
                "name": "GWDG", "provider_type": "openai_compatible", "enabled": True,
                "preferred_model": "", "available_models": ["llama-3.3-70b", "qwen"],
            }],
            "pipeline_default_provider": "GWDG",
            "pipeline_default_model": "",  # empty → must be filled on save
        })
        cfg = self.cm.load_config()
        self.assertTrue(self.cm.save_config(cfg))

        on_disk = json.loads(self.config_file.read_text())["unified_config"]
        self.assertEqual(on_disk["pipeline_default_model"], "llama-3.3-70b",
                         "empty pipeline_default_model must be filled and persisted")

    def test_scope_enum_and_string_resolve_identically(self):
        self._seed({
            "providers": [{
                "name": "GWDG", "provider_type": "openai_compatible", "enabled": True,
                "preferred_model": "llama-3.3-70b", "available_models": ["llama-3.3-70b"],
            }],
            "pipeline_default_provider": "GWDG",
        })
        uc = self.cm.load_config().unified_config
        self.assertEqual(
            uc.resolve_default_provider_model(scope=ProviderScope.PIPELINE),
            uc.resolve_default_provider_model(scope="pipeline"),
        )


if __name__ == "__main__":
    unittest.main()
