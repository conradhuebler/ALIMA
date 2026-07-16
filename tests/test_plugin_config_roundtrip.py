"""End-to-end config migration + round-trip idempotency for the plugin model - Claude Generated.

The highest-risk part of the plugin refactor is that ~53 legacy readers keep
seeing an exact ``CatalogConfig`` / ``SystemConfig`` mirror. This locks in:

* a pre-plugin config synthesises search + input instances on load, and
* ``load → save → load`` is diff-free for the mirrors (plus a ``plugins`` section).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from src.utils.config_manager import ConfigManager


class PluginConfigRoundTripTest(unittest.TestCase):
    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self.tmp.write_text(json.dumps({
            "unified_config": {"providers": [], "gemini_api_key": "K"},
            "catalog_config": {
                "catalog_token": "TOK",
                "catalog_details_url": "https://d",
                "finc_base_url": "https://finc",
                "finc_default_limit": 25,
                "finc_dk_enabled": True,
                "sru_preset": "dnb",
            },
            "search_provider_config": {"providers": {"finc": False, "swb": True}},
            "system_config": {"contact_email": "me@x.org", "doi_use_openalex": False},
        }))
        self.cm = ConfigManager()
        self.cm.config_file = self.tmp

    def tearDown(self):
        ConfigManager.reset()

    def _reload(self):
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        return cm.load_config()

    def test_search_instances_synthesised(self):
        cfg = self.cm.load_config()
        insts = {p.instance_id: p for p in cfg.instances_for("search_provider")}
        self.assertEqual(set(insts), {"lobid", "swb", "catalog", "finc", "sru", "gnd_local"})
        self.assertEqual(insts["catalog"].settings["token"], "TOK")
        self.assertEqual(insts["finc"].settings["default_limit"], 25)
        self.assertFalse(insts["finc"].enabled)
        self.assertTrue(insts["swb"].enabled)

    def test_input_instances_synthesised(self):
        cfg = self.cm.load_config()
        insts = {p.instance_id: p for p in cfg.instances_for("input_source")}
        self.assertEqual(set(insts), {"doi_crossref", "doi_openalex", "doi_datacite", "url_fetch"})
        self.assertEqual(insts["doi_crossref"].settings["contact_email"], "me@x.org")
        self.assertFalse(insts["doi_openalex"].enabled)

    def test_round_trip_mirrors_are_diff_free(self):
        cfg = self.cm.load_config()
        cat_before = asdict(cfg.catalog_config)
        sys_before = asdict(cfg.system_config)
        self.cm.save_config(cfg)
        cfg2 = self._reload()
        self.assertEqual(asdict(cfg2.catalog_config), cat_before)
        self.assertEqual(asdict(cfg2.system_config), sys_before)
        # plugins section persisted
        raw = json.loads(self.tmp.read_text())
        self.assertIn("plugins", raw)
        self.assertTrue(len(raw["plugins"]) >= 6)

    def test_gate_and_flags_preserved(self):
        cfg = self.cm.load_config()
        self.cm.save_config(cfg)
        cfg2 = self._reload()
        self.assertFalse(cfg2.search_provider_config.is_enabled("finc"))
        self.assertTrue(cfg2.search_provider_config.is_enabled("swb"))
        self.assertFalse(cfg2.system_config.doi_use_openalex)
        self.assertEqual(cfg2.system_config.contact_email, "me@x.org")


if __name__ == "__main__":
    unittest.main()
