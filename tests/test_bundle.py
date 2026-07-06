"""Tests for institutional bundle deploy (build/install/list/remove) - Claude Generated.

Uses an injected fake ConfigManager so the tests are deterministic and never touch
the real ~/.config/alima (and dodge the "no LLM providers" load guard). The fake's
``save_config`` re-runs ``derive_search_mirrors`` exactly like the real one, so the
tests catch the derived-mirror interaction (a profile must toggle instances, not the
gate). Bundles are built on the fly under a temp dir.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from src.utils import bundle as B
from src.utils.config_models import AlimaConfig
from src.utils.plugin_migration import derive_search_mirrors, synthesize_search_instances

_DECL_SRU = (
    '[plugin]\nid = "demo_sru"\nlabel = "Demo SRU"\n'
    'category = "search_provider"\ntype = "declarative"\nkind = "sru"\n'
    '[settings]\npreset = "k10plus"\n'
)


class _FakeCM:
    """In-memory ConfigManager stub with a real save-side mirror derivation."""

    def __init__(self, config, plugins_dir):
        self._config = config
        self._pd = Path(plugins_dir)

    @property
    def plugins_dir(self) -> Path:
        return self._pd

    def load_config(self, force_reload: bool = False):
        return self._config

    def save_config(self, config, **kw) -> bool:
        try:
            derive_search_mirrors(
                config.plugins, config.catalog_config, config.search_provider_config
            )
        except Exception:
            pass
        self._config = config
        return True


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class BundleTest(unittest.TestCase):
    def setUp(self):
        from src.core.plugins import loader

        loader._reset_for_tests()
        self.tmp = Path(tempfile.mkdtemp())
        self.plugins_dir = self.tmp / "installed_plugins"
        cfg = AlimaConfig()
        cfg.plugins = synthesize_search_instances(cfg.catalog_config, cfg.search_provider_config)
        self.cm = _FakeCM(cfg, self.plugins_dir)

    def tearDown(self):
        from src.core.plugins import loader

        loader._reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_bundle(self, *, profile=None, secrets="", plugins=None) -> Path:
        bd = self.tmp / "src_bundle"
        if bd.exists():
            shutil.rmtree(bd)
        _write(bd / "bundle.toml",
               '[bundle]\nid = "testb"\nversion = "1.0"\nlabel = "Test"\n' + secrets)
        for name, toml in (plugins or {"demo_sru": _DECL_SRU}).items():
            _write(bd / "plugins" / name / "plugin.toml", toml)
        if profile is not None:
            _write(bd / "profile.json", json.dumps(profile))
        return bd

    # -- build --------------------------------------------------------------
    def test_build_writes_approvals_and_zip(self):
        bd = self._make_bundle()
        B.build_bundle(bd)
        approvals = json.loads((bd / "approvals.json").read_text())
        self.assertIn("demo_sru", approvals)
        self.assertEqual(len(approvals["demo_sru"]), 64)  # sha256 hex

        zip_path = self.tmp / "out.zip"
        out = B.build_bundle(bd, zip_path)
        self.assertEqual(out, zip_path)
        self.assertTrue(zip_path.is_file())

    # -- install ------------------------------------------------------------
    def test_install_seeds_instance_and_records_ledger(self):
        bd = self._make_bundle()
        report = B.install_bundle(bd, config_manager=self.cm)

        self.assertEqual(report.bundle_id, "testb")
        self.assertEqual(report.plugins, [("demo_sru", "loaded", "none")])
        cfg = self.cm.load_config()
        self.assertIn("demo_sru", [p.instance_id for p in cfg.plugins])
        rec = cfg.installed_bundles["testb"]
        self.assertEqual(rec["plugin_ids"], ["demo_sru"])
        self.assertEqual(rec["instance_ids"], ["demo_sru"])  # instance_id == plugin id
        self.assertTrue((self.plugins_dir / "demo_sru" / "plugin.toml").is_file())

    def test_install_disables_builtin_via_profile(self):
        # A profile that "disables swb" must toggle the swb *instance* — the gate
        # alone would be clobbered by derive_search_mirrors on save.
        bd = self._make_bundle(profile={"search_provider_config": {"providers": {"swb": False}}})
        B.install_bundle(bd, config_manager=self.cm)
        cfg = self.cm.load_config()
        swb = [p for p in cfg.plugins if p.provider_id == "swb"]
        self.assertTrue(swb and all(not p.enabled for p in swb))
        self.assertFalse(cfg.search_provider_config.is_enabled("swb"))  # derived mirror agrees

    def test_install_reports_required_secret(self):
        secrets = ('\n[secrets]\nrequired = [{ plugin = "demo_sru", key = "token", '
                   'hint = "personal token" }]\n')
        bd = self._make_bundle(secrets=secrets)
        report = B.install_bundle(bd, config_manager=self.cm)
        self.assertEqual(len(report.required_secrets), 1)
        s = report.required_secrets[0]
        self.assertEqual(s["env_var"], "ALIMA_PLUGIN_DEMO_SRU_TOKEN")
        self.assertFalse(s["satisfied"])

    def test_integrity_mismatch_reported(self):
        bd = self._make_bundle()
        B.build_bundle(bd)  # writes approvals.json with the pristine hash
        # tamper a plugin file after building → hash no longer matches approvals
        _write(bd / "plugins" / "demo_sru" / "extra.txt", "changed")
        report = B.install_bundle(bd, config_manager=self.cm)
        self.assertIn("demo_sru", report.integrity_mismatches)

    # -- profile whitelist (user-config protection) -------------------------
    def test_profile_rejects_forbidden_section(self):
        bd = self._make_bundle(profile={"unified_config": {"gemini_api_key": "leak"}})
        with self.assertRaises(B.BundleError):
            B.install_bundle(bd, config_manager=self.cm)

    def test_profile_rejects_secret_system_key(self):
        bd = self._make_bundle(profile={"system_config": {"anthropic_api_key": "leak"}})
        with self.assertRaises(B.BundleError):
            B.install_bundle(bd, config_manager=self.cm)

    # -- list / remove ------------------------------------------------------
    def test_list_and_remove_roundtrip(self):
        bd = self._make_bundle(
            profile={
                "search_provider_config": {"providers": {"swb": False}},
                "system_config": {"url_fetch_allowlist": ["x.example"]},
            }
        )
        B.install_bundle(bd, config_manager=self.cm)

        listed = B.list_bundles(config_manager=self.cm)
        self.assertEqual([b["id"] for b in listed], ["testb"])

        B.remove_bundle("testb", config_manager=self.cm)
        cfg = self.cm.load_config()
        self.assertNotIn("testb", cfg.installed_bundles)
        self.assertEqual([p.instance_id for p in cfg.plugins if p.instance_id == "demo_sru"], [])
        self.assertNotIn("demo_sru", cfg.approved_plugins)
        self.assertFalse((self.plugins_dir / "demo_sru").exists())
        # profile keys restored to their pre-install values
        self.assertTrue(cfg.search_provider_config.is_enabled("swb"))
        self.assertEqual(cfg.system_config.url_fetch_allowlist, [])

    def test_remove_unknown_raises(self):
        with self.assertRaises(B.BundleError):
            B.remove_bundle("nope", config_manager=self.cm)

    # -- config field serialization ----------------------------------------
    def test_installed_bundles_survives_serialization(self):
        cfg = AlimaConfig()
        cfg.installed_bundles = {"b": {"version": "1", "plugin_ids": ["p"]}}
        restored = json.loads(json.dumps(asdict(cfg), default=str))
        self.assertEqual(restored["installed_bundles"], {"b": {"version": "1", "plugin_ids": ["p"]}})


if __name__ == "__main__":
    unittest.main()
