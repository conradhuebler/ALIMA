"""End-to-end proof of the blueprint workflow - Claude Generated.

Copy a built-in plugin dir → rename its id → drop it into a plugins root →
load it through the REAL code-plugin path (scan, approval, package import,
registration) → run a search on the loaded provider. This is exactly what a
third party does with `cp -r lobid ~/.config/alima/plugins/mine`.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import src.core.search  # noqa: F401  side-effect: registers category + built-ins
from src.core.plugins import loader as loader_mod
from src.core.search.provider import SearchCapability
from src.core.search.registry import PROVIDER_REGISTRY, get_provider

BLUEPRINT = Path(__file__).resolve().parent.parent / "src" / "core" / "search" / "providers" / "lobid"
COPY_ID = "lobid_copy"


class _FakeSuggester:
    """Canned suggester so the E2E search needs no network. - Claude Generated"""

    last_errors = {}
    last_raw = {}
    last_http_status = {}

    def search(self, terms, **kwargs):
        return {t: {"Wasserstoff": {"count": 3, "gnd_ids": {"4064784-5"}, "classifications": {"DDC": {"546"}}}} for t in terms}


class BlueprintEndToEndTest(unittest.TestCase):
    def setUp(self):
        loader_mod._reset_for_tests()
        self._pre_registry = dict(PROVIDER_REGISTRY)
        self.root = Path(tempfile.mkdtemp())

    def tearDown(self):
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(self._pre_registry)
        loader_mod._reset_for_tests()
        for name in [m for m in sys.modules if m.startswith("alima_plugin_")]:
            sys.modules.pop(name, None)
        shutil.rmtree(self.root, ignore_errors=True)

    def _copy_and_rename(self) -> Path:
        dst = self.root / "myplugin"
        shutil.copytree(BLUEPRINT, dst, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("plugin.toml", "provider.py"):
            f = dst / name
            f.write_text(f.read_text(encoding="utf-8").replace('"lobid"', f'"{COPY_ID}"'), encoding="utf-8")
        return dst

    def test_copy_rename_load_search(self):
        self._copy_and_rename()

        approvals = []

        def approve(manifest, findings, digest):
            approvals.append((manifest.id, digest))
            return True

        result = loader_mod.discover(
            self.root, approved_plugins={}, approve_cb=approve, enable_code_plugins=True
        )
        self.assertEqual(len(result.plugins), 1)
        self.assertEqual(result.plugins[0].status, "loaded", result.plugins[0].detail)
        self.assertEqual(result.plugins[0].type_id, COPY_ID)
        self.assertEqual(approvals[0][0], COPY_ID)
        # the seeded instance is what makes the plugin visible + tool-exposed
        self.assertEqual([i.provider_id for i in result.instances], [COPY_ID])
        self.assertTrue(result.instances[0].enabled)

        # The copy's intra-plugin relative import must resolve through the
        # synthetic package (this is what breaks with single-file loading).
        import importlib

        sug_mod = importlib.import_module(f"alima_plugin_{COPY_ID}.suggester")
        self.assertTrue(hasattr(sug_mod, "LobidSuggester"))

        cls = get_provider(COPY_ID)
        provider = cls()
        provider._cache_raw = False  # no UKM in this test
        provider._suggester = _FakeSuggester()

        out = provider.search(SearchCapability.GND_KEYWORDS, ["Wasser"])
        keywords = out.to_gnd_keywords()
        self.assertIn("Wasser", keywords)
        self.assertIn("Wasserstoff", keywords["Wasser"])
        self.assertEqual(out.errors, {})

    def test_toml_only_rename_is_refused_before_import(self):
        """Operator regression (July 6): id renamed in plugin.toml but NOT in
        provider.py → must fail with the precise fix hint, statically, without
        touching the registry. - Claude Generated"""
        dst = self.root / "halfrenamed"
        shutil.copytree(BLUEPRINT, dst, ignore=shutil.ignore_patterns("__pycache__"))
        toml = dst / "plugin.toml"
        toml.write_text(
            toml.read_text(encoding="utf-8").replace('id = "lobid"', 'id = "lobid_plugin"'),
            encoding="utf-8",
        )
        result = loader_mod.discover(
            self.root, approved_plugins={}, approve_cb=lambda *a: True, enable_code_plugins=True
        )
        self.assertEqual(result.plugins[0].status, "error")
        self.assertIn('still has id = "lobid"', result.plugins[0].detail)
        self.assertIn('id = "lobid_plugin"', result.plugins[0].detail)
        self.assertIs(get_provider("lobid"), self._pre_registry["lobid"])
        self.assertFalse([m for m in sys.modules if m.startswith("alima_plugin_lobid_plugin")])

    def test_unrenamed_copy_is_rejected_with_hint(self):
        dst = self.root / "unrenamed"
        shutil.copytree(BLUEPRINT, dst, ignore=shutil.ignore_patterns("__pycache__"))
        result = loader_mod.discover(
            self.root, approved_plugins={}, approve_cb=lambda *a: True, enable_code_plugins=True
        )
        self.assertEqual(result.plugins[0].status, "error")
        self.assertIn("rename the plugin id", result.plugins[0].detail)
        # built-in registration untouched
        self.assertIs(get_provider("lobid"), self._pre_registry["lobid"])

    def test_second_discover_skips_reimport(self):
        self._copy_and_rename()
        approved = {}
        kw = dict(approved_plugins=approved, approve_cb=lambda *a: True, enable_code_plugins=True)
        first = loader_mod.discover(self.root, **kw)
        self.assertEqual(first.plugins[0].status, "loaded")
        second = loader_mod.discover(self.root, **kw)
        self.assertEqual(second.plugins[0].status, "loaded")
        self.assertIn("already loaded", second.plugins[0].detail)


if __name__ == "__main__":
    unittest.main()
