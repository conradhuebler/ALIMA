"""Built-in provider dirs double as copyable blueprints - Claude Generated.

Every package dir under ``src/core/search/providers/`` must carry a *valid*
code-plugin manifest (plugin.toml) + README.md, and the manifest must be
consistent with the registered provider class (entry module/class exist,
class ``id`` == manifest ``id``). Built-in manifests are never scanned at
runtime — this test is what keeps them loadable.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import src.core.search  # noqa: F401  side-effect: registers all providers
from src.core.plugins.manifest import load_manifest_file
from src.core.search.registry import get_provider

PROVIDERS_DIR = Path(__file__).resolve().parent.parent / "src" / "core" / "search" / "providers"


def _blueprint_dirs():
    return sorted(
        d for d in PROVIDERS_DIR.iterdir()
        if d.is_dir() and d.name != "__pycache__" and (d / "plugin.toml").is_file()
    )


class BuiltinPluginManifestTest(unittest.TestCase):
    def test_blueprint_dirs_exist(self):
        names = {d.name for d in _blueprint_dirs()}
        self.assertEqual(
            names, {"lobid", "swb", "catalog", "finc", "sru", "gnd_local"},
            "every built-in provider is a self-contained blueprint dir",
        )

    def test_manifests_parse_and_match_registered_class(self):
        for d in _blueprint_dirs():
            with self.subTest(plugin=d.name):
                manifest = load_manifest_file(d / "plugin.toml")
                self.assertTrue(manifest.is_code, f"{d.name}: blueprint must be a code plugin")
                self.assertEqual(manifest.category, "search_provider")
                self.assertEqual(manifest.id, d.name, "manifest id should match dir name")
                entry = d / manifest.entry_module
                self.assertTrue(entry.is_file(), f"{d.name}: entry module missing")
                cls = get_provider(manifest.id)  # registered via built-in import
                self.assertEqual(cls.__name__, manifest.entry_class)
                self.assertEqual(getattr(cls, "id", None), manifest.id)
                self.assertTrue(manifest.description, f"{d.name}: [doc] description required")

    def test_readme_present(self):
        for d in _blueprint_dirs():
            with self.subTest(plugin=d.name):
                self.assertTrue((d / "README.md").is_file(), f"{d.name}: README.md missing")

    def test_init_is_pure_reexport(self):
        # __init__.py is built-in-mode glue and must stay trivial (it is NOT
        # executed for external copies).
        for d in _blueprint_dirs():
            with self.subTest(plugin=d.name):
                src = (d / "__init__.py").read_text(encoding="utf-8")
                self.assertIn("from .provider import", src)


if __name__ == "__main__":
    unittest.main()
