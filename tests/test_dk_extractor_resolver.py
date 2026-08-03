"""Claude Generated - Tests for the capability-driven DK/RVK extractor resolver.

``resolve_dk_extractor`` (src/core/search/factory.py) replaced the hand-wired
FincCatalogClient/MarcXmlClient/BiblioClient if-elif in the classic DK step
(``execute_notation_search``). Since WP P4 every backend is built through the factory
from its own instance settings (no more CatalogConfig kwargs). These tests pin:

1. **Built-in parity** — each config selects the same backend the old if-elif did
   (finc opt-in via its ``dk_enabled`` → SRU opt-in via its ``dk_enabled`` /
   legacy ``catalog_type`` → Libero default), and finc/SRU stay opt-in.
2. **Extensibility** — a NEW, non-built-in provider declaring the CLASSIFICATION
   capability + a ``dk_extractor()`` is picked up as a DK source with no core
   change, taking precedence over the Libero default.
"""

import unittest

from src.core.search.factory import resolve_dk_extractor
from src.core.search.provider import SearchCapability
from src.core.search.registry import PROVIDER_REGISTRY, register_provider


class _FakeExtractor:
    def __init__(self):
        self.calls = []

    def extract_dk_classifications_for_keywords(self, keywords, max_results=50):
        self.calls.append((tuple(keywords), max_results))
        return [
            {"keyword": k, "source": "mylib",
             "classifications": [{"dk": "530", "type": "DK", "count": 1}]}
            for k in keywords
        ]


class _StubInstance:
    def __init__(self, provider_id, enabled=True, settings=None):
        self.provider_id = provider_id
        self.instance_id = provider_id
        self.enabled = enabled
        self.settings = settings or {}


class _StubConfig:
    def __init__(self, instances):
        self._instances = instances

    def enabled_instances_for(self, category):
        return [i for i in self._instances if getattr(i, "enabled", True)]


def _cfg(*instances):
    return _StubConfig(list(instances))


class BuiltinParityTest(unittest.TestCase):
    """Each config selects the same backend the old catalog_type if-elif did —
    now driven by per-instance settings (WP P4)."""

    def test_finc_opt_in_selects_finc(self):
        cfg = _cfg(_StubInstance("finc", settings={
            "dk_enabled": True, "base_url": "https://ex.org/proxy.php"}))
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "FincCatalogClient")

    def test_finc_not_opted_in_falls_through_to_libero(self):
        # A finc base_url alone (without dk_enabled) must NOT hijack the DK backend.
        cfg = _cfg(
            _StubInstance("finc", settings={
                "dk_enabled": False, "base_url": "https://ex.org/proxy.php"}),
            _StubInstance("catalog", settings={"catalog_type": "libero_soap"}),
        )
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "BiblioClient")

    def test_sru_dk_enabled_selects_sru(self):
        cfg = _cfg(_StubInstance("sru", settings={"dk_enabled": True, "preset": "dnb"}))
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "MarcXmlClient")

    def test_legacy_catalog_type_marcxml_sru_still_selects_sru(self):
        # Transition fallback: configs migrated before sru.dk_enabled existed keep
        # their SRU backend via the catalog instance's catalog_type. - Claude Generated
        cfg = _cfg(
            _StubInstance("sru", settings={"preset": "dnb"}),
            _StubInstance("catalog", settings={"catalog_type": "marcxml_sru"}),
        )
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "MarcXmlClient")

    def test_legacy_catalog_type_auto_with_sru_configured_selects_sru(self):
        cfg = _cfg(
            _StubInstance("sru", settings={"base_url": "https://sru.example/x"}),
            _StubInstance("catalog", settings={"catalog_type": "auto"}),
        )
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "MarcXmlClient")

    def test_default_selects_libero(self):
        cfg = _cfg(_StubInstance("catalog", settings={
            "catalog_type": "libero_soap", "token": "tok"}))
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "BiblioClient")

    def test_no_catalog_instance_still_builds_libero_default(self):
        # Empty config → the registry-default catalog (web-scraping fallback).
        ex = resolve_dk_extractor(config=_cfg())
        self.assertEqual(type(ex).__name__, "BiblioClient")

    def test_precedence_finc_over_sru(self):
        cfg = _cfg(
            _StubInstance("finc", settings={
                "dk_enabled": True, "base_url": "https://ex.org/proxy.php"}),
            _StubInstance("sru", settings={"dk_enabled": True, "preset": "dnb"}),
        )
        ex = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ex).__name__, "FincCatalogClient")


class ExtensibilityTest(unittest.TestCase):
    """A library without finc/Libero ships its own DK/RVK plugin — no core edit."""

    def setUp(self):
        self._added = []

    def tearDown(self):
        for pid in self._added:
            PROVIDER_REGISTRY.pop(pid, None)

    def _register(self, pid, *, capabilities, extractor=None):
        ext = extractor or _FakeExtractor()
        caps = set(capabilities)

        @register_provider
        class _P:
            id = pid
            label = pid

            def __init__(self, **cfg):
                self._cfg = cfg

            def is_available(self, cfg=None):
                return True

            def dk_extractor(self, **_ignore):
                return ext

        _P.capabilities = caps  # set outside the class body (class scope can't see `caps`)
        self._added.append(pid)
        return ext

    def test_custom_classification_plugin_picked_over_libero(self):
        ext = self._register("mylib_catalog", capabilities={SearchCapability.CLASSIFICATION})
        cfg = _cfg(_StubInstance("mylib_catalog"),
                   _StubInstance("catalog", settings={"catalog_type": "libero_soap"}))
        got = resolve_dk_extractor(config=cfg)
        self.assertIs(got, ext)

    def test_custom_plugin_yields_to_finc_opt_in(self):
        self._register("mylib_catalog", capabilities={SearchCapability.CLASSIFICATION})
        cfg = _cfg(
            _StubInstance("mylib_catalog"),
            _StubInstance("finc", settings={
                "dk_enabled": True, "base_url": "https://ex.org/proxy.php"}),
        )
        got = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(got).__name__, "FincCatalogClient")

    def test_non_classification_custom_plugin_ignored(self):
        # A custom provider WITHOUT the CLASSIFICATION capability is not a DK source.
        self._register("mylib_titles", capabilities={SearchCapability.TITLE_RECORDS})
        cfg = _cfg(_StubInstance("mylib_titles"),
                   _StubInstance("catalog", settings={"catalog_type": "libero_soap"}))
        got = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(got).__name__, "BiblioClient")

    def test_custom_extractor_honours_shared_contract(self):
        ext = self._register("mylib_catalog", capabilities={SearchCapability.CLASSIFICATION})
        cfg = _cfg(_StubInstance("mylib_catalog"))
        got = resolve_dk_extractor(config=cfg)
        out = got.extract_dk_classifications_for_keywords(["Physik"], max_results=10)
        self.assertEqual(out[0]["classifications"][0]["dk"], "530")
        self.assertEqual(ext.calls, [(("Physik",), 10)])


if __name__ == "__main__":
    unittest.main()
