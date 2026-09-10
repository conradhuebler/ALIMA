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

    def test_default_selects_libero(self):
        cfg = _cfg(_StubInstance("catalog", settings={"token": "tok"}))
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


class LegacyCatalogTypeMigrationTest(unittest.TestCase):
    """``catalog_type`` chose the DK backend before sru had its own ``dk_enabled``.

    It was declared vestigial in July 2026 but stayed in the settings form as a
    prominent "DK-Backend" combo, so the form claimed to pick a backend that the
    precedence chain in ``resolve_dk_extractor`` really decided. The field is gone;
    ``migrate_catalog_type`` carries the operator's old choice over once, at load,
    so no library silently changes backend. - Claude Generated
    """

    def _plugins(self, catalog_settings, sru_settings=None):
        from src.utils.config_models import PluginInstanceConfig

        out = [PluginInstanceConfig(
            "catalog", "search_provider", "catalog", settings=dict(catalog_settings))]
        if sru_settings is not None:
            out.append(PluginInstanceConfig(
                "sru", "search_provider", "sru", settings=dict(sru_settings)))
        return out

    def _migrate(self, plugins):
        from src.utils.plugin_migration import migrate_catalog_type

        migrate_catalog_type(plugins)
        return {p.provider_id: p for p in plugins}

    def test_marcxml_sru_becomes_dk_enabled(self):
        by_id = self._migrate(self._plugins({"catalog_type": "marcxml_sru"}, {"preset": "dnb"}))
        self.assertIs(by_id["sru"].settings["dk_enabled"], True)
        self.assertNotIn("catalog_type", by_id["catalog"].settings)

    def test_auto_with_a_configured_endpoint_becomes_dk_enabled(self):
        by_id = self._migrate(self._plugins(
            {"catalog_type": "auto"}, {"base_url": "https://sru.example/x"}))
        self.assertIs(by_id["sru"].settings["dk_enabled"], True)

    def test_auto_without_an_endpoint_stays_off(self):
        by_id = self._migrate(self._plugins({"catalog_type": "auto"}, {}))
        self.assertFalse(by_id["sru"].settings.get("dk_enabled"))

    def test_libero_soap_stays_off(self):
        by_id = self._migrate(self._plugins({"catalog_type": "libero_soap"}, {"preset": "dnb"}))
        self.assertFalse(by_id["sru"].settings.get("dk_enabled"))
        self.assertNotIn("catalog_type", by_id["catalog"].settings)

    def test_migration_never_disables_an_enabled_backend(self):
        """It only ever turns DK on: a legacy 'libero_soap' cannot switch sru off."""
        by_id = self._migrate(self._plugins(
            {"catalog_type": "libero_soap"}, {"dk_enabled": True}))
        self.assertIs(by_id["sru"].settings["dk_enabled"], True)

    def test_migration_is_idempotent(self):
        plugins = self._plugins({"catalog_type": "marcxml_sru"}, {"preset": "dnb"})
        from src.utils.plugin_migration import migrate_catalog_type

        self.assertTrue(migrate_catalog_type(plugins))
        self.assertFalse(migrate_catalog_type(plugins))

    def test_the_migrated_config_still_resolves_to_sru(self):
        """End to end: the whole point is that the backend does not change."""
        plugins = self._plugins({"catalog_type": "marcxml_sru"}, {"preset": "dnb"})
        self._migrate(plugins)
        cfg = _cfg(*[_StubInstance(p.provider_id, settings=p.settings) for p in plugins])
        self.assertEqual(type(resolve_dk_extractor(config=cfg)).__name__, "MarcXmlClient")


if __name__ == "__main__":
    unittest.main()
