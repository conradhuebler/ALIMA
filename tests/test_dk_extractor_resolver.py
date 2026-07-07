"""Claude Generated - Tests for the capability-driven DK/RVK extractor resolver.

``resolve_dk_extractor`` (src/core/search/factory.py) replaced the hand-wired
FincCatalogClient/MarcXmlClient/BiblioClient if-elif in the classic DK step
(``execute_dk_search``). These tests pin:

1. **Built-in parity** — each config selects the same backend the old if-elif did
   (finc opt-in → SRU via catalog_type → Libero default), and finc stays opt-in.
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


class BuiltinParityTest(unittest.TestCase):
    def test_finc_opt_in_selects_finc(self):
        ex = resolve_dk_extractor(
            finc_dk_enabled=True, finc_base_url="https://ex.org/proxy.php"
        )
        self.assertEqual(type(ex).__name__, "FincCatalogClient")

    def test_finc_not_opted_in_falls_through_to_libero(self):
        # A finc base_url alone (without the opt-in flag) must NOT hijack the DK
        # backend — that opt-in gate is the whole point of finc_dk_enabled.
        ex = resolve_dk_extractor(
            finc_dk_enabled=False, finc_base_url="https://ex.org/proxy.php",
            catalog_type="libero_soap",
        )
        self.assertEqual(type(ex).__name__, "BiblioClient")

    def test_marcxml_sru_selects_sru(self):
        ex = resolve_dk_extractor(catalog_type="marcxml_sru", sru_preset="dnb")
        self.assertEqual(type(ex).__name__, "MarcXmlClient")

    def test_default_selects_libero(self):
        ex = resolve_dk_extractor(catalog_type="libero_soap", catalog_token="tok")
        self.assertEqual(type(ex).__name__, "BiblioClient")

    def test_precedence_finc_over_sru(self):
        # finc opt-in wins even when catalog_type would select SRU.
        ex = resolve_dk_extractor(
            finc_dk_enabled=True, finc_base_url="https://ex.org/proxy.php",
            catalog_type="marcxml_sru",
        )
        self.assertEqual(type(ex).__name__, "FincCatalogClient")


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
        cfg = _StubConfig([_StubInstance("mylib_catalog")])
        got = resolve_dk_extractor(config=cfg, catalog_type="libero_soap")
        self.assertIs(got, ext)

    def test_custom_plugin_yields_to_finc_opt_in(self):
        self._register("mylib_catalog", capabilities={SearchCapability.CLASSIFICATION})
        cfg = _StubConfig([_StubInstance("mylib_catalog")])
        got = resolve_dk_extractor(
            config=cfg, finc_dk_enabled=True,
            finc_base_url="https://ex.org/proxy.php",
        )
        self.assertEqual(type(got).__name__, "FincCatalogClient")

    def test_non_classification_custom_plugin_ignored(self):
        # A custom provider WITHOUT the CLASSIFICATION capability is not a DK source.
        self._register("mylib_titles", capabilities={SearchCapability.TITLE_RECORDS})
        cfg = _StubConfig([_StubInstance("mylib_titles")])
        got = resolve_dk_extractor(config=cfg, catalog_type="libero_soap")
        self.assertEqual(type(got).__name__, "BiblioClient")

    def test_custom_extractor_honours_shared_contract(self):
        ext = self._register("mylib_catalog", capabilities={SearchCapability.CLASSIFICATION})
        cfg = _StubConfig([_StubInstance("mylib_catalog")])
        got = resolve_dk_extractor(config=cfg)
        out = got.extract_dk_classifications_for_keywords(["Physik"], max_results=10)
        self.assertEqual(out[0]["classifications"][0]["dk"], "530")
        self.assertEqual(ext.calls, [(("Physik",), 10)])


if __name__ == "__main__":
    unittest.main()
