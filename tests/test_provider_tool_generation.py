"""Tests for registry-driven MCP search-tool generation (P3) - Claude Generated.

The 5 library search tools (search_lobid/swb/catalog/catalog_titles/finc) are no
longer hand-written: their ToolDefinition + handler are generated from each
provider's ProviderToolSpec. These tests pin the generated schemas and the
handler output shapes (the per-tool nuances that were proven byte-identical to the
former hand-written handlers): lobid/swb enrich rows with gnd_urls + carry errors
and use the cache for default options (raw suggester otherwise); catalog has
neither gnd_urls nor an errors block; catalog_titles returns raw records.
"""

import json
import types
import unittest

try:
    from src.mcp.tool_registry import ToolRegistry
    from src.core.search import provider_tool_specs
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


class _Raw:
    def __init__(self, out):
        self.out = out
        self.last_errors = {}
        self.recorded = None

    def search(self, terms, **kw):
        self.recorded = kw
        return self.out


class _Meta:
    """Mock MetaSuggester: cached .search(terms) + raw_suggester()."""

    def __init__(self, cached_out, raw_out):
        self._cached = cached_out
        self.raw = _Raw(raw_out)
        self.last_errors = {"lobid:term": "boom"}

    def search(self, terms):
        return self._cached

    def raw_suggester(self, pid=None):
        return self.raw


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class GeneratedSearchToolTest(unittest.TestCase):
    GND = {"t": {"Kw": {"count": 5, "gndid": {"4074335-4"}, "ddc": {"333"}, "dk": set()}}}
    RAW = {"t": {"KwRaw": {"count": 2, "gndid": {"4055747-6"}, "ddc": set(), "dk": set()}}}

    def _registry(self):
        reg = ToolRegistry()
        reg._suggesters_initialized = True
        reg._lobid = _Meta(self.GND, self.RAW)
        reg._swb = _Meta(self.GND, self.RAW)
        reg._biblio = types.SimpleNamespace(
            search=lambda terms, search_type="kw": self.GND,
            search_titles=lambda terms, search_type="title", max_results=25: {"t": [{"title": "B", "id": "1"}]},
            last_errors={},
        )
        reg._finc = _Raw({"t": {"records": [], "result_count": 0, "facets": {}, "errors": []}})
        return reg

    def _handlers(self, reg):
        return {td.name: h for td, h in reg._generated_search_tools()}

    def test_all_five_tools_generated(self):
        names = {td.name for td, _ in self._registry()._generated_search_tools()}
        self.assertEqual(
            names,
            {"search_lobid", "search_swb", "search_catalog", "search_catalog_titles", "search_finc"},
        )

    def test_schemas_have_expected_params(self):
        specs = {s.name: s for s in provider_tool_specs()}
        self.assertIn("search_type", specs["search_lobid"].parameters["properties"])
        self.assertIn("max_pages", specs["search_swb"].parameters["properties"])
        for p in ("filters", "facets", "limit", "availability"):
            self.assertIn(p, specs["search_finc"].parameters["properties"])
        self.assertEqual(specs["search_lobid"].parameters["required"], ["terms"])

    def test_lobid_default_uses_cache_with_gnd_urls_and_errors(self):
        reg = self._registry()
        out = json.loads(self._handlers(reg)["search_lobid"](terms=["t"], search_type="kw"))
        self.assertEqual(out["source"], "lobid")
        self.assertIn("Kw", out["results"]["t"])          # came from cached .search()
        self.assertIn("gnd_urls", out["results"]["t"]["Kw"])  # gnd_url enrichment
        self.assertEqual(out["errors"], {"lobid:term": "boom"})

    def test_lobid_non_default_uses_raw_suggester(self):
        reg = self._registry()
        out = json.loads(self._handlers(reg)["search_lobid"](terms=["t"], search_type="title"))
        self.assertIn("KwRaw", out["results"]["t"])  # came from raw_suggester()
        self.assertEqual(reg._lobid.raw.recorded, {"search_type": "title"})

    def test_swb_non_default_passes_max_pages_to_raw(self):
        reg = self._registry()
        self._handlers(reg)["search_swb"](terms=["t"], search_type="kw", max_pages=3)
        self.assertEqual(reg._swb.raw.recorded, {"search_type": "kw", "max_pages": 3})

    def test_catalog_has_no_gnd_urls_and_no_errors_block(self):
        reg = self._registry()
        out = json.loads(self._handlers(reg)["search_catalog"](terms=["t"], search_type="kw"))
        self.assertEqual(out["source"], "catalog")
        self.assertNotIn("gnd_urls", out["results"]["t"]["Kw"])
        self.assertNotIn("errors", out)

    def test_catalog_titles_returns_records(self):
        reg = self._registry()
        out = json.loads(self._handlers(reg)["search_catalog_titles"](terms=["t"]))
        self.assertEqual(out["source"], "catalog_titles")
        self.assertEqual(out["results"]["t"], [{"title": "B", "id": "1"}])
        self.assertNotIn("errors", out)

    def test_unavailable_source_message(self):
        reg = self._registry()
        reg._biblio = None
        out = json.loads(self._handlers(reg)["search_catalog"](terms=["t"]))
        self.assertEqual(out["error"], "BiblioSuggester not available")


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class ProviderConfigGatingTest(unittest.TestCase):
    def test_disabled_provider_tool_not_generated(self):
        from src.utils.config_models import SearchProviderConfig

        reg = ToolRegistry()
        reg._suggesters_initialized = True
        reg._config_manager = types.SimpleNamespace(
            get_search_provider_config=lambda: SearchProviderConfig(
                providers={"finc": False, "swb": False}
            )
        )
        names = {td.name for td, _ in reg._generated_search_tools()}
        self.assertNotIn("search_finc", names)
        self.assertNotIn("search_swb", names)
        self.assertIn("search_lobid", names)
        self.assertIn("search_catalog", names)
        self.assertIn("search_catalog_titles", names)

    def test_default_all_enabled_and_roundtrip(self):
        from dataclasses import asdict
        from src.utils.config_models import SearchProviderConfig

        cfg = SearchProviderConfig()
        self.assertTrue(cfg.is_enabled("anything"))  # absent → enabled
        cfg.set_enabled("finc", False)
        self.assertFalse(cfg.is_enabled("finc"))
        # survives serialize → reload (asdict is how AlimaConfig persists)
        restored = SearchProviderConfig(**asdict(cfg))
        self.assertFalse(restored.is_enabled("finc"))
        self.assertTrue(restored.is_enabled("lobid"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CodePluginToolGenerationTest(unittest.TestCase):
    """A loaded code-plugin copy must get a working, non-shadowing tool - Claude Generated.

    Regression (July 6): an un-adapted blueprint copy keeps the built-in's
    ProviderToolSpec name (e.g. ``search_lobid``). It must be suffixed instead of
    silently shadowing the built-in tool, and its canonical handler must be the
    generic factory-built one (the hand-wired handlers only know built-in types).
    """

    def setUp(self):
        from src.core.search.registry import PROVIDER_REGISTRY
        from src.core.search.providers.lobid.provider import LobidProvider

        class LobidCopy(LobidProvider):
            id = "lobid_copy"

        self._copy_cls = LobidCopy
        PROVIDER_REGISTRY["lobid_copy"] = LobidCopy

    def tearDown(self):
        from src.core.search.registry import PROVIDER_REGISTRY

        PROVIDER_REGISTRY.pop("lobid_copy", None)

    def _registry_with_instances(self):
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("lobid", "search_provider", "lobid", enabled=True, is_primary=True),
            PluginInstanceConfig("lobid_copy", "search_provider", "lobid_copy", enabled=True),
        ]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        return reg

    def test_copy_tool_is_suffixed_not_shadowing(self):
        reg = self._registry_with_instances()
        tools = reg._generated_search_tools()
        names = [td.name for td, _ in tools]
        self.assertEqual(names.count("search_lobid"), 1)
        self.assertIn("search_lobid_lobid_copy", names)

    def test_copy_handler_uses_its_own_provider_class(self):
        from unittest.mock import patch

        reg = self._registry_with_instances()
        handlers = {td.name: h for td, h in reg._generated_search_tools()}
        built = {}

        def fake_build(inst, **kw):
            from src.core.search.registry import get_provider

            cls = get_provider(inst.provider_id)
            built["cls"] = cls

            class _Stub:
                def is_available(self):
                    return False

            return _Stub()

        with patch("src.core.search.build_provider", side_effect=fake_build):
            out = handlers["search_lobid_lobid_copy"](["t"])
        self.assertIs(built["cls"], self._copy_cls)
        self.assertIn("error", json.loads(out))  # unavailable stub → guarded JSON
