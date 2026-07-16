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
    """Fake underlying suggester: records search kwargs, returns canned output. - Claude Generated"""

    def __init__(self, out, titles=None):
        self.out = out
        self.titles = titles if titles is not None else {"t": [{"title": "B", "id": "1"}]}
        self.last_errors = {}
        self.last_raw = {}
        self.recorded = None

    def search(self, terms, **kw):
        self.recorded = kw
        return self.out

    def search_titles(self, terms, search_type="title", max_results=25):
        return self.titles


class _FakeProvider:
    """Fake factory provider seeded into ToolRegistry._provider_cache (no network).

    Serves both cache keys: the ``search(capability, terms)`` path (default cached
    tool) returns a typed ProviderResult; ``.suggester`` returns the raw stub for
    the non-default / catalog passthrough. - Claude Generated"""

    def __init__(self, gnd_result, raw, available=True):
        self._gnd_result = gnd_result
        self._raw = _Raw(raw)
        self._available = available

    def is_available(self, cfg=None):
        return self._available

    def search(self, capability, terms, **kw):
        return self._gnd_result

    @property
    def suggester(self):
        return self._raw


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class GeneratedSearchToolTest(unittest.TestCase):
    GND = {"t": {"Kw": {"count": 5, "gndid": {"4074335-4"}, "ddc": {"333"}, "dk": set()}}}
    RAW = {"t": {"KwRaw": {"count": 2, "gndid": {"4055747-6"}, "ddc": set(), "dk": set()}}}

    def _registry(self):
        from unittest.mock import MagicMock
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig, SearchProviderConfig
        from src.core.search.provider import ProviderResult, ResultItem, SearchCapability

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("lobid", "search_provider", "lobid", enabled=True, is_primary=True),
            PluginInstanceConfig("swb", "search_provider", "swb", enabled=True, is_primary=True),
            PluginInstanceConfig("catalog", "search_provider", "catalog", enabled=True, is_primary=True),
            PluginInstanceConfig("finc", "search_provider", "finc", enabled=True, is_primary=True),
        ]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(
            load_config=lambda **k: cfg,
            get_search_provider_config=lambda: SearchProviderConfig(),
        )
        reg._tools = {}
        reg._handlers = {}
        km = MagicMock()
        km.get_raw_response.return_value = None
        reg._knowledge_manager = km
        reg._provider_cache = {}

        def _gnd_result(errors=None):
            return ProviderResult(
                SearchCapability.GND_KEYWORDS,
                per_term={"t": [ResultItem(label="Kw", gnd_ids={"4074335-4"}, count=5, ddc={"333"})]},
                errors=errors or {},
            )

        fakes = {
            "lobid": _FakeProvider(_gnd_result({"term": "boom"}), self.RAW),
            "swb": _FakeProvider(_gnd_result({"term": "boom"}), self.RAW),
            "catalog": _FakeProvider(_gnd_result(), self.GND),
        }
        for pid, fake in fakes.items():
            reg._provider_cache[(pid, True)] = fake
            reg._provider_cache[(pid, False)] = fake
        reg._fakes = fakes
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
        self.assertIn("Kw", out["results"]["t"])          # came from the cached provider
        self.assertIn("gnd_urls", out["results"]["t"]["Kw"])  # gnd_url enrichment
        self.assertEqual(out["errors"], {"lobid:term": "boom"})

    def test_lobid_non_default_uses_raw_suggester(self):
        reg = self._registry()
        out = json.loads(self._handlers(reg)["search_lobid"](terms=["t"], search_type="title"))
        self.assertIn("KwRaw", out["results"]["t"])  # came from the raw suggester
        self.assertEqual(reg._fakes["lobid"].suggester.recorded, {"search_type": "title"})

    def test_swb_non_default_passes_max_pages_to_raw(self):
        reg = self._registry()
        self._handlers(reg)["search_swb"](terms=["t"], search_type="kw", max_pages=3)
        self.assertEqual(reg._fakes["swb"].suggester.recorded, {"search_type": "kw", "max_pages": 3})

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
        reg._fakes["catalog"]._available = False  # token-less catalog → unavailable
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
    silently shadowing the built-in tool, and the collision branch hands it the
    generic factory-built handler.

    Note the copy here is *not* canonical-with-its-own-name, so it never reaches
    the nuanced handler — see ``CanonicalCodePluginNuancesTest`` for that path.
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


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CanonicalCodePluginNuancesTest(unittest.TestCase):
    """A canonical code plugin gets the nuanced handler too (WP P1) - Claude Generated.

    Before P1 a ``hand_wired = {"lobid","swb","catalog","finc"}`` literal reserved
    the nuanced handlers (gnd_url enrichment, agent_view, non-default raw
    passthrough) for built-in ids; any other type fell to the generic handler even
    when canonical. The nuanced path is factory-backed, so it answers from the
    plugin's own class — the id test bought nothing.

    ``CodePluginToolGenerationTest`` cannot cover this: its copy keeps the
    blueprint's tool name, so the *collision* branch forces the generic handler
    regardless. This one renames the tool.
    """

    def setUp(self):
        from src.core.search.provider import ProviderToolSpec, SearchCapability
        from src.core.search.providers.lobid.provider import LobidProvider
        from src.core.search.registry import PROVIDER_REGISTRY

        self._saved = dict(PROVIDER_REGISTRY)

        class RenamedCopy(LobidProvider):
            id = "renamed_copy"

            @classmethod
            def mcp_tool_specs(cls):
                return [
                    ProviderToolSpec(
                        name="search_renamed",  # no collision → canonical stays canonical
                        capability=SearchCapability.GND_KEYWORDS,
                        description="renamed copy",
                        parameters={"type": "object", "properties": {}},
                        source_label="renamed",
                        cached=True,
                        add_gnd_urls=True,
                        default_opts={"search_type": "kw", "max_pages": 5},
                    )
                ]

        self._cls = RenamedCopy
        PROVIDER_REGISTRY["renamed_copy"] = RenamedCopy

    def tearDown(self):
        from src.core.search.registry import PROVIDER_REGISTRY

        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(self._saved)

    def _registry(self, provider_id="renamed_copy"):
        from unittest.mock import MagicMock

        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(provider_id, "search_provider", provider_id,
                                 enabled=True, is_primary=True)
        ]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        reg._knowledge_manager = MagicMock()
        reg._provider_cache = {}
        return reg

    def test_canonical_copy_gets_the_nuanced_factory_handler(self):
        from unittest.mock import patch

        reg = self._registry()
        handlers = {td.name: h for td, h in reg._generated_search_tools()}
        self.assertIn("search_renamed", handlers)

        built = {}

        def fake_build(inst, **kw):
            from src.core.search.registry import get_provider

            built["cls"] = get_provider(inst.provider_id)

            class _Stub:
                def is_available(self):
                    return False

            return _Stub()

        # _provider_for imports from src.core.search.factory — the nuanced path's
        # import site, distinct from the generic handler's re-export. - Claude Generated
        with patch("src.core.search.factory.build_provider", side_effect=fake_build):
            out = json.loads(handlers["search_renamed"](["t"]))

        self.assertIs(built["cls"], self._cls, "canonical copy did not build its own class")
        self.assertIn("error", out)
        self.assertIn("not available", out["error"])

    def test_unknown_result_shape_does_not_break_tool_generation(self):
        """One odd plugin spec must not take the whole tool list down.

        _make_search_handler used to raise ValueError on an unknown result_shape;
        unreachable while only built-ins reached it, reachable once every canonical
        spec does.
        """
        from src.core.search.provider import ProviderToolSpec, SearchCapability
        from src.core.search.providers.lobid.provider import LobidProvider
        from src.core.search.registry import PROVIDER_REGISTRY

        class OddShape(LobidProvider):
            id = "odd_shape"

            @classmethod
            def mcp_tool_specs(cls):
                return [
                    ProviderToolSpec(
                        name="search_odd",
                        capability=SearchCapability.GND_KEYWORDS,
                        description="odd",
                        parameters={"type": "object", "properties": {}},
                        result_shape="records",  # not a shape the layer knows
                        source_label="odd",
                    )
                ]

        PROVIDER_REGISTRY["odd_shape"] = OddShape
        reg = self._registry("odd_shape")
        tools = reg._generated_search_tools()  # must not raise
        self.assertIn("search_odd", [td.name for td, _ in tools])
