"""Tests for search-provider config schema, factory + config migration - Claude Generated."""

from __future__ import annotations

import unittest

from src.core.search import (
    build_enabled,
    build_provider,
    enabled_gnd_provider_ids,
    get_provider,
    list_providers,
)
from src.core.search.provider import SearchCapability
from src.utils.config_models import CatalogConfig, PluginInstanceConfig, SearchProviderConfig
from src.utils import plugin_migration as pm


class ConfigFieldsAndGatingTest(unittest.TestCase):
    def test_all_providers_declare_fields(self):
        for pid in list_providers():
            self.assertIsInstance(get_provider(pid).config_fields(), list)

    def test_every_provider_has_complete_doc(self):
        """Design requirement: each plugin self-describes (description + input + output)."""
        for pid in list_providers():
            doc = get_provider(pid).doc()
            self.assertTrue(doc.is_complete(), f"{pid} doc incomplete: {doc}")

    def test_catalog_gated_by_token(self):
        self.assertFalse(get_provider("catalog")().is_available())
        self.assertTrue(get_provider("catalog")(token="t").is_available())

    def test_finc_gated_by_base_url(self):
        self.assertFalse(get_provider("finc")().is_available())
        self.assertTrue(get_provider("finc")(base_url="http://x").is_available())

    def test_sru_or_gate(self):
        self.assertFalse(get_provider("sru")().is_available())
        self.assertTrue(get_provider("sru")(preset="dnb").is_available())
        self.assertTrue(get_provider("sru")(base_url="http://s").is_available())

    def test_lobid_swb_gnd_local_always_available(self):
        for pid in ("lobid", "swb", "gnd_local"):
            self.assertTrue(get_provider(pid)().is_available())


class FactoryTest(unittest.TestCase):
    def test_build_provider_passes_settings(self):
        inst = PluginInstanceConfig("finc", "search_provider", "finc", settings={"base_url": "http://x"})
        p = build_provider(inst)
        self.assertEqual(p.id, "finc")
        self.assertTrue(p.is_available())

    def test_build_provider_caches_gnd_keywords(self):
        inst = PluginInstanceConfig("lobid", "search_provider", "lobid")
        wrapped = build_provider(inst, cache=True)
        self.assertEqual(type(wrapped).__name__, "CachingProvider")
        # non-GND provider is not wrapped
        finc = build_provider(PluginInstanceConfig("finc", "search_provider", "finc"), cache=True)
        self.assertEqual(finc.id, "finc")
        self.assertNotEqual(type(finc).__name__, "CachingProvider")

    def test_enabled_gnd_provider_ids_gates_by_instance(self):
        from src.utils.config_models import AlimaConfig
        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("lobid", "search_provider", "lobid", enabled=True),
            PluginInstanceConfig("swb", "search_provider", "swb", enabled=False),
            PluginInstanceConfig("finc", "search_provider", "finc", enabled=True),  # TITLE_RECORDS only
        ]
        ids = enabled_gnd_provider_ids(cfg)
        self.assertIn("lobid", ids)
        self.assertNotIn("swb", ids)   # disabled
        self.assertNotIn("finc", ids)  # no GND_KEYWORDS capability

    def test_build_enabled_skips_disabled_and_unknown(self):
        insts = [
            PluginInstanceConfig("lobid", "search_provider", "lobid", enabled=True),
            PluginInstanceConfig("off", "search_provider", "finc", enabled=False),
            PluginInstanceConfig("weird", "search_provider", "does_not_exist", enabled=True),
        ]
        built = build_enabled(insts)
        self.assertEqual(set(built), {"lobid"})


class MigrationTest(unittest.TestCase):
    def _catalog(self):
        return CatalogConfig(
            catalog_token="TOK",
            catalog_search_url="https://s",
            catalog_details_url="https://d",
            finc_base_url="https://finc",
            finc_default_limit=25,
            finc_dk_enabled=True,
            sru_preset="dnb",
            sru_max_records=40,
        )

    def test_synthesize_maps_fields(self):
        insts = {i.instance_id: i for i in pm.synthesize_search_instances(self._catalog(), SearchProviderConfig())}
        self.assertEqual(insts["catalog"].settings["token"], "TOK")
        self.assertEqual(insts["catalog"].settings["catalog_details"], "https://d")
        self.assertEqual(insts["finc"].settings["base_url"], "https://finc")
        self.assertEqual(insts["finc"].settings["default_limit"], 25)
        self.assertTrue(insts["finc"].settings["dk_enabled"])
        self.assertEqual(insts["sru"].settings["preset"], "dnb")
        self.assertTrue(all(i.is_primary for i in insts.values()))

    def test_enabled_gate_respected(self):
        spc = SearchProviderConfig(providers={"finc": False})
        insts = {i.instance_id: i for i in pm.synthesize_search_instances(self._catalog(), spc)}
        self.assertFalse(insts["finc"].enabled)
        self.assertTrue(insts["lobid"].enabled)

    def test_derive_round_trip(self):
        cat = self._catalog()
        spc = SearchProviderConfig()
        insts = pm.synthesize_search_instances(cat, spc)
        fresh = CatalogConfig()
        fresh_spc = SearchProviderConfig()
        pm.derive_search_mirrors(insts, fresh, fresh_spc)
        for attr in ("catalog_token", "catalog_details_url", "finc_base_url", "finc_default_limit", "sru_preset"):
            self.assertEqual(getattr(fresh, attr), getattr(cat, attr), attr)

    def test_reverse_sync_captures_mirror_edit(self):
        cat = self._catalog()
        insts = pm.synthesize_search_instances(cat, SearchProviderConfig())
        # Simulate a legacy Catalog-tab edit
        cat.catalog_token = "EDITED"
        pm.sync_instances_from_mirrors(insts, cat, SearchProviderConfig(), None)
        catalog_inst = next(i for i in insts if i.provider_id == "catalog")
        self.assertEqual(catalog_inst.settings["token"], "EDITED")


class ListPluginsToolTest(unittest.TestCase):
    def test_list_plugins_returns_active_with_docs(self):
        import json
        import types
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig("lobid", "search_provider", "lobid", enabled=True, is_primary=True),
            PluginInstanceConfig("off", "search_provider", "swb", enabled=False),
            PluginInstanceConfig("doi_crossref", "input_source", "doi_crossref", enabled=True,
                                 settings={"contact_email": "a@b.c"}),
        ]
        tr = ToolRegistry.__new__(ToolRegistry)
        tr._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        out = json.loads(tr._handle_list_plugins())

        search = {p["instance_id"]: p for p in out["plugins"]["search_provider"]}
        self.assertIn("lobid", search)
        self.assertNotIn("off", search)  # disabled excluded by default
        self.assertTrue(search["lobid"]["description"])  # self-doc present
        self.assertIn("gnd_keywords", search["lobid"]["capabilities"])
        inp = {p["instance_id"]: p for p in out["plugins"]["input_source"]}
        self.assertIn("doi_crossref", inp)
        self.assertTrue(inp["doi_crossref"]["input"] and inp["doi_crossref"]["output"])
        # secrets never leak; note distinguishes plugins from workflows
        self.assertNotIn("token", str(out))
        self.assertIn("workflow", out["note"].lower())

    def test_list_plugins_include_disabled(self):
        import json
        import types
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = [PluginInstanceConfig("off", "search_provider", "swb", enabled=False)]
        tr = ToolRegistry.__new__(ToolRegistry)
        tr._config_manager = types.SimpleNamespace(load_config=lambda: cfg)
        out = json.loads(tr._handle_list_plugins(include_disabled=True))
        ids = [p["instance_id"] for p in out["plugins"]["search_provider"]]
        self.assertIn("off", ids)


class InputToolGenerationTest(unittest.TestCase):
    def _registry(self, plugins):
        import types
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.config_models import AlimaConfig, SearchProviderConfig

        cfg = AlimaConfig()
        cfg.plugins = plugins
        tr = ToolRegistry.__new__(ToolRegistry)
        tr._config_manager = types.SimpleNamespace(
            load_config=lambda force_reload=False: cfg,
            get_search_provider_config=lambda: SearchProviderConfig(),
        )
        tr._tools = {}
        tr._handlers = {}
        tr._suggesters_initialized = True
        return tr, cfg

    def test_three_doi_tools_generated(self):
        plugins = [
            PluginInstanceConfig("doi_crossref", "input_source", "doi_crossref", enabled=True),
            PluginInstanceConfig("doi_openalex", "input_source", "doi_openalex", enabled=True),
            PluginInstanceConfig("doi_datacite", "input_source", "doi_datacite", enabled=True),
        ]
        tr, _ = self._registry(plugins)
        names = {td.name for td, _ in tr._generated_input_tools()}
        self.assertEqual(
            names, {"resolve_doi_crossref", "resolve_doi_openalex", "resolve_doi_datacite"}
        )

    def test_refresh_removes_disabled_plugin_tool(self):
        plugins = [
            PluginInstanceConfig("doi_crossref", "input_source", "doi_crossref", enabled=True),
            PluginInstanceConfig("doi_openalex", "input_source", "doi_openalex", enabled=True),
        ]
        tr, cfg = self._registry(plugins)
        tr.register_all_tools()
        self.assertIn("resolve_doi_openalex", tr._tools)
        cfg.plugins[1].enabled = False  # disable openalex
        tr.refresh()
        self.assertIn("resolve_doi_crossref", tr._tools)
        self.assertNotIn("resolve_doi_openalex", tr._tools)  # gone at runtime


class GndSourceIdsNoneVsEmptyTest(unittest.TestCase):
    """The None-vs-[] contract at the find_keywords call site - Claude Generated.

    ``enabled_gnd_provider_ids`` returns None when the config can not be read
    (callers keep their own default) and [] when the operator disabled every GND
    source. Collapsing them (``ids if ids else default``) searches the built-ins
    against an explicit disable. The tab's source-checkbox builder is Qt-free
    apart from ``self.logger``, so it is callable unbound.
    """

    def _call(self, return_value):
        from unittest.mock import MagicMock, patch

        from src.ui.find_keywords import SearchTab

        stub = MagicMock()  # only self.logger is touched
        with patch(
            "src.core.search.factory.enabled_gnd_provider_ids", return_value=return_value
        ):
            return SearchTab._gnd_source_ids(stub)

    def test_unreadable_config_falls_back(self):
        self.assertEqual(self._call(None), ["lobid", "swb"])

    def test_empty_means_all_disabled_and_is_respected(self):
        self.assertEqual(self._call([]), [])

    def test_enabled_ids_pass_through(self):
        self.assertEqual(self._call(["poc_lobid"]), ["poc_lobid"])


if __name__ == "__main__":
    unittest.main()
