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
from src.utils.config_models import PluginInstanceConfig
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


class PrimarySettingsTest(unittest.TestCase):
    """WP P7: the read-side successor of the CatalogConfig mirror. - Claude Generated"""

    def _cfg(self, *insts):
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = list(insts)
        return cfg

    def test_settings_layered_over_configfield_defaults(self):
        # The mirror used to supply the dataclass default for an unset field; the
        # instance side must supply the ConfigField default instead.
        from src.core.search.factory import primary_settings

        cfg = self._cfg(PluginInstanceConfig(
            "catalog", "search_provider", "catalog", enabled=True, is_primary=True,
            settings={"token": "TOK"},
        ))
        out = primary_settings(cfg, "catalog")
        self.assertEqual(out["token"], "TOK")                            # instance wins
        self.assertEqual(out["catalog_type"], "libero_soap")             # declared default
        self.assertIs(out["strict_gnd_validation_for_dk_search"], True)  # declared default

    def test_enabled_only_gates_source_reads_but_not_policy_reads(self):
        from src.core.search.factory import primary_settings

        cfg = self._cfg(PluginInstanceConfig(
            "finc", "search_provider", "finc", enabled=False, is_primary=True,
            settings={"harvest_enabled": True},
        ))
        # source gate (default): disabled instance contributes nothing
        self.assertEqual(primary_settings(cfg, "finc"), {})
        # policy read: enable state ignored, mirroring derive_search_mirrors
        self.assertIs(primary_settings(cfg, "finc", enabled_only=False)["harvest_enabled"], True)

    def test_unknown_provider_and_unreadable_config_are_empty(self):
        from src.core.search.factory import primary_settings

        self.assertEqual(primary_settings(self._cfg(), "catalog"), {})

        class _Boom:
            def primary_instance(self, *a, **k):
                raise RuntimeError("unreadable")

        self.assertEqual(primary_settings(_Boom(), "catalog"), {})


class SetPrimarySettingsTest(unittest.TestCase):
    """WP P7: the write seam the setup wizards use instead of the mirror.

    The wizards assemble an AlimaConfig from scratch and used to set
    ``catalog_config``, relying on save_config to lift it into instances. - Claude Generated
    """

    def _fresh(self):
        from src.utils.config_models import AlimaConfig

        return AlimaConfig()

    def test_seeds_the_full_builtin_set_not_just_the_written_type(self):
        # The synthesis guard is per *category*: a lone hand-made catalog instance
        # would strand the other five built-ins forever.
        from src.core.search.factory import set_primary_settings

        cfg = self._fresh()
        set_primary_settings(cfg, "catalog", {"token": "TOK"})
        self.assertEqual(
            {p.instance_id for p in cfg.instances_for("search_provider")},
            {"lobid", "swb", "catalog", "finc", "sru", "gnd_local"},
        )

    def test_writes_onto_the_primary_and_merges(self):
        from src.core.search.factory import primary_settings, set_primary_settings

        cfg = self._fresh()
        set_primary_settings(cfg, "catalog", {"token": "TOK"})
        set_primary_settings(cfg, "catalog", {"catalog_search_url": "https://s"})
        cat = primary_settings(cfg, "catalog", enabled_only=False)
        self.assertEqual(cat["token"], "TOK")            # first write survives
        self.assertEqual(cat["catalog_search_url"], "https://s")

    def test_equivalent_to_the_legacy_upgrade_path(self):
        # A/B: an operator who set these values in a pre-plugin config (upgraded on
        # load) must end up with the same instances as one who runs the wizard now.
        from src.core.search.factory import set_primary_settings

        legacy_section = {
            "catalog_token": "TOK", "catalog_search_url": "https://s",
            "catalog_details_url": "https://d", "finc_base_url": "https://finc",
            "finc_default_limit": 25, "finc_dk_enabled": True,
        }
        old = {p.instance_id: dict(p.settings or {})
               for p in pm.synthesize_search_instances(legacy_section)}

        cfg = self._fresh()
        set_primary_settings(cfg, "catalog", {
            "token": "TOK", "catalog_search_url": "https://s", "catalog_details": "https://d",
        })
        set_primary_settings(cfg, "finc", {
            "base_url": "https://finc", "default_limit": 25, "dk_enabled": True,
        })
        new = {p.instance_id: dict(p.settings or {})
               for p in cfg.instances_for("search_provider")}
        self.assertEqual(old, new)


class CatalogWebBasesTest(unittest.TestCase):
    """WP P7: explicit catalog-before-finc precedence for the OPAC link base.

    Both plugins declare ``catalog_web_record_url``; they used to mirror onto one
    CatalogConfig field where dict order let finc win. - Claude Generated
    """

    def _cfg(self, cat_url, finc_url, *, cat_search=""):
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(
                "catalog", "search_provider", "catalog", enabled=True, is_primary=True,
                settings={"catalog_web_record_url": cat_url,
                          "catalog_web_search_url": cat_search},
            ),
            PluginInstanceConfig(
                "finc", "search_provider", "finc", enabled=True, is_primary=True,
                settings={"catalog_web_record_url": finc_url},
            ),
        ]
        return cfg

    def test_catalog_wins_over_finc(self):
        from src.core.search.factory import catalog_web_bases

        record, _ = catalog_web_bases(self._cfg("https://cat/Record/", "https://finc/Record/"))
        self.assertEqual(record, "https://cat/Record/")

    def test_empty_finc_no_longer_blanks_the_base(self):
        # The regression the mirror caused: catalog URL set, finc's left empty →
        # finc overwrote the shared field with "" → no OPAC links at all.
        from src.core.search.factory import catalog_web_bases

        record, _ = catalog_web_bases(self._cfg("https://cat/Record/", ""))
        self.assertEqual(record, "https://cat/Record/")

    def test_finc_fills_in_when_catalog_has_none(self):
        from src.core.search.factory import catalog_web_bases

        record, _ = catalog_web_bases(self._cfg("", "https://finc/Record/"))
        self.assertEqual(record, "https://finc/Record/")

    def test_search_base_comes_from_catalog_only(self):
        from src.core.search.factory import catalog_web_bases

        record, search = catalog_web_bases(
            self._cfg("https://cat/Record/", "", cat_search="https://cat/Search")
        )
        self.assertEqual(search, "https://cat/Search")


class MigrationTest(unittest.TestCase):
    """One-way upgrade of a pre-plugin config's raw JSON sections. - Claude Generated"""

    def _catalog(self):
        """The legacy ``catalog_config`` JSON section of a pre-plugin config."""
        return {
            "catalog_token": "TOK",
            "catalog_search_url": "https://s",
            "catalog_details_url": "https://d",
            "finc_base_url": "https://finc",
            "finc_default_limit": 25,
            "finc_dk_enabled": True,
            "sru_preset": "dnb",
            "sru_max_records": 40,
        }

    def test_synthesize_maps_fields(self):
        insts = {i.instance_id: i for i in pm.synthesize_search_instances(self._catalog())}
        self.assertEqual(insts["catalog"].settings["token"], "TOK")
        self.assertEqual(insts["catalog"].settings["catalog_details"], "https://d")
        self.assertEqual(insts["finc"].settings["base_url"], "https://finc")
        self.assertEqual(insts["finc"].settings["default_limit"], 25)
        self.assertTrue(insts["finc"].settings["dk_enabled"])
        self.assertEqual(insts["sru"].settings["preset"], "dnb")
        self.assertTrue(all(i.is_primary for i in insts.values()))

    def test_enabled_gate_respected(self):
        gate = {"providers": {"finc": False}}
        insts = {i.instance_id: i for i in pm.synthesize_search_instances(self._catalog(), gate)}
        self.assertFalse(insts["finc"].enabled)
        self.assertTrue(insts["lobid"].enabled)

    def test_absent_legacy_keys_are_omitted_not_none(self):
        """The upgrade must not hand ``None`` to a provider constructor.

        Until WP P7 the unset fields came back as the CatalogConfig dataclass
        defaults (``getattr`` on an unset attribute → ``catalog_type='libero_soap'``).
        Reading the raw JSON section with ``dict.get`` instead would silently produce
        ``None`` for every key the operator never set, and build_provider passes
        settings straight into ``cls(**settings)``. So absent ⇒ *omitted*, and the
        plugin's own ConfigField/constructor default applies. - Claude Generated
        """
        insts = {i.instance_id: i for i in pm.synthesize_search_instances({"catalog_token": "TOK"})}

        cat = insts["catalog"].settings
        self.assertEqual(cat["token"], "TOK")
        for absent in ("catalog_type", "strict_gnd_validation_for_dk_search",
                       "catalog_search_url", "catalog_web_record_url"):
            self.assertNotIn(absent, cat, f"{absent} must be omitted, not None")
        self.assertEqual(insts["finc"].settings, {})
        self.assertEqual(insts["sru"].settings, {})

        # …and the declared defaults are what the readers then see.
        from src.core.search.factory import primary_settings
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = list(insts.values())
        eff = primary_settings(cfg, "catalog", enabled_only=False)
        self.assertEqual(eff["catalog_type"], "libero_soap")
        self.assertIs(eff["strict_gnd_validation_for_dk_search"], True)

    def test_fresh_config_seeds_all_builtins(self):
        insts = pm.synthesize_search_instances({})
        self.assertEqual(
            {i.instance_id for i in insts},
            {"lobid", "swb", "catalog", "finc", "sru", "gnd_local"},
        )
        self.assertTrue(all(i.enabled for i in insts))


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
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.plugins = plugins
        tr = ToolRegistry.__new__(ToolRegistry)
        tr._config_manager = types.SimpleNamespace(
            load_config=lambda force_reload=False: cfg,
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


class ResolveGndSourceToolsTest(unittest.TestCase):
    """WP P6a: the agentic source→tool map is derived from the enabled GND
    providers, not hardcoded, and a requested built-in id resolves to the copy
    backing the same tool name (own-plugins POC). - Claude Generated"""

    def _resolve(self, requested, enabled_ids):
        from unittest.mock import patch
        from src.core.search.factory import resolve_gnd_source_tools
        with patch("src.core.search.factory.enabled_gnd_provider_ids", return_value=enabled_ids):
            return resolve_gnd_source_tools(requested)

    def test_default_is_every_enabled_provider(self):
        ids, mp = self._resolve(None, ["lobid", "swb"])
        self.assertEqual(ids, ["lobid", "swb"])
        self.assertEqual(mp, {"lobid": "search_lobid", "swb": "search_swb"})

    def test_requested_subset_is_filtered(self):
        ids, mp = self._resolve(["lobid"], ["lobid", "swb"])
        self.assertEqual(ids, ["lobid"])
        self.assertEqual(mp, {"lobid": "search_lobid"})

    def test_unreadable_config_returns_none(self):
        self.assertIsNone(self._resolve(["lobid"], None))

    def test_all_disabled_returns_empty(self):
        self.assertEqual(self._resolve(["lobid"], []), ([], {}))

    def test_requested_builtin_resolves_to_enabled_copy(self):
        # Own-plugins POC: lobid is disabled; a copy backs the same tool name.
        from src.core.search.provider import ProviderToolSpec, SearchCapability
        from src.core.search.registry import PROVIDER_REGISTRY, register_provider

        @register_provider
        class _PocLobid:
            id = "poc_lobid_x"
            label = "poc lobid"
            capabilities = {SearchCapability.GND_KEYWORDS}

            @classmethod
            def mcp_tool_specs(cls):
                return [ProviderToolSpec(
                    name="search_lobid", capability=SearchCapability.GND_KEYWORDS,
                    description="", parameters={},
                )]

        try:
            ids, mp = self._resolve(["lobid"], ["poc_lobid_x"])
            self.assertEqual(ids, ["poc_lobid_x"])
            self.assertEqual(mp, {"poc_lobid_x": "search_lobid"})
        finally:
            PROVIDER_REGISTRY.pop("poc_lobid_x", None)


if __name__ == "__main__":
    unittest.main()
