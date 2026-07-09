"""Lookup plugin category (src/utils/lookups/) - Claude Generated.

The third plugin category: small external-API lookups exposed as agent tools.
Covers the registry/spec, category adapter (incl. the cache field), MCP tool
generation, the handler dispatch, and per-plugin raw-response caching.
"""

import os
import tempfile
import types
import unittest

try:
    from src.utils.lookups.registry import (
        LOOKUP_REGISTRY, LookupToolSpec, register_lookup, list_lookups, lookup_tool_specs,
        get_lookup,
    )
    import src.utils.lookups  # noqa: F401 — registers category + built-in rvk_api
    from src.core.plugins.category import get_category
    from src.core.plugins.schema import PluginDoc
    from src.mcp.tool_registry import ToolRegistry
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import AlimaConfig, DatabaseConfig, PluginInstanceConfig
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


class _FakeLookup:
    id = "fake_lk"
    label = "Fake"

    def __init__(self, **config):
        self._config = config

    @classmethod
    def config_fields(cls):
        return []

    @classmethod
    def doc(cls):
        return PluginDoc(description="x", input="y", output="z")

    @classmethod
    def mcp_tool_specs(cls):
        return [LookupToolSpec(
            name="fake_search",
            description="d",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["q"],
            },
            method="run",
            cache_key_param="q",
        )]

    def run(self, q, n=5):
        return {"q": q, "n": n, "hits": [q.upper()]}


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LookupRegistryTest(unittest.TestCase):
    def setUp(self):
        LOOKUP_REGISTRY["fake_lk"] = _FakeLookup

    def tearDown(self):
        LOOKUP_REGISTRY.pop("fake_lk", None)

    def test_builtin_rvk_registered(self):
        self.assertIn("rvk_api", list_lookups())

    def test_specs_stamped_with_provider_id(self):
        specs = {s.name: s for s in lookup_tool_specs()}
        self.assertEqual(specs["fake_search"].provider_id, "fake_lk")
        self.assertIn("rvk_search", specs)  # built-in

    def test_category_meta_has_cache_field(self):
        keys = [f.key for f in get_category("lookup").type_meta("fake_lk").config_fields]
        self.assertIn("cache_responses", keys)

    def test_every_lookup_config_fields_and_meta_build(self):
        # Guards the class of bug where a ConfigField references an undefined kind
        # symbol (config_fields() raising NameError) — tests that never call it miss
        # it, but the GUI plugin form does. Exercise it for every registered lookup.
        cat = get_category("lookup")
        for lid in list_lookups():
            cls = get_lookup(lid)
            fields = cls.config_fields() if hasattr(cls, "config_fields") else []
            self.assertIsInstance(fields, list)
            meta = cat.type_meta(lid)  # builds config_fields + cache_field
            keys = [f.key for f in meta.config_fields]
            self.assertIn("cache_responses", keys)
        # k10plus specifically exposes its dir-cache setting.
        k10_keys = [f.key for f in get_lookup("k10plus").config_fields()]
        self.assertIn("cache_dir", k10_keys)


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LookupToolHandlerTest(unittest.TestCase):
    def setUp(self):
        LOOKUP_REGISTRY["fake_lk"] = _FakeLookup
        UnifiedKnowledgeManager.reset()
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(self.tmp.name))

    def tearDown(self):
        LOOKUP_REGISTRY.pop("fake_lk", None)
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _registry(self, cache_pref):
        cfg = AlimaConfig()
        cfg.system_config.enable_response_cache = True
        cfg.plugins = [PluginInstanceConfig(
            "fake_lk", "lookup", "fake_lk", enabled=True, is_primary=True,
            settings={"cache_responses": cache_pref},
        )]
        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = types.SimpleNamespace(load_config=lambda **k: cfg)
        reg._knowledge_manager = self.km
        reg._tools = {}
        reg._handlers = {}
        return reg

    def _handler(self, reg):
        return {td.name: h for td, h in reg._generated_lookup_tools()}["fake_search"]

    def test_dispatch_and_cache_on(self):
        import json
        reg = self._registry("on")
        out = json.loads(self._handler(reg)(q="wasser", n=3))
        self.assertEqual(out["hits"], ["WASSER"])       # plugin.run called
        self.assertEqual(out["n"], 3)
        # raw response cached under (tool_name, key, other-args)
        self.assertIsNotNone(self.km.get_raw_response("fake_search", "wasser", {"n": 3}))

    def test_cache_off_setting(self):
        import json
        reg = self._registry("off")
        self._handler(reg)(q="klima", n=2)
        self.assertIsNone(self.km.get_raw_response("fake_search", "klima", {"n": 2}))

    def test_stray_kwargs_ignored(self):
        import json
        reg = self._registry("off")
        out = json.loads(self._handler(reg)(q="x", n=1, bogus="drop me"))
        self.assertEqual(out["hits"], ["X"])            # bogus filtered, no TypeError


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class K10PlusLookupTest(unittest.TestCase):
    """k10plus package harvester as a lookup tool (mocked SRU — no network). - Claude Generated"""

    def test_fetch_package_caps_and_serializes(self):
        from unittest.mock import patch
        from src.utils.k10plus_resolver import K10PlusRecord
        from src.utils.lookups.k10plus import K10PlusLookup

        recs = [K10PlusRecord(ppn=f"p{i}", title=f"T{i}", doi=f"10.x/{i}") for i in range(5)]
        with patch("src.utils.k10plus_resolver.fetch_records_for_siegel", return_value=recs):
            out = K10PlusLookup(max_records=3).fetch_package("ZDB-2-CMS")
        self.assertEqual(out["total"], 5)
        self.assertEqual(out["returned"], 3)            # capped
        self.assertEqual(out["records"][0]["title"], "T0")
        self.assertEqual(out["records"][0]["ppn"], "p0")

    def test_tool_generated(self):
        reg = ToolRegistry(); reg.register_all_tools()
        self.assertIn("k10plus_package", reg.get_tool_names())


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class DnbLookupTest(unittest.TestCase):
    """DNB GND-classification as a lookup tool (mocked RDF client — no network/rdflib)."""

    def _fake_dnb_module(self, captured):
        import sys, types
        mod = types.ModuleType("src.core.dnb_utils")

        def _get(gnd_id, timeout=10):
            captured.append((gnd_id, timeout))
            return {"status": "success", "preferred_name": "Grasfrosch",
                    "ddc": [{"code": "597.8", "determinancy": "4"}],
                    "gnd_subject_categories": [], "category": "SubjectHeading", "types": []}

        mod.get_dnb_classification = _get
        self._saved = sys.modules.get("src.core.dnb_utils")
        sys.modules["src.core.dnb_utils"] = mod

    def tearDown(self):
        import sys
        if getattr(self, "_saved", None) is not None:
            sys.modules["src.core.dnb_utils"] = self._saved
        else:
            sys.modules.pop("src.core.dnb_utils", None)

    def test_registered_and_tool_generated(self):
        from src.utils.lookups.registry import list_lookups
        self.assertIn("dnb", list_lookups())
        reg = ToolRegistry(); reg.register_all_tools()
        self.assertIn("dnb_classification", reg.get_tool_names())

    def test_classify_passes_timeout_and_returns_data(self):
        captured = []
        self._fake_dnb_module(captured)
        from src.utils.lookups.dnb import DnbLookup
        out = DnbLookup(timeout=5).classify("4045956-1")
        self.assertEqual(out["preferred_name"], "Grasfrosch")
        self.assertEqual(captured, [("4045956-1", 5)])  # timeout flows to the client


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LookupRawCacheHelperTest(unittest.TestCase):
    """C3b: direct lookup callers (e.g. the RVK anchor) reuse the WP2 raw cache."""

    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(self.tmp.name))

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _config(self, pref):
        cfg = AlimaConfig()
        cfg.system_config.enable_response_cache = True
        cfg.plugins = [PluginInstanceConfig(
            "rvk_api", "lookup", "rvk_api", enabled=True, is_primary=True,
            settings={"cache_responses": pref},
        )]
        return cfg

    def test_enabled_reads_plugin_setting_and_global(self):
        from src.utils.lookups.cache import lookup_cache_enabled
        self.assertTrue(lookup_cache_enabled(self._config("auto"), "rvk_api"))
        self.assertTrue(lookup_cache_enabled(self._config("on"), "rvk_api"))
        self.assertFalse(lookup_cache_enabled(self._config("off"), "rvk_api"))
        cfg = self._config("auto")
        cfg.system_config.enable_response_cache = False
        self.assertFalse(lookup_cache_enabled(cfg, "rvk_api"))  # auto follows global
        self.assertFalse(lookup_cache_enabled(None, "rvk_api"))

    def test_cached_call_serves_second_hit_from_cache(self):
        from src.utils.lookups.cache import cached_call
        calls = {"n": 0}

        def fetch():
            calls["n"] += 1
            return [{"code": "WI 1000"}]

        r1 = cached_call(self.km, True, "rvk_search", "wirtschaft", {"max_results": 6}, fetch)
        r2 = cached_call(self.km, True, "rvk_search", "wirtschaft", {"max_results": 6}, fetch)
        self.assertEqual(r1, r2)
        self.assertEqual(calls["n"], 1)  # 2nd served from cache

    def test_cached_call_disabled_always_live(self):
        from src.utils.lookups.cache import cached_call
        calls = {"n": 0}
        cached_call(self.km, False, "rvk_validate", "WI 1000", {}, lambda: calls.__setitem__("n", calls["n"] + 1))
        cached_call(self.km, False, "rvk_validate", "WI 1000", {}, lambda: calls.__setitem__("n", calls["n"] + 1))
        self.assertEqual(calls["n"], 2)


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LookupSeedingTest(unittest.TestCase):
    """C2: lookups are auto-seeded into config.plugins so the GUI list is populated."""

    def test_synthesize_covers_all_registered_lookups(self):
        from src.utils.plugin_migration import synthesize_lookup_instances, LOOKUP_CATEGORY

        insts = synthesize_lookup_instances()
        ids = {i.provider_id for i in insts}
        self.assertIn("rvk_api", ids)
        self.assertIn("k10plus", ids)
        self.assertIn("dnb", ids)
        for i in insts:
            self.assertEqual(i.category, LOOKUP_CATEGORY)
            self.assertTrue(i.enabled)
            self.assertTrue(i.is_primary)
            self.assertTrue(i.label)  # non-empty display label from the plugin class

    def test_load_config_seeds_lookup_instances(self):
        """A config with no lookup section gets lookup instances on parse."""
        from src.utils.config_manager import ConfigManager
        from src.utils.plugin_migration import LOOKUP_CATEGORY

        cm = ConfigManager.__new__(ConfigManager)
        import logging
        cm.logger = logging.getLogger("test")
        # Minimal config dict with a provider so parsing doesn't bail early.
        data = {
            "unified_config": {
                "providers": [
                    {"name": "local", "provider_type": "ollama", "host": "localhost", "enabled": True}
                ]
            }
        }
        cfg = cm._parse_config(data)
        lookup_ids = {p.provider_id for p in cfg.plugins if p.category == LOOKUP_CATEGORY}
        self.assertIn("rvk_api", lookup_ids)
        self.assertIn("k10plus", lookup_ids)


if __name__ == "__main__":
    unittest.main()
