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


if __name__ == "__main__":
    unittest.main()
