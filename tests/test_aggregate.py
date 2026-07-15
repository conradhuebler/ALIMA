"""WP2 P4.2: aggregate_gnd_results (counter + provenance over the raw cache).

Verifies the aggregation engine derives the ranked pool from the raw cache with:
* cross-source merge + provenance (``sources`` / ``source_count``);
* the count-landmine — pool ``count`` forced to 1, real Häufigkeit in
  ``display_count`` (max-merged across sources), never used for ranking;
* ranking by ``(source_count, count)`` descending;
* missing-raw terms recorded, unknown sources skipped.
- Claude Generated
"""

import json
import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.search.aggregate import aggregate_gnd_results
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class AggregateTest(unittest.TestCase):
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

    def test_pool_counter_and_provenance(self):
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        self.km.store_raw_response("swb", "wasser", {"search_type": "kw", "max_pages": 5}, "{}")
        transforms = {
            "lobid": lambda raw: {
                "Wasser": {"count": 47, "gndid": {"g1"}, "ddc": set(), "dk": set()},
            },
            "swb": lambda raw: {
                "Wasser": {"count": 1, "gndid": {"g1"}, "ddc": set(), "dk": set()},
                "Klima": {"count": 1, "gndid": {"g2"}, "ddc": set(), "dk": set()},
            },
        }
        out = aggregate_gnd_results(
            ["wasser"], ["lobid", "swb"], self.km, transforms,
            params_by_source={"lobid": {"search_type": "kw"},
                              "swb": {"search_type": "kw", "max_pages": 5}},
        )
        by_title = {e["title"]: e for e in out["pool"]}
        # Confirmed by both sources → ranked first, source_count 2.
        self.assertEqual(out["pool"][0]["title"], "Wasser")
        self.assertEqual(by_title["Wasser"]["source_count"], 2)
        self.assertEqual(by_title["Wasser"]["sources"], ["lobid", "swb"])
        # Count-landmine: pool count stays 1; real count rides in display_count.
        self.assertEqual(by_title["Wasser"]["count"], 1)
        self.assertEqual(by_title["Wasser"]["display_count"], 47)
        self.assertEqual(by_title["Klima"]["source_count"], 1)
        self.assertEqual(by_title["Klima"]["count"], 1)
        self.assertEqual(by_title["Klima"]["display_count"], 1)
        self.assertEqual(out["missing"], {})
        # terms_map records which query produced each title (per-keyword display).
        self.assertEqual(out["terms_map"]["Wasser"], ["wasser"])
        self.assertEqual(out["terms_map"]["Klima"], ["wasser"])

    def test_mapping_fallback_when_raw_absent(self):
        # No raw stored, but a fresh mapping + gnd facts exist → aggregate must
        # fall back to the mapping index (size-capped/pruned/pre-WP2 case).
        self.km.store_gnd_fact("g1", {"title": "Wasser", "description": "",
                                      "synonyms": "", "ddcs": ""})
        self.km.update_search_mapping("wasser", "swb", found_gnd_ids=["g1"],
                                      gnd_counts={"g1": 12})
        out = aggregate_gnd_results(
            ["wasser"], ["swb"], self.km,
            {"swb": lambda raw: {}},  # transform never called (no raw)
            params_by_source={"swb": {"search_type": "kw", "max_pages": 5}},
        )
        self.assertEqual(len(out["pool"]), 1)
        e = out["pool"][0]
        self.assertEqual(e["title"], "Wasser")
        self.assertEqual(e["count"], 1)           # count-landmine
        self.assertEqual(e["display_count"], 12)  # real count from gnd_counts
        self.assertEqual(out["missing"], {})       # fallback covered it

    def test_missing_raw_recorded(self):
        out = aggregate_gnd_results(
            ["x"], ["lobid"], self.km, {"lobid": lambda raw: {}},
            params_by_source={"lobid": {"search_type": "kw"}},
        )
        self.assertEqual(out["missing"], {"lobid": ["x"]})
        self.assertEqual(out["pool"], [])

    def test_unknown_source_skipped(self):
        out = aggregate_gnd_results(["x"], ["mystery"], self.km, {})
        self.assertEqual(out["pool"], [])
        self.assertEqual(out["missing"], {})

    def test_bad_blob_skipped(self):
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        # A transform that chokes on the blob must not sink the aggregation.
        def _boom(raw):
            raise ValueError("bad")
        out = aggregate_gnd_results(
            ["wasser"], ["lobid"], self.km, {"lobid": _boom},
            params_by_source={"lobid": {"search_type": "kw"}},
        )
        self.assertEqual(out["pool"], [])


class _FakeSuggester:
    def transform(self, raw):
        return {"Wasser": {"count": 9, "gndid": {"g1"}, "ddc": set(), "dk": set()}}


class _FakeProvider:
    """Factory-provider stub whose underlying suggester carries a canned transform.

    Seeded into ``ToolRegistry._provider_cache`` so ``_source_transform`` resolves
    it without building a real (network) provider. - Claude Generated"""

    @property
    def suggester(self):
        return _FakeSuggester()


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class AggregateMcpToolTest(unittest.TestCase):
    """The aggregate_gnd_results MCP handler wires suggesters → engine (no network)."""

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

    def test_handler_builds_pool_json(self):
        from src.mcp.tool_registry import ToolRegistry

        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        reg = ToolRegistry()
        reg._suggesters_initialized = True  # skip network suggester init
        # Seed the factory provider so _source_transform reads the canned transform
        # off the underlying suggester (no network build). - Claude Generated
        reg._provider_cache = {("lobid", False): _FakeProvider()}
        out = json.loads(reg._handle_aggregate_gnd_results(["wasser"], sources=["lobid"]))
        self.assertEqual(out["pool"][0]["title"], "Wasser")
        self.assertEqual(out["pool"][0]["count"], 1)          # count-landmine
        self.assertEqual(out["pool"][0]["display_count"], 9)   # real Häufigkeit
        self.assertEqual(out["pool"][0]["sources"], ["lobid"])


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class AggregateDefaultSourcesTest(unittest.TestCase):
    """The default source list is derived from the enabled instances (WP P1).

    Replaces the hardcoded ``["lobid","swb","catalog"]`` so an external GND plugin
    reaches the pool's provenance. - Claude Generated
    """

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

    def _registry(self, cfg):
        from unittest.mock import MagicMock

        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = MagicMock(load_config=lambda: cfg)
        reg._knowledge_manager = self.km
        reg._suggesters_initialized = True
        reg._provider_cache = {
            ("lobid", False): _FakeProvider(),
            ("swb", False): _FakeProvider(),
            ("catalog", False): _FakeProvider(),
        }
        return reg

    def _standard_config(self):
        """All six built-ins enabled — what synthesize_search_instances produces."""
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(pid, "search_provider", pid, enabled=True, is_primary=True)
            for pid in ("lobid", "swb", "catalog", "finc", "sru", "gnd_local")
        ]
        return cfg

    def test_derived_default_matches_the_legacy_literal(self):
        """The Vergleichslauf: gnd_local is GND-capable and enabled, but has no
        suggester → the pre-existing transform filter drops it → provenance,
        order and ranking stay byte-identical to the old hardcoded default."""
        from src.core.search.factory import enabled_gnd_provider_ids

        cfg = self._standard_config()
        # Guard against a false green: if the config read failed, the handler would
        # fall back to the very literal we compare against and this test would pass
        # while proving nothing. Pin that the derivation really runs — and really
        # yields a *fourth* id that the transform filter then has to drop.
        self.assertEqual(
            enabled_gnd_provider_ids(config=cfg), ["lobid", "swb", "catalog", "gnd_local"]
        )

        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        reg = self._registry(cfg)

        derived = json.loads(reg._handle_aggregate_gnd_results(["wasser"]))
        literal = json.loads(
            reg._handle_aggregate_gnd_results(["wasser"], sources=["lobid", "swb", "catalog"])
        )
        self.assertEqual(derived, literal)
        self.assertEqual(derived["sources"], ["lobid", "swb", "catalog"])

    def test_external_plugin_joins_the_default(self):
        """P1's point: a copied plugin reaches the provenance with no core edit."""
        from src.core.search.providers.lobid.provider import LobidProvider
        from src.core.search.registry import PROVIDER_REGISTRY, register_provider
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        saved = dict(PROVIDER_REGISTRY)
        try:
            @register_provider
            class _PocLobid(LobidProvider):
                id = "poc_lobid"

            cfg = AlimaConfig()
            cfg.plugins = [
                PluginInstanceConfig("poc_lobid", "search_provider", "poc_lobid",
                                     enabled=True, is_primary=True)
            ]
            self.km.store_raw_response("poc_lobid", "wasser", {"search_type": "kw"}, "{}")
            reg = self._registry(cfg)
            reg._provider_cache = {("poc_lobid", False): _FakeProvider()}

            out = json.loads(reg._handle_aggregate_gnd_results(["wasser"]))
            self.assertEqual(out["sources"], ["poc_lobid"])
            self.assertEqual(out["pool"][0]["sources"], ["poc_lobid"])
        finally:
            PROVIDER_REGISTRY.clear()
            PROVIDER_REGISTRY.update(saved)

    def test_unreadable_config_keeps_the_legacy_default(self):
        """None (not []) means 'config unreadable' → never search nothing."""
        from unittest.mock import patch

        reg = self._registry(self._standard_config())
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        with patch("src.core.search.factory.enabled_gnd_provider_ids", return_value=None):
            out = json.loads(reg._handle_aggregate_gnd_results(["wasser"]))
        self.assertEqual(out["sources"], ["lobid", "swb", "catalog"])

    def test_all_sources_disabled_aggregates_nothing(self):
        """[] means the operator disabled every GND source — respect it."""
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        cfg = AlimaConfig()
        cfg.plugins = [
            PluginInstanceConfig(pid, "search_provider", pid, enabled=False, is_primary=True)
            for pid in ("lobid", "swb", "catalog", "finc", "sru", "gnd_local")
        ]
        reg = self._registry(cfg)
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")

        out = json.loads(reg._handle_aggregate_gnd_results(["wasser"]))
        self.assertEqual(out["sources"], [])
        self.assertEqual(out["pool"], [])


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class DefaultAggregateToggleTest(unittest.TestCase):
    """default_aggregate_from_raw reads SystemConfig.aggregate_from_raw (toggle)."""

    def test_reads_config_flag(self):
        from unittest.mock import patch, MagicMock
        from src.core.search.aggregate import default_aggregate_from_raw

        cfg = MagicMock()
        cfg.system_config.aggregate_from_raw = False
        with patch("src.utils.config_manager.ConfigManager") as CM:
            CM.return_value.load_config.return_value = cfg
            self.assertFalse(default_aggregate_from_raw())
        cfg.system_config.aggregate_from_raw = True
        with patch("src.utils.config_manager.ConfigManager") as CM:
            CM.return_value.load_config.return_value = cfg
            self.assertTrue(default_aggregate_from_raw())

    def test_defaults_true_on_error(self):
        from unittest.mock import patch
        from src.core.search.aggregate import default_aggregate_from_raw

        with patch("src.utils.config_manager.ConfigManager", side_effect=Exception("boom")):
            self.assertTrue(default_aggregate_from_raw())


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class NestedFromAggregateTest(unittest.TestCase):
    """Reshape the ranked pool back to the classic {term:{title:{...}}} contract."""

    def test_reshape_and_fresh_sets(self):
        from src.core.search.aggregate import nested_from_aggregate

        agg = {
            "pool": [
                {"title": "Wasser", "gnd_ids": ["g1"], "ddc_codes": ["540"],
                 "dk_codes": [], "count": 1, "display_count": 9},
            ],
            "terms_map": {"Wasser": ["wasser", "h2o"]},
        }
        nested = nested_from_aggregate(agg)
        self.assertEqual(set(nested.keys()), {"wasser", "h2o"})
        w = nested["wasser"]["Wasser"]
        self.assertEqual(w["gndid"], {"g1"})
        self.assertEqual(w["ddc"], {"540"})
        self.assertEqual(w["count"], 1)
        self.assertEqual(w["display_count"], 9)
        # Sets are per-term copies (no shared mutation across terms).
        w["gndid"].add("x")
        self.assertNotIn("x", nested["h2o"]["Wasser"]["gndid"])


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class SearchFromRawTest(unittest.TestCase):
    """Classic SearchCLI.search_from_raw derives the nested view from raw."""

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

    def test_search_from_raw_nested_shape(self):
        """SearchCLI.search_from_raw derives the nested view from the raw cache
        through the unified provider service (no MetaSuggester). - Claude Generated"""
        from src.core.search_cli import SearchCLI
        from src.core.search.provider import ProviderResult, SearchCapability
        from src.core.search.registry import PROVIDER_REGISTRY

        self.km.store_raw_response("fake_l", "wasser", {"search_type": "kw"}, "{}")

        class _Sugg:
            @staticmethod
            def transform(raw):
                return {"Wasser": {"count": 7, "gndid": {"g1"}, "ddc": set(), "dk": set()}}

        class _FakeGnd:
            id = "fake_l"
            label = "fake_l"
            capabilities = {SearchCapability.GND_KEYWORDS}

            def __init__(self, **config):
                pass

            def is_available(self, cfg=None):
                return True

            def search(self, capability, query, *, progress=None, **opts):
                return ProviderResult.from_gnd_keywords({t: {} for t in query})

            @property
            def suggester(self):
                return _Sugg()

        PROVIDER_REGISTRY["fake_l"] = _FakeGnd
        try:
            cli = SearchCLI(self.km)
            nested = cli.search_from_raw(["wasser"], ["fake_l"])
        finally:
            PROVIDER_REGISTRY.pop("fake_l", None)

        self.assertIn("Wasser", nested["wasser"])
        w = nested["wasser"]["Wasser"]
        self.assertEqual(w["count"], 1)           # count-landmine
        self.assertEqual(w["display_count"], 7)   # real count
        self.assertEqual(w["gndid"], {"g1"})


if __name__ == "__main__":
    unittest.main()
