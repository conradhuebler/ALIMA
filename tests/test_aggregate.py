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


class _FakeMeta:
    def raw_suggester(self, pid=None):
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
        reg._lobid = _FakeMeta()
        reg._swb = None
        reg._biblio = None
        out = json.loads(reg._handle_aggregate_gnd_results(["wasser"], sources=["lobid"]))
        self.assertEqual(out["pool"][0]["title"], "Wasser")
        self.assertEqual(out["pool"][0]["count"], 1)          # count-landmine
        self.assertEqual(out["pool"][0]["display_count"], 9)   # real Häufigkeit
        self.assertEqual(out["pool"][0]["sources"], ["lobid"])


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
        from unittest.mock import patch
        from src.core.search_cli import SearchCLI

        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")

        class _Sugg:
            def transform(self, raw):
                return {"Wasser": {"count": 7, "gndid": {"g1"}, "ddc": set(), "dk": set()}}

        class _Meta:
            last_errors: dict = {}

            def __init__(self, **kw):
                pass

            def search(self, terms):
                return {}

            def raw_suggester(self, pid=None):
                return _Sugg()

        with patch("src.core.search_cli.MetaSuggester", _Meta):
            cli = SearchCLI(self.km)
            nested = cli.search_from_raw(["wasser"], ["lobid"])

        self.assertIn("Wasser", nested["wasser"])
        w = nested["wasser"]["Wasser"]
        self.assertEqual(w["count"], 1)           # count-landmine
        self.assertEqual(w["display_count"], 7)   # real count
        self.assertEqual(w["gndid"], {"g1"})


if __name__ == "__main__":
    unittest.main()
