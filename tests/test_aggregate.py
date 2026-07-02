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


if __name__ == "__main__":
    unittest.main()
