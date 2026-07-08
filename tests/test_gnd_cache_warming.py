"""GND cache warming: mapping-cache hits resolve after a search - Claude Generated.

Pins the A1/A2 fixes:
* a standalone GND search warms minimal gnd_entries facts (INSERT OR IGNORE), so a
  mapping-cache hit resolves titles instead of returning empty, and search_local_gnd
  sees searched terms;
* warming never clobbers a richer enrichment fact;
* search_local_gnd returns 1-2 local hits instead of discarding them.
"""

import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.search.caching import CachingProvider
    from src.core.search.provider import ProviderResult, ResultItem, SearchCapability
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


class _FakeInner:
    id = "fake"
    label = "fake"
    capabilities = {SearchCapability.GND_KEYWORDS}

    def __init__(self):
        self.live_calls = 0

    def is_available(self, cfg=None):
        return True

    def search(self, capability, query, *, progress=None, **opts):
        self.live_calls += 1
        per_term = {
            t: [ResultItem(label="Grasfrosch", gnd_ids={"g1", "g2"}, count=5)] for t in query
        }
        return ProviderResult(SearchCapability.GND_KEYWORDS, per_term=per_term)


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class GndCacheWarmingTest(unittest.TestCase):
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

    def test_cache_hit_resolves_after_warming(self):
        inner = _FakeInner()
        cp = CachingProvider(inner, ukm=self.km)
        r1 = cp.search(SearchCapability.GND_KEYWORDS, ["x"])
        r2 = cp.search(SearchCapability.GND_KEYWORDS, ["x"])  # cache hit
        self.assertEqual(inner.live_calls, 1)                  # 2nd served from cache
        self.assertEqual(len(r2.per_term["x"]), 1)             # was 0 before the fix
        self.assertEqual(r2.per_term["x"][0].gnd_ids, {"g1", "g2"})
        self.assertEqual(r2.per_term["x"][0].display_count, 5)  # real count restored

    def test_warm_does_not_clobber_enriched_fact(self):
        self.km.store_gnd_fact("g1", {"title": "Grasfrosch", "description": "RICH", "ddcs": "597.8"})
        self.km.warm_gnd_entries({"g1": "Grasfrosch", "g2": "Springfrosch"})
        f1 = self.km.get_gnd_fact("g1")
        f2 = self.km.get_gnd_fact("g2")
        self.assertEqual(f1.description, "RICH")   # enriched fact preserved
        self.assertEqual(f1.ddcs, "597.8")
        self.assertIsNotNone(f2)                    # new id warmed in

    def test_search_local_gnd_returns_partial_hits(self):
        self.km.warm_gnd_entries({"g1": "Grasfrosch"})  # single local hit
        hits = self.km.search_local_gnd("Grasfrosch", min_results=3)
        self.assertEqual([h.gnd_id for h in hits], ["g1"])  # was [] under the old gate


if __name__ == "__main__":
    unittest.main()
