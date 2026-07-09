"""Search cache is self-contained; local GND store stays independent - Claude Generated.

Pins WP Phase C1:
* a mapping-cache hit rebuilds items from the mapping's own denormalized ``titles``
  (no read of the local GND store), so ``display_count`` is restored and results are
  non-empty (the F1 symptom stays fixed);
* a standalone GND search does **not** write into the local GND store — ``warm_gnd_entries``
  is gone, so ``search_local_gnd`` / ``get_gnd_fact`` never see terms that were only
  searched online (the operator's chosen decoupling);
* a pre-C1a mapping row (GND IDs but no titles) is treated as a cache miss;
* ``search_local_gnd`` returns 1-2 local hits instead of discarding them (F2).
"""

import os
import shutil
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
class SearchCacheDecouplingTest(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        # Temp *directory* so a sibling gnd_local.db (Phase C1c) is isolated + cleaned.
        self.tmpdir = tempfile.mkdtemp()
        db = os.path.join(self.tmpdir, "alima_knowledge.db")
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(db))

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_hit_resolves_from_mapping(self):
        inner = _FakeInner()
        cp = CachingProvider(inner, ukm=self.km)
        cp.search(SearchCapability.GND_KEYWORDS, ["x"])
        r2 = cp.search(SearchCapability.GND_KEYWORDS, ["x"])  # cache hit
        self.assertEqual(inner.live_calls, 1)                  # 2nd served from cache
        self.assertEqual(len(r2.per_term["x"]), 1)             # rebuilt from mapping titles
        self.assertEqual(r2.per_term["x"][0].gnd_ids, {"g1", "g2"})
        self.assertEqual(r2.per_term["x"][0].display_count, 5)  # real count restored

    def test_standalone_search_does_not_touch_local_store(self):
        inner = _FakeInner()
        cp = CachingProvider(inner, ukm=self.km)
        cp.search(SearchCapability.GND_KEYWORDS, ["Grasfrosch"])
        # The online-searched term/ids never leak into the local authority copy.
        self.assertEqual(self.km.search_local_gnd("Grasfrosch", min_results=1), [])
        self.assertIsNone(self.km.get_gnd_fact("g1"))

    def test_pre_c1a_mapping_without_titles_is_a_miss(self):
        # Simulate a legacy row: GND IDs stored, but no denormalized titles.
        self.km.update_search_mapping("y", "fake", found_gnd_ids=["g1", "g2"], gnd_counts={})
        inner = _FakeInner()
        cp = CachingProvider(inner, ukm=self.km)
        cp.search(SearchCapability.GND_KEYWORDS, ["y"])
        self.assertEqual(inner.live_calls, 1)  # re-fetched live instead of a broken hit

    def test_search_local_gnd_returns_partial_hits(self):
        # F2: fewer than min_results local matches must still be returned.
        self.km.store_gnd_fact("g1", {"title": "Grasfrosch", "description": "RICH", "ddcs": "597.8"})
        hits = self.km.search_local_gnd("Grasfrosch", min_results=3)
        self.assertEqual([h.gnd_id for h in hits], ["g1"])  # was [] under the old gate

    def test_gnd_entries_live_in_separate_store_db(self):
        # WP Phase C1c: the local GND copy is a physically separate DB file.
        import sqlite3
        self.km.store_gnd_fact("g1", {"title": "Grasfrosch", "ddcs": ""})
        gnd_local = os.path.join(self.tmpdir, "gnd_local.db")
        cache_db = os.path.join(self.tmpdir, "alima_knowledge.db")
        self.assertTrue(os.path.exists(gnd_local))

        def tables(p):
            con = sqlite3.connect(p)
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            con.close()
            return names

        self.assertIn("gnd_entries", tables(gnd_local))
        self.assertNotIn("gnd_entries", tables(cache_db))
        self.assertIn("search_mappings", tables(cache_db))


if __name__ == "__main__":
    unittest.main()
