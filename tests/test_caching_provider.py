"""Tests for the mapping-first CachingProvider + F-4 display_count (P2) - Claude Generated.

Verifies:
* cache miss → live search + write-back storing per-GND-ID counts (``gnd_counts``);
* cache hit → pool ``count`` stays 1 (landmine) while ``display_count`` restores the
  real count; inner provider not re-queried;
* source failures are NOT cached as "no hit";
* old rows without counts fall back to display_count=None (→ pool count);
* the display value flows through gnd_search_core + flatten_gnd_hits, while
  ``rank_pool`` never reads it.
"""

import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.search.caching import CachingProvider
    from src.core.search.provider import ProviderResult, ResultItem, SearchCapability
    from src.core.gnd_search_core import _entry_from_kw_data, merge_into_pool, rank_pool
    from src.utils.pipeline_formatters import PipelineResultFormatter
    from src.core.agents.shared_context import SharedContext
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


class _FakeProvider:
    """Fake GND_KEYWORDS provider returning canned items; records its calls."""

    id = "lobid"
    label = "Fake Lobid"
    capabilities = {SearchCapability.GND_KEYWORDS}

    def __init__(self, per_term_items, errors=None):
        self._items = per_term_items
        self._errors = errors or {}
        self.calls = []

    def is_available(self, cfg=None):
        return True

    def search(self, capability, query, *, progress=None, **opts):
        self.calls.append(list(query))
        per_term = {t: list(self._items.get(t, [])) for t in query}
        errs = {t: self._errors[t] for t in query if t in self._errors}
        return ProviderResult(
            SearchCapability.GND_KEYWORDS, per_term=per_term, errors=errs
        )


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CachingProviderTest(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()
        self.km = UnifiedKnowledgeManager(database_config=_sqlite_config(self.tmp.name))
        self.km.store_gnd_fact("g1", {"title": "Wassermanagement", "description": "",
                                      "synonyms": "", "ddcs": ""})
        self.km.store_gnd_fact("g2", {"title": "Klima", "description": "",
                                      "synonyms": "", "ddcs": ""})

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_gnd_counts_column_migrated(self):
        rows = self.km.db_manager.fetch_all("PRAGMA table_info(search_mappings)")
        cols = {r["name"] for r in rows}
        self.assertIn("gnd_counts", cols)
        self.assertIn("titles", cols)  # WP Phase C1a denormalized titles column

    def test_miss_then_hit_restores_display_count(self):
        inner = _FakeProvider(
            {"wasser": [ResultItem(label="Wassermanagement", gnd_ids={"g1"}, count=47)]}
        )
        cp = CachingProvider(inner, ukm=self.km)

        # 1) Miss: live search, write-back with real counts.
        res1 = cp.search(SearchCapability.GND_KEYWORDS, ["wasser"])
        self.assertEqual(inner.calls, [["wasser"]])
        item1 = res1.per_term["wasser"][0]
        self.assertEqual(item1.count, 47)  # live count untouched

        mapping = self.km.get_search_mapping("wasser", "lobid")
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.found_gnd_ids, ["g1"])
        self.assertEqual(mapping.gnd_counts, {"g1": 47})

        # 2) Hit: served from cache, inner NOT called again.
        res2 = cp.search(SearchCapability.GND_KEYWORDS, ["wasser"])
        self.assertEqual(inner.calls, [["wasser"]])  # unchanged
        item2 = res2.per_term["wasser"][0]
        self.assertEqual(item2.label, "Wassermanagement")
        self.assertEqual(item2.gnd_ids, {"g1"})
        self.assertEqual(item2.count, 1)          # POOL count stays 1 (landmine)
        self.assertEqual(item2.display_count, 47)  # display-only real count restored

    def test_row_with_titles_but_no_counts_falls_back(self):
        # A cache row carrying denormalized titles but no gnd_counts (e.g. a
        # pre-F-4-style row upgraded to C1a titles) is still a hit; display_count
        # falls back to None → the display layer uses the pool count.
        self.km.update_search_mapping(
            "klima", "lobid", found_gnd_ids=["g2"], titles={"g2": "Klima"}
        )
        cp = CachingProvider(_FakeProvider({}), ukm=self.km)
        res = cp.search(SearchCapability.GND_KEYWORDS, ["klima"])
        item = res.per_term["klima"][0]
        self.assertEqual(item.label, "Klima")     # resolved from the mapping's titles
        self.assertEqual(item.count, 1)
        self.assertIsNone(item.display_count)  # → display falls back to pool count

    def test_pre_c1a_row_without_titles_is_a_miss(self):
        # A legacy row with GND IDs but no denormalized titles is treated as a miss
        # (the cache no longer reads the local GND store to resolve titles).
        self.km.update_search_mapping("klima", "lobid", found_gnd_ids=["g2"])
        inner = _FakeProvider(
            {"klima": [ResultItem(label="Klima", gnd_ids={"g2"}, count=9)]}
        )
        cp = CachingProvider(inner, ukm=self.km)
        res = cp.search(SearchCapability.GND_KEYWORDS, ["klima"])
        self.assertEqual(inner.calls, [["klima"]])  # re-fetched live
        self.assertEqual(res.per_term["klima"][0].count, 9)

    def test_source_failure_not_cached(self):
        inner = _FakeProvider({"x": []}, errors={"x": "network down"})
        cp = CachingProvider(inner, ukm=self.km)
        res = cp.search(SearchCapability.GND_KEYWORDS, ["x"])
        self.assertEqual(res.errors.get("x"), "network down")
        # A source failure must not be stored as a confirmed "no hit".
        self.assertIsNone(self.km.get_search_mapping("x", "lobid"))


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class DisplayCountFlowTest(unittest.TestCase):
    """display_count must flow to the display layer but never into ranking."""

    def test_entry_carries_display_count_but_rank_ignores_it(self):
        kw_data = {"count": 1, "gndid": {"g1"}, "ddc": set(), "dk": set(), "display_count": 47}
        entry = _entry_from_kw_data("Wassermanagement", kw_data)
        self.assertEqual(entry["count"], 1)
        self.assertEqual(entry["display_count"], 47)

        pool = {}
        merge_into_pool(pool, {"Wassermanagement": entry})
        ranked = rank_pool(pool, {"wassermanagement": {"lobid"}})
        # Ranking key is (source_count, count) — count is still the pool's 1.
        self.assertEqual(ranked[0]["count"], 1)
        self.assertEqual(ranked[0]["display_count"], 47)

    def test_merge_max_merges_display_count(self):
        cached = _entry_from_kw_data(
            "T", {"count": 1, "gndid": {"g1"}, "ddc": set(), "dk": set(), "display_count": 47}
        )
        live = _entry_from_kw_data(
            "T", {"count": 3, "gndid": {"g2"}, "ddc": set(), "dk": set()}
        )
        pool = {}
        merge_into_pool(pool, {"T": cached})
        merge_into_pool(pool, {"T": live})
        merged = pool["t"]
        self.assertEqual(merged["count"], 3)          # pool count = max(1, 3)
        self.assertEqual(merged["display_count"], 47)  # real count wins for display

    def test_flatten_uses_display_count_dict_form(self):
        results = {"wasser": {"Wassermanagement": {
            "gnd_ids": ["g1"], "count": 1, "display_count": 47, "classifications": {}}}}
        rows = PipelineResultFormatter.flatten_gnd_hits(results)
        self.assertEqual(rows[0]["count"], 47)

    def test_flatten_uses_display_count_agentic_list(self):
        entries = [{"gnd_id": "g1", "title": "Wassermanagement", "count": 1, "display_count": 47}]
        rows = PipelineResultFormatter.flatten_gnd_hits(entries)
        self.assertEqual(rows[0]["count"], 47)

    def test_flatten_no_display_count_uses_pool_count(self):
        entries = [{"gnd_id": "g1", "title": "X", "count": 5}]
        rows = PipelineResultFormatter.flatten_gnd_hits(entries)
        self.assertEqual(rows[0]["count"], 5)

    def test_agentic_kas_projection_preserves_display_count_end_to_end(self):
        # The counter bug's exact seam: gnd_entries → to_keyword_analysis_state →
        # flatten_gnd_hits (what _populate_gnd_hits feeds the GUI table on agentic
        # completion + reload). Must surface 87, not the pool placeholder 1. - Claude Generated
        ctx = SharedContext(abstract="x", initial_keywords=["Halbleiter"])
        ctx.extracted_keywords = ["Halbleiter"]
        ctx.gnd_entries_per_keyword = {"Halbleiter": ["Halbleiter"]}
        ctx.gnd_entries = [
            {"title": "Halbleiter", "gnd_id": "4129772-7", "gnd_ids": ["4129772-7"],
             "ddc_codes": ["530"], "count": 1, "display_count": 87},
        ]
        state = ctx.to_keyword_analysis_state()
        rows = PipelineResultFormatter.flatten_gnd_hits(state.search_results)
        halbleiter = [r for r in rows if r.get("begriff") == "Halbleiter"]
        self.assertTrue(halbleiter, f"no Halbleiter row in {rows}")
        self.assertEqual(halbleiter[0]["count"], 87)


if __name__ == "__main__":
    unittest.main()
