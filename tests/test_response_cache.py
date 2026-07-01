"""Tests for the WP2 raw-first response cache (P1) - Claude Generated.

Covers the additive infrastructure:
* the ``search_response_cache`` table is created on init;
* ``store_raw_response`` / ``get_raw_response`` round-trip;
* ``params_hash`` is order-independent, drops ``None`` values, and separates
  ``kw`` from ``title`` (no cross-search-type collision);
* the size cap rejects oversized blobs;
* the TTL gate returns ``None`` for stale rows (and can be disabled);
* the soft row cap prunes the oldest rows per source;
* the ``SuggesterBackedProvider`` fetch seam dual-writes ``suggester.last_raw``
  without disturbing the reduced GND-keyword output, and respects the enable gate.
"""

import json
import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.search.providers._base import SuggesterBackedProvider
    from src.core.search.provider import SearchCapability
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


class _FakeSuggester:
    """Minimal suggester exposing ``last_raw`` like the real ones do post-capture."""

    def __init__(self, per_term_raw):
        self._raw = per_term_raw
        self.last_errors = {}
        self.last_raw = {}
        self.last_http_status = {}

    def search(self, terms, **kwargs):
        out = {}
        for t in terms:
            blob = self._raw.get(t)
            if blob is not None:
                self.last_raw[t] = blob
                self.last_http_status[t] = 200
            out[t] = {}  # reduced view — empty is fine for the seam test
        return out


_SeamBase = SuggesterBackedProvider if IMPORT_ERROR is None else object


class _SeamProvider(_SeamBase):
    id = "lobid"
    label = "Fake Lobid"
    capabilities = {SearchCapability.GND_KEYWORDS} if IMPORT_ERROR is None else set()

    def __init__(self, suggester, provider_id="lobid", **config):
        super().__init__(**config)
        self._suggester = suggester
        self.id = provider_id  # per-instance override of the class attr

    def _build_suggester(self):
        return self._suggester


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class RawResponseCacheTest(unittest.TestCase):
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

    def test_table_created(self):
        rows = self.km.db_manager.fetch_all("PRAGMA table_info(search_response_cache)")
        cols = {r["name"] for r in rows}
        self.assertTrue(
            {"source", "query", "normalized_query", "params_hash", "raw_json",
             "http_status", "byte_size", "last_updated"}.issubset(cols)
        )

    def test_store_get_roundtrip(self):
        blob = json.dumps({"member": [1, 2], "totalItems": 9})
        self.assertTrue(
            self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"},
                                       blob, http_status=200, result_count=2)
        )
        got = self.km.get_raw_response("lobid", "wasser", {"search_type": "kw"})
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["totalItems"], 9)
        self.assertEqual(got["http_status"], 200)
        self.assertEqual(got["result_count"], 2)

    def test_params_hash_orderless_and_drops_none(self):
        h = UnifiedKnowledgeManager.params_hash
        self.assertEqual(h({"a": 1, "b": 2}), h({"b": 2, "a": 1}))
        self.assertEqual(h({"a": 1, "b": None}), h({"a": 1}))
        self.assertEqual(h(None), h({}))

    def test_search_type_no_collision(self):
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, '{"m":"kw"}')
        self.km.store_raw_response("lobid", "wasser", {"search_type": "title"}, '{"m":"title"}')
        kw = self.km.get_raw_response("lobid", "wasser", {"search_type": "kw"})
        title = self.km.get_raw_response("lobid", "wasser", {"search_type": "title"})
        self.assertEqual(json.loads(kw["raw_json"])["m"], "kw")
        self.assertEqual(json.loads(title["raw_json"])["m"], "title")

    def test_size_cap_rejects_large_blob(self):
        big = "x" * 2048
        self.assertFalse(
            self.km.store_raw_response("lobid", "huge", {"search_type": "kw"},
                                       big, max_bytes=1024)
        )
        self.assertIsNone(self.km.get_raw_response("lobid", "huge", {"search_type": "kw"}))

    def test_ttl_expiry(self):
        self.km.store_raw_response("lobid", "old", {"search_type": "kw"}, '{"x":1}')
        # Age the row well past any TTL.
        self.km.db_manager.execute_query(
            "UPDATE search_response_cache SET last_updated = ? WHERE source = ?",
            ["2000-01-01 00:00:00", "lobid"],
        )
        self.assertIsNone(
            self.km.get_raw_response("lobid", "old", {"search_type": "kw"}, max_age_hours=24)
        )
        # TTL disabled → still retrievable.
        self.assertIsNotNone(
            self.km.get_raw_response("lobid", "old", {"search_type": "kw"}, max_age_hours=None)
        )

    def test_row_cap_prune(self):
        # Seed 5 rows with distinct (manually-aged) timestamps.
        for i in range(5):
            self.km.store_raw_response("swb", f"t{i}", {"search_type": "kw"}, "{}")
            self.km.db_manager.execute_query(
                "UPDATE search_response_cache SET last_updated = ? "
                "WHERE source = 'swb' AND normalized_query = ?",
                [f"2020-01-0{i + 1} 00:00:00", f"t{i}"],
            )
        # A 6th write (newest) with cap=3 triggers the prune of the oldest rows.
        self.km.store_raw_response("swb", "t5", {"search_type": "kw"}, "{}",
                                   max_rows_per_source=3)
        n = self.km.db_manager.fetch_one(
            "SELECT COUNT(*) AS n FROM search_response_cache WHERE source = 'swb'"
        )["n"]
        self.assertEqual(n, 4)  # soft cap keeps the boundary tie (t2..t5)
        # The two oldest were pruned; the newest survives.
        self.assertIsNone(self.km.get_raw_response("swb", "t0", {"search_type": "kw"},
                                                   max_age_hours=None))
        self.assertIsNotNone(self.km.get_raw_response("swb", "t5", {"search_type": "kw"},
                                                      max_age_hours=None))

    def test_seam_dual_writes_raw(self):
        blob = json.dumps({"member": [1, 2, 3], "totalItems": 42})
        prov = _SeamProvider(_FakeSuggester({"wasser": blob}))
        prov._ukm_ref = self.km
        prov._cache_raw = True
        prov._gnd_search(["wasser"], None, search_type="kw")
        got = self.km.get_raw_response("lobid", "wasser", {"search_type": "kw"})
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["totalItems"], 42)
        self.assertEqual(got["http_status"], 200)

    def test_seam_respects_disable(self):
        prov = _SeamProvider(_FakeSuggester({"wasser": "{}"}), cache_responses=False)
        prov._ukm_ref = self.km
        prov._cache_raw = True  # global on, but per-instance override wins
        prov._gnd_search(["wasser"], None, search_type="kw")
        self.assertIsNone(self.km.get_raw_response("lobid", "wasser", {"search_type": "kw"}))

    def test_seam_skips_when_no_last_raw(self):
        # A suggester that never populates last_raw must not error or write.
        sugg = _FakeSuggester({})
        prov = _SeamProvider(sugg)
        prov._ukm_ref = self.km
        prov._cache_raw = True
        prov._gnd_search(["nope"], None, search_type="kw")
        self.assertIsNone(self.km.get_raw_response("lobid", "nope", {"search_type": "kw"}))

    def test_seam_swb_params_include_max_pages(self):
        # swb passes both search_type and max_pages → both belong in the key.
        blob = json.dumps({"pages": ["<html/>"], "totalItems": 3})
        prov = _SeamProvider(_FakeSuggester({"klima": blob}), provider_id="swb")
        prov._ukm_ref = self.km
        prov._cache_raw = True
        prov._gnd_search(["klima"], None, search_type="kw", max_pages=5)
        got = self.km.get_raw_response("swb", "klima", {"search_type": "kw", "max_pages": 5})
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["totalItems"], 3)
        # A different max_pages is a distinct cache key → miss.
        self.assertIsNone(
            self.km.get_raw_response("swb", "klima", {"search_type": "kw", "max_pages": 9})
        )


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class AgentViewSurfacingTest(unittest.TestCase):
    """P2 transform-on-read: the MCP handler attaches member/totalItems from the raw cache."""

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

    def test_attach_agent_view_from_raw_cache(self):
        from src.mcp.tool_registry import ToolRegistry

        blob = json.dumps({"totalItems": 77, "member": [{"title": "X"}], "aggregation": {}})
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, blob)
        reg = ToolRegistry()
        out = {"source": "lobid", "results": {}}
        reg._attach_agent_view(out, "lobid", ["wasser"], "kw")
        self.assertIn("agent_view", out)
        self.assertEqual(out["agent_view"]["wasser"]["totalItems"], 77)
        self.assertEqual(len(out["agent_view"]["wasser"]["member"]), 1)

    def test_attach_agent_view_noop_for_unsupported_source(self):
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry()
        out = {"source": "catalog", "results": {}}
        reg._attach_agent_view(out, None, ["x"], "kw")  # catalog: raw_id None
        self.assertNotIn("agent_view", out)

    def test_attach_agent_view_miss_omits_term(self):
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry()
        out = {"source": "lobid", "results": {}}
        reg._attach_agent_view(out, "lobid", ["never-cached"], "kw")
        self.assertNotIn("agent_view", out)


if __name__ == "__main__":
    unittest.main()
