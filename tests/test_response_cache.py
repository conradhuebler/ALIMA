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
import logging
import os
import tempfile
import unittest

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.core.search.providers._base import SuggesterBackedProvider
    from src.core.search.provider import SearchCapability
    from src.utils.suggesters.biblio_suggester import BiblioSuggester
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

    def test_clear_search_cache_clears_mappings_and_raw(self):
        # Both the mapping index and the raw cache must be cleared, and the call
        # must succeed (no bogus commit_transaction → "no transaction is active").
        self.km.update_search_mapping("wasser", "lobid", found_gnd_ids=["g1"],
                                      gnd_counts={"g1": 5})
        self.km.store_raw_response("lobid", "wasser", {"search_type": "kw"}, "{}")
        ok, msg = self.km.clear_search_cache()
        self.assertTrue(ok, msg)
        self.assertIsNone(self.km.get_search_mapping("wasser", "lobid"))
        self.assertIsNone(
            self.km.get_raw_response("lobid", "wasser", {"search_type": "kw"},
                                     max_age_hours=None)
        )

    def test_store_suggester_raw_non_default(self):
        # The non-default MCP search path (bare raw_suggester, bypasses the seam)
        # must still populate raw, keyed by the same params a reader would use.
        from src.mcp.tool_registry import ToolRegistry

        class _Sugg:
            last_raw = {"wasser": '{"m": 1}'}
            last_http_status = {"wasser": 200}

        reg = ToolRegistry()
        reg._store_suggester_raw("lobid", ["wasser"], {"search_type": "title"}, _Sugg())
        got = self.km.get_raw_response("lobid", "wasser", {"search_type": "title"})
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["m"], 1)


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class RawCacheParamsTest(unittest.TestCase):
    """raw_cache_params_for is the single source of truth for cache-key params."""

    def test_per_source_keys(self):
        from src.core.search.provider import raw_cache_params_for

        # lobid ignores max_pages; a non-default search_type is kept.
        self.assertEqual(raw_cache_params_for("lobid", search_type="title"),
                         {"search_type": "title"})
        self.assertEqual(raw_cache_params_for("lobid", search_type="kw", max_pages=9),
                         {"search_type": "kw"})
        # swb keys on max_pages too.
        self.assertEqual(raw_cache_params_for("swb", search_type="kw", max_pages=9),
                         {"search_type": "kw", "max_pages": 9})
        # finc keys on its facet set; None facets dropped.
        self.assertEqual(raw_cache_params_for("finc", search_type="kw", facets=["udk"]),
                         {"search_type": "kw", "facets": ["udk"]})
        self.assertEqual(raw_cache_params_for("finc", search_type="kw"),
                         {"search_type": "kw"})
        # Unknown source → search_type only.
        self.assertEqual(raw_cache_params_for("mystery"), {"search_type": "kw"})


class _FakeBiblioExtractor:
    """Stand-in BiblioClient exposing last_raw after search_subjects."""

    def __init__(self, last_raw):
        self._last_raw = last_raw

    def search_subjects(self, searches, search_type="kw"):
        self.last_raw = self._last_raw
        return {t: {} for t in searches}


class _FakeCurrentTerm:
    def emit(self, *a, **k):
        pass


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class BiblioRawPropagationTest(unittest.TestCase):
    """BiblioSuggester.search must lift the client's parsed records into last_raw."""

    def _fake_self(self, extractor):
        obj = BiblioSuggester.__new__(BiblioSuggester)  # skip network __init__
        obj.extractor = extractor
        obj.logger = logging.getLogger("test_biblio")
        obj.currentTerm = _FakeCurrentTerm()
        return obj

    def test_records_propagated_as_json(self):
        records = {"klima": [{"title": "X", "subjects": ["Klimawandel"]}]}
        obj = self._fake_self(_FakeBiblioExtractor(records))
        BiblioSuggester.search(obj, ["klima"], search_type="kw")
        self.assertIn("klima", obj.last_raw)
        parsed = json.loads(obj.last_raw["klima"])
        self.assertEqual(parsed["totalItems"], 1)
        self.assertEqual(parsed["records"][0]["subjects"], ["Klimawandel"])

    def test_no_client_last_raw_is_safe(self):
        # An extractor that never sets last_raw must not break search.
        class _Bare:
            def search_subjects(self, searches, search_type="kw"):
                return {t: {} for t in searches}

        obj = self._fake_self(_Bare())
        BiblioSuggester.search(obj, ["x"])
        self.assertEqual(obj.last_raw, {})


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


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class RecordRawCaptureTest(unittest.TestCase):
    """finc + catalog-title record searches also populate the raw cache (P5)."""

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

    def test_catalog_titles_raw_stored(self):
        from src.core.search.providers.catalog_provider import CatalogProvider

        class _FakeBiblio:
            def search_titles(self, terms, search_type="title", max_results=25):
                return {t: [{"title": "Rec1"}, {"title": "Rec2"}] for t in terms}

        prov = CatalogProvider()
        prov._suggester = _FakeBiblio()
        prov._ukm_ref = self.km
        prov._cache_raw = True
        prov.search(SearchCapability.TITLE_RECORDS, ["wasser"], search_type="title")
        got = self.km.get_raw_response("catalog_titles", "wasser", {"search_type": "title"})
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["totalItems"], 2)

    def test_finc_raw_stored(self):
        from src.core.search.providers.finc_provider import FincProvider

        class _FakeFinc:
            last_errors = {}

            def search(self, terms, **kw):
                return {t: {"records": [{"id": "1"}], "facets": {}} for t in terms}

        prov = FincProvider()
        prov._suggester = _FakeFinc()
        prov._ukm_ref = self.km
        prov._cache_raw = True
        prov.search(SearchCapability.SUBJECT_FACETS, ["wasser"], search_type="kw",
                    facets=["udk_raw_de105"])
        got = self.km.get_raw_response(
            "finc", "wasser", {"search_type": "kw", "facets": ["udk_raw_de105"]}
        )
        self.assertIsNotNone(got)
        self.assertEqual(json.loads(got["raw_json"])["records"][0]["id"], "1")


class _DoiInst:
    provider_id = "doi_crossref"
    instance_id = "doi_crossref"
    settings: dict = {}


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class InputToolCacheTest(unittest.TestCase):
    """Input tools (e.g. DOI) read-through the raw cache when cacheable (P5)."""

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

    def _handler(self, spec):
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry()
        reg._response_cache_enabled = lambda: True  # deterministic (no real config)
        return reg._make_input_handler(spec, _DoiInst())

    def test_read_through_caches(self):
        from unittest.mock import patch
        from src.utils.input_sources.registry import InputToolSpec

        calls = {"n": 0}

        class _FakeSource:
            def __init__(self, **kw):
                pass

            def mcp_execute(self, value):
                calls["n"] += 1
                return {"doi": value, "title": "Rec"}

        spec = InputToolSpec(name="resolve_doi_crossref", description="d", param="doi")
        handler = self._handler(spec)
        with patch("src.utils.input_sources.get_input_source", return_value=_FakeSource):
            r1 = json.loads(handler(doi="10.1/x"))
            r2 = json.loads(handler(doi="10.1/x"))
        self.assertEqual(r1["title"], "Rec")
        self.assertEqual(r2["title"], "Rec")
        self.assertEqual(calls["n"], 1)  # second call served from cache

    def test_not_cacheable_bypasses(self):
        from unittest.mock import patch
        from src.utils.input_sources.registry import InputToolSpec

        calls = {"n": 0}

        class _FakeSource:
            def __init__(self, **kw):
                pass

            def mcp_execute(self, value):
                calls["n"] += 1
                return {"doi": value}

        spec = InputToolSpec(name="x", description="d", param="doi", cacheable=False)
        handler = self._handler(spec)
        with patch("src.utils.input_sources.get_input_source", return_value=_FakeSource):
            handler(doi="10.1/y")
            handler(doi="10.1/y")
        self.assertEqual(calls["n"], 2)  # never cached


if __name__ == "__main__":
    unittest.main()
