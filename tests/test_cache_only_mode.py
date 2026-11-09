"""Claude Generated - Cache-only mode (``cache_only``) of the catalog and finc sources.

The switch has to reach every path into these sources: the plugin hull built
by the search factory (``CachingProvider`` + WP2 raw cache + aggregate), the
classic DK step that calls the extractor directly (``catalog_dk_cache``), the
generated agent tools, and the per-run reporting.

Every test runs behind a network tripwire: an attempt to send an HTTP request
(``requests.Session.request``) or open a socket is recorded and raises. A broad
``except`` in the code under test may swallow the raise, the record stays, so
the cache-only tests assert that it is empty. The off-mode tests pin today's
behaviour: the live path is taken and the age limits apply.

All data lives in a throwaway SQLite file; nothing touches the operator's
configuration or the network.
"""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.core.search.cache_only import (
    CACHE_MISS_MESSAGE,
    TOOL_DESCRIPTION_NOTE,
    is_cache_miss,
    is_cache_only,
    split_cache_misses,
)
from src.core.search.provider import SearchCapability, raw_cache_params_for
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import DatabaseConfig
from src.utils.config_models import AlimaConfig, PluginInstanceConfig

STALE = "2020-01-01 00:00:00"  # far beyond every age limit (24 h raw, 30 min DK TTL)

CATALOG_RECORDS = {
    "records": [
        {"subjects": ["Photosynthese", "Pflanzenphysiologie"], "mab_subjects": []},
        {"subjects": ["Photosynthese"], "mab_subjects": []},
    ],
    "totalItems": 2,
}
CATALOG_TITLES = {
    "records": [
        {"rsn": "1", "title": "Faust I", "classifications": {}},
        {"rsn": "2", "title": "Faust II", "classifications": {}},
    ],
    "totalItems": 2,
}
FINC_ENTRY = {
    "records": [{"id": "0-1", "title": "Graphen A"}, {"id": "0-2", "title": "Graphen B"}],
    "result_count": 2,
    "facets": {},
    "errors": [],
}
FINC_FAILED_ENTRY = {"records": [], "result_count": 0, "facets": {}, "errors": ["HTTP 500"]}
DK_TITLES = [{"rsn": "1", "title": "Pflanzen", "classifications": ["DK 581.1"]}]


class _NetworkTripwire:
    """Records (and refuses) every HTTP request and socket connect."""

    def __init__(self):
        self.calls = []
        self._patches = []

    def start(self):
        def _request(_session, method, url, *a, **k):
            self.calls.append(f"{method} {url}")
            raise AssertionError(f"network request attempted: {method} {url}")

        def _connect(_sock, address):
            self.calls.append(f"connect {address}")
            raise AssertionError(f"socket connect attempted: {address}")

        self._patches = [
            patch.object(requests.sessions.Session, "request", _request),
            patch.object(socket.socket, "connect", _connect),
        ]
        for p in self._patches:
            p.start()
        return self

    def stop(self):
        for p in reversed(self._patches):
            p.stop()


class _TempKnowledgeDB(unittest.TestCase):
    """A fresh UnifiedKnowledgeManager on a throwaway SQLite file + tripwire."""

    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.km = UnifiedKnowledgeManager(
            database_config=DatabaseConfig(
                db_type="sqlite", sqlite_path=str(Path(self._tmp.name) / "knowledge.db")
            )
        )
        self.net = _NetworkTripwire().start()
        self.addCleanup(self.net.stop)

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        self._tmp.cleanup()

    # --- seeding helpers ------------------------------------------------------
    def seed_raw(self, source, term, params, payload):
        self.km.store_raw_response(source, term, params, json.dumps(payload))

    def age_raw(self, source):
        self.km.db_manager.execute_query(
            "UPDATE search_response_cache SET last_updated = ? WHERE source = ?", [STALE, source]
        )

    def raw_last_updated(self, source, term, params):
        row = self.km.get_raw_response(source, term, params, max_age_hours=None)
        return str(row["last_updated"]) if row else None

    def seed_dk(self):
        self.km.store_catalog_dk_cache("Photosynthese", DK_TITLES, status="success")
        self.km.store_catalog_dk_cache("Leer", [], status="no_results", ttl_minutes=30)
        self.km.store_catalog_dk_cache("Kaputt", [], status="error", error_message="timeout",
                                       ttl_minutes=60)
        # Every TTL long expired: cache-only must still read these rows.
        self.km.db_manager.execute_query(
            "UPDATE catalog_dk_cache SET retry_after = ?, last_updated = ?",
            ["2020-01-01T00:00:00", STALE],
        )


def _catalog_instance(cache_only):
    return PluginInstanceConfig(
        "catalog", "search_provider", "catalog", enabled=True,
        settings={
            "token": "t",
            "catalog_search_url": "http://catalog.invalid/soap",
            "catalog_details": "http://catalog.invalid/details",
            "catalog_web_search_url": "http://catalog.invalid/web",
            "cache_only": cache_only,
        },
    )


def _finc_instance(cache_only):
    return PluginInstanceConfig(
        "finc", "search_provider", "finc", enabled=True,
        settings={"base_url": "http://finc.invalid/proxy.php", "cache_only": cache_only},
    )


class _FakeSuggester:
    """Stands in for the live suggester in off-mode tests (records its calls)."""

    def __init__(self, result=None):
        self.calls = []
        self.last_raw = {}
        self.last_errors = {}
        self._result = result or {}

    def search(self, terms, **kw):
        self.calls.append(list(terms))
        return {t: self._result.get(t, {}) for t in terms}


# ---------------------------------------------------------------------------
# Setting + helpers
# ---------------------------------------------------------------------------
class TestTripwireIsArmed(unittest.TestCase):
    """Positive control: without it an empty ``net.calls`` would prove nothing."""

    def test_records_requests_and_sockets(self):
        net = _NetworkTripwire().start()
        try:
            with self.assertRaises(AssertionError):
                requests.Session().get("http://catalog.invalid/x", timeout=1)
            with socket.socket() as sock, self.assertRaises(AssertionError):
                sock.connect(("catalog.invalid", 80))
        finally:
            net.stop()
        self.assertEqual(len(net.calls), 2)


class TestSetting(unittest.TestCase):
    def test_field_is_declared_for_the_sources_with_a_cache_path(self):
        """Since Nov 6 also lobid, swb and kvk, since Nov 9 sru (raw cache of
        its title and PICA DK answers); gnd_local is local. - Claude Generated"""
        from src.core.plugins.schema import BOOL
        from src.core.search.factory import SearchProviderCategory

        cat = SearchProviderCategory()
        for pid in ("catalog", "finc", "lobid", "swb", "kvk", "sru"):
            fields = {f.key: f for f in cat.type_meta(pid).config_fields}
            self.assertIn("cache_only", fields, pid)
            self.assertEqual(fields["cache_only"].kind, BOOL)
            self.assertIs(fields["cache_only"].default, False)
            self.assertTrue(fields["cache_only"].help)
        for pid in ("gnd_local",):
            keys = {f.key for f in cat.type_meta(pid).config_fields}
            self.assertNotIn("cache_only", keys, pid)

    def test_coercion_and_miss_split(self):
        self.assertFalse(is_cache_only(None))
        self.assertFalse(is_cache_only({}))
        self.assertTrue(is_cache_only({"cache_only": True}))
        self.assertTrue(is_cache_only({"cache_only": "true"}))
        self.assertFalse(is_cache_only({"cache_only": "off"}))
        failures, misses = split_cache_misses({
            "catalog:a": CACHE_MISS_MESSAGE,
            "catalog:b": CACHE_MISS_MESSAGE,
            "lobid:a": "HTTP 500",
        })
        self.assertEqual(failures, {"lobid:a": "HTTP 500"})
        self.assertEqual(misses, {"catalog": ["a", "b"]})


# ---------------------------------------------------------------------------
# Plugin hull: catalog through the search factory
# ---------------------------------------------------------------------------
class TestCatalogPluginHull(_TempKnowledgeDB):
    def setUp(self):
        super().setUp()
        self.kw = {"search_type": "kw"}
        self.seed_raw("catalog", "Photosynthese", self.kw, CATALOG_RECORDS)
        self.age_raw("catalog")

    def _provider(self, cache_only, cache=True):
        from src.core.search.factory import build_provider

        return build_provider(_catalog_instance(cache_only), cache=cache, ukm=self.km)

    def test_stale_entry_is_served_without_network(self):
        res = self._provider(True).search(SearchCapability.GND_KEYWORDS, ["Photosynthese"])
        labels = {it.label: it.count for it in res.per_term["Photosynthese"]}
        self.assertEqual(labels, {"Photosynthese": 2, "Pflanzenphysiologie": 1})
        self.assertEqual(res.errors, {})
        self.assertEqual(self.net.calls, [])

    def test_miss_is_empty_with_reason_and_writes_no_cache(self):
        provider = self._provider(True)
        res = provider.search(SearchCapability.GND_KEYWORDS, ["Photosynthese", "Fehlt"])
        self.assertEqual(res.per_term["Fehlt"], [])
        self.assertTrue(is_cache_miss(res.errors.get("Fehlt")))
        # Neither the miss nor the hit reaches the mapping cache, and the stale raw
        # row is not re-stamped as fresh.
        self.assertIsNone(self.km.get_search_mapping("Fehlt", "catalog"))
        self.assertIsNone(self.km.get_search_mapping("Photosynthese", "catalog"))
        self.assertTrue(
            self.raw_last_updated("catalog", "Photosynthese", self.kw).startswith("2020")
        )
        self.assertEqual(self.net.calls, [])

    def test_raw_first_service_path_serves_stale_entry(self):
        from src.core.search.service import search_gnd_keywords

        nested, errors = search_gnd_keywords(
            ["Photosynthese", "Fehlt"], [_catalog_instance(True)],
            cache=True, aggregate_from_raw=True, ukm=self.km,
        )
        self.assertEqual(set(nested["Photosynthese"]), {"Photosynthese", "Pflanzenphysiologie"})
        self.assertEqual(nested.get("Fehlt", {}), {})
        self.assertEqual(list(errors), ["catalog:Fehlt"])
        self.assertTrue(is_cache_miss(errors["catalog:Fehlt"]))
        self.assertEqual(self.net.calls, [])

    def test_title_records_from_raw_cache(self):
        params = raw_cache_params_for("catalog_titles", search_type="title")
        self.seed_raw("catalog_titles", "Faust", params, CATALOG_TITLES)
        self.age_raw("catalog_titles")
        res = self._provider(True, cache=False).search(
            SearchCapability.TITLE_RECORDS, ["Faust", "Fehlt"],
            search_type="title", max_results=1,
        )
        self.assertEqual([it.record["title"] for it in res.per_term["Faust"]], ["Faust I"])
        self.assertEqual(res.per_term["Fehlt"], [])
        self.assertEqual(list(res.errors), ["Fehlt"])
        self.assertTrue(is_cache_miss(res.errors["Fehlt"]))
        self.assertEqual(self.net.calls, [])

    def test_off_mode_takes_the_live_suggester_and_age_applies(self):
        from src.core.search.aggregate import aggregate_gnd_results
        from src.core.search.providers.catalog.suggester import BiblioSuggester

        provider = self._provider(False, cache=False)
        fake = _FakeSuggester({"Photosynthese": {"Live": {"count": 3, "gnd_ids": set()}}})
        provider._suggester = fake
        res = provider.search(SearchCapability.GND_KEYWORDS, ["Photosynthese"])
        self.assertEqual(fake.calls, [["Photosynthese"]])
        self.assertEqual([it.label for it in res.per_term["Photosynthese"]], ["Live"])
        # Without the mode the stale raw entry is outside the 24 h window.
        agg = aggregate_gnd_results(
            ["Photosynthese"], ["catalog"], self.km, {"catalog": BiblioSuggester().transform},
            params_by_source={"catalog": self.kw},
        )
        self.assertEqual(agg["missing"], {"catalog": ["Photosynthese"]})
        self.assertEqual(agg["pool"], [])


# ---------------------------------------------------------------------------
# Direct DK path: BiblioClient + catalog_dk_cache
# ---------------------------------------------------------------------------
class TestCatalogDkDirectPath(_TempKnowledgeDB):
    def setUp(self):
        super().setUp()
        self.seed_dk()

    def _client(self, cache_only, **kw):
        from src.utils.clients.biblio_client import BiblioClient

        return BiblioClient(
            token="t",
            soap_search_url="http://catalog.invalid/soap",
            soap_details_url="http://catalog.invalid/details",
            web_search_url="http://catalog.invalid/web",
            web_record_url="http://catalog.invalid/Record/",
            cache_only=cache_only,
            **kw,
        )

    def test_cache_only_extraction_reads_any_age_and_lists_misses(self):
        client = self._client(True)
        res = client.extract_dk_classifications_for_keywords(
            ["Photosynthese", "Leer", "Kaputt", "Fehlt"]
        )
        self.assertEqual([r["keyword"] for r in res], ["Photosynthese"])
        self.assertEqual(res[0]["source"], "cache")
        self.assertEqual(res[0]["classifications"][0]["dk"], "581.1")
        # "Leer" is a known empty answer; a cached failure and a missing row are misses.
        self.assertEqual(client.last_cache_misses, ["Kaputt", "Fehlt"])
        # Nothing written: no row for the miss, the failure row untouched.
        self.assertIsNone(self.km.get_catalog_dk_cache("Fehlt", ignore_ttl=True))
        self.assertEqual(self.km.get_catalog_dk_cache("Kaputt", ignore_ttl=True)[1], "error")
        self.assertEqual(self.net.calls, [])

    def test_every_request_method_is_blocked(self):
        client = self._client(True, use_json_cache=False)
        self.assertEqual(client.search("x"), [])
        self.assertEqual(client._search_web("x"), [])
        self.assertIsNone(client.get_title_details("123"))
        self.assertIsNone(client._get_title_details_web("123"))
        self.assertEqual(client.search_subjects(["x"]), {"x": {}})
        self.assertEqual(self.net.calls, [])

    def test_off_mode_miss_goes_live_and_caches_no_results(self):
        client = self._client(False, rate_limit_delay_ms=0)
        client.enable_web_fallback = False
        live_calls = []
        client.search = lambda term, search_type="ku": live_calls.append(term) or []
        client.extract_dk_classifications_for_keywords(["Fehlt"])
        self.assertEqual(live_calls, ["Fehlt"])
        self.assertEqual(self.km.get_catalog_dk_cache("Fehlt")[1], "no_results")

    def test_ttl_applies_unless_ignored(self):
        self.assertIsNone(self.km.get_catalog_dk_cache("Leer"))
        self.assertEqual(self.km.get_catalog_dk_cache("Leer", ignore_ttl=True)[1], "no_results")


# ---------------------------------------------------------------------------
# Where the direct path gets the switch from: the instance settings
# ---------------------------------------------------------------------------
class _StubConfig:
    def __init__(self, *instances):
        self._instances = list(instances)

    def enabled_instances_for(self, category):
        return [i for i in self._instances if i.enabled]


class TestDkExtractorResolution(unittest.TestCase):
    def test_enabled_catalog_instance_carries_the_switch(self):
        from src.core.search.factory import resolve_dk_extractor

        on = resolve_dk_extractor(config=_StubConfig(_catalog_instance(True)))
        off = resolve_dk_extractor(config=_StubConfig(_catalog_instance(False)))
        self.assertEqual(type(on).__name__, "BiblioClient")
        self.assertTrue(on.cache_only)
        self.assertFalse(off.cache_only)

    def test_enabled_finc_dk_instance_carries_the_switch(self):
        from src.core.search.factory import resolve_dk_extractor

        inst = _finc_instance(True)
        inst.settings["dk_enabled"] = True
        ext = resolve_dk_extractor(config=_StubConfig(inst))
        self.assertEqual(type(ext).__name__, "FincCatalogClient")
        self.assertTrue(ext.cache_only)
        self.assertTrue(ext._client.cache_only)

    def test_disabled_instance_is_not_used_at_all(self):
        """A disabled catalog instance contributes none of its settings; the
        fallback extractor has no endpoints, so it cannot reach the catalog."""
        from src.core.search.factory import resolve_dk_extractor

        inst = _catalog_instance(True)
        inst.enabled = False
        ext = resolve_dk_extractor(config=_StubConfig(inst))
        self.assertFalse(ext.cache_only)
        self.assertEqual(ext.SEARCH_URL, "")
        self.assertEqual(ext.WEB_SEARCH_URL, "")


# ---------------------------------------------------------------------------
# finc: records/facets and the DK extractor
# ---------------------------------------------------------------------------
class TestFinc(_TempKnowledgeDB):
    def _provider(self, cache_only):
        from src.core.search.providers.finc.provider import FincProvider

        p = FincProvider(base_url="http://finc.invalid/proxy.php", cache_only=cache_only)
        p._ukm_ref = self.km
        return p

    def test_records_from_raw_cache_any_age(self):
        params = raw_cache_params_for("finc", search_type="subject")
        self.seed_raw("finc", "Graphen", params, FINC_ENTRY)
        self.seed_raw("finc", "Kaputt", params, FINC_FAILED_ENTRY)
        self.age_raw("finc")
        res = self._provider(True).search(
            SearchCapability.TITLE_RECORDS, ["Graphen", "Kaputt", "Fehlt"],
            search_type="subject", limit=1,
        )
        self.assertEqual([it.label for it in res.per_term["Graphen"]], ["Graphen A"])
        self.assertEqual(sorted(res.errors), ["Fehlt", "Kaputt"])
        self.assertTrue(all(is_cache_miss(m) for m in res.errors.values()))
        self.assertTrue(self.raw_last_updated("finc", "Graphen", params).startswith("2020"))
        self.assertEqual(self.net.calls, [])

    def test_client_sends_no_request(self):
        from src.core.search.providers.finc.finc_client import FincClient

        client = FincClient(base_url="http://finc.invalid/proxy.php", cache_only=True)
        out = client.search("x")
        self.assertEqual(out["status"], "ERROR")
        self.assertTrue(is_cache_miss(out["error"]))
        self.assertEqual(client.get_records(["0-1"])["status"], "ERROR")
        self.assertEqual(client.discover_fields(), {})
        self.assertEqual(self.net.calls, [])

    def test_dk_extractor_reads_dk_cache_any_age(self):
        self.seed_dk()
        ext = self._provider(True).dk_extractor(knowledge_manager=self.km)
        res = ext.extract_dk_classifications_for_keywords(["Photosynthese", "Leer", "Fehlt"])
        self.assertEqual([r["keyword"] for r in res], ["Photosynthese"])
        self.assertEqual(ext.last_cache_misses, ["Fehlt"])
        self.assertIsNone(self.km.get_catalog_dk_cache("Fehlt", ignore_ttl=True))
        self.assertEqual(self.net.calls, [])

    def test_off_mode_takes_the_live_paths(self):
        provider = self._provider(False)
        fake = _FakeSuggester()
        provider._suggester = fake
        provider.search(SearchCapability.TITLE_RECORDS, ["Graphen"], search_type="subject")
        self.assertEqual(fake.calls, [["Graphen"]])

        ext = provider.dk_extractor(knowledge_manager=self.km)
        ext._client = MagicMock()
        ext._client.search.return_value = {"status": "OK", "records": []}
        ext.extract_dk_classifications_for_keywords(["Fehlt"])
        self.assertEqual(ext._client.search.call_count, 1)
        self.assertEqual(self.km.get_catalog_dk_cache("Fehlt")[1], "no_results")


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------
class TestAgentTools(_TempKnowledgeDB):
    def setUp(self):
        super().setUp()
        self.seed_raw("catalog", "Photosynthese", {"search_type": "kw"}, CATALOG_RECORDS)
        self.seed_raw("catalog_titles", "Faust",
                      raw_cache_params_for("catalog_titles", search_type="title"), CATALOG_TITLES)
        self.seed_raw("finc", "Graphen", raw_cache_params_for("finc", search_type="kw"), FINC_ENTRY)
        for source in ("catalog", "catalog_titles", "finc"):
            self.age_raw(source)

    def _registry(self, cache_only):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = [_catalog_instance(cache_only), _finc_instance(cache_only)]
        reg = ToolRegistry(config_manager=MagicMock(load_config=lambda *a, **k: cfg))
        reg._knowledge_manager = self.km
        return reg

    @staticmethod
    def _tools(reg):
        return {td.name: (td, handler) for td, handler in reg._generated_search_tools()}

    def test_descriptions_say_cache_only(self):
        on = self._tools(self._registry(True))
        off = self._tools(self._registry(False))
        for name in ("search_catalog", "search_catalog_titles", "search_finc"):
            self.assertIn(TOOL_DESCRIPTION_NOTE, on[name][0].description, name)
            self.assertNotIn(TOOL_DESCRIPTION_NOTE, off[name][0].description, name)

    def test_handlers_answer_from_cache_and_name_missing_terms(self):
        tools = self._tools(self._registry(True))
        out = json.loads(tools["search_catalog"][1](terms=["Photosynthese", "Fehlt"]))
        self.assertIn("Pflanzenphysiologie", out["results"]["Photosynthese"])
        self.assertEqual(out["cache_only"]["missing_terms"], ["Fehlt"])

        out = json.loads(tools["search_catalog_titles"][1](terms=["Faust", "Fehlt"]))
        self.assertEqual([r["title"] for r in out["results"]["Faust"]], ["Faust I", "Faust II"])
        self.assertEqual(out["cache_only"]["missing_terms"], ["Fehlt"])

        out = json.loads(tools["search_finc"][1](terms=["Graphen", "Fehlt"]))
        self.assertEqual(len(out["results"]["Graphen"]["records"]), 2)
        self.assertEqual(out["cache_only"]["missing_terms"], ["Fehlt"])
        self.assertEqual(self.net.calls, [])

    def test_off_mode_handler_uses_the_suggester_without_block(self):
        from src.core.search.factory import build_provider

        reg = self._registry(False)
        provider = build_provider(_catalog_instance(False), ukm=self.km)
        fake = _FakeSuggester()
        provider._suggester = fake
        reg._provider_cache[("catalog", False)] = provider
        out = json.loads(self._tools(reg)["search_catalog"][1](terms=["Photosynthese"]))
        self.assertEqual(fake.calls, [["Photosynthese"]])
        self.assertNotIn("cache_only", out)

    def test_aggregate_tool_reads_stale_raw_only_in_mode(self):
        on = json.loads(self._registry(True)._handle_aggregate_gnd_results(
            ["Photosynthese"], sources=["catalog"]))
        off = json.loads(self._registry(False)._handle_aggregate_gnd_results(
            ["Photosynthese"], sources=["catalog"]))
        self.assertEqual({e["title"] for e in on["pool"]}, {"Photosynthese", "Pflanzenphysiologie"})
        self.assertEqual(off["pool"], [])
        self.assertEqual(off["missing"], {"catalog": ["Photosynthese"]})
        self.assertEqual(self.net.calls, [])


# ---------------------------------------------------------------------------
# Reporting: one summary per run, no per-term failure lines
# ---------------------------------------------------------------------------
class BiblioClient:
    """Fake extractor. Named like the real Libero client on purpose: the DK step
    labels its source by the extractor's type name ("catalog")."""

    def __init__(self, cache_only, hits):
        self.cache_only = cache_only
        self.last_cache_misses = []
        self._hits = hits

    def extract_dk_classifications_for_keywords(self, keywords, max_results=50, force_update=False):
        kw = keywords[0]
        if kw in self._hits:
            self.last_cache_misses = []
            return [{"keyword": kw, "source": "cache",
                     "classifications": [{"dk": "581.1", "type": "DK", "count": 1,
                                          "titles": ["Pflanzen"]}]}]
        self.last_cache_misses = [kw]
        return []


class TestReporting(unittest.TestCase):
    KEYWORDS = [f"K{i} (GND-ID: 400000{i}-{i})" for i in range(5)]

    def _run_dk_step(self, cache_only):
        from src.utils.pipeline_utils import PipelineStepExecutor

        lines = []
        executor = PipelineStepExecutor(
            alima_manager=None, cache_manager=None, logger=None, config_manager=None
        )
        with patch("src.core.search.factory.resolve_dk_extractor",
                   return_value=BiblioClient(cache_only, hits={"K0"})), \
                patch("src.utils.config_manager.ConfigManager", MagicMock()):
            result = executor.execute_notation_search(
                list(self.KEYWORDS), stream_callback=lambda m, _s: lines.append(m),
                rvk_enabled=False,
            )
        return lines, result

    def test_dk_step_summarises_misses_once(self):
        lines, result = self._run_dk_step(cache_only=True)
        summary = [m for m in lines if "nicht im Cache" in m]
        self.assertEqual(len(summary), 1)
        self.assertIn("catalog: 4 von 5", summary[0])
        self.assertEqual([m for m in lines if "fehlgeschlagen" in m], [])
        self.assertEqual(len(result["statistics"]["cache_only_misses"]), 4)
        self.assertEqual([m for m in lines if "Web-Fallback" in m], [])

    def test_dk_step_off_mode_reports_per_keyword_failures(self):
        lines, result = self._run_dk_step(cache_only=False)
        self.assertEqual(len([m for m in lines if "Suche fehlgeschlagen" in m]), 4)
        self.assertEqual([m for m in lines if "nicht im Cache" in m], [])
        self.assertNotIn("cache_only_misses", result["statistics"])

    def _run_gnd_step(self, message):
        from src.utils.pipeline_utils import PipelineStepExecutor

        terms = ["a", "b", "c"]
        lines = []

        def fake_search(search_terms, instances, **kw):
            errors = {f"catalog:{t}": message for t in search_terms if t != "c"}
            return {t: {} for t in search_terms}, errors

        executor = PipelineStepExecutor(
            alima_manager=None, cache_manager=MagicMock(), logger=None, config_manager=None
        )
        with patch("src.utils.pipeline_utils.enabled_gnd_provider_ids", return_value=None), \
                patch("src.core.search.service.resolve_gnd_instances", return_value=[]), \
                patch("src.core.search.service.search_gnd_keywords", fake_search):
            executor.execute_gnd_search(
                terms, suggesters=["lobid"], aggregate_from_raw=False,
                stream_callback=lambda m, _s: lines.append(m),
            )
        return lines

    def test_gnd_step_summarises_misses_once(self):
        lines = self._run_gnd_step(CACHE_MISS_MESSAGE)
        summary = [m for m in lines if "nicht im Cache" in m]
        self.assertEqual(len(summary), 1)
        self.assertIn("catalog: 2 von 3", summary[0])
        self.assertEqual([m for m in lines if "fehlgeschlagen" in m or "Quellen-Fehler" in m], [])

    def test_gnd_step_off_mode_reports_real_failures_per_keyword(self):
        lines = self._run_gnd_step("HTTP 500")
        self.assertEqual(len([m for m in lines if "Quelle(n) fehlgeschlagen" in m]), 2)
        self.assertEqual(len([m for m in lines if "Quellen-Fehler" in m]), 1)
        self.assertEqual([m for m in lines if "nicht im Cache" in m], [])

    def test_search_tab_status_counts_misses_instead_of_listing_failures(self):
        from src.ui.find_keywords import SearchTab

        tab = MagicMock()
        tab._rows_from_results.return_value = ([], 0)
        SearchTab._on_search_finished(
            tab, {}, {"catalog:a": CACHE_MISS_MESSAGE, "catalog:b": CACHE_MISS_MESSAGE}
        )
        msg, tone = tab._set_status.call_args[0]
        self.assertIn("catalog: 2 Begriffe nicht im Cache", msg)
        self.assertNotIn("fehlgeschlagen", msg)
        self.assertEqual(tone, "success")


if __name__ == "__main__":
    unittest.main()
