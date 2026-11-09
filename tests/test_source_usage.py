"""Claude Generated - the per-run source overview (``source_usage``).

Pinned here (Nov 9, 2026):

A. Counting and lines: requests, cache, network, hits/records, errors,
   skipped, cache-only misses, down; the line format; only sources with a
   request or a skip appear; reset at a run start together with
   ``source_health`` (outermost scope only).
B. Where it is counted: the provider base (GND keyword search, live, skipped,
   cache-only), the mapping cache, the DK extractors (finc, MARC, PICA,
   Libero), the title tool handler, the lookup tool handler.
C. The result: ``KeywordAnalysisState.source_usage``, both file formats
   (state file, export with ``results.source_usage``), files without the
   field load and are written back unchanged; the result card (collapsed
   above five rows), ``show-protocol``; a plain workflow run carries it into
   its state; the chat closes a turn with one notice line.

No test reaches the network (fakes, or a tripwire on requests/sockets); HOME
and the knowledge DB are temporary. Data invented.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from src.core.data_models import KeywordAnalysisState
from src.core.search import source_health, source_usage
from src.core.search.cache_only import CACHE_MISS_MESSAGE
from src.core.search.provider import SearchCapability


def _row(source):
    return {r["source"]: r for r in source_usage.rows()}.get(source)


class _Fresh(unittest.TestCase):
    def setUp(self):
        source_usage.reset()
        notice = patch.object(source_health, "_emit_notice", lambda _t: None)
        notice.start()
        self.addCleanup(notice.stop)
        self.addCleanup(source_usage.reset)


# ===========================================================================
# A. Counting and lines
# ===========================================================================
class TestCounting(_Fresh):
    def test_record_terms_by_origin(self):
        source_usage.record_terms("finc", ["a", "b", "c", "d", "e", "f"], errors={
            "c": "HTTP 500", "d": source_health.down_message("finc"),
            "e": CACHE_MISS_MESSAGE,
        }, hits={"a": {"x": 1}})
        row = _row("finc")
        self.assertEqual((row["requests"], row["cache"], row["network"], row["hits"],
                          row["errors"], row["skipped"], row["cache_misses"]),
                         (6, 0, 4, 1, 1, 1, 1))
        source_usage.record_terms("kvk", ["a", "b"], records={"a": 3}, cache_only=True,
                                  kind=source_usage.KIND_RECORDS)
        row = _row("kvk")
        self.assertEqual((row["requests"], row["cache"], row["network"], row["records"]),
                         (2, 2, 0, 3))

    def test_institution_not_set_is_an_error_without_request(self):
        from src.utils.institution import NOT_SET_MESSAGE

        source_usage.record_terms("finc", ["a"], errors={"a": NOT_SET_MESSAGE})
        row = _row("finc")
        self.assertEqual((row["requests"], row["network"], row["errors"]), (1, 0, 1))

    def test_line_format(self):
        for _ in range(31):
            source_usage.record("finc", source_usage.CACHE)
        for i in range(7):
            source_usage.record("finc", source_usage.NETWORK, hit=i < 3)
        source_usage.set_label("finc", "finc")
        self.assertEqual(_row("finc")["text"],
                         "finc · 38 Anfragen · 31 Cache · 7 Netz · 3 Treffer-Schlagwörter · 0 Fehler")
        source_usage.set_label("catalog", "Libero")
        source_usage.mark_down("catalog")
        self.assertEqual(_row("catalog")["text"], "Libero · übersprungen nach 3 Verbindungsfehlern")
        source_usage.record("catalog", source_usage.NETWORK, error=True)
        for _ in range(13):
            source_usage.record("catalog", source_usage.SKIPPED)
        self.assertEqual(_row("catalog")["text"],
                         "Libero · 14 Anfragen · 0 Cache · 1 Netz · 0 Treffer-Schlagwörter · "
                         "1 Fehler · 13 übersprungen nach 3 Verbindungsfehlern")
        source_usage.set_label("kvk", "kvk")  # labels outlive a run; other tests set them too
        source_usage.set_label("rvk", "rvk")
        source_usage.record("kvk", source_usage.NETWORK, kind=source_usage.KIND_RECORDS, records=12)
        self.assertEqual(_row("kvk")["text"],
                         "kvk · 1 Anfragen · 0 Cache · 1 Netz · 12 Datensätze · 0 Fehler")
        source_usage.record("rvk", source_usage.CALL, kind=source_usage.KIND_LOOKUP)
        self.assertEqual(_row("rvk")["text"], "rvk · 1 Anfragen · 0 Cache · 0 Fehler")
        source_usage.record("sru", source_usage.CACHE_MISS)
        self.assertIn("1 nicht im Cache", _row("sru")["text"])

    def test_only_used_sources_appear_in_first_use_order(self):
        source_usage.set_label("lobid", "Lobid")
        source_usage.record("swb", source_usage.NETWORK)
        source_usage.record("finc", source_usage.CACHE)
        self.assertEqual([r["source"] for r in source_usage.rows()], ["swb", "finc"])
        source_usage.record(None, source_usage.NETWORK)
        self.assertEqual(len(source_usage.rows()), 2)

    def test_down_comes_from_source_health(self):
        for _ in range(source_health.FAILURE_THRESHOLD):
            source_health.record_failure("catalog", requests.exceptions.ConnectionError("x"))
        self.assertTrue(_row("catalog")["down"])

    def test_reset_with_the_run(self):
        source_usage.record("finc", source_usage.NETWORK)
        with source_health.run_scope():
            self.assertIsNone(_row("finc"))
            source_usage.record("finc", source_usage.NETWORK)
            with source_health.run_scope():  # nested: keeps the counts
                source_usage.record("finc", source_usage.NETWORK)
            self.assertEqual(_row("finc")["requests"], 2)
        source_health.begin_run()
        self.assertEqual(source_usage.rows(), [])

    def test_summary_line(self):
        source_usage.set_label("a", "a")
        self.assertEqual(source_usage.summary_line(), "")
        source_usage.record("a", source_usage.NETWORK)
        source_usage.record("b", source_usage.CACHE)
        self.assertEqual(source_usage.summary_line().count(" | "), 1)
        self.assertTrue(source_usage.summary_line().startswith("Quellen dieses Laufs: a · 1"))


# ===========================================================================
# B. Where it is counted
# ===========================================================================
class _Sugg:
    def __init__(self, hits):
        self.hits = hits
        self.last_errors = {}
        self.calls = 0

    def search(self, terms, **kw):
        self.calls += 1
        self.last_errors = {t: "HTTP 503" for t in terms if t == "kaputt"}
        return {t: ({"Treffer": {"count": 1, "gnd_ids": {"1"}}} if t in self.hits else {})
                for t in terms}

    @staticmethod
    def transform(raw):
        return {"Treffer": {"count": 1, "gnd_ids": {"1"}}}


def _gnd_provider(sugg, **settings):
    from src.core.search.cache_only import cache_only_field
    from src.core.search.provider_base import SuggesterBackedProvider

    class P(SuggesterBackedProvider):
        id = "fakegnd"
        label = "Fake"
        capabilities = {SearchCapability.GND_KEYWORDS}

        @classmethod
        def config_fields(cls):
            return [cache_only_field()]

        def _build_suggester(self):
            return sugg

        def _store_raw_responses(self, *a, **k):
            pass

    p = P(**settings)
    p._source_id = "gnd-a"
    return p


class TestProviderBase(_Fresh):
    def test_live_search(self):
        p = _gnd_provider(_Sugg({"a"}))
        p._gnd_search(["a", "b", "kaputt"], None)
        row = _row("gnd-a")
        self.assertEqual((row["requests"], row["network"], row["hits"], row["errors"]), (3, 3, 1, 1))

    def test_down_source_is_skipped(self):
        sugg = _Sugg({"a"})
        p = _gnd_provider(sugg)
        for _ in range(source_health.FAILURE_THRESHOLD):
            source_health.record_failure("gnd-a")
        p._gnd_search(["a", "b"], None)
        self.assertEqual(sugg.calls, 0)
        self.assertEqual((_row("gnd-a")["skipped"], _row("gnd-a")["network"]), (2, 0))

    def test_cache_only(self):
        p = _gnd_provider(_Sugg(set()), cache_only=True)
        with patch.object(type(p), "_cached_raw",
                          lambda self, s, t, params: {"raw": 1} if t == "a" else None):
            p._gnd_search(["a", "b"], None)
        row = _row("gnd-a")
        self.assertEqual((row["cache"], row["hits"], row["cache_misses"], row["network"]),
                         (1, 1, 1, 0))

    def test_mapping_cache_hit(self):
        from src.core.search.caching import CachingProvider

        inner = MagicMock()
        inner.source_id = "lobid"
        inner.id = "lobid"
        mapping = SimpleNamespace(last_updated=datetime.now(), found_gnd_ids=["1"],
                                  titles={"1": "Gestein"}, gnd_counts={"1": 4})
        ukm = MagicMock()
        ukm.get_search_mapping.return_value = mapping
        CachingProvider(inner, ukm=ukm).search(SearchCapability.GND_KEYWORDS, ["Gestein"])
        inner.search.assert_not_called()
        self.assertEqual((_row("lobid")["cache"], _row("lobid")["hits"]), (1, 1))


class _FakeKm:
    def __init__(self, cache=None):
        self.cache = cache or {}

    def get_catalog_dk_cache(self, kw, ignore_ttl=False):
        return self.cache.get(kw)

    def store_catalog_dk_cache(self, *a, **k):
        pass

    def extract_classifications_from_titles(self, titles, matched_keywords=None):
        return [{"dk": "552", "type": "DK", "count": len(titles)}] if titles else []


class TestDkExtractors(_Fresh):
    TITLES = [{"rsn": "1", "title": "T", "classifications": ["DK 552"]}]

    def test_finc(self):
        from src.core.search.providers.finc.finc_catalog_client import FincCatalogClient

        client = FincCatalogClient(base_url="https://finc.example", source_id="finc",
                                   knowledge_manager=_FakeKm({"k1": (self.TITLES, "success", "")}))
        client._client.search = lambda **kw: {"status": "OK", "records": [{"id": "1", "title": "T"}]}
        client._classify_titles = lambda records: list(self.TITLES)
        client.extract_dk_classifications_for_keywords(["k1", "k2"])
        row = _row("finc")
        self.assertEqual((row["requests"], row["cache"], row["network"], row["hits"]), (2, 1, 1, 2))
        for _ in range(source_health.FAILURE_THRESHOLD):
            source_health.record_failure("finc")
        client.extract_dk_classifications_for_keywords(["k3"])
        self.assertEqual(_row("finc")["skipped"], 1)

    def test_finc_cache_only_and_no_institution(self):
        from src.core.search.providers.finc.finc_catalog_client import FincCatalogClient
        from src.utils import institution

        client = FincCatalogClient(base_url="https://finc.example", source_id="finc",
                                   cache_only=True,
                                   knowledge_manager=_FakeKm({"k1": (self.TITLES, "success", "")}))
        client.extract_dk_classifications_for_keywords(["k1", "k2"])
        row = _row("finc")
        self.assertEqual((row["cache"], row["cache_misses"], row["network"]), (1, 1, 0))
        live = FincCatalogClient(base_url="https://finc.example", source_id="finc-b",
                                 knowledge_manager=_FakeKm())
        institution.set_current("", "")
        live.extract_dk_classifications_for_keywords(["k1"])
        self.assertEqual((_row("finc-b")["errors"], _row("finc-b")["network"]), (1, 0))

    def test_marc(self):
        from src.utils.clients.marcxml_client import MarcXmlClient

        client = MarcXmlClient(preset="dnb", source_id="sru", cache_only=True)
        client.extract_dk_classifications_for_keywords(["k1"])
        self.assertEqual(_row("sru")["cache_misses"], 1)
        live = MarcXmlClient(preset="dnb", source_id="sru-b")
        live.search = lambda term, search_type="keyword": [
            {"title": "T", "decimal_classifications": ["552"], "rvk_classifications": []}]
        live.extract_dk_classifications_for_keywords(["k1"])
        self.assertEqual((_row("sru-b")["network"], _row("sru-b")["hits"]), (1, 1))

    def test_pica(self):
        from src.utils.clients.picaxml_client import PicaDkExtractor

        client = MagicMock()
        client.source_id = "k10"
        client.timeout = 5
        client.search.side_effect = requests.exceptions.ReadTimeout("langsam")
        ext = PicaDkExtractor(client, "105")
        ext.extract_dk_classifications_for_keywords(["k1"])
        self.assertEqual((_row("k10")["network"], _row("k10")["errors"]), (1, 1))
        cached = PicaDkExtractor(client, "105", cache_only=True,
                                 cache_read=lambda term, st: {"records": []} if term == "k1" else None)
        cached.extract_dk_classifications_for_keywords(["k1", "k2"])
        self.assertEqual((_row("k10")["cache"], _row("k10")["cache_misses"]), (1, 1))

    def test_libero(self):
        from src.utils.clients.biblio_client import BiblioClient

        km = _FakeKm({"k1": (self.TITLES, "success", ""), "k2": ([], "no_results", "")})
        with patch("src.core.unified_knowledge_manager.UnifiedKnowledgeManager",
                   return_value=km):
            BiblioClient(cache_only=True, source_id="catalog") \
                .extract_dk_classifications_for_keywords(["k1", "k2", "k3"])
            row = _row("catalog")
            self.assertEqual((row["cache"], row["hits"], row["cache_misses"], row["network"]),
                             (2, 1, 1, 0))
            source_usage.reset()
            client = BiblioClient(source_id="catalog", rate_limit_delay_ms=0)
            client.search = lambda term, search_type="ku": []
            with patch("src.utils.classification_lookup_service.get_classification_lookup_service",
                       return_value=MagicMock(get_titles_for_classification=lambda kw: None)):
                client.extract_dk_classifications_for_keywords(["k1", "k4"])
        row = _row("catalog")
        self.assertEqual((row["cache"], row["network"], row["hits"]), (1, 1, 1))


class TestToolHandlers(_Fresh):
    def test_title_handler(self):
        from src.core.search.provider import ProviderResult, ProviderToolSpec, ResultItem
        from src.mcp.tool_registry import ToolRegistry

        provider = MagicMock()
        provider.source_id = "kvk"
        provider.cache_only = False
        provider.is_available.return_value = True
        provider.search.return_value = ProviderResult(
            SearchCapability.TITLE_RECORDS,
            per_term={"A": [ResultItem(label="x", record={"title": "x"})], "B": []},
            errors={"B": "HTTP 500"})
        del provider.tool_notice
        del provider.inner
        reg = ToolRegistry(config_manager=MagicMock())
        reg._provider_for = lambda inst, cache=False: provider
        spec = ProviderToolSpec(name="search_kvk", capability=SearchCapability.TITLE_RECORDS,
                                description="", parameters={}, result_shape="title_records")
        with patch("src.core.search.service.underlying_suggester", return_value=None):
            reg._make_title_records_handler(spec, SimpleNamespace(instance_id="kvk"))(terms=["A", "B"])
        row = _row("kvk")
        self.assertEqual((row["requests"], row["network"], row["records"], row["errors"]),
                         (2, 2, 1, 1))

    def test_lookup_handler(self):
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.lookups import LookupToolSpec

        reg = ToolRegistry(config_manager=MagicMock())
        reg._lookup_for = lambda inst: SimpleNamespace(find=lambda term: {"hits": [term]})
        spec = LookupToolSpec(name="rvk_search", description="", method="find",
                              parameters={"properties": {"term": {}}}, cacheable=False)
        inst = SimpleNamespace(instance_id="rvk_api", label="RVK-API", settings={})
        reg._make_lookup_handler(spec, inst)(term="Geologie")
        self.assertEqual(_row("rvk_api")["text"], "RVK-API · 1 Anfragen · 0 Cache · 0 Fehler")


# ===========================================================================
# C. The result
# ===========================================================================
ROWS = [
    {"source": "finc", "label": "finc", "requests": 38, "cache": 31, "network": 7, "hits": 3,
     "records": 0, "errors": 0, "skipped": 0, "cache_misses": 0, "down": False,
     "kinds": ["keywords"],
     "text": "finc · 38 Anfragen · 31 Cache · 7 Netz · 3 Treffer-Schlagwörter · 0 Fehler"},
    {"source": "catalog", "label": "Libero", "requests": 14, "cache": 0, "network": 1, "hits": 0,
     "records": 0, "errors": 1, "skipped": 13, "cache_misses": 0, "down": True,
     "kinds": ["keywords"],
     "text": "Libero · 14 Anfragen · 0 Cache · 1 Netz · 0 Treffer-Schlagwörter · 1 Fehler · "
             "13 übersprungen nach 3 Verbindungsfehlern"},
]


def _state(rows=None):
    state = KeywordAnalysisState(original_abstract="Ein Text über Gesteine.",
                                 initial_keywords=["Gestein"], search_suggesters_used=["lobid"],
                                 initial_gnd_classes=[], search_results=[],
                                 initial_llm_call_details=None, final_llm_analysis=None)
    state.dk_classifications = ["DK 552"]
    if rows is not None:
        state.source_usage = [dict(r) for r in rows]
    return state


class TestResultFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "state.json")

    def _load(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        return PipelineJsonManager.load_analysis_state(self.path)

    def test_default_is_empty(self):
        self.assertEqual(KeywordAnalysisState(original_abstract="", initial_keywords=[],
                                              search_suggesters_used=[], initial_gnd_classes=[],
                                              search_results=[], initial_llm_call_details=None,
                                              final_llm_analysis=None).source_usage, [])

    def test_state_file_roundtrip(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        PipelineJsonManager.save_analysis_state(_state(ROWS), self.path)
        self.assertEqual(self._load().source_usage, ROWS)

    def test_state_file_without_the_field_is_written_back_unchanged(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        data = dataclasses.asdict(_state())
        data.pop("source_usage")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=4)
        before = Path(self.path).read_text(encoding="utf-8")
        state = self._load()
        self.assertEqual(state.source_usage, [])
        PipelineJsonManager.save_analysis_state(state, self.path)
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), before)

    def test_export_file(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(_state(ROWS), self.path, validate_rvk=False)
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.assertEqual(data["results"]["source_usage"], ROWS)
        self.assertEqual(self._load().source_usage, ROWS)

    def test_export_without_rows_has_no_field(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(_state(), self.path, validate_rvk=False)
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.assertNotIn("source_usage", data["results"])
        self.assertEqual(self._load().source_usage, [])

    def test_cli_protocol(self):
        from src.cli.formatters.protocol_formatters import display_protocol
        from src.utils.pipeline_persistence import PipelineJsonManager

        PipelineJsonManager.save_analysis_state(_state(ROWS), self.path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            display_protocol(self.path, ["dk_classification"])
        out = buffer.getvalue().splitlines()
        start = out.index("Quellen dieses Laufs:")
        self.assertEqual(out[start + 1:start + 3], ["  " + r["text"] for r in ROWS])
        PipelineJsonManager.save_analysis_state(_state(), self.path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            display_protocol(self.path, ["dk_classification"])
        self.assertNotIn("Quellen dieses Laufs", buffer.getvalue())


class TestCard(unittest.TestCase):
    def card(self, rows):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        return PipelineResultFormatter.format_final_result_card_html(_state(rows))

    def test_block_with_up_to_five_rows(self):
        html, plain = self.card(ROWS)
        self.assertIn('<ul class="alima-result__sources">', html)
        self.assertNotIn("alima-result__sources-more", html)
        self.assertIn("<li>finc · 38 Anfragen · 31 Cache · 7 Netz", html)
        lines = plain.splitlines()
        start = lines.index("Quellen dieses Laufs:")
        self.assertEqual(lines[start + 1:start + 3], ["  " + r["text"] for r in ROWS])

    def test_collapsed_above_five_rows(self):
        rows = [dict(ROWS[0], source=f"s{i}", text=f"s{i} · 1 Anfragen") for i in range(6)]
        html, _plain = self.card(rows)
        start = html.index('<details class="alima-result__more alima-result__sources-more"')
        self.assertNotIn("open", html[start:html.index(">", start)])
        self.assertIn("<summary>Quellen dieses Laufs (6)</summary>", html)

    def test_no_block_without_rows(self):
        html, plain = self.card(None)
        self.assertNotIn("Quellen dieses Laufs", html + plain)


class TestRunsCarryIt(_Fresh):
    def test_workflow_run_puts_it_into_the_state(self):
        from src.core.agents.registry import TOOL_FN_REGISTRY, register_tool_fn
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import load_workflow

        snapshot = dict(TOOL_FN_REGISTRY)
        self.addCleanup(lambda: (TOOL_FN_REGISTRY.clear(), TOOL_FN_REGISTRY.update(snapshot)))

        @register_tool_fn("ask_a_source")
        def _ask(text):
            source_usage.set_label("finc", "finc")
            source_usage.record("finc", source_usage.NETWORK, hit=True)
            return {"ok": True}

        source_usage.record("stale", source_usage.NETWORK)  # before the run: reset
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.yaml"
            path.write_text(textwrap.dedent("""
                name: "Quellen"
                version: "4.0"
                steps:
                  - id: ask
                    type: deterministic
                    function: "ask_a_source"
                    inputs:
                      text: "${abstract}"
            """), encoding="utf-8")
            wf = load_workflow(path)
        ctx = SharedContext(abstract="x")
        WorkflowExecutor().run(wf, ctx)
        state = ctx.to_keyword_analysis_state()
        self.assertEqual([r["source"] for r in state.source_usage], ["finc"])
        self.assertEqual(state.source_usage[0]["text"],
                         "finc · 1 Anfragen · 0 Cache · 1 Netz · 1 Treffer-Schlagwörter · 0 Fehler")

    def test_pipeline_manager_stores_it(self):
        from src.core.pipeline_manager import PipelineManager

        source_usage.record("finc", source_usage.CACHE)
        holder = SimpleNamespace(current_analysis_state=_state(), logger=logging.getLogger("t"))
        PipelineManager._finish_source_usage(holder)
        self.assertEqual(holder.current_analysis_state.source_usage[0]["source"], "finc")

    def test_chat_turn_announces_one_line(self):
        events = []
        bus = MagicMock()
        bus.return_value.emit_event = lambda name, payload: events.append((name, payload))
        with patch("src.core.state_bus.AlimaStateBus", bus):
            self.assertEqual(source_usage.announce(), "")
            source_usage.record("finc", source_usage.NETWORK)
            source_usage.record("kvk", source_usage.CACHE)
            line = source_usage.announce()
        self.assertEqual(events, [("state.notice", {"text": line, "level": "info"})])
        self.assertTrue(line.startswith("Quellen dieses Laufs: finc"))

    def test_headless_chat_turn_is_one_run(self):
        from src.core import headless_agent

        seen = {}

        class Loop:
            def __init__(self, **kw):
                pass

            def run(self, **kw):
                seen["before"] = source_usage.rows()
                source_usage.record("finc", source_usage.NETWORK)
                return "ok"

        source_usage.record("stale", source_usage.NETWORK)  # before the turn: reset
        runner = headless_agent.HeadlessAgentRunner(
            llm_service=MagicMock(), mcp_registry=MagicMock(get_tool_names=lambda: []),
            kb_manager=MagicMock(), mode="general",
            chat_config=SimpleNamespace(temperature=0.5, system_prompt_tier="auto",
                                        institution_context=""))
        announced = []
        with patch.object(headless_agent, "AgentLoop", Loop), \
                patch("src.ui.chat_tools.build_chat_toolset", return_value=MagicMock()), \
                patch.object(source_usage, "announce",
                             lambda: announced.append(source_usage.rows())):
            self.assertEqual(runner.run("Hallo", provider="p", model="m"), "ok")
        self.assertEqual(seen["before"], [])
        self.assertEqual([r["source"] for r in announced[0]], ["finc"])


if __name__ == "__main__":
    unittest.main()
