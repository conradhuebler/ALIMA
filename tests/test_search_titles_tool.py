"""Claude Generated - the agent tool ``search_titles`` (one title search, fallback per term).

Pinned here (Nov 9, 2026): the sources are asked in the order of the plugin
list; the first one with a record answers; ``empty`` / ``down`` /
``cache_miss`` / ``unsupported`` (axis or not available) / ``error`` go on to
the next; each axis is mapped per source (``title_search_axes``); records carry
``source``; ``total`` / ``truncated`` / ``notice`` of the answering source are
passed through; the tool exists only with an active title source; and a
source in the state "Nur Cache" answers from the raw cache without network.

Fakes stand in for the providers, except in the last class, which builds the
real registry and the real KVK/Libero providers against a temporary knowledge
DB with a network tripwire. Records invented.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.search import source_health, source_usage
from src.core.search.cache_only import CACHE_MISS_MESSAGE
from src.core.search.provider import ProviderResult, ResultItem, SearchCapability
from src.mcp._title_search import build_title_search_tool, search_titles
from src.utils.config_models import AlimaConfig, PluginInstanceConfig


def _inst(pid, *, iid=None, enabled=True, label="", **settings):
    return PluginInstanceConfig(instance_id=iid or pid, category="search_provider",
                                provider_id=pid, label=label, enabled=enabled,
                                settings=dict(settings))


class FakeTitles:
    """A title provider with scripted answers per term."""

    def __init__(self, sid, records=None, *, errors=None, meta=None, available=True,
                 cache_only=False, raises=False, notice=None, finc_base=False):
        self.source_id = sid
        self.cache_only = cache_only
        self.records = records or {}
        self.errors = errors or {}
        self.meta = meta or {}
        self.available = available
        self.raises = raises
        self.notice = notice
        self.calls = []
        if finc_base:
            self.catalog_web_record_base = lambda: "https://opac.example/Record"

    def is_available(self, cfg=None):
        return self.available

    def search(self, capability, query, *, search_type=None, max_results=None, limit=None, **_):
        assert capability is SearchCapability.TITLE_RECORDS
        self.calls.append((list(query), search_type, max_results, limit))
        if self.raises:
            raise RuntimeError("Verbindung abgebrochen")
        per_term = {t: [ResultItem(label=r.get("title", ""), record=r)
                        for r in self.records.get(t, [])] for t in query}
        return ProviderResult(SearchCapability.TITLE_RECORDS, per_term=per_term,
                              per_term_meta={t: self.meta[t] for t in query if t in self.meta},
                              errors={t: self.errors[t] for t in query if t in self.errors})

    def tool_notice(self, result, *, tools=()):
        return self.notice or ""


class StubRegistry:
    """What search_titles reads from the ToolRegistry."""

    def __init__(self, instances, providers):
        self.instances = instances
        self.providers = providers

    def _search_instances(self):
        return list(self.instances)

    def _provider_for(self, inst, cache=False):
        return self.providers[inst.instance_id]

    def _finc_record_base(self, provider):
        return provider.catalog_web_record_base()

    def _known_tool_names(self):
        return {"search_k10plus"}


def _setup(**providers):
    order = list(providers)
    instances = [_inst(pid, label=pid.upper()) for pid in order]
    return StubRegistry(instances, providers)


def _rec(title, **extra):
    return {"title": title, **extra}


class TestOrderAndFallback(unittest.TestCase):
    def test_first_source_with_records_answers_later_ones_are_not_asked(self):
        finc = FakeTitles("finc", {"A": [_rec("A1", id="0-1", web_url="https://w/1")]})
        catalog = FakeTitles("catalog", {"A": [_rec("A2")]})
        out = search_titles(_setup(finc=finc, catalog=catalog), ["A"])
        term = out["results"]["A"]
        self.assertEqual(term["answered_by"], "finc")
        self.assertEqual(term["records"], [{"title": "A1", "id": "0-1",
                                            "web_url": "https://w/1", "source": "finc"}])
        self.assertEqual(term["tried"], [{"source": "finc", "status": "ok", "records": 1}])
        self.assertEqual(catalog.calls, [])
        self.assertEqual(out["sources"], ["finc", "catalog"])

    def test_list_order_decides(self):
        finc = FakeTitles("finc", {"A": [_rec("A1")]})
        catalog = FakeTitles("catalog", {"A": [_rec("A2")]})
        reg = _setup(catalog=catalog, finc=finc)
        self.assertEqual(search_titles(reg, ["A"])["results"]["A"]["answered_by"], "catalog")

    def test_empty_down_cache_miss_and_error_go_on(self):
        down = source_health.down_message("catalog")
        reg = _setup(
            finc=FakeTitles("finc"),
            catalog=FakeTitles("catalog", errors={"A": down}),
            sru=FakeTitles("sru", errors={"A": CACHE_MISS_MESSAGE}, cache_only=True),
            kvk=FakeTitles("kvk", raises=True),
            other=FakeTitles("other", errors={"A": "HTTP 500"}),
            last=FakeTitles("last", {"A": [_rec("A9")]}),
        )
        reg.instances[-2] = _inst("kvk", iid="other")
        reg.instances[-1] = _inst("kvk", iid="last")
        term = search_titles(reg, ["A"])["results"]["A"]
        self.assertEqual([(t["source"], t["status"]) for t in term["tried"]], [
            ("finc", "empty"), ("catalog", "down"), ("sru", "cache_miss"),
            ("kvk", "error"), ("other", "error"), ("last", "ok")])
        self.assertEqual(term["answered_by"], "last")
        self.assertEqual(term["tried"][4]["error"], "HTTP 500")
        self.assertIn("Verbindung abgebrochen", term["tried"][3]["error"])

    def test_nobody_has_it(self):
        term = search_titles(_setup(finc=FakeTitles("finc"), kvk=FakeTitles("kvk")),
                             ["X"])["results"]["X"]
        self.assertIsNone(term["answered_by"])
        self.assertEqual(term["records"], [])
        self.assertEqual([t["status"] for t in term["tried"]], ["empty", "empty"])

    def test_terms_are_independent(self):
        finc = FakeTitles("finc", {"A": [_rec("A1")]})
        kvk = FakeTitles("kvk", {"A": [_rec("A2")], "B": [_rec("B2")]})
        out = search_titles(_setup(finc=finc, kvk=kvk), ["A", "B"])["results"]
        self.assertEqual(out["A"]["answered_by"], "finc")
        self.assertEqual(out["B"]["answered_by"], "kvk")


class TestAxes(unittest.TestCase):
    def test_each_source_gets_its_own_search_type(self):
        catalog = FakeTitles("catalog")
        sru = FakeTitles("sru")
        kvk = FakeTitles("kvk")
        finc = FakeTitles("finc")
        reg = _setup(finc=finc, catalog=catalog, sru=sru, kvk=kvk)
        search_titles(reg, ["Hübler, Conrad"], search_type="author")
        self.assertEqual([c[1] for c in (finc.calls + catalog.calls + sru.calls + kvk.calls)],
                         ["author", "kb", "author", "author"])
        for p in (finc, catalog, sru, kvk):
            p.calls.clear()
        search_titles(reg, ["Gestein"], search_type="subject")
        self.assertEqual(catalog.calls[0][1], "sk")

    def test_unsupported_axis_is_skipped_and_noted(self):
        finc = FakeTitles("finc", {"978-3": [_rec("X")]})
        catalog = FakeTitles("catalog", {"978-3": [_rec("Y")]})
        term = search_titles(_setup(finc=finc, catalog=catalog), ["978-3"],
                             search_type="isbn")["results"]["978-3"]
        self.assertEqual(term["tried"][0],
                         {"source": "finc", "status": "unsupported",
                          "reason": "Suchart isbn nicht unterstützt"})
        self.assertEqual(finc.calls, [])
        self.assertEqual(catalog.calls[0][1], "i")
        self.assertEqual(term["answered_by"], "catalog")

    def test_not_available_is_skipped_and_noted(self):
        finc = FakeTitles("finc", {"A": [_rec("X")]}, available=False)
        kvk = FakeTitles("kvk", {"A": [_rec("Y")]})
        term = search_titles(_setup(finc=finc, kvk=kvk), ["A"])["results"]["A"]
        self.assertEqual(term["tried"][0], {"source": "finc", "status": "unsupported",
                                            "reason": "nicht verfügbar"})
        self.assertEqual(finc.calls, [])

    def test_unknown_axis_and_limit(self):
        reg = _setup(kvk=FakeTitles("kvk"))
        self.assertIn("error", search_titles(reg, ["A"], search_type="dk"))
        search_titles(reg, ["A"], max_results=500)
        self.assertEqual(reg.providers["kvk"].calls[-1][2:], (50, 50))
        search_titles(reg, ["A"])
        self.assertEqual(reg.providers["kvk"].calls[-1][2], 10)


class TestPassThrough(unittest.TestCase):
    def test_total_truncated_and_notice_of_the_answering_source(self):
        sru = FakeTitles("sru", {"A": [_rec("A1"), _rec("A2"), _rec("A3")]},
                         meta={"A": {"total": 120, "returned": 3, "truncated": True}})
        term = search_titles(_setup(sru=sru), ["A"], max_results=2)["results"]["A"]
        self.assertEqual(term["total"], 120)
        self.assertIs(term["truncated"], True)
        self.assertEqual(len(term["records"]), 2)
        kvk = FakeTitles("kvk", {"A": [_rec("A1")]}, notice="K10plus: 1 von 30 geliefert.")
        term = search_titles(_setup(kvk=kvk), ["A"])["results"]["A"]
        self.assertEqual(term["notice"], "K10plus: 1 von 30 geliefert.")
        self.assertNotIn("truncated", term)
        self.assertNotIn("total", term)

    def test_finc_records_get_the_catalog_link(self):
        finc = FakeTitles("finc", {"A": [_rec("A1", id="0-7")]}, finc_base=True)
        term = search_titles(_setup(finc=finc), ["A"])["results"]["A"]
        self.assertEqual(term["records"][0]["web_url"], "https://opac.example/Record/0-7")

    def test_answers_are_counted_in_the_source_overview(self):
        source_usage.reset()
        reg = _setup(finc=FakeTitles("finc"), kvk=FakeTitles("kvk", {"A": [_rec("1"), _rec("2")]}))
        search_titles(reg, ["A"])
        rows = {r["source"]: r for r in source_usage.rows()}
        self.assertEqual((rows["finc"]["requests"], rows["finc"]["network"], rows["finc"]["records"]),
                         (1, 1, 0))
        self.assertEqual((rows["kvk"]["requests"], rows["kvk"]["records"]), (1, 2))


class _StubConfigManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_config(self, *a, **k):
        return self.cfg


_KEEP_ALIVE: list = []


class _Isolated(unittest.TestCase):
    """Temporary HOME and knowledge DB, network tripwire."""

    def setUp(self):
        from PyQt6.QtWidgets import QApplication

        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
        from src.utils.config_manager import DatabaseConfig

        _KEEP_ALIVE.append(QApplication.instance() or QApplication(["alima-tests"]))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        home = patch.dict(os.environ, {"HOME": self._tmp.name})
        home.start()
        self.addCleanup(home.stop)
        UnifiedKnowledgeManager.reset()
        self.addCleanup(UnifiedKnowledgeManager.reset)
        self.km = UnifiedKnowledgeManager(database_config=DatabaseConfig(
            db_type="sqlite", sqlite_path=str(Path(self._tmp.name) / "knowledge.db")))
        _KEEP_ALIVE.append(self.km)
        self.net = []

        def trip(*a, **k):
            self.net.append(a)
            raise AssertionError("network request attempted")

        for target in ("requests.sessions.Session.request", "requests.get",
                       "urllib.request.urlopen", "socket.socket.connect"):
            p = patch(target, trip)
            p.start()
            self.addCleanup(p.stop)
        p = patch("src.core.search.factory._global_response_cache_enabled", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def _registry(self, *instances):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = list(instances)
        reg = ToolRegistry(config_manager=_StubConfigManager(cfg))
        reg._knowledge_manager = self.km
        reg.register_all_tools()
        return reg


class TestRealRegistry(_Isolated):
    def test_tool_exists_only_with_a_title_source(self):
        reg = self._registry(_inst("lobid"), _inst("gnd_local"))
        self.assertNotIn("search_titles", reg.get_tool_names())
        self.assertIsNone(build_title_search_tool(reg))
        reg = self._registry(_inst("lobid"), _inst("kvk", label="KVK"),
                             _inst("finc", label="finc", base_url="https://finc.example"))
        self.assertIn("search_titles", reg.get_tool_names())
        td, _handler = build_title_search_tool(reg)
        self.assertIn("KVK [kvk], finc [finc]", td.description)
        self.assertIn("search_finc", reg.get_tool_names())  # the single tools stay

    def test_cache_only_answers_without_network(self):
        from src.core.search.provider import raw_cache_params_for

        record = {"title": "Testtitel der Gesteinskunde", "author": "Muster, Max",
                  "catalog": "K10plus", "ppn": "123456789"}
        self.km.store_raw_response(
            "kvk", "Gesteinskunde", raw_cache_params_for("kvk", search_type="title"),
            json.dumps({"records": [record], "catalog_errors": [], "catalog_stats": []}))
        reg = self._registry(
            _inst("finc", label="finc"),  # no base URL: not available
            _inst("catalog", label="Libero", token="T", cache_only=True),
            _inst("kvk", label="KVK", cache_only=True),
        )
        source_usage.reset()
        out = json.loads(reg.execute("search_titles", {"terms": ["Gesteinskunde"],
                                                       "search_type": "title"}))
        term = out["results"]["Gesteinskunde"]
        self.assertEqual([(t["source"], t["status"]) for t in term["tried"]],
                         [("finc", "unsupported"), ("catalog", "cache_miss"), ("kvk", "ok")])
        self.assertEqual(term["records"][0]["ppn"], "123456789")
        self.assertEqual(term["records"][0]["source"], "kvk")
        self.assertEqual(self.net, [])
        rows = {r["source"]: r for r in source_usage.rows()}
        self.assertEqual((rows["kvk"]["cache"], rows["kvk"]["network"], rows["kvk"]["records"]),
                         (1, 0, 1))
        self.assertEqual(rows["catalog"]["cache_misses"], 1)
        self.assertEqual(rows["kvk"]["label"], "KVK")  # the instance label (build_provider)
        self.assertNotIn("finc", rows)


if __name__ == "__main__":
    unittest.main()
