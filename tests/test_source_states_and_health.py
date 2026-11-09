"""Claude Generated - instance state, per-instance timeout, run-wide source skip, rules in the tab.

Four things are pinned here (Nov 6, 2026):

A. The plugin tab shows a state per instance (An / Nur Cache / Aus) that is
   stored exactly as before (``enabled`` + ``settings["cache_only"]``); an
   untouched config is saved unchanged, by the tab and by the config manager.
B. ``timeout_s`` per network source: the default is the limit each client had,
   and a set value reaches the request.
C. ``source_health``: after three connection failures in a row a source is
   skipped for the run, without network, with one message, never cached as
   "no match"; HTTP errors do not count, a success resets, a run start and a
   300 s expiry forget it.
D. The tab states the rules: "Wirkung" per type and the DK source of the
   current list, computed like ``resolve_dk_extractor``.

No test reaches the network (requests/sockets are mocked or tripwired) and none
reads or writes ``~/.config/alima``: HOME points at a temporary directory, the
knowledge DB is a temporary SQLite file.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.core.search import source_health
from src.core.search.provider import SearchCapability, raw_cache_params_for
from src.utils.config_models import PluginInstanceConfig


def _inst(pid, *, iid=None, enabled=True, **settings):
    return PluginInstanceConfig(
        instance_id=iid or pid, category="search_provider", provider_id=pid,
        label="", enabled=enabled, settings=dict(settings),
    )


class _StubConfig:
    def __init__(self, *instances):
        self._instances = list(instances)

    def enabled_instances_for(self, category):
        return [i for i in self._instances if i.enabled]


_KEEP_ALIVE: list = []


def _app():
    """A referenced QApplication, created before any knowledge DB.

    ``DatabaseManager`` creates a bare ``QCoreApplication`` when no Qt
    application exists; widgets built after that abort the process. - Claude Generated
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["alima-tests"])
    _KEEP_ALIVE.append(app)
    return app


class _Isolated(unittest.TestCase):
    """Temporary HOME (suggester data dirs) and a temporary knowledge DB.

    Each knowledge manager is kept referenced until the module ends: its Qt
    connection name is built from ``id(self)``, and a freed instance's id can
    be handed to the next test's manager, which then reuses a connection name
    Qt still holds (segfault on open).
    """

    def setUp(self):
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
        from src.utils.config_manager import DatabaseConfig

        _app()
        self._tmp = tempfile.TemporaryDirectory()
        home = patch.dict(os.environ, {"HOME": self._tmp.name})
        home.start()
        self.addCleanup(home.stop)
        UnifiedKnowledgeManager.reset()
        self.km = UnifiedKnowledgeManager(database_config=DatabaseConfig(
            db_type="sqlite", sqlite_path=str(Path(self._tmp.name) / "knowledge.db")))
        _KEEP_ALIVE.append(self.km)
        self.notices = []
        notice = patch.object(source_health, "_emit_notice", self.notices.append)
        notice.start()
        self.addCleanup(notice.stop)
        source_health.begin_run()

    def tearDown(self):
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

        UnifiedKnowledgeManager.reset()
        source_health.begin_run()
        self._tmp.cleanup()


class _TempHome(unittest.TestCase):
    """HOME in a temporary directory (suggester data dirs, config dir)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.dict(os.environ, {"HOME": tmp.name})
        home.start()
        self.addCleanup(home.stop)


def _conn_error():
    return requests.exceptions.ConnectionError("Failed to resolve 'catalog.invalid'")


# ===========================================================================
# C. source_health itself
# ===========================================================================
class TestSourceHealthCore(unittest.TestCase):
    def setUp(self):
        source_health.begin_run()
        self.notices = []
        p = patch.object(source_health, "_emit_notice", self.notices.append)
        p.start()
        self.addCleanup(p.stop)
        self.clock = [1000.0]
        c = patch.object(source_health, "_now", lambda: self.clock[0])
        c.start()
        self.addCleanup(c.stop)
        self.addCleanup(source_health.begin_run)

    def _fail(self, source="lobid", exc=None):
        def boom():
            raise exc or _conn_error()
        with self.assertRaises(Exception):
            source_health.call(source, boom)

    def test_constants(self):
        self.assertEqual(source_health.FAILURE_THRESHOLD, 3)
        self.assertEqual(source_health.DOWN_EXPIRY_S, 300)

    def test_three_connection_failures_mark_down_and_stop_requests(self):
        for _ in range(2):
            self._fail()
        self.assertFalse(source_health.is_down("lobid"))
        self._fail()
        self.assertTrue(source_health.is_down("lobid"))
        fn = MagicMock()
        with self.assertRaises(source_health.SourceDownError) as ctx:
            source_health.call("lobid", fn)
        fn.assert_not_called()
        self.assertTrue(source_health.is_source_down(str(ctx.exception)))
        self.assertFalse(source_health.is_down("swb"))  # per source

    def test_one_message_per_source(self):
        with self.assertLogs("src.core.search.source_health", level="WARNING") as logs:
            for _ in range(5):
                try:
                    source_health.call("lobid", lambda: (_ for _ in ()).throw(_conn_error()))
                except Exception:
                    pass
        self.assertEqual(self.notices, [source_health.announcement("lobid")])
        self.assertIn("lobid: nach 3 Verbindungsfehlern für diesen Lauf übersprungen",
                      self.notices[0])
        self.assertEqual(len([r for r in logs.records if "übersprungen" in r.getMessage()]), 1)

    def test_http_error_does_not_count_and_resets(self):
        self._fail()
        self._fail()
        response = MagicMock(status_code=500)
        self.assertIs(source_health.call("lobid", lambda: response), response)
        self._fail()
        self.assertFalse(source_health.is_down("lobid"))
        for _ in range(5):
            self._fail(exc=urllib.error.HTTPError("http://x", 503, "down", {}, None))
        self.assertFalse(source_health.is_down("lobid"))
        self.assertEqual(source_health.status(), {})

    def test_read_timeout_does_not_count(self):
        for _ in range(4):
            self._fail(exc=requests.exceptions.ReadTimeout("slow"))
        self.assertFalse(source_health.is_down("lobid"))

    def test_success_resets_the_counter(self):
        self._fail()
        self._fail()
        source_health.call("lobid", lambda: None)
        self._fail()
        self._fail()
        self.assertFalse(source_health.is_down("lobid"))

    def test_connection_error_kinds(self):
        yes = [
            _conn_error(),
            requests.exceptions.ConnectTimeout("connect timed out"),
            urllib.error.URLError(socket.gaierror(-2, "Name or service not known")),
            urllib.error.URLError(TimeoutError("timed out")),
            ConnectionRefusedError(111, "refused"),
        ]
        no = [
            requests.exceptions.ReadTimeout("read"),
            requests.exceptions.HTTPError("500"),
            urllib.error.HTTPError("http://x", 500, "err", {}, None),
            ValueError("parse"),
            source_health.SourceDownError("lobid"),
        ]
        for exc in yes:
            self.assertTrue(source_health.is_connection_error(exc), repr(exc))
        for exc in no:
            self.assertFalse(source_health.is_connection_error(exc), repr(exc))

    def test_begin_run_resets(self):
        for _ in range(3):
            self._fail()
        source_health.begin_run()
        self.assertFalse(source_health.is_down("lobid"))

    def test_down_expires_after_300_seconds(self):
        for _ in range(3):
            self._fail()
        self.clock[0] += 299
        self.assertTrue(source_health.is_down("lobid"))
        self.clock[0] += 1
        self.assertFalse(source_health.is_down("lobid"))
        self.assertEqual(source_health.call("lobid", lambda: "ok"), "ok")

    def test_old_partial_count_expires_too(self):
        self._fail()
        self._fail()
        self.clock[0] += 300
        self._fail()
        self.assertFalse(source_health.is_down("lobid"))

    def test_untracked_source_is_never_gated(self):
        for _ in range(5):
            self._fail(source=None)
        self.assertEqual(source_health.call(None, lambda: 1), 1)
        self.assertEqual(source_health.status(), {})

    def test_only_the_outermost_run_scope_resets(self):
        for _ in range(3):
            self._fail()
        with source_health.run_scope():          # outermost: resets
            self.assertFalse(source_health.is_down("lobid"))
            for _ in range(3):
                self._fail()
            with source_health.run_scope():      # nested: keeps the state
                self.assertTrue(source_health.is_down("lobid"))
        with source_health.run_scope():
            self.assertFalse(source_health.is_down("lobid"))


class TestRunStartsReset(unittest.TestCase):
    def test_start_pipeline_resets(self):
        from src.core.pipeline_manager import PipelineManager

        pm = MagicMock()
        with patch.object(source_health, "begin_run") as begin, \
                patch("src.core.state_bus.AlimaStateBus"):
            PipelineManager.start_pipeline(pm, "Text")
        begin.assert_called_once_with()

    def test_executor_and_meta_agent_runs_are_scoped(self):
        from src.core.agents.meta_agent import MetaAgent
        from src.core.agents.workflow_executor import WorkflowExecutor

        for cls in (WorkflowExecutor, MetaAgent):
            calls = []
            with patch.object(source_health, "run_scope",
                              side_effect=lambda: _Recorder(calls)):
                try:
                    cls.run(MagicMock(), MagicMock(), MagicMock())
                except Exception:
                    pass
            self.assertEqual(calls, ["enter", "exit"], cls.__name__)


class _Recorder:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append("enter")

    def __exit__(self, *exc):
        self.calls.append("exit")
        return False


# ===========================================================================
# C. through the providers and the direct DK paths
# ===========================================================================
class TestLobidSkip(_Isolated):
    """lobid (urllib): 3 failed terms, the rest without network, nothing cached."""

    def _provider(self):
        from src.core.search.factory import build_provider

        return build_provider(_inst("lobid"), cache=True, ukm=self.km, cache_raw=True)

    def test_skip_after_three_and_no_cache_write(self):
        terms = ["A", "B", "C", "D", "E"]
        opened = []

        def refuse(url, timeout=None):
            opened.append(url)
            raise urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))

        with patch("urllib.request.urlopen", refuse):
            res = self._provider().search(SearchCapability.GND_KEYWORDS, terms)
        self.assertEqual(len(opened), 3)
        self.assertEqual(set(res.errors), set(terms))
        self.assertTrue(all(source_health.is_source_down(res.errors[t]) for t in "DE"))
        self.assertFalse(any(source_health.is_source_down(res.errors[t]) for t in "ABC"))
        self.assertEqual(len(self.notices), 1)
        for term in terms:
            self.assertIsNone(self.km.get_search_mapping(term, "lobid"), term)
            self.assertIsNone(self.km.get_raw_response(
                "lobid", term, raw_cache_params_for("lobid"), max_age_hours=None))

    def test_a_down_source_never_reaches_the_suggester(self):
        for _ in range(3):
            source_health.record_failure("lobid")
        provider = self._provider()
        provider.inner._suggester = MagicMock()
        res = provider.search(SearchCapability.GND_KEYWORDS, ["X"])
        provider.inner._suggester.search.assert_not_called()
        self.assertTrue(source_health.is_source_down(res.errors["X"]))
        self.assertIsNone(self.km.get_search_mapping("X", "lobid"))


class TestSwbSkip(_Isolated):
    def test_swb_skips_without_request(self):
        from src.core.search.factory import build_provider

        provider = build_provider(_inst("swb"), cache=False, ukm=self.km, cache_raw=False)
        with patch("requests.get", side_effect=_conn_error()) as get:
            res = provider.search(SearchCapability.GND_KEYWORDS, ["a", "b", "c", "d"])
        self.assertEqual(get.call_count, 3)
        self.assertTrue(source_health.is_source_down(res.errors["d"]))


class TestCatalogDkDirectPath(_Isolated):
    """Libero via the DK extractor: each SOAP attempt counts, then skip."""

    def _client(self, session):
        from src.utils.clients.biblio_client import BiblioClient

        client = BiblioClient(
            token="t", soap_search_url="http://catalog.invalid/soap",
            soap_details_url="http://catalog.invalid/details",
            web_search_url="http://catalog.invalid/web",
            web_record_url="http://catalog.invalid/Record/",
            rate_limit_delay_ms=0, source_id="catalog",
        )
        client.session = session
        client.use_json_cache = False
        return client

    def test_skip_and_nothing_cached(self):
        session = MagicMock()
        session.post.side_effect = _conn_error()
        session.get.side_effect = _conn_error()
        client = self._client(session)
        with patch("src.utils.clients._biblio_soap.time.sleep"):
            out = client.extract_dk_classifications_for_keywords(["K1", "K2", "K3", "K4"])
        self.assertEqual(out, [])
        self.assertEqual(session.post.call_count, 3)   # three attempts for K1
        self.assertEqual(session.get.call_count, 0)    # web fallback skipped
        self.assertEqual(client.last_source_down, ["K2", "K3", "K4"])
        for kw in ("K1", "K2", "K3", "K4"):
            self.assertIsNone(self.km.get_catalog_dk_cache(kw, ignore_ttl=True), kw)
        self.assertEqual(len(self.notices), 1)

    def test_http_error_is_not_a_connection_failure(self):
        response = MagicMock(status_code=500, text="error")
        session = MagicMock()
        session.post.return_value = response
        client = self._client(session)
        client.enable_web_fallback = False
        client.extract_dk_classifications_for_keywords(["K1", "K2", "K3", "K4", "K5"])
        self.assertEqual(session.post.call_count, 5)
        self.assertFalse(source_health.is_down("catalog"))
        # A server that answered with an error is a real "no results" answer.
        self.assertEqual(self.km.get_catalog_dk_cache("K5", ignore_ttl=True)[1], "no_results")

    def test_gnd_path_reports_unknown_not_empty(self):
        from src.core.search.factory import build_provider

        provider = build_provider(
            _inst("catalog", token="t", catalog_search_url="http://catalog.invalid/soap"),
            cache=True, ukm=self.km, cache_raw=True,
        )
        provider.inner.suggester.extractor.session = MagicMock(
            post=MagicMock(side_effect=_conn_error()))
        with patch("src.utils.clients._biblio_soap.time.sleep"):
            res = provider.search(SearchCapability.GND_KEYWORDS, ["A", "B"])
        self.assertEqual(set(res.errors), {"A", "B"})
        self.assertTrue(source_health.is_source_down(res.errors["B"]))
        for term in ("A", "B"):
            self.assertIsNone(self.km.get_search_mapping(term, "catalog"))


class TestFincSkip(_Isolated):
    def test_skipped_term_keeps_its_earlier_raw_entry(self):
        from src.core.search.factory import build_provider

        params = raw_cache_params_for("finc", search_type="kw")
        good = {"records": [{"id": "1", "title": "Alt"}], "result_count": 1,
                "facets": {}, "errors": []}
        self.km.store_raw_response("finc", "D", params, json.dumps(good))
        provider = build_provider(_inst("finc", base_url="http://finc.invalid/proxy.php"),
                                  ukm=self.km, cache_raw=True)
        provider.suggester.client.session = MagicMock(get=MagicMock(side_effect=_conn_error()))
        with patch("src.core.search.providers.finc.finc_client.time.sleep"):
            res = provider.search(SearchCapability.TITLE_RECORDS, ["A", "B", "C", "D"])
        self.assertEqual(provider.suggester.client.session.get.call_count, 3)
        self.assertTrue(source_health.is_source_down(res.errors["D"]))
        kept = json.loads(self.km.get_raw_response("finc", "D", params,
                                                   max_age_hours=None)["raw_json"])
        self.assertEqual(kept, good)

    def test_dk_extractor_skips_without_caching(self):
        for _ in range(3):
            source_health.record_failure("finc")
        from src.core.search.providers.finc.provider import FincProvider

        provider = FincProvider(base_url="http://finc.invalid/proxy.php")
        provider._ukm_ref = self.km
        ext = provider.dk_extractor(knowledge_manager=self.km)
        ext._client.session = MagicMock()
        self.assertEqual(ext.extract_dk_classifications_for_keywords(["K1"]), [])
        ext._client.session.get.assert_not_called()
        self.assertEqual(ext.last_source_down, ["K1"])
        self.assertIsNone(self.km.get_catalog_dk_cache("K1", ignore_ttl=True))


class TestSruSkip(_Isolated):
    def test_dk_extractor_skips_without_request(self):
        from src.core.search.factory import build_provider

        ext = build_provider(_inst("sru", preset="dnb", dk_enabled=True),
                             cache_raw=False).dk_extractor()
        ext.session = MagicMock(get=MagicMock(side_effect=_conn_error()))
        self.assertEqual(ext.extract_dk_classifications_for_keywords(["a", "b", "c", "d"]), [])
        self.assertEqual(ext.session.get.call_count, 3)
        self.assertEqual(ext.last_source_down, ["d"])


class TestReporting(unittest.TestCase):
    """One message per source: no per-keyword failure lines for skipped terms."""

    def setUp(self):
        source_health.begin_run()
        self.addCleanup(source_health.begin_run)

    def test_dk_step(self):
        from src.utils.pipeline_utils import PipelineStepExecutor

        class Extractor:
            cache_only = False

            def __init__(self):
                self.last_source_down = []

            def extract_dk_classifications_for_keywords(self, keywords, max_results=50):
                self.last_source_down = list(keywords)
                return []

        lines = []
        executor = PipelineStepExecutor(alima_manager=None, cache_manager=None,
                                        logger=None, config_manager=None)
        keywords = [f"K{i} (GND-ID: 400000{i}-{i})" for i in range(4)]
        with patch("src.core.search.factory.resolve_dk_extractor", return_value=Extractor()), \
                patch("src.utils.config_manager.ConfigManager", MagicMock()):
            result = executor.execute_notation_search(
                list(keywords), stream_callback=lambda m, _s: lines.append(m), rvk_enabled=False)
        self.assertEqual([m for m in lines if "fehlgeschlagen" in m and "❌" in m], [])
        self.assertEqual(len(result["statistics"]["source_down_skipped"]), 4)

    def test_gnd_step(self):
        from src.utils.pipeline_utils import PipelineStepExecutor

        down = source_health.down_message("lobid")

        def fake_search(search_terms, instances, **kw):
            return {t: {} for t in search_terms}, {f"lobid:{t}": down for t in search_terms}

        lines = []
        executor = PipelineStepExecutor(alima_manager=None, cache_manager=MagicMock(),
                                        logger=None, config_manager=None)
        with patch("src.utils.pipeline_utils.enabled_gnd_provider_ids", return_value=None), \
                patch("src.core.search.service.resolve_gnd_instances", return_value=[]), \
                patch("src.core.search.service.search_gnd_keywords", fake_search):
            executor.execute_gnd_search(["a", "b", "c"], suggesters=["lobid"],
                                        aggregate_from_raw=False,
                                        stream_callback=lambda m, _s: lines.append(m))
        self.assertEqual([m for m in lines if "Quelle(n) fehlgeschlagen" in m], [])
        self.assertEqual(len([m for m in lines if "3 Quellen-Fehler" in m]), 1)

    def test_search_tab_status_counts_the_skip_once(self):
        from src.ui.find_keywords import SearchTab

        down = source_health.down_message("lobid")
        tab = MagicMock()
        tab._rows_from_results.return_value = ([], 0)
        SearchTab._on_search_finished(tab, {}, {"lobid:a": down, "lobid:b": down})
        msg, tone = tab._set_status.call_args[0]
        self.assertIn("lobid: nach 3 Verbindungsfehlern übersprungen (2 Begriffe)", msg)
        self.assertNotIn("fehlgeschlagen", msg)
        self.assertEqual(tone, "warning")

    def test_gnd_batch_search_has_no_partial_failure_line_for_a_skip(self):
        from src.core.agents.deterministic_functions import gnd_batch_search

        down = source_health.down_message("lobid")
        registry = MagicMock()
        registry.execute.return_value = json.dumps(
            {"source": "lobid", "results": {}, "errors": {"lobid:a": down, "lobid:b": down}})
        lines = []
        with patch("src.core.search.factory.resolve_gnd_source_tools",
                   return_value=(["lobid"], {"lobid": "search_lobid"})):
            try:
                gnd_batch_search(["a", "b"], tool_registry=registry,
                                 stream_callback=lines.append,
                                 config={"aggregate_from_raw": False,
                                         "enrich_from_local_db": False,
                                         "enrich_authority_from_dnb": False})
            except RuntimeError:
                pass  # no hits at all is reported as an error; the lines are what count
        self.assertEqual([m for m in lines if "Teilfehler" in m], [])

    def test_tool_block_names_skipped_terms(self):
        from src.mcp._tool_generation import ToolGenerationMixin

        block = ToolGenerationMixin._source_down_block(
            {"a": source_health.down_message("kvk"), "b": "HTTP 500"})
        self.assertEqual(block["skipped_terms"], ["a"])
        self.assertIsNone(ToolGenerationMixin._source_down_block({"b": "HTTP 500"}))


# ===========================================================================
# B. timeouts
# ===========================================================================
class TestTimeouts(_Isolated):
    OLD = {"lobid": 30, "swb": 15, "catalog": 10, "finc": 30, "sru": 30, "kvk": 30}

    def _client_timeout(self, pid, **settings):
        from src.core.search.factory import build_provider

        provider = build_provider(_inst(pid, **settings), cache_raw=False, ukm=self.km)
        if pid == "lobid" or pid == "swb":
            return provider.suggester.timeout
        if pid == "catalog":
            return provider.suggester.extractor.timeout
        if pid == "finc":
            dk = provider.dk_extractor(knowledge_manager=self.km)
            self.assertEqual(dk.timeout, dk._client.timeout)
            self.assertEqual(provider.suggester.client.timeout, dk.timeout)
            return dk.timeout
        return provider.client.timeout  # sru, kvk

    def test_defaults_are_the_old_limits(self):
        from src.core.search.factory import SearchProviderCategory

        cat = SearchProviderCategory()
        for pid, old in self.OLD.items():
            self.assertEqual(self._client_timeout(pid), old, pid)
            key = "timeout" if pid in ("finc", "kvk") else "timeout_s"
            fields = {f.key: f for f in cat.type_meta(pid).config_fields}
            self.assertEqual(fields[key].default, old, pid)
            self.assertEqual((fields[key].minimum, fields[key].maximum), (1, 120), pid)

    def test_a_set_value_reaches_the_client_and_is_bounded(self):
        for pid in self.OLD:
            key = "timeout" if pid in ("finc", "kvk") else "timeout_s"
            self.assertEqual(self._client_timeout(pid, **{key: 7}), 7, pid)
            self.assertEqual(self._client_timeout(pid, **{key: 500}), 120, pid)

    def test_the_request_carries_it(self):
        from src.core.search.factory import build_provider

        seen = {}

        def urlopen(url, timeout=None):
            seen["lobid"] = timeout
            raise urllib.error.URLError("stop")

        with patch("urllib.request.urlopen", urlopen):
            build_provider(_inst("lobid", timeout_s=7), cache_raw=False).search(
                SearchCapability.GND_KEYWORDS, ["x"])
        with patch("requests.get", side_effect=_conn_error()) as get:
            build_provider(_inst("swb", timeout_s=8), cache_raw=False).search(
                SearchCapability.GND_KEYWORDS, ["x"])
            seen["swb"] = get.call_args.kwargs["timeout"]
        cat = build_provider(_inst("catalog", token="t", timeout_s=9,
                                   catalog_search_url="http://catalog.invalid/soap"),
                             cache_raw=False)
        client = cat.suggester.extractor
        client.session = MagicMock(post=MagicMock(return_value=MagicMock(status_code=500)))
        client.search("x")
        seen["catalog"] = client.session.post.call_args.kwargs["timeout"]
        self.assertEqual(seen, {"lobid": 7, "swb": 8, "catalog": 9})

    def test_direct_dk_paths_get_the_instance_timeout(self):
        from src.core.search.factory import resolve_dk_extractor

        cases = [
            (_inst("catalog", timeout_s=11), "BiblioClient"),
            (_inst("finc", base_url="http://finc.invalid/p", dk_enabled=True, timeout=12),
             "FincCatalogClient"),
            (_inst("sru", preset="dnb", dk_enabled=True, timeout_s=13), "MarcXmlClient"),
        ]
        for inst, name in cases:
            with patch("src.core.search.factory._global_response_cache_enabled",
                       return_value=False):
                ext = resolve_dk_extractor(config=_StubConfig(inst))
            self.assertEqual(type(ext).__name__, name)
            self.assertEqual(ext.timeout, next(v for k, v in inst.settings.items()
                                               if k.startswith("timeout")))
            self.assertEqual(ext.source_id, inst.instance_id)


# ===========================================================================
# A. cache-only for the newly offered sources
# ===========================================================================
class TestCacheOnlyNewSources(_Isolated):
    def setUp(self):
        super().setUp()
        self.net = []

        def trip(*a, **k):
            self.net.append(a)
            raise AssertionError("network request attempted")

        for target in ("requests.sessions.Session.request", "urllib.request.urlopen",
                       "urllib.request.OpenerDirector.open", "requests.get",
                       "socket.socket.connect"):
            p = patch(target, trip)
            p.start()
            self.addCleanup(p.stop)

    def test_tripwire_is_armed(self):
        import urllib.request

        with self.assertRaises(AssertionError):
            urllib.request.build_opener().open("http://data.invalid/x", timeout=1)
        with self.assertRaises(AssertionError):
            requests.Session().get("http://data.invalid/x", timeout=1)
        self.assertEqual(len(self.net), 2)
        self.net.clear()

    def test_agent_tool_with_non_default_options_stays_offline(self):
        from src.mcp.tool_registry import ToolRegistry
        from src.utils.config_models import AlimaConfig

        params = raw_cache_params_for("swb", search_type="title", max_pages=5)
        blob = {"subjects": {"Faust": {"count": 1, "gnd_ids": ["4016818-9"]}}}
        self.km.store_raw_response("swb", "Faust", params, json.dumps(blob))
        cfg = AlimaConfig()
        cfg.plugins = [_inst("swb", cache_only=True)]
        reg = ToolRegistry(config_manager=MagicMock(load_config=lambda *a, **k: cfg))
        reg._knowledge_manager = self.km
        tools = {td.name: h for td, h in reg._generated_search_tools()}
        out = json.loads(tools["search_swb"](terms=["Faust", "Fehlt"], search_type="title"))
        self.assertIn("Faust", out["results"]["Faust"])
        self.assertEqual(out["cache_only"]["missing_terms"], ["Fehlt"])
        self.assertEqual(self.net, [])

    def test_swb_answers_from_the_raw_cache(self):
        from src.core.search.factory import build_provider

        params = raw_cache_params_for("swb", search_type="kw", max_pages=5)
        blob = {"subjects": {"Graphen": {"count": 1, "gnd_ids": ["4021842-9"]}}}
        self.km.store_raw_response("swb", "Graph", params, json.dumps(blob))
        provider = build_provider(_inst("swb", cache_only=True), cache=True, ukm=self.km,
                                  cache_raw=True)
        res = provider.search(SearchCapability.GND_KEYWORDS, ["Graph", "Fehlt"])
        self.assertEqual([it.label for it in res.per_term["Graph"]], ["Graphen"])
        self.assertIn("Fehlt", res.errors)
        self.assertEqual(self.net, [])

    def test_lobid_without_local_subject_table_is_a_miss_not_a_download(self):
        from src.core.search.cache_only import is_cache_miss
        from src.core.search.factory import build_provider

        params = raw_cache_params_for("lobid", search_type="kw")
        self.km.store_raw_response("lobid", "Graph", params, json.dumps({"aggregation": {}}))
        provider = build_provider(_inst("lobid", cache_only=True), ukm=self.km, cache_raw=True)
        res = provider.search(SearchCapability.GND_KEYWORDS, ["Graph"])
        self.assertTrue(is_cache_miss(res.errors["Graph"]))
        self.assertEqual(self.net, [])

    def test_kvk_writes_through_the_factory_and_reads_in_cache_only(self):
        from src.core.search.cache_only import is_cache_miss
        from src.core.search.factory import build_provider

        payload = "\n".join([
            json.dumps({"type": "catalog", "data": {"name": "K10plus", "results": 1,
                        "items": [{"title": "Faust", "url": "http://x?bibtip_docid=1"}]}}),
        ])
        live = build_provider(_inst("kvk"), ukm=self.km, cache_raw=True)
        response = MagicMock(status_code=200, text=payload)
        live.client._session = MagicMock(get=MagicMock(return_value=response))
        live.search(SearchCapability.TITLE_RECORDS, ["Faust"], search_type="title")

        offline = build_provider(_inst("kvk", cache_only=True), ukm=self.km, cache_raw=True)
        offline.client._session = MagicMock()
        res = offline.search(SearchCapability.TITLE_RECORDS, ["Faust", "Fehlt"],
                             search_type="title")
        offline.client._session.get.assert_not_called()
        self.assertEqual([it.label for it in res.per_term["Faust"]], ["Faust"])
        self.assertTrue(is_cache_miss(res.errors["Fehlt"]))

    def test_a_directly_built_kvk_provider_writes_nothing(self):
        from src.core.search.providers.kvk.provider import KvkProvider

        provider = KvkProvider()
        provider._ukm_ref = self.km
        provider.client._session = MagicMock(get=MagicMock(return_value=MagicMock(
            status_code=200, text="")))
        provider.search(SearchCapability.TITLE_RECORDS, ["Faust"])
        self.assertIsNone(self.km.get_raw_response(
            "kvk", "Faust", raw_cache_params_for("kvk"), max_age_hours=None))

    def test_cache_only_is_ignored_where_the_type_has_no_cache_path(self):
        from src.core.search.factory import build_provider

        # sru has a cache path since Nov 9 (tests/test_k10plus_sru_tool.py).
        provider = build_provider(_inst("gnd_local", cache_only=True), cache_raw=False)
        self.assertIsNot(getattr(provider, "cache_only", False), True)


# ===========================================================================
# A + D. the settings tab
# ===========================================================================
class TestStateMapping(unittest.TestCase):
    def test_state_reads_and_writes_the_existing_storage(self):
        from src.ui.plugin_settings_tab import (OFFER_CACHE_ONLY, OFFER_NONE, STATE_CACHE_ONLY,
                                                STATE_OFF, STATE_ON, apply_state, instance_state)

        cases = [
            (_inst("lobid"), STATE_ON),
            (_inst("lobid", cache_only=True), STATE_CACHE_ONLY),
            (_inst("lobid", enabled=False, cache_only=True), STATE_OFF),
        ]
        for inst, state in cases:
            self.assertEqual(instance_state(inst, OFFER_CACHE_ONLY), state)

        inst = _inst("lobid")
        apply_state(inst, STATE_ON, OFFER_CACHE_ONLY)
        self.assertEqual(inst.settings, {})                       # no key added
        apply_state(inst, STATE_CACHE_ONLY, OFFER_CACHE_ONLY)
        self.assertEqual((inst.enabled, inst.settings), (True, {"cache_only": True}))
        apply_state(inst, STATE_OFF, OFFER_CACHE_ONLY)
        self.assertEqual((inst.enabled, inst.settings), (False, {"cache_only": True}))
        apply_state(inst, STATE_ON, OFFER_CACHE_ONLY)
        self.assertEqual((inst.enabled, inst.settings), (True, {"cache_only": False}))

        other = _inst("gnd_local", cache_only=True)               # hand-written key
        self.assertEqual(instance_state(other, OFFER_NONE), STATE_ON)
        apply_state(other, STATE_OFF, OFFER_NONE)
        apply_state(other, STATE_ON, OFFER_NONE)
        self.assertEqual(other.settings, {"cache_only": True})    # left as it was


class TestSettingsTab(unittest.TestCase):
    def _panel(self, category, instances):
        from PyQt6.QtWidgets import QApplication

        from src.ui.plugin_settings_tab import _CategoryPanel, _ensure_categories

        if not isinstance(_app(), QApplication):
            self.skipTest("a bare QCoreApplication exists; widgets cannot be built")
        _ensure_categories()
        panel = _CategoryPanel(category)
        panel.load(instances)
        return panel

    def _states(self, panel, row):
        panel.list.setCurrentRow(row)
        combo = panel._state_combo
        out = {}
        for i in range(combo.count()):
            item = combo.model().item(i)
            out[combo.itemData(i)] = item.isEnabled()
        return out

    def test_state_entries_per_type(self):
        search = self._panel("search_provider", [
            _inst("lobid"), _inst("sru"), _inst("gnd_local"), _inst("kvk")])
        self.assertEqual(self._states(search, 0), {"on": True, "cache_only": True, "off": True})
        self.assertEqual(self._states(search, 1), {"on": True, "cache_only": True, "off": True})
        self.assertEqual(self._states(search, 2), {"on": True, "off": True})
        self.assertEqual(self._states(search, 3), {"on": True, "cache_only": True, "off": True})
        lookup = self._panel("lookup", [PluginInstanceConfig(
            "k10plus", "lookup", "k10plus", settings={})])
        self.assertEqual(self._states(lookup, 0), {"on": True, "off": True})

    def test_cache_only_is_the_state_not_a_second_field(self):
        panel = self._panel("search_provider", [_inst("catalog", cache_only=True)])
        panel.list.setCurrentRow(0)
        self.assertNotIn("cache_only", panel._field_widgets)
        self.assertEqual(panel._state_combo.currentData(), "cache_only")
        panel._state_combo.setCurrentIndex(panel._state_combo.findData("off"))
        out = panel.collect()[0]
        self.assertEqual((out.enabled, out.settings["cache_only"]), (False, True))

    def test_untouched_instances_round_trip_without_diff(self):
        originals = [
            _inst("lobid", page_size=30),
            _inst("swb"),
            _inst("catalog", token="t", cache_only=True, strict_gnd_validation_for_dk_search=True),
            _inst("finc", enabled=False, base_url="http://f", timeout=30, dk_enabled=True),
            _inst("sru", preset="dnb", max_records=50),
            _inst("kvk", timeout=45),
        ]
        panel = self._panel("search_provider", originals)
        for row in range(len(originals)):
            panel.list.setCurrentRow(row)
        panel.list.setCurrentRow(0)
        out = panel.collect()
        declared = {}
        from src.core.search.factory import SearchProviderCategory

        for inst in originals:
            declared[inst.instance_id] = {
                f.key: f.default for f in SearchProviderCategory().type_meta(
                    inst.provider_id).config_fields}
        for before, after in zip(originals, out):
            self.assertEqual(after.enabled, before.enabled, before.instance_id)
            self.assertNotIn("timeout_s", set(after.settings) - set(before.settings))
            self.assertNotIn("cache_only", set(after.settings) - set(before.settings))
            for key, value in before.settings.items():
                self.assertEqual(after.settings[key], value, (before.instance_id, key))

    def test_effect_row_and_multi_instance_rule(self):
        one = self._panel("search_provider", [_inst("lobid")])
        one.list.setCurrentRow(0)
        self.assertIn("ergänzt die anderen GND-Quellen", one._effect_label.text())
        self.assertNotIn("Mehrere Instanzen", one._effect_label.text())
        two = self._panel("search_provider", [_inst("lobid"), _inst("lobid", iid="lobid-2")])
        two.list.setCurrentRow(1)
        self.assertIn("erste aktive Instanz in der Liste", two._effect_label.text())

    def test_dk_line_follows_the_list_and_the_unsaved_form(self):
        from src.ui.plugin_settings_tab import ORDER_RULE_TEXT

        panel = self._panel("search_provider", [
            _inst("catalog"), _inst("finc", base_url="http://f", dk_enabled=True)])
        self.assertTrue(panel._dk_label.text().startswith("DK-Quelle im Moment: finc"))
        panel.list.setCurrentRow(1)
        panel._state_combo.setCurrentIndex(panel._state_combo.findData("off"))
        self.assertTrue(panel._dk_label.text().startswith("DK-Quelle im Moment: Katalog"))
        texts = [w.text() for w in panel.findChildren(__import__(
            "PyQt6.QtWidgets", fromlist=["QLabel"]).QLabel)]
        self.assertIn(ORDER_RULE_TEXT, texts)


class TestConfigFileRoundTrip(_TempHome):
    """load → save leaves the plugin entries of an existing config.json as they were."""

    def test_search_entries_unchanged(self):
        from src.utils.config_manager import ConfigManager

        plugins = [
            {"instance_id": "lobid", "category": "search_provider", "provider_id": "lobid",
             "label": "Lobid", "enabled": True, "usage_hint": "", "settings": {"page_size": 30}},
            {"instance_id": "catalog", "category": "search_provider", "provider_id": "catalog",
             "label": "Katalog", "enabled": True, "usage_hint": "",
             "settings": {"token": "T", "cache_only": True}},
            {"instance_id": "finc", "category": "search_provider", "provider_id": "finc",
             "label": "finc", "enabled": False, "usage_hint": "",
             "settings": {"base_url": "http://f", "timeout": 30}},
            {"instance_id": "kvk", "category": "search_provider", "provider_id": "kvk",
             "label": "KVK", "enabled": True, "usage_hint": "", "settings": {"timeout": 45}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "unified_config": {"providers": [], "gemini_api_key": "K"},
                "plugins": plugins,
            }))
            ConfigManager.reset()
            try:
                cm = ConfigManager()
                cm.config_file = path
                cm.save_config(cm.load_config())
                saved = json.loads(path.read_text())["plugins"]
            finally:
                ConfigManager.reset()
        by_id = {p["instance_id"]: p for p in saved if p["category"] == "search_provider"}
        for original in plugins:
            got = by_id[original["instance_id"]]
            self.assertEqual(got["settings"], original["settings"], original["instance_id"])
            self.assertEqual(got["enabled"], original["enabled"])


# ===========================================================================
# D. DK source line == resolve_dk_extractor
# ===========================================================================
class TestDkSourceDescription(_TempHome):
    TYPE_OF = {"FincCatalogClient": "finc", "MarcXmlClient": "sru", "BiblioClient": "catalog"}

    def _configs(self):
        return {
            "finc_on": [_inst("finc", base_url="http://f", dk_enabled=True), _inst("catalog")],
            "finc_on_without_url": [_inst("finc", dk_enabled=True), _inst("catalog")],
            "finc_off": [_inst("finc", base_url="http://f"), _inst("catalog")],
            "finc_disabled": [_inst("finc", enabled=False, base_url="http://f", dk_enabled=True),
                              _inst("catalog")],
            "sru_on": [_inst("sru", preset="dnb", dk_enabled=True), _inst("catalog")],
            "sru_on_without_endpoint": [_inst("sru", dk_enabled=True), _inst("catalog")],
            "finc_and_sru": [_inst("sru", preset="dnb", dk_enabled=True),
                             _inst("finc", base_url="http://f", dk_enabled=True)],
            "catalog_only": [_inst("catalog", token="t")],
            "nothing": [],
            "second_catalog": [_inst("catalog", iid="cat-a"), _inst("catalog", iid="cat-b")],
        }

    def test_matches_the_resolver_for_every_config(self):
        from src.core.search.factory import describe_dk_source, resolve_dk_extractor

        for name, instances in self._configs().items():
            cfg = _StubConfig(*instances)
            with patch("src.core.search.factory._global_response_cache_enabled",
                       return_value=False):
                ext = resolve_dk_extractor(config=cfg)
            info = describe_dk_source(cfg)
            # Two DK switches on form a fallback chain (Nov 2026): compare the
            # whole chain. - Claude Generated
            parts = ext.candidates if type(ext).__name__ == "FallbackDkExtractor" else None
            first = parts[0].extractor if parts else ext
            self.assertEqual(info["provider_id"], self.TYPE_OF[type(first).__name__], name)
            if name == "nothing":
                self.assertIsNone(info["instance_id"])
            else:
                self.assertEqual(info["instance_id"], first.source_id, name)
                self.assertEqual([c["instance_id"] for c in info["chain"]],
                                 [c.source_id for c in parts] if parts else [ext.source_id], name)

    def test_custom_classification_plugin(self):
        from src.core.search.factory import describe_dk_source, resolve_dk_extractor
        from src.core.search.registry import PROVIDER_REGISTRY, register_provider

        marker = object()

        @register_provider
        class _Mine:
            id = "mylib_dk"
            label = "Meine DK"
            capabilities = {SearchCapability.CLASSIFICATION}

            def __init__(self, **cfg):
                pass

            def is_available(self, cfg=None):
                return True

            def dk_extractor(self, **_):
                return marker

        self.addCleanup(PROVIDER_REGISTRY.pop, "mylib_dk", None)
        cfg = _StubConfig(_inst("catalog"), _inst("mylib_dk"))
        with patch("src.core.search.factory._global_response_cache_enabled", return_value=False):
            self.assertIs(resolve_dk_extractor(config=cfg), marker)
        self.assertEqual(describe_dk_source(cfg)["provider_id"], "mylib_dk")

    def test_effect_texts(self):
        from src.core.search.factory import SearchProviderCategory

        cat = SearchProviderCategory()
        effect = {t: cat.type_meta(t).effect for t in ("lobid", "catalog", "kvk", "sru")}
        self.assertIn("ergänzt die anderen GND-Quellen (Vereinigung der Treffer)", effect["lobid"])
        self.assertIn("Reihenfolge der Liste", effect["catalog"])
        self.assertIn("geht an die nächste", effect["catalog"])
        self.assertIn("Werkzeug für den Agenten", effect["kvk"])
        # sru has an agent tool and a cache path since Nov 9.
        self.assertIn("Werkzeug für den Agenten", effect["sru"])
        self.assertFalse(cat.type_meta("sru").cache_only_note)
        self.assertFalse(cat.type_meta("lobid").cache_only_note)


if __name__ == "__main__":
    unittest.main()
