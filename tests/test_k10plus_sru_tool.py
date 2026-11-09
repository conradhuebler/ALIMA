"""Claude Generated - search_k10plus: K10plus SRU in PICA-XML for chat and agents.

Pinned here (Nov 9, 2026):

* the query: one CQL index per ``search_type``, values escaped, an author
  "Vorname Nachname" turned into "Nachname, Vorname" (only without comma and
  with exactly two words), always ``recordSchema=picaxml``;
* the PICA parser on invented records of the structure of a real K10plus
  record (003@, 021A, 028A/028C, 033A, 011@, 004A, 013D, 037C, 044K chains
  closed by ``$A<ISIL>``, 045F, 101@ blocks, 209A copies, 245Z notations);
* ``local`` only with an ELN and a copy of that ELN, otherwise ``local_hint``;
* ``total`` from numberOfRecords and ``truncated``;
* the tool name per preset, the Stufe-1 markers (cache-only miss, source
  down), no request in cache-only mode;
* the DK extractor with an ELN: only notations of the library's copies count.

No test reaches the network (a fake session answers, a tripwire guards
``requests``/``socket``); the knowledge DB is a temporary SQLite file and
HOME a temporary directory. All records are invented.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.sax.saxutils import escape

import requests

from src.core.search import source_health
from src.core.search.cache_only import is_cache_miss
from src.core.search.provider import SearchCapability
from src.utils import institution
from src.utils.clients import picaxml_client as pica
from src.utils.config_models import AlimaConfig, PluginInstanceConfig


# ── invented PICA records ─────────────────────────────────────────────────


def _df(tag, subs, occ=None):
    attr = f' occurrence="{occ}"' if occ else ""
    inner = "".join(f'<subfield code="{c}">{escape(v)}</subfield>' for c, v in subs)
    return f'<datafield tag="{tag}"{attr}>{inner}</datafield>'


def _record(*fields):
    return '<record xmlns="info:srw/schema/5/picaXML-v1.0">' + "".join(fields) + "</record>"


def _response(records, total):
    recs = "".join(
        "<zs:record><zs:recordSchema>picaxml</zs:recordSchema><zs:recordPacking>xml"
        f"</zs:recordPacking><zs:recordData>{r}</zs:recordData>"
        f"<zs:recordPosition>{i}</zs:recordPosition></zs:record>"
        for i, r in enumerate(records, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><zs:searchRetrieveResponse '
        'xmlns:zs="http://www.loc.gov/zing/srw/"><zs:version>1.1</zs:version>'
        f"<zs:numberOfRecords>{total}</zs:numberOfRecords><zs:records>{recs}</zs:records>"
        "</zs:searchRetrieveResponse>"
    ).encode("utf-8")


#: A record with a copy of ELN 105 and a copy of another library (ELN 999).
RECORD_A = _record(
    _df("003@", [("0", "900000001")]),
    _df("021A", [("a", "Grundlagen der @Testchemie"), ("d", "ein Lehrbuch")]),
    _df("028A", [("9", "900000010"), ("7", "gnd/100000001"), ("A", "Muster"),
                 ("D", "Erika"), ("B", "VerfasserIn")]),
    _df("028C", [("A", "Beispiel"), ("D", "Max"), ("B", "HerausgeberIn")]),
    _df("033A", [("p", "Teststadt"), ("n", "Testverlag")]),
    _df("011@", [("a", "2024")]),
    _df("004A", [("0", "978-3-00-000000-1")]),
    _df("013D", [("9", "900000020"), ("7", "gnd/4000000-0"), ("a", "Lehrbuch")]),
    _df("037C", [("d", "Dissertation"), ("e", "Testuniversität"), ("f", "2023")]),
    _df("045F", [("a", "540")]),
    _df("044K", [("9", "900000100"), ("7", "gnd/4000001-1"), ("a", "Testchemie")]),
    _df("044K", [("9", "900000101"), ("7", "gnd/4000002-2"), ("a", "Analyse")], occ="01"),
    _df("044K", [("9", "900000102"), ("7", "gnd/4000003-3"), ("a", "Methode")], occ="01"),
    _df("044K", [("A", "DE-105")], occ="01"),
    _df("044K", [("9", "900000103"), ("7", "gnd/4000004-4"), ("a", "Fremdkette")], occ="02"),
    _df("044K", [("A", "DE-999")], occ="02"),
    _df("101@", [("a", "2027")]),
    _df("203@", [("0", "4000000001")], occ="01"),
    _df("209A", [("B", "105"), ("a", "TC 100 A"), ("x", "00")], occ="01"),
    _df("245Z", [("a", "FGAUT"), ("x", "00")], occ="01"),
    _df("245Z", [("9", "900000500"), ("a", "DK 54"), ("x", "00")], occ="01"),
    _df("245Z", [("9", "900000501"), ("a", "DK 543.2"), ("x", "00")], occ="01"),
    _df("101@", [("a", "0001")]),
    _df("209A", [("B", "999"), ("a", "X 1")], occ="01"),
    _df("245Z", [("9", "900000600"), ("a", "DK 999.9")], occ="01"),
)
#: Only a copy of another library.
RECORD_B = _record(
    _df("003@", [("0", "900000002")]),
    _df("021A", [("a", "Ein anderes Buch")]),
    _df("101@", [("a", "0001")]),
    _df("209A", [("B", "999"), ("a", "Y 2")], occ="01"),
    _df("245Z", [("9", "900000700"), ("a", "DK 54")], occ="01"),
)
#: No copy at all.
RECORD_C = _record(
    _df("003@", [("0", "900000003")]),
    _df("021A", [("a", "Ohne Bestand")]),
)


class _Resp:
    def __init__(self, content, status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class _FakeSession:
    """Answers every request with one payload and records the calls."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        return _Resp(self.payload, self.status)


def _inst(pid="sru", *, iid=None, enabled=True, **settings):
    return PluginInstanceConfig(instance_id=iid or pid, category="search_provider",
                                provider_id=pid, label="", enabled=enabled,
                                settings=dict(settings))


_KEEP_ALIVE: list = []


class _Isolated(unittest.TestCase):
    """Temporary HOME + knowledge DB, network tripwire, Freiberg institution."""

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
        notice = patch.object(source_health, "_emit_notice", lambda _t: None)
        notice.start()
        self.addCleanup(notice.stop)
        source_health.begin_run()
        self.addCleanup(source_health.begin_run)
        institution.set_current("DE-105", "105")

    def _provider(self, payload=None, **settings):
        from src.core.search.factory import build_provider

        provider = build_provider(_inst(**{"preset": "k10plus", **settings}),
                                  ukm=self.km, cache_raw=True)
        session = _FakeSession(payload if payload is not None else _response([RECORD_A], 1))
        provider._pica_client = pica.PicaXmlClient(session=session, source_id=provider.source_id)
        return provider, session

    def _registry(self, *instances):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = list(instances)
        reg = ToolRegistry(config_manager=MagicMock(load_config=lambda *a, **k: cfg))
        reg._knowledge_manager = self.km
        return reg


# ── query ─────────────────────────────────────────────────────────────────


class TestQuery(unittest.TestCase):
    def test_index_per_search_type(self):
        cases = {
            "kw": 'pica.all="Quantenchemie"',
            "title": 'pica.tit="Quantenchemie"',
            "subject": 'pica.slw="Quantenchemie"',
            "isbn": 'pica.isb="Quantenchemie"',
            "ppn": 'pica.ppn="Quantenchemie"',
            "author": 'pica.per="Quantenchemie"',
        }
        for search_type, expected in cases.items():
            self.assertEqual(pica.build_cql("Quantenchemie", search_type), expected, search_type)
        self.assertEqual(pica.build_cql("x", None), 'pica.all="x"')
        with self.assertRaises(ValueError):
            pica.build_cql("x", "dk")

    def test_values_are_escaped(self):
        self.assertEqual(pica.escape_cql('a "b" \\c'), 'a \\"b\\" \\\\c')
        self.assertEqual(pica.build_cql('Der "Titel"', "title"), 'pica.tit="Der \\"Titel\\""')
        self.assertEqual(pica.build_cql("a\\", "kw"), 'pica.all="a\\\\"')

    def test_author_is_turned_only_for_two_words_without_comma(self):
        cases = {
            "Conrad Hübler": "Hübler, Conrad",
            "  Conrad   Hübler ": "Hübler, Conrad",
            "Hübler, Conrad": "Hübler, Conrad",
            "Hübler": "Hübler",
            "Hans Peter Müller": "Hans Peter Müller",
            "Müller,Hans Peter": "Müller,Hans Peter",
        }
        for given, sent in cases.items():
            self.assertEqual(pica.normalize_author(given), sent, given)
        self.assertEqual(pica.build_cql("Conrad Hübler", "author"), 'pica.per="Hübler, Conrad"')
        # Only the author axis is turned.
        self.assertEqual(pica.build_cql("Conrad Hübler", "title"), 'pica.tit="Conrad Hübler"')

    def test_params_always_ask_for_picaxml(self):
        params = pica.build_params("x", "title", 7)
        self.assertEqual(params["recordSchema"], "picaxml")
        self.assertEqual(params["version"], "1.1")
        self.assertEqual(params["operation"], "searchRetrieve")
        self.assertEqual(params["maximumRecords"], "7")

    def test_max_results_default_and_bound(self):
        self.assertEqual(pica.clamp_max_results(None), 10)
        self.assertEqual(pica.clamp_max_results(99), 50)
        self.assertEqual(pica.clamp_max_results(0), 1)
        self.assertEqual(pica.clamp_max_results("x"), 10)


# ── parser ────────────────────────────────────────────────────────────────


class TestParser(unittest.TestCase):
    def setUp(self):
        self.records, self.total, self.diag = pica.parse_response(
            _response([RECORD_A, RECORD_B, RECORD_C], 25))
        self.a, self.b, self.c = self.records

    def test_total_and_count(self):
        self.assertEqual(self.total, 25)
        self.assertEqual(len(self.records), 3)
        self.assertEqual(self.diag, [])

    def test_bibliographic_fields(self):
        a = self.a
        self.assertEqual(a["ppn"], "900000001")
        self.assertEqual(a["title"], "Grundlagen der Testchemie : ein Lehrbuch")
        self.assertEqual(a["authors"], [
            {"name": "Muster, Erika", "role": "VerfasserIn", "gnd_id": "100000001"},
            {"name": "Beispiel, Max", "role": "HerausgeberIn", "gnd_id": ""},
        ])
        self.assertEqual(a["publisher"], "Teststadt : Testverlag")
        self.assertEqual(a["year"], "2024")
        self.assertEqual(a["isbn"], ["978-3-00-000000-1"])
        self.assertEqual(a["form"], ["Lehrbuch"])
        self.assertEqual(a["thesis"], "Dissertation, Testuniversität, 2023")
        self.assertEqual(a["ddc"], ["540"])

    def test_several_chains_split_by_occurrence_and_source_line(self):
        chains = self.a["subjects"]
        self.assertEqual([[t["term"] for t in c["terms"]] for c in chains],
                         [["Testchemie"], ["Analyse", "Methode"], ["Fremdkette"]])
        self.assertEqual([c["source"] for c in chains], ["", "DE-105", "DE-999"])
        self.assertEqual(chains[1]["terms"][0], {"term": "Analyse", "gnd_id": "4000002-2",
                                                 "ppn": "900000101"})

    def test_two_chains_in_one_occurrence_are_split_by_the_source_line(self):
        rec = pica.parse_response(_response([_record(
            _df("044K", [("a", "Eins")], occ="03"), _df("044K", [("A", "DE-105")], occ="03"),
            _df("044K", [("a", "Zwei")], occ="03"), _df("044K", [("A", "DE-105")], occ="03"),
        )], 1))[0][0]
        self.assertEqual([[t["term"] for t in c["terms"]] for c in rec["subjects"]],
                         [["Eins"], ["Zwei"]])

    def test_copies_belong_to_their_library_block(self):
        self.assertEqual(self.a["copies"], [
            {"eln": "105", "signature": "TC 100 A", "notations": [
                {"text": "FGAUT", "ppn": ""},
                {"text": "DK 54", "ppn": "900000500"},
                {"text": "DK 543.2", "ppn": "900000501"}]},
            {"eln": "999", "signature": "X 1", "notations": [
                {"text": "DK 999.9", "ppn": "900000600"}]},
        ])
        self.assertEqual(self.b["copies"][0]["eln"], "999")
        self.assertEqual(self.c["copies"], [])

    def test_unreadable_answer_is_an_error(self):
        with self.assertRaises(ValueError):
            pica.parse_response(b"<not xml")

    def test_diagnostic_without_records_is_an_error(self):
        payload = (b'<?xml version="1.0"?><zs:searchRetrieveResponse xmlns:zs="http://www.loc.gov/'
                   b'zing/srw/"><zs:numberOfRecords>0</zs:numberOfRecords><zs:diagnostics>'
                   b'<diag:diagnostic xmlns:diag="http://www.loc.gov/zing/srw/diagnostic/">'
                   b'<diag:message>Query syntax error</diag:message></diag:diagnostic>'
                   b'</zs:diagnostics></zs:searchRetrieveResponse>')
        client = pica.PicaXmlClient(session=_FakeSession(payload))
        with self.assertRaises(ValueError) as ctx:
            client.search("x")
        self.assertIn("Query syntax error", str(ctx.exception))


class TestLocalBlock(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = pica.parse_response(
            _response([RECORD_A, RECORD_B, RECORD_C], 3))[0]

    def test_with_eln_and_copy(self):
        shaped = pica.shape_record(self.a, "105", "DE-105")
        self.assertNotIn("copies", shaped)
        self.assertNotIn("local_hint", shaped)
        self.assertEqual(shaped["local"], {
            "signature": "TC 100 A",
            "notations": [{"text": "FGAUT", "ppn": ""},
                          {"text": "DK 54", "ppn": "900000500"},
                          {"text": "DK 543.2", "ppn": "900000501"}],
            "subject_chains": [{"terms": [
                {"term": "Analyse", "gnd_id": "4000002-2", "ppn": "900000101"},
                {"term": "Methode", "gnd_id": "4000003-3", "ppn": "900000102"}]}],
        })

    def test_copy_of_another_eln_and_no_copy(self):
        for record in (self.b, self.c):
            shaped = pica.shape_record(record, "105", "DE-105")
            self.assertNotIn("local", shaped)
            self.assertEqual(shaped["local_hint"], "kein Exemplar der Bibliothek (ELN 105) im Satz")
        # The other library's copy is found under its own ELN.
        self.assertEqual(pica.shape_record(self.b, "999", "")["local"]["signature"], "Y 2")

    def test_without_eln(self):
        shaped = pica.shape_record(self.a, None, "DE-105")
        self.assertNotIn("local", shaped)
        self.assertTrue(shaped["local_hint"].startswith("ELN nicht gesetzt"))

    def test_eln_without_isil_has_no_local_chains(self):
        local = pica.shape_record(self.a, "105", None)["local"]
        self.assertEqual(local["subject_chains"], [])
        self.assertTrue(local["subject_chains_hint"].startswith("ISIL nicht gesetzt"))


# ── provider and tool ─────────────────────────────────────────────────────


class TestToolSpec(unittest.TestCase):
    def _names(self, *instances):
        from src.mcp.tool_registry import ToolRegistry

        cfg = AlimaConfig()
        cfg.plugins = list(instances)
        reg = ToolRegistry(config_manager=MagicMock(load_config=lambda *a, **k: cfg))
        return {td.name: td for td, _ in reg._generated_search_tools()}

    def test_name_per_preset(self):
        self.assertIn("search_k10plus", self._names(_inst(preset="k10plus")))
        names = self._names(_inst(preset="dnb"))
        self.assertIn("search_sru", names)
        self.assertNotIn("search_k10plus", names)
        both = self._names(_inst(preset="k10plus"), _inst(iid="sru-dnb", preset="dnb"))
        self.assertIn("search_k10plus", both)
        self.assertIn("search_sru_sru_dnb", both)
        self.assertIn("search_sru", self._names(_inst(base_url="https://sru.example/db")))

    def test_no_tool_for_an_instance_without_endpoint(self):
        names = self._names(_inst())
        self.assertFalse([n for n in names if n.startswith(("search_sru", "search_k10plus"))])

    def test_description_names_the_rules(self):
        institution.set_current("DE-105", "105")
        td = self._names(_inst(preset="k10plus"))["search_k10plus"]
        self.assertIn("'Conrad Hübler' -> 'Hübler, Conrad'", td.description)
        self.assertIn("ELN 105", td.description)
        self.assertIn("truncated", td.description)
        props = td.parameters["properties"]
        self.assertEqual(props["search_type"]["enum"],
                         ["kw", "title", "author", "subject", "isbn", "ppn"])
        self.assertEqual((props["max_results"]["default"], props["max_results"]["maximum"]),
                         (10, 50))
        institution.set_current("", "")
        td = self._names(_inst(preset="k10plus"))["search_k10plus"]
        self.assertIn("no ELN is set", td.description)

    def test_capability_stays_title_records_and_classification(self):
        from src.core.search.providers.sru.provider import SruProvider

        self.assertEqual(SruProvider.capabilities,
                         {SearchCapability.TITLE_RECORDS, SearchCapability.CLASSIFICATION})


class TestK10plusTool(_Isolated):
    def _call(self, reg, inst, payload, **args):
        provider = reg._provider_for(inst, cache=False)
        session = _FakeSession(payload)
        provider._pica_client = pica.PicaXmlClient(session=session, source_id=provider.source_id)
        tools = {td.name: h for td, h in reg._generated_search_tools()}
        return json.loads(tools["search_k10plus"](**args)), session

    def test_request_result_and_truncated(self):
        inst = _inst(preset="k10plus")
        reg = self._registry(inst)
        out, session = self._call(reg, inst, _response([RECORD_A, RECORD_C], 25),
                                  terms=["Erika Muster"], search_type="author")
        self.assertEqual(session.calls[0]["url"], "https://sru.k10plus.de/k10plus")
        params = session.calls[0]["params"]
        self.assertEqual(params["query"], 'pica.per="Muster, Erika"')
        self.assertEqual(params["maximumRecords"], "10")        # default
        self.assertEqual(params["recordSchema"], "picaxml")
        self.assertEqual(out["source"], "k10plus")
        self.assertEqual(out["errors"], {})
        records = out["results"]["Erika Muster"]
        self.assertEqual([r["ppn"] for r in records], ["900000001", "900000003"])
        self.assertEqual(records[0]["local"]["signature"], "TC 100 A")
        self.assertIn("local_hint", records[1])
        self.assertEqual(out["meta"]["Erika Muster"],
                         {"result_count": 2, "returned": 2, "total": 25, "truncated": True})
        self.assertEqual(self.net, [])

    def test_no_truncated_when_all_came_back_and_max_is_bounded(self):
        inst = _inst(preset="k10plus")
        reg = self._registry(inst)
        out, session = self._call(reg, inst, _response([RECORD_A], 1),
                                  terms=["x"], max_results=99)
        self.assertEqual(session.calls[0]["params"]["maximumRecords"], "50")
        self.assertEqual(session.calls[0]["params"]["query"], 'pica.all="x"')
        self.assertNotIn("truncated", out["meta"]["x"])
        self.assertEqual(out["meta"]["x"]["total"], 1)

    def test_without_eln_no_record_has_local(self):
        institution.set_current("DE-105", "")
        inst = _inst(preset="k10plus")
        reg = self._registry(inst)
        out, _ = self._call(reg, inst, _response([RECORD_A], 1), terms=["x"])
        record = out["results"]["x"][0]
        self.assertNotIn("local", record)
        self.assertTrue(record["local_hint"].startswith("ELN nicht gesetzt"))

    def test_http_error_is_an_error_not_an_empty_hit(self):
        inst = _inst(preset="k10plus")
        reg = self._registry(inst)
        provider = reg._provider_for(inst, cache=False)
        provider._pica_client = pica.PicaXmlClient(session=_FakeSession(b"", status=503),
                                                   source_id=provider.source_id)
        tools = {td.name: h for td, h in reg._generated_search_tools()}
        out = json.loads(tools["search_k10plus"](terms=["x"]))
        self.assertIn("503", out["errors"]["x"])
        self.assertEqual(out["results"]["x"], [])

    def test_timeout_reaches_the_request(self):
        provider, session = self._provider(timeout_s=17)
        provider._pica_client = None
        client = provider.pica_client
        self.assertEqual(client.timeout, 17)
        client._session = session
        provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        self.assertEqual(session.calls[0]["timeout"], 17)


class TestStageOneMarkers(_Isolated):
    def test_cache_only_reads_what_a_live_search_stored_and_makes_no_request(self):
        live, _ = self._provider(_response([RECORD_A], 4))
        live.search(SearchCapability.TITLE_RECORDS, ["Testchemie"], search_type="subject")

        offline, session = self._provider(cache_only=True)
        self.assertTrue(offline.cache_only)
        res = offline.search(SearchCapability.TITLE_RECORDS, ["Testchemie", "Fehlt"],
                             search_type="subject")
        self.assertEqual(session.calls, [])
        self.assertEqual(self.net, [])
        record = res.per_term["Testchemie"][0].record
        self.assertEqual(record["local"]["signature"], "TC 100 A")
        self.assertEqual(res.per_term_meta["Testchemie"]["total"], 4)
        self.assertTrue(res.per_term_meta["Testchemie"]["truncated"])
        self.assertTrue(is_cache_miss(res.errors["Fehlt"]))

    def test_cache_only_tool_lists_missing_terms(self):
        inst = _inst(preset="k10plus", cache_only=True)
        reg = self._registry(inst)
        tools = {td.name: (td, h) for td, h in reg._generated_search_tools()}
        td, handler = tools["search_k10plus"]
        self.assertIn("Cache-only mode", td.description)
        out = json.loads(handler(terms=["Fehlt"]))
        self.assertEqual(out["cache_only"]["missing_terms"], ["Fehlt"])
        self.assertEqual(self.net, [])

    def test_a_down_source_is_skipped_with_the_marker(self):
        provider, session = self._provider()
        for _ in range(source_health.FAILURE_THRESHOLD):
            source_health.record_failure(provider.source_id)
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        self.assertEqual(session.calls, [])
        self.assertTrue(source_health.is_source_down(res.errors["x"]))

    def test_a_directly_built_provider_writes_nothing(self):
        from src.core.search.providers.sru.provider import SruProvider

        provider = SruProvider(preset="k10plus")
        provider._ukm_ref = self.km
        provider._pica_client = pica.PicaXmlClient(session=_FakeSession(_response([RECORD_A], 1)))
        provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        params = provider._raw_params("kw", "picaxml")
        self.assertIsNone(self.km.get_raw_response("sru", "x", params, max_age_hours=None))


# ── DK extractor ──────────────────────────────────────────────────────────


class TestDkExtractor(_Isolated):
    def test_with_eln_only_the_librarys_copies_count(self):
        provider, session = self._provider(
            _response([RECORD_A, RECORD_B, RECORD_C], 3), dk_enabled=True)
        ext = provider.dk_extractor()
        self.assertIsInstance(ext, pica.PicaDkExtractor)
        out = ext.extract_dk_classifications_for_keywords(["Testchemie (GND-ID: 4000001-1)"],
                                                         max_results=20)
        self.assertEqual(session.calls[0]["params"]["query"], 'pica.slw="Testchemie"')
        self.assertEqual(session.calls[0]["params"]["maximumRecords"], "20")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["keyword"], "Testchemie (GND-ID: 4000001-1)")
        codes = {(c["classification_type"], c["dk"]): c for c in out[0]["classifications"]}
        self.assertEqual(set(codes), {("DK", "54"), ("DK", "543.2")})   # not 999.9, not FGAUT
        self.assertEqual(codes[("DK", "54")]["count"], 1)                # record B: other library
        self.assertEqual(codes[("DK", "54")]["titles"], ["Grundlagen der Testchemie : ein Lehrbuch"])
        self.assertEqual(set(codes[("DK", "54")]),
                         {"dk", "classification_type", "count", "titles", "keywords"})

    def test_without_eln_or_with_another_preset_the_marc_path_stays(self):
        from src.core.search.factory import build_provider
        from src.utils.clients.marcxml_client import MarcXmlClient

        institution.set_current("DE-105", "")
        ext = build_provider(_inst(preset="k10plus"), ukm=self.km, cache_raw=True).dk_extractor()
        self.assertIsInstance(ext, MarcXmlClient)
        institution.set_current("DE-105", "105")
        ext = build_provider(_inst(preset="dnb"), ukm=self.km, cache_raw=True).dk_extractor()
        self.assertIsInstance(ext, MarcXmlClient)

    def test_cache_only_reads_the_stored_answer_without_request(self):
        live, _ = self._provider(_response([RECORD_A], 1))
        live.dk_extractor().extract_dk_classifications_for_keywords(["Testchemie"])

        offline, session = self._provider(cache_only=True)
        ext = offline.dk_extractor()
        out = ext.extract_dk_classifications_for_keywords(["Testchemie"])
        self.assertEqual({c["dk"] for c in out[0]["classifications"]}, {"54", "543.2"})
        self.assertEqual(ext.extract_dk_classifications_for_keywords(["Fehlt"]), [])
        self.assertEqual(ext.last_cache_misses, ["Fehlt"])
        self.assertEqual(session.calls, [])
        self.assertEqual(self.net, [])

    def test_marc_dk_path_in_cache_only_makes_no_request(self):
        from src.core.search.factory import build_provider

        institution.set_current("DE-105", "")
        ext = build_provider(_inst(preset="k10plus", cache_only=True), ukm=self.km,
                             cache_raw=True).dk_extractor()
        ext.session = MagicMock()
        self.assertEqual(ext.extract_dk_classifications_for_keywords(["a", "b"]), [])
        ext.session.get.assert_not_called()
        self.assertEqual(ext.last_cache_misses, ["a", "b"])

    def test_dk_source_is_still_off_by_default(self):
        from src.core.search.factory import describe_dk_source

        class _Cfg:
            def enabled_instances_for(self, _c):
                return [_inst(preset="k10plus"), _inst("catalog")]

        self.assertEqual(describe_dk_source(_Cfg())["provider_id"], "catalog")


if __name__ == "__main__":
    unittest.main()
