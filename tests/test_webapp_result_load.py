"""Loading a saved result into a webapp session - Claude Generated.

``POST /api/session/{id}/load`` (``src/webapp/routers/load.py``, validation
and rendering in ``src/webapp/result_loading.py``): both file formats ALIMA
writes (the webapp/API export and a saved ``KeywordAnalysisState``), the
limits (size, JSON, object, known shape), foreign text in every card field
(escaped, nothing executable), no path from the file opened, the session state
afterwards (poll, export, WebSocket replay), and a new run replacing the loaded
result. The state is the synthetic geo_sachdb fixture of
``tests/test_geo_sachdb_result.py`` (chains, DK, WinIBW entry with local PPNs,
geo block); all data invented. No network: the RVK check of the export is
patched.
"""
from __future__ import annotations

import builtins
import dataclasses
import html
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HOME = {}


def setUpModule():
    _HOME["old"] = os.environ.get("HOME")
    _HOME["tmp"] = tempfile.mkdtemp()
    os.environ["HOME"] = _HOME["tmp"]


def tearDownModule():
    if _HOME.get("old") is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _HOME["old"]
    shutil.rmtree(_HOME["tmp"], ignore_errors=True)


EVIL = "<img src=x onerror=\"window.__xss=1\"><script>window.__xss=2</script>"


def fixture_state():
    """The geo fixture state plus a core and a shelf place (as a 105 run has)."""
    from tests.test_geo_sachdb_result import geo_state, payload

    state = geo_state(payload())
    state.working_title = "Testgestein und Lagerstätten"
    state.core_keywords = ["Lagerstätte", "Testgestein"]
    state.shelf = {"placement": {"notation": "GEO 410", "label": "Lagerstättenkunde",
                                 "location": "Freihand"}, "candidates": [], "unplaced": []}
    return state


def state_bytes(state, **extra) -> bytes:
    from src.utils.pipeline_persistence import PipelineJsonManager

    data = PipelineJsonManager.convert_sets_to_lists(dataclasses.asdict(state))
    data.update(extra)
    return json.dumps(data).encode("utf-8")


def export_bytes(state) -> bytes:
    from src.utils.pipeline_persistence import export_analysis_state_to_file

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        export_analysis_state_to_file(state, path, validate_rvk=False)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def _no_rvk_network(code):
    return {"status": "standard", "is_standard": True, "canonical_code": code,
            "label": None, "message": None}


def card_html(events, card):
    return "".join(e["html"] for e in events if e.get("card") == card)


def winibw_text(events):
    match = re.search(r'<pre class="alima-result__winibw">(.*?)</pre>',
                      card_html(events, "final_result"), re.S)
    return html.unescape(match.group(1)) if match else None


class ParseResultFile(unittest.TestCase):
    """``result_loading.parse_result_file`` / ``display_filename``."""

    def _reject(self, raw, status=422):
        from src.webapp.result_loading import ResultFileError, parse_result_file

        with self.assertRaises(ResultFileError) as ctx:
            parse_result_file(raw)
        self.assertEqual(ctx.exception.status, status)
        return ctx.exception.message

    def test_empty_file(self):
        self.assertIn("leer", self._reject(b"  \n"))

    def test_broken_json_names_the_place(self):
        self.assertIn("Zeile 2, Spalte 5", self._reject(b'{"a": 1,\n    x}'))

    def test_not_utf8(self):
        self.assertIn("UTF-8", self._reject(b"\xff\xfe\x00{"))

    def test_json_without_object(self):
        self.assertIn("kein JSON-Objekt", self._reject(b"[1, 2]"))

    def test_object_of_another_shape(self):
        self.assertIn("kein ALIMA-Ergebnis", self._reject(b'{"title": "x", "items": []}'))

    def test_deep_nesting(self):
        self.assertIn("verschachtelt", self._reject(b'{"a":' * 200000 + b"1" + b"}" * 200000))

    def test_size_limit(self):
        from src.webapp import result_loading

        with patch.object(result_loading, "MAX_RESULT_FILE_BYTES", 64):
            message = self._reject(b'{"results": {}}' + b" " * 100, status=413)
        self.assertIn("größer als", message)

    def test_both_formats_are_recognised(self):
        from src.webapp.result_loading import (
            FORMAT_ANALYSIS_STATE, FORMAT_WEBAPP_EXPORT, parse_result_file)

        state = fixture_state()
        self.assertEqual(parse_result_file(state_bytes(state))[1], FORMAT_ANALYSIS_STATE)
        self.assertEqual(parse_result_file(export_bytes(state))[1], FORMAT_WEBAPP_EXPORT)

    def test_display_filename_is_a_bare_name(self):
        from src.webapp.result_loading import display_filename

        self.assertEqual(display_filename("../../etc/passwd.json"), "passwd.json")
        self.assertEqual(display_filename("C:\\Users\\x\\erg.json"), "erg.json")
        self.assertEqual(display_filename("a\x00b\nc.json"), "abc.json")
        self.assertEqual(display_filename(""), "ergebnis.json")
        self.assertLessEqual(len(display_filename("x" * 500 + ".json")), 120)


class LoadEndpoint(unittest.TestCase):
    """``POST /api/session/{id}/load`` against the real app."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod

        self.appmod = appmod
        self.client = TestClient(appmod.app)
        self.sid = f"load-{self._testMethodName}"[:60]
        appmod.sessions.pop(self.sid, None)
        self._rvk = patch("src.webapp.result_serialization.validate_rvk_notation", _no_rvk_network)
        self._rvk.start()

    def tearDown(self):
        self._rvk.stop()
        self.appmod.sessions.pop(self.sid, None)

    def _load(self, raw, name="ergebnis.json"):
        return self.client.post(f"/api/session/{self.sid}/load",
                                files={"file": (name, raw, "application/json")})

    def test_both_formats_show_the_same_result_cards(self):
        from src.utils.winibw_record import build_winibw_record

        state = fixture_state()
        expected = "\n".join(build_winibw_record(state)["lines"])
        seen = {}
        for name, raw in (("zustand.json", state_bytes(state)), ("export.json", export_bytes(state))):
            resp = self._load(raw, name)
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            cards = [e["card"] for e in data["render_events"] if e.get("card")]
            self.assertEqual(cards, ["final_result", "dk_classifications"], name)
            self.assertEqual(winibw_text(data["render_events"]), expected, name)
            self.assertTrue(expected.startswith("5550 "))
            seen[name] = data["format"]
        self.assertEqual(seen, {"zustand.json": "analysis_state", "export.json": "webapp_export"})

    def test_export_keeps_the_core_rank_of_the_notations(self):
        # The shared loader keeps only the flat code strings of an export; the
        # structured entries carry "core", which the DK card shows as "Kern".
        resp = self._load(export_bytes(fixture_state()), "export.json")
        dk = card_html(resp.json()["render_events"], "dk_classifications")
        self.assertIn(">Kern</span>", dk)

    def test_the_log_names_the_file_and_claims_no_run(self):
        events = self._load(state_bytes(fixture_state()), "mein ergebnis.json").json()["render_events"]
        log = "".join(e["html"] for e in events if not e.get("card"))
        self.assertIn("mein ergebnis.json", log)
        self.assertNotIn("Pipeline vollständig abgeschlossen", log)

    def test_session_after_loading(self):
        resp = self._load(state_bytes(fixture_state()), "../../tmp/zustand.json")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["filename"], "zustand.json")
        session = self.appmod.sessions[self.sid]
        self.assertEqual((session.status, session.loaded_from), ("completed", "zustand.json"))
        self.assertEqual(session.working_title, "Testgestein und Lagerstätten")
        self.assertEqual([e["seq"] for e in session.render_buffer], list(range(len(session.render_buffer))))

        poll = self.client.get(f"/api/session/{self.sid}").json()
        self.assertEqual(poll["loaded_from"], "zustand.json")
        self.assertNotIn("results", poll)  # full results stay out of the poll, as for a run

        exported = self.client.get(f"/api/export/{self.sid}")
        self.assertEqual(exported.status_code, 200)
        payload = exported.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["input"]["loaded_from"], "zustand.json")
        self.assertTrue(payload["results"]["winibw"]["lines"][0].startswith("5550 "))

    def test_a_reconnecting_client_gets_the_same_events(self):
        answered = self._load(state_bytes(fixture_state())).json()["render_events"]
        with self.client.websocket_connect(f"/ws/{self.sid}") as ws:
            msg = ws.receive_json()
        self.assertEqual(msg["type"], "complete")
        self.assertEqual(msg["render_events"], answered)
        self.assertIn("final_result", [e.get("card") for e in msg["render_events"]])
        self.assertEqual(msg["results"]["working_title"], "Testgestein und Lagerstätten")

    def test_foreign_text_is_escaped_everywhere(self):
        state = fixture_state()
        state.working_title = "Titel " + EVIL
        state.core_keywords = ["Kern " + EVIL]
        state.final_llm_analysis.extracted_gnd_keywords = ["GND " + EVIL]
        state.keyword_chains = [{"chain": ["Kette " + EVIL, "Glied"]}]
        state.rule_output = "5550 </pre>" + EVIL
        state.report_markdown = "Bericht " + EVIL + " [x](javascript:window.__xss=3)"
        state.shelf = {"placement": {"notation": "GEO " + EVIL, "label": 'Etikett " ' + EVIL,
                                     "location": EVIL, "selected_reason": EVIL}}
        resp = self._load(state_bytes(state), EVIL + ".json")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        everything = "".join(e.get("html", "") for e in data["render_events"]).lower()
        self.assertNotIn("<img", everything)
        self.assertNotIn("<script", everything)
        self.assertNotIn('href="javascript', everything)
        self.assertIn("&lt;img src=x onerror=", everything)
        # The file name reaches the page only via textContent; it carries no path.
        self.assertNotIn("/", data["filename"])

    def test_no_path_named_in_the_file_is_opened(self):
        state = fixture_state()
        state.source_value = "/nonexistent/alima-fremd/quelle.pdf"
        geo = dict(state.geo_sachdb)
        geo["source"] = {"source": "/nonexistent/alima-fremd/geo.db", "path": "/nonexistent/alima-fremd/geo.db"}
        state.geo_sachdb = geo
        raw = state_bytes(state, extra_path="/nonexistent/alima-fremd/x.json")
        opened = []
        real_open, real_io_open = builtins.open, io.open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", spy), patch("io.open", spy):
            resp = self._load(raw, "../../nonexistent/alima-fremd/datei.json")
            self.client.get(f"/api/export/{self.sid}")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual([p for p in opened if "alima-fremd" in p], [])
        self.assertIs(builtins.open, real_open)
        self.assertIs(io.open, real_io_open)

    def test_unknown_fields_of_a_state_file_are_named_not_fatal(self):
        resp = self._load(state_bytes(fixture_state(), zukunftsfeld=1))
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["ignored_fields"], ["zukunftsfeld"])

    def test_a_state_without_any_result_is_refused(self):
        from src.core.data_models import KeywordAnalysisState

        empty = KeywordAnalysisState(original_abstract="Text", initial_keywords=[],
                                     search_suggesters_used=[])
        resp = self._load(state_bytes(empty))
        self.assertEqual(resp.status_code, 422)
        self.assertIn("kein Erschließungsergebnis", resp.json()["detail"])

    def test_values_the_cards_cannot_show_leave_the_session_alone(self):
        session = self.appmod.Session(self.sid)
        session.append_render_event({"type": "block", "html": "vorher"})
        self.appmod.sessions[self.sid] = session
        state = fixture_state()
        state.dk_statistics = {"most_frequent": [{"dk": "1", "count": "viele"}]}
        resp = self._load(state_bytes(state))
        self.assertEqual(resp.status_code, 422)
        self.assertIn("nicht anzeigen", resp.json()["detail"])
        self.assertEqual([e["html"] for e in session.render_buffer], ["vorher"])

    def test_a_structure_the_loader_cannot_take_is_a_clear_message(self):
        raw = state_bytes(fixture_state(), final_llm_analysis=["keine", "Analyse"])
        resp = self._load(raw)
        self.assertEqual(resp.status_code, 422)
        self.assertIn("passt nicht zum Analysezustand", resp.json()["detail"])

    def test_broken_file_over_http(self):
        resp = self._load(b"{ kaputt")
        self.assertEqual(resp.status_code, 422)
        self.assertIn("kein gültiges JSON", resp.json()["detail"])

    def test_without_a_file(self):
        resp = self.client.post(f"/api/session/{self.sid}/load")
        self.assertEqual(resp.status_code, 400)

    def test_not_while_running(self):
        session = self.appmod.Session(self.sid)
        session.status = "running"
        self.appmod.sessions[self.sid] = session
        with patch("src.webapp.routers.load.load_result_file") as loader:
            resp = self._load(state_bytes(fixture_state()))
        self.assertEqual(resp.status_code, 409)
        loader.assert_not_called()  # refused before the file is even read
        self.assertEqual(session.render_buffer, [])

    def test_a_run_started_while_reading_wins(self):
        from src.webapp import result_loading

        session = self.appmod.Session(self.sid)
        self.appmod.sessions[self.sid] = session
        real = result_loading.load_result_file

        def slow_load(raw, name):
            loaded = real(raw, name)
            session.status = "running"  # an analysis started meanwhile
            return loaded

        with patch("src.webapp.routers.load.load_result_file", slow_load):
            resp = self._load(state_bytes(fixture_state()))
        self.assertEqual(resp.status_code, 409)
        self.assertIsNone(session.loaded_from)
        self.assertEqual(session.render_buffer, [])

    def test_a_new_run_replaces_the_loaded_result(self):
        self._load(state_bytes(fixture_state()))
        session = self.appmod.sessions[self.sid]
        self.assertTrue(session.render_buffer)

        async def _no_run(*_args, **_kwargs):
            return None

        with patch("src.webapp.routers.analysis.run_analysis", _no_run):
            resp = self.client.post(f"/api/analyze/{self.sid}",
                                    data={"input_type": "text", "content": "Neuer Text"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(session.render_buffer, [])
        self.assertEqual(session.results, {})
        self.assertIsNone(session.loaded_from)
        self.assertIsNone(session.current_analysis_state)
        self.assertEqual(session.status, "running")


class StateFromDict(unittest.TestCase):
    """``analysis_state_from_dict`` is the file loader without the file."""

    def test_same_state_as_the_file_loader(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        state = fixture_state()
        for raw in (state_bytes(state), export_bytes(state)):
            fd, path = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                with open(path, "wb") as fh:
                    fh.write(raw)
                from_file = PipelineJsonManager.load_analysis_state(path)
            finally:
                os.unlink(path)
            from_dict = PipelineJsonManager.analysis_state_from_dict(json.loads(raw))
            a, b = dataclasses.asdict(from_file), dataclasses.asdict(from_dict)
            a.pop("timestamp"), b.pop("timestamp")
            self.assertEqual(a, b)

    def test_file_loader_still_wraps_errors(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"original_abstract": "x", "initial_keywords": [], '
                         '"search_suggesters_used": [], "kein_feld": 1}')
            with self.assertRaises(ValueError) as ctx:
                PipelineJsonManager.load_analysis_state(path)
            self.assertIn("incompatible with KeywordAnalysisState", str(ctx.exception))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
