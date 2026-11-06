"""geo_sachdb im Ergebnis: Zustand, Anzeige, Speichern und Laden - Claude Generated.

Der Schritt ``geo_candidates`` legte seine Zeilen nur in ``extra.geo_sachdb``
ab; kein Ergebnis, keine Anzeige und keine Ergebnisdatei trug sie. Hier wird
die Übernahme nach dem Muster der Aufstellung (``shelf``) festgehalten: das
Feld in ``KeywordAnalysisState``, die Abschlusskarte (GUI-Log und Webapp über
``render_pipeline_result``), die CLI-Protokollausgabe, der Webapp-Export und das
Laden alter Dateien ohne das Feld. Dazu die Nummerierung der lokalen
Schlagwörter nach den Ketten des Ergebnisses (Obergrenze 5599), die Liste, was
jede Zeile bedeutet, und Ergebnisse aus dem Stand vor dieser Liste. Die
Nutzlast kommt aus der echten Funktion gegen die synthetische Datenbank der
Plugin-Tests. Daten erfunden.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from src.core.agents import shared_context as shared_context_mod
from src.core.agents.deterministic_functions import geo_sachdb_candidates
from src.core.agents.shared_context import SharedContext
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup

_spec = importlib.util.spec_from_file_location(
    "geo_lookup_fixture_result", Path(__file__).resolve().parent / "test_geo_sachdb_lookup.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
_HOME = {}

KEYWORDS = [{"keyword": "Testgestein", "gnd_id": "x"}, {"keyword": "Lagerstätte"},
            {"keyword": "Gestein"}, {"keyword": "Unbekanntwort"}]
NOTATIONS = [{"code": "DK 999.1", "rank": "core"}, {"code": "DK 888.8", "rank": "additional"}]
#: What the plugin function emits: every heading line as 5550, it does not know the chains.
PLUGIN_LINES = ["5550 !111111111!", "5550 !333333333!", "5550 |s|Gestein",
                "6700 DK 999.21", "6700 !555555555!"]
NOTATION_LINES = ["6700 DK 999.21", "6700 !555555555!"]
HEADING_VALUES = ["!111111111!", "!333333333!", "|s|Gestein"]


def _state_lines(first: int) -> list:
    """The lines a state holds when its local headings start at ``first``."""
    return [f"{first + i} {value}" for i, value in enumerate(HEADING_VALUES)] + NOTATION_LINES


#: The lines of a result without chains: the headings count up from 5550.
EXPECTED_LINES = _state_lines(5550)


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


def _payload() -> dict:
    """What ``geo_candidates`` puts into ``extra.geo_sachdb`` for the fixture."""
    tmp = tempfile.mkdtemp()
    try:
        db = Path(tmp) / "geo.db"
        _fixture.build_synthetic_db(db)
        lookup = GeoSachdbLookup(**_fixture.settings(db))
        with patch("src.utils.lookups.build_lookup", return_value=lookup):
            return geo_sachdb_candidates(keywords=KEYWORDS, notations=NOTATIONS, config={})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _chains(count: int) -> list:
    return [{"chain": [f"Kette{i}", "Glied"], "reason": ""} for i in range(count)]


def _state(payload=None, *, dk=("DK 999.1",), chains=0):
    context = SharedContext()
    context.abstract = "Testgestein in einer Lagerstätte."
    if payload is not None:
        context.extra["geo_sachdb"] = payload
    context.keyword_chains = chains if isinstance(chains, list) else _chains(chains)
    state = context.to_keyword_analysis_state()
    state.dk_classifications = list(dk)
    return state


def _old_geo(payload: dict) -> dict:
    """``geo_sachdb`` as results saved before ``entries`` existed carried it."""
    return {
        "lines": list(payload["lines"]),
        "not_found": list(payload["not_found"]),
        "candidates": [dict(c) for c in payload["candidates"]],
        "source": {k: payload["source"][k] for k in ("source", "source_date", "source_sha256",
                                                     "schema_version") if payload["source"].get(k)},
    }


def _rendered(state) -> str:
    from src.utils.pipeline_formatters import render_pipeline_result

    out = []

    class Renderer:
        def render_pipeline_log(self, text, kind="info"):
            out.append(text)

        def render_html_block(self, html, *, kind=None, plain_text=""):
            out.append(html)

        def render_markdown_block(self, markdown_text, *, kind=None):
            out.append(markdown_text)

    render_pipeline_result(Renderer(), state)
    return "\n".join(out)


def _cli(state) -> list:
    from src.cli.formatters.protocol_formatters import format_step_dk_classification

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        format_step_dk_classification(state)
    return buffer.getvalue().splitlines()


class _FirstByClass(HTMLParser):
    """Text of the first element carrying ``css_class``, like querySelector;
    ``items`` the text of each ``<li>`` inside it."""

    def __init__(self, css_class: str):
        super().__init__(convert_charrefs=True)
        self.css_class = css_class
        self.depth = 0
        self.done = False
        self.text = None
        self.items = []
        self._in_li = False

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if self.depth:
            self.depth += 1
            if tag == "li":
                self.items.append("")
                self._in_li = True
            return
        classes = (dict(attrs).get("class") or "").split()
        if self.css_class in classes:
            self.depth = 1
            self.text = ""

    def handle_endtag(self, tag):
        if self.depth and not self.done:
            if tag == "li":
                self._in_li = False
            self.depth -= 1
            if self.depth == 0:
                self.done = True

    def handle_data(self, data):
        if self.depth and not self.done:
            self.text += data
            if self._in_li:
                self.items[-1] += data


def _first_text(html: str, css_class: str):
    parser = _FirstByClass(css_class)
    parser.feed(html)
    return parser.text


def _list_items(html: str, css_class: str = "alima-result__geo-entries") -> list:
    parser = _FirstByClass(css_class)
    parser.feed(html)
    return [" ".join(item.split()) for item in parser.items]


class GeoPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = _payload()

    def test_fixture_payload_is_what_the_step_delivers(self):
        # Guards the fixture itself: the tests below rely on these values. The
        # plugin function stays deterministic: every heading line is 5550.
        self.assertEqual(self.payload["status"], "ok")
        self.assertEqual(self.payload["lines"], PLUGIN_LINES)
        self.assertEqual(self.payload["not_found"], ["Unbekanntwort"])


class GeoReachesTheState(GeoPayload):
    def test_the_context_carries_lines_misses_and_candidates(self):
        geo = _state(self.payload).geo_sachdb
        self.assertEqual(geo["lines"], EXPECTED_LINES)
        self.assertEqual(geo["not_found"], ["Unbekanntwort"])
        by_code = {c["code"]: c["assigned"] for c in geo["candidates"]}
        self.assertEqual(by_code, {"999.1": True, "999.21": False})

    def test_the_source_keeps_the_version_not_the_local_path(self):
        self.assertIn("db_path", self.payload["source"])
        source = _state(self.payload).geo_sachdb["source"]
        self.assertNotIn("db_path", source)
        self.assertNotIn("rows", source)
        self.assertEqual(source.get("schema_version"), self.payload["source"]["schema_version"])

    def test_without_the_step_the_field_stays_empty(self):
        self.assertEqual(_state(None).geo_sachdb, {})

    def test_an_empty_step_result_leaves_the_field_empty(self):
        for status in ("disabled", "unavailable", "no_input", "error"):
            empty = {**self.payload, "status": status, "lines": [], "not_found": [],
                     "candidates": [], "keyword_matches": []}
            self.assertEqual(_state(empty).geo_sachdb, {}, status)
        self.assertEqual(_state("kaputt").geo_sachdb, {})

    def test_only_misses_still_count_as_a_result(self):
        geo = _state({"status": "ok", "lines": [], "not_found": ["Unbekanntwort"]}).geo_sachdb
        self.assertEqual((geo["lines"], geo["not_found"]), ([], ["Unbekanntwort"]))


class GeoNumbering(GeoPayload):
    """Local headings are single headings, numbered on after the result's chains."""

    def test_without_chains_the_headings_count_up_from_5550(self):
        geo = _state(self.payload, chains=0).geo_sachdb
        self.assertEqual(geo["lines"], _state_lines(5550))
        self.assertEqual(geo["exhausted"], [])

    def test_four_chains_put_the_first_heading_at_5554(self):
        geo = _state(self.payload, chains=4).geo_sachdb
        self.assertEqual(geo["lines"], _state_lines(5554))
        self.assertEqual(geo["numbering"], {"chains": 4, "first_tag": 5554, "last_tag": 5599})

    def test_chains_without_terms_do_not_count(self):
        chains = _chains(2) + [{"chain": [], "reason": "leer"}, {"chain": ["", ""]}, {"reason": "ohne"}]
        self.assertEqual(_state(self.payload, chains=chains).geo_sachdb["lines"], _state_lines(5552))

    def test_6700_lines_are_never_numbered(self):
        for count in (0, 4, 47, 60):
            lines = _state(self.payload, chains=count).geo_sachdb["lines"]
            self.assertEqual([l for l in lines if not l.startswith("55")], NOTATION_LINES, count)

    def test_the_numbers_may_reach_5599(self):
        geo = _state(self.payload, chains=47).geo_sachdb
        self.assertEqual(geo["lines"], _state_lines(5597))
        self.assertEqual(geo["exhausted"], [])

    def test_a_heading_beyond_5599_gets_no_line_and_is_named(self):
        geo = _state(self.payload, chains=48).geo_sachdb
        self.assertEqual(geo["lines"], ["5598 !111111111!", "5599 !333333333!"] + NOTATION_LINES)
        self.assertEqual(geo["exhausted"],
                         [{"type": "heading", "label": "Gestein", "kind": "Sachschlagwort",
                           "value": "|s|Gestein"}])
        self.assertFalse(any(l.startswith("56") for l in geo["lines"]))

    def test_with_many_chains_only_the_6700_lines_remain(self):
        geo = _state(self.payload, chains=60).geo_sachdb
        self.assertEqual(geo["lines"], NOTATION_LINES)
        self.assertEqual([e["label"] for e in geo["exhausted"]], ["Testgestein", "Lagerstätte", "Gestein"])

    def test_the_start_number_comes_from_one_function(self):
        # The numbering assumption lives in ``_geo_first_heading_tag`` alone:
        # changing it there changes the result.
        with patch.object(shared_context_mod, "_geo_first_heading_tag", return_value=5570):
            lines = _state(self.payload, chains=4).geo_sachdb["lines"]
        self.assertEqual(lines, _state_lines(5570))
        self.assertEqual(shared_context_mod._geo_first_heading_tag(4), 5554)


class GeoEntries(GeoPayload):
    def test_each_line_carries_what_it_stands_for(self):
        geo = _state(self.payload, chains=4).geo_sachdb
        self.assertEqual([e["line"] for e in geo["entries"]], geo["lines"])
        self.assertEqual(geo["entries"][:3], [
            {"type": "heading", "label": "Testgestein", "kind": "Sachschlagwort", "line": "5554 !111111111!"},
            {"type": "heading", "label": "Lagerstätte", "kind": "Sachschlagwort", "line": "5555 !333333333!"},
            {"type": "heading", "label": "Gestein", "kind": "Sachschlagwort", "line": "5556 |s|Gestein"},
        ])
        self.assertEqual(geo["entries"][3:], [
            {"type": "notation", "line": "6700 DK 999.21", "system": "DK", "notation": "DK 999.21",
             "label": "Lagerstaetten (Test)", "assigned": False},
            {"type": "notation", "line": "6700 !555555555!", "system": "DK", "notation": "DK 999.1",
             "label": "Testgesteine", "assigned": True},
        ])

    def test_a_payload_without_records_still_numbers_its_lines(self):
        geo = _state({"status": "ok", "lines": ["5550 !123456789!"], "not_found": []}, chains=1).geo_sachdb
        self.assertEqual(geo["lines"], ["5551 !123456789!"])
        self.assertEqual(geo["entries"], [{"type": "heading", "label": "", "kind": "", "line": "5551 !123456789!"}])


class GeoIsShown(GeoPayload):
    TITLE = "Lokale Sacherschließung (geo_sachdb)"
    NOT_GIVEN_ROW = "6700, im Lauf nicht vergeben"

    def test_the_card_shows_the_lines_one_per_line(self):
        html = _rendered(_state(self.payload))
        self.assertIn(self.TITLE, html)
        self.assertEqual(_first_text(html, "alima-result__geo-lines"), "\n".join(EXPECTED_LINES))

    def test_the_card_lists_what_each_line_stands_for(self):
        html = _rendered(_state(self.payload, chains=4))
        self.assertEqual(_list_items(html), [
            "5554 !111111111! Testgestein Sachschlagwort",
            "5555 !333333333! Lagerstätte Sachschlagwort",
            "5556 |s|Gestein Gestein Sachschlagwort",
            "6700 DK 999.21 DK 999.21 Lagerstaetten (Test) im Lauf nicht vergeben",
            "6700 !555555555! DK 999.1 Testgesteine",
        ])
        self.assertLess(html.index("alima-result__geo-lines"), html.index("alima-result__geo-entries"))

    def test_the_copy_block_holds_only_the_lines(self):
        html = _rendered(_state(self.payload, chains=4))
        copied = _first_text(html, "alima-result__geo-lines")
        self.assertEqual(copied, "\n".join(_state_lines(5554)))
        for word in ("Sachschlagwort", "Testgesteine", "nicht vergeben", "Unbekanntwort"):
            self.assertNotIn(word, copied)

    def test_the_misses_follow_the_lines(self):
        html = _rendered(_state(self.payload))
        self.assertIn("Ohne lokales Schlagwort", html)
        self.assertLess(html.index("alima-result__geo-lines"), html.index("Unbekanntwort"))

    def test_notations_the_run_did_not_give_are_marked_in_the_list(self):
        html = _rendered(_state(self.payload))
        items = _list_items(html)
        self.assertTrue(items[3].startswith("6700 DK 999.21") and items[3].endswith("im Lauf nicht vergeben"))
        self.assertNotIn("nicht vergeben", items[4])
        # With the list there is no separate row repeating them.
        self.assertNotIn(self.NOT_GIVEN_ROW, html)

    def test_exhausted_numbers_are_named_on_the_card(self):
        html = _rendered(_state(self.payload, chains=48))
        self.assertIn("Ohne Feldnummer (5599 erreicht)", html)
        start = html.index("Ohne Feldnummer (5599 erreicht)")
        self.assertIn("|s|Gestein = Gestein · Sachschlagwort", html[start:start + 300])
        self.assertNotIn("|s|Gestein", _first_text(html, "alima-result__geo-lines"))

    def test_only_exhausted_headings_still_yield_a_card(self):
        payload = {"status": "ok", "lines": ["5550 !123456789!"], "not_found": [],
                   "keyword_matches": [{"keyword": "Beispiel", "records": [
                       {"label": "Beispiel", "kind": "Geographikum", "line": "5550 !123456789!"}]}]}
        state = _state(payload, dk=(), chains=50)
        self.assertEqual(state.geo_sachdb["lines"], [])
        html = _rendered(state)
        self.assertIn(self.TITLE, html)
        self.assertIn("!123456789! = Beispiel · Geographikum", html)
        self.assertNotIn("alima-result__geo-lines", html)
        out = _cli(state)
        self.assertIn(self.TITLE + ":", out)
        self.assertIn("  ohne feldnummer (5599 erreicht): !123456789! = Beispiel · Geographikum", out)

    def test_the_lines_have_their_own_copy_button(self):
        html = _rendered(_state(self.payload))
        self.assertIn('data-alima-copy=".alima-result__geo-lines"', html)

    def test_the_rule_snippet_button_still_copies_the_rule(self):
        # The rule button selects the first ``.alima-result__pre`` of the card;
        # the geo block sits before it and must not be that element.
        state = _state(self.payload)
        state.rule_output = "5553 Regelzeile"
        html = _rendered(state)
        self.assertEqual(_first_text(html, "alima-result__pre"), "5553 Regelzeile")
        self.assertEqual(_first_text(html, "alima-result__geo-lines"), "\n".join(EXPECTED_LINES))

    def test_the_plain_text_carries_the_lines_verbatim(self):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        _html, plain = PipelineResultFormatter.format_final_result_card_html(_state(self.payload))
        plain_lines = plain.splitlines()
        start = plain_lines.index(self.TITLE + ":")
        self.assertEqual(plain_lines[start + 1:start + 1 + len(EXPECTED_LINES)], EXPECTED_LINES)
        self.assertIn("  5550 !111111111! = Testgestein · Sachschlagwort", plain_lines)
        self.assertIn("  6700 DK 999.21 = DK 999.21 Lagerstaetten (Test) · im Lauf nicht vergeben",
                      plain_lines)
        self.assertIn("Ohne lokales Schlagwort: Unbekanntwort", plain_lines)

    def test_only_misses_yield_a_card_without_a_copy_button(self):
        state = _state({"status": "ok", "lines": [], "not_found": ["Unbekanntwort"]}, dk=())
        html = _rendered(state)
        self.assertIn(self.TITLE, html)
        self.assertIn("Unbekanntwort", html)
        self.assertNotIn("alima-result__geo-lines", html)
        self.assertNotIn("alima-result__geo-entries", html)

    def test_without_data_nothing_is_shown(self):
        html = _rendered(_state(None))
        self.assertNotIn("geo_sachdb", html)
        self.assertNotIn("Ohne lokales Schlagwort", html)

    def test_the_cli_prints_the_lines_unindented(self):
        out = _cli(_state(self.payload))
        self.assertIn(self.TITLE + ":", out)
        start = out.index(self.TITLE + ":")
        self.assertEqual(out[start + 1:start + 1 + len(EXPECTED_LINES)], EXPECTED_LINES)
        self.assertIn("  ohne lokales Schlagwort: Unbekanntwort", out)

    def test_the_cli_lists_what_each_line_stands_for(self):
        out = _cli(_state(self.payload, chains=4))
        start = out.index(self.TITLE + ":") + 1 + len(EXPECTED_LINES)
        self.assertEqual(out[start:start + 5], [
            "  5554 !111111111! = Testgestein · Sachschlagwort",
            "  5555 !333333333! = Lagerstätte · Sachschlagwort",
            "  5556 |s|Gestein = Gestein · Sachschlagwort",
            "  6700 DK 999.21 = DK 999.21 Lagerstaetten (Test) · im Lauf nicht vergeben",
            "  6700 !555555555! = DK 999.1 Testgesteine",
        ])
        self.assertFalse(any(line.startswith("  " + self.NOT_GIVEN_ROW) for line in out))

    def test_the_cli_names_exhausted_numbers(self):
        out = _cli(_state(self.payload, chains=48))
        self.assertIn("  ohne feldnummer (5599 erreicht): |s|Gestein = Gestein · Sachschlagwort", out)
        self.assertNotIn("5600 |s|Gestein", out)

    def test_the_cli_prints_them_without_classifications_too(self):
        self.assertIn("5550 !111111111!", _cli(_state(self.payload, dk=())))

    def test_the_cli_prints_nothing_without_data(self):
        for dk in (("DK 999.1",), ()):
            self.assertNotIn("geo_sachdb", "\n".join(_cli(_state(None, dk=dk))))

    def test_the_webapp_payload_carries_it(self):
        from src.webapp.result_serialization import extract_results_from_analysis_state

        payload = extract_results_from_analysis_state(_state(self.payload, chains=4))
        self.assertEqual(payload["geo_sachdb"]["lines"], _state_lines(5554))
        self.assertEqual(payload["geo_sachdb"]["entries"][0]["label"], "Testgestein")
        json.dumps(payload["geo_sachdb"])
        self.assertEqual(extract_results_from_analysis_state(_state(None))["geo_sachdb"], {})


class GeoFromBeforeTheEntries(GeoPayload):
    """Results saved before ``entries`` existed: lines all 5550, no list."""

    def _old_state(self):
        state = _state(None)
        state.geo_sachdb = _old_geo(self.payload)
        return state

    def test_the_card_shows_the_stored_lines_and_the_old_row(self):
        html = _rendered(self._old_state())
        self.assertEqual(_first_text(html, "alima-result__geo-lines"), "\n".join(PLUGIN_LINES))
        self.assertNotIn("alima-result__geo-entries", html)
        start = html.index("6700, im Lauf nicht vergeben")
        self.assertIn("DK 999.21", html[start:start + 200])
        self.assertNotIn("DK 999.1<", html[start:start + 200])

    def test_the_cli_shows_the_stored_lines_and_the_old_row(self):
        out = _cli(self._old_state())
        start = out.index("Lokale Sacherschließung (geo_sachdb):")
        self.assertEqual(out[start + 1:start + 1 + len(PLUGIN_LINES)], PLUGIN_LINES)
        self.assertIn("  6700, im Lauf nicht vergeben: DK 999.21", out)
        self.assertFalse(any(" = " in line for line in out))


class GeoIsSavedAndLoaded(GeoPayload):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "state.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        return PipelineJsonManager.load_analysis_state(self.path)

    def test_state_file_roundtrip(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        state = _state(self.payload, chains=4)
        PipelineJsonManager.save_analysis_state(state, self.path)
        self.assertEqual(self._load().geo_sachdb, state.geo_sachdb)

    def test_export_file_roundtrip(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        state = _state(self.payload, chains=4)
        export_analysis_state_to_file(state, self.path, validate_rvk=False)
        self.assertEqual(self._load().geo_sachdb, state.geo_sachdb)

    def test_the_cli_protocol_shows_a_saved_result(self):
        from src.cli.formatters.protocol_formatters import display_protocol
        from src.utils.pipeline_persistence import PipelineJsonManager

        PipelineJsonManager.save_analysis_state(_state(self.payload, chains=4), self.path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            display_protocol(self.path, ["dk_classification"])
        out = buffer.getvalue().splitlines()
        self.assertIn("6700 !555555555!", out)
        self.assertIn("5554 !111111111!", out)
        self.assertIn("  5554 !111111111! = Testgestein · Sachschlagwort", out)

    def test_an_old_state_file_without_the_field_loads(self):
        data = dataclasses.asdict(_state(None))
        data.pop("geo_sachdb")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.assertEqual(self._load().geo_sachdb, {})

    def test_an_old_export_file_without_the_field_loads(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(_state(None), self.path, validate_rvk=False)
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["results"].pop("geo_sachdb")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, {})
        self.assertEqual(_rendered(state).count("geo_sachdb"), 0)

    def test_a_state_file_from_before_the_entries_loads_and_renders(self):
        data = dataclasses.asdict(_state(None))
        data["geo_sachdb"] = _old_geo(self.payload)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, _old_geo(self.payload))
        html = _rendered(state)
        self.assertEqual(_first_text(html, "alima-result__geo-lines"), "\n".join(PLUGIN_LINES))
        self.assertIn("6700, im Lauf nicht vergeben", html)

    def test_an_export_file_from_before_the_entries_loads_and_renders(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(_state(None), self.path, validate_rvk=False)
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["results"]["geo_sachdb"] = _old_geo(self.payload)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, _old_geo(self.payload))
        out = _cli(state)
        self.assertIn("5550 !333333333!", out)
        self.assertIn("  6700, im Lauf nicht vergeben: DK 999.21", out)


if __name__ == "__main__":
    unittest.main()
