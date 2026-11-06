"""geo_sachdb im Ergebnis: Zustand, WinIBW-Eintrag, Anzeige, Speichern und Laden - Claude Generated.

Der Schritt ``geo_candidates`` legt seine Treffer in ``extra.geo_sachdb`` ab;
``SharedContext.to_keyword_analysis_state`` übernimmt daraus je Schlagwort die
lokalen Sätze und die verwendbare PPN (``terms``) und je vergebener Notation die
lokalen Einträge und die verwendbare PPN (``notations``). Der WinIBW-Eintrag
(``src/utils/winibw_record.py``) setzt eine eindeutige PPN statt des Textes ein;
eine eigene Nummerierung der lokalen Schlagwörter gibt es nicht mehr. Geprüft
werden: Eindeutigkeit (ein Satz, oder genau ein Sachschlagwort), Mehrdeutigkeit
(zwei Sätze „Geschichte“), Satz ohne PPN, kein Treffer, DK lokal mit einer PPN,
mit zwei PPN, ohne PPN, nicht lokal; der Geo-Block der Abschlusskarte und der
CLI (Treffer, weitere Vorschläge mit eigenem Kopierblock, ohne lokales
Schlagwort); Speichern/Laden und Dateien aus dem Stand vor ``terms``. Die
Nutzlast kommt aus der echten Funktion gegen eine synthetische Datenbank.
Daten erfunden.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from src.core.agents.deterministic_functions import geo_sachdb_candidates
from src.core.agents.shared_context import SharedContext
from src.utils.lookups.geo_sachdb import schema as schema_mod
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup

_spec = importlib.util.spec_from_file_location(
    "geo_lookup_fixture_result", Path(__file__).resolve().parent / "test_geo_sachdb_lookup.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
_HOME = {}

#: Zusätzlich zur Datenbank der Plugin-Tests: zwei Sachschlagwörter
#: „Geschichte“ (mehrdeutig) und „Bergbau“ einmal als Sachschlagwort, einmal als
#: Geographikum (eindeutig über die Art).
EXTRA_HEADINGS = [
    (8, "Geschichte", "Sachschlagwort", "100000001", "|s|Geschichte !100000001!", None),
    (9, "Geschichte", "Sachschlagwort", "100000002", "|s|Geschichte !100000002!", None),
    (10, "Bergbau", "Sachschlagwort", "200000001", "|s|Bergbau !200000001!", None),
    (11, "Bergbau", "Geographikum", "200000002", "|g|Bergbau !200000002!", None),
]

KEYWORDS = [{"keyword": "Testgestein", "gnd_id": "x"}, {"keyword": "Lagerstätte"},
            {"keyword": "Gestein"}, {"keyword": "Geschichte"}, {"keyword": "Bergbau"},
            {"keyword": "Unbekanntwort"}]
#: DK 999.1: zwei lokale Einträge mit verschiedener PPN; 999.3: eine PPN;
#: 999: lokal ohne PPN; 888.8: nicht lokal; RVK prüft der Schritt (scheme DK) nicht.
NOTATIONS = [{"code": "DK 999.1", "rank": "core"}, {"code": "DK 999.3"}, {"code": "DK 999"},
             {"code": "DK 888.8"}, {"code": "QX 200", "type": "RVK"}]
CHAINS = [{"chain": ["Lagerstätte", "Geschichte"], "reason": ""}]

#: Der WinIBW-Eintrag dieses Zustands.
EXPECTED_RECORD = [
    "5550 !333333333!",
    "5550 Geschichte",
    "5550 $ADE-105",
    "",
    "5551 !111111111!",
    "5552 Gestein",
    "5553 !200000001!",
    "5554 Unbekanntwort",
    "",
    "6700 DK 999.1",
    "6700 !666666666!",
    "6700 DK 999",
    "6700 DK 888.8",
    "6700 QX 200",
]
#: Derselbe Zustand ohne Geo-Daten: alles als Text.
EXPECTED_RECORD_WITHOUT_GEO = [
    "5550 Lagerstätte", "5550 Geschichte", "5550 $ADE-105", "",
    "5551 Testgestein", "5552 Gestein", "5553 Bergbau", "5554 Unbekanntwort", "",
    "6700 DK 999.1", "6700 DK 999.3", "6700 DK 999", "6700 DK 888.8", "6700 QX 200",
]
PROPOSAL_LINES = ["6700 DK 999.21"]


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


def build_db(path: Path) -> None:
    """Die Datenbank der Plugin-Tests plus ``EXTRA_HEADINGS``."""
    _fixture.build_synthetic_db(path)
    conn = sqlite3.connect(path)
    conn.executemany("INSERT INTO heading VALUES (?,?,?,?,?,?)", EXTRA_HEADINGS)
    conn.commit()
    conn.close()


def payload(keywords=KEYWORDS, notations=NOTATIONS) -> dict:
    """What ``geo_candidates`` puts into ``extra.geo_sachdb``."""
    tmp = tempfile.mkdtemp()
    try:
        db = Path(tmp) / "geo.db"
        build_db(db)
        lookup = GeoSachdbLookup(**_fixture.settings(db))
        with patch("src.utils.lookups.build_lookup", return_value=lookup):
            return geo_sachdb_candidates(keywords=keywords, notations=notations, config={})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def geo_state(geo_payload=None, *, keywords=KEYWORDS, notations=NOTATIONS, chains=CHAINS,
              form_keywords=()):
    """A result state as the agentic path builds it."""
    context = SharedContext()
    context.abstract = "Testgestein in einer Lagerstätte."
    context.working_title = "Testtitel"
    context.extra["final_keywords"] = [dict(k) for k in keywords]
    if form_keywords:
        context.extra["form_keywords"] = [{"keyword": k} for k in form_keywords]
    if geo_payload is not None:
        context.extra["geo_sachdb"] = geo_payload
    context.keyword_chains = [dict(c) for c in chains]
    context.dk_classifications = [dict(n) for n in notations]
    return context.to_keyword_analysis_state()


def _old_geo(geo_payload: dict) -> dict:
    """``geo_sachdb`` as results saved before ``terms``/``notations`` carried it
    (Nov 6: lines numbered on from the chains, entries, numbering)."""
    return {
        "lines": ["5551 !111111111!", "5552 !333333333!", "6700 DK 999.21"],
        "entries": [{"type": "heading", "label": "Testgestein", "kind": "Sachschlagwort",
                     "line": "5551 !111111111!"}],
        "exhausted": [],
        "numbering": {"chains": 1, "first_tag": 5551, "last_tag": 5599},
        "not_found": list(geo_payload["not_found"]),
        "candidates": [dict(c) for c in geo_payload["candidates"]],
        "source": {"source": "TEST.mdb"},
    }


def rendered(state) -> str:
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


def cli(state) -> list:
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


def first_text(html: str, css_class: str):
    parser = _FirstByClass(css_class)
    parser.feed(html)
    return parser.text


def list_items(html: str, css_class: str) -> list:
    parser = _FirstByClass(css_class)
    parser.feed(html)
    return [" ".join(item.split()) for item in parser.items]


class GeoPayload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = payload()

    def test_fixture_payload_is_what_the_step_delivers(self):
        # Guards the fixture itself: the tests below rely on these values.
        self.assertEqual(self.payload["status"], "ok")
        self.assertEqual([m["keyword"] for m in self.payload["keyword_matches"]],
                         ["Testgestein", "Lagerstätte", "Gestein", "Geschichte", "Bergbau"])
        self.assertEqual(self.payload["not_found"], ["Unbekanntwort"])
        self.assertEqual([m["code"] for m in self.payload["notation_matches"]],
                         ["DK 999.1", "DK 999.3", "DK 999", "DK 888.8"])
        self.assertTrue(all(m["system"] == "DK" for m in self.payload["notation_matches"]))


class GeoReachesTheState(GeoPayload):
    def test_terms_carry_the_local_headings_and_the_usable_ppn(self):
        terms = {t["keyword"]: t for t in geo_state(self.payload).geo_sachdb["terms"]}
        self.assertEqual(set(terms), {"Testgestein", "Lagerstätte", "Gestein", "Geschichte", "Bergbau"})
        self.assertEqual((terms["Testgestein"]["status"], terms["Testgestein"]["ppn"]), ("unique", "111111111"))
        self.assertEqual(terms["Testgestein"]["record"],
                         {"label": "Testgestein", "kind": "Sachschlagwort", "ppn": "111111111"})

    def test_two_local_headings_of_the_same_kind_are_ambiguous(self):
        term = {t["keyword"]: t for t in geo_state(self.payload).geo_sachdb["terms"]}["Geschichte"]
        self.assertEqual((term["status"], term["ppn"], term["record"]), ("ambiguous", "", None))
        self.assertEqual([r["ppn"] for r in term["records"]], ["100000001", "100000002"])

    def test_exactly_one_sachschlagwort_among_several_is_unique(self):
        term = {t["keyword"]: t for t in geo_state(self.payload).geo_sachdb["terms"]}["Bergbau"]
        self.assertEqual(len(term["records"]), 2)
        self.assertEqual((term["status"], term["ppn"], term["record"]["kind"]),
                         ("unique", "200000001", "Sachschlagwort"))

    def test_a_single_local_heading_of_another_kind_is_unique(self):
        only = [{"keyword": "Musterland"}]
        geo = geo_state(payload(keywords=only, notations=[]), keywords=only, notations=[], chains=[]).geo_sachdb
        term = geo["terms"][0]
        self.assertEqual((term["keyword"], term["status"], term["ppn"], term["record"]["kind"]),
                         ("Musterland", "unique", "444444444", "Geographikum"))

    def test_a_unique_heading_without_ppn_is_marked(self):
        term = {t["keyword"]: t for t in geo_state(self.payload).geo_sachdb["terms"]}["Gestein"]
        self.assertEqual((term["status"], term["ppn"]), ("no_ppn", ""))

    def test_notations_carry_the_local_entries_and_the_usable_ppn(self):
        notes = {n["code"]: n for n in geo_state(self.payload).geo_sachdb["notations"]}
        self.assertEqual({c: (n["status"], n["ppn"]) for c, n in notes.items()}, {
            "DK 999.1": ("ambiguous", ""),
            "DK 999.3": ("unique", "666666666"),
            "DK 999": ("no_ppn", ""),
            "DK 888.8": ("not_local", ""),
        })
        self.assertEqual((notes["DK 999.3"]["system"], notes["DK 999.3"]["notation"]), ("DK", "999.3"))

    def test_the_old_numbering_fields_are_gone(self):
        geo = geo_state(self.payload).geo_sachdb
        self.assertEqual(set(geo), {"terms", "notations", "not_found", "candidates", "source"})

    def test_the_source_keeps_the_version_not_the_local_path(self):
        self.assertIn("db_path", self.payload["source"])
        source = geo_state(self.payload).geo_sachdb["source"]
        self.assertNotIn("db_path", source)
        self.assertNotIn("rows", source)
        self.assertEqual(source.get("schema_version"), self.payload["source"]["schema_version"])

    def test_without_the_step_the_field_stays_empty(self):
        self.assertEqual(geo_state(None).geo_sachdb, {})

    def test_an_empty_step_result_leaves_the_field_empty(self):
        for status in ("disabled", "unavailable", "no_input", "error"):
            empty = {**self.payload, "status": status}
            self.assertEqual(geo_state(empty).geo_sachdb, {}, status)
        self.assertEqual(geo_state("kaputt").geo_sachdb, {})

    def test_only_misses_still_count_as_a_result(self):
        geo = geo_state({"status": "ok", "not_found": ["Unbekanntwort"]}).geo_sachdb
        self.assertEqual((geo["terms"], geo["not_found"]), ([], ["Unbekanntwort"]))


class GeoEntersTheRecord(GeoPayload):
    """A unique local PPN replaces the text, in chains and single keywords alike."""

    def test_the_record_uses_unique_ppns_and_keeps_text_otherwise(self):
        from src.utils.winibw_record import build_winibw_record

        self.assertEqual(build_winibw_record(geo_state(self.payload))["lines"], EXPECTED_RECORD)

    def test_without_geo_data_the_same_record_is_all_text(self):
        from src.utils.winibw_record import build_winibw_record

        self.assertEqual(build_winibw_record(geo_state(None))["lines"], EXPECTED_RECORD_WITHOUT_GEO)

    def test_each_line_says_which_local_record_it_uses(self):
        from src.utils.winibw_record import build_winibw_record, winibw_entry_text

        texts = [winibw_entry_text(e) for e in build_winibw_record(geo_state(self.payload))["entries"]]
        self.assertIn("5550 !333333333! = Lagerstätte · Kette 1 · lokal: Lagerstätte · Sachschlagwort", texts)
        self.assertIn("5550 Geschichte · Kette 1 · lokal mehrdeutig (2 Sätze), als Text", texts)
        self.assertIn("5552 Gestein · Einzelschlagwort · lokal: Gestein · Sachschlagwort · "
                      "lokaler Satz ohne PPN, als Text", texts)
        self.assertIn("5553 !200000001! = Bergbau · Einzelschlagwort · lokal: Bergbau · Sachschlagwort", texts)
        self.assertIn("5554 Unbekanntwort · Einzelschlagwort", texts)
        self.assertIn("6700 DK 999.1 · lokal mehrdeutig (2 Einträge mit verschiedener PPN), als Text", texts)
        self.assertIn("6700 !666666666! = DK 999.3 · lokal: Musterland", texts)
        self.assertIn("6700 DK 999 · lokal: Testklasse · lokaler Eintrag ohne PPN", texts)
        self.assertIn("6700 DK 888.8", texts)

    def test_proposals_never_enter_the_record(self):
        from src.utils.winibw_record import build_winibw_record

        self.assertNotIn("6700 DK 999.21", build_winibw_record(geo_state(self.payload))["lines"])


class GeoIsShown(GeoPayload):
    TITLE = "Lokale Sacherschließung (geo_sachdb)"

    def test_the_card_lists_the_local_hits_per_keyword_and_notation(self):
        items = list_items(rendered(geo_state(self.payload)), "alima-result__geo-hits")
        self.assertIn("Testgestein → Testgestein · Sachschlagwort · PPN 111111111", items)
        self.assertIn("Gestein → Gestein · Sachschlagwort · ohne PPN", items)
        self.assertIn("Geschichte → mehrdeutig, als Text: Geschichte · Sachschlagwort · PPN 100000001; "
                      "Geschichte · Sachschlagwort · PPN 100000002", items)
        self.assertIn("Bergbau → Bergbau · Sachschlagwort · PPN 200000001", items)
        self.assertIn("DK 999.1 → mehrdeutig, als Text: Testgesteine · PPN 555555555; "
                      "Testgesteine (zweiter Eintrag) · PPN 999999999", items)
        self.assertIn("DK 999.3 → Musterland · PPN 666666666", items)
        self.assertFalse(any(item.startswith("DK 888.8") for item in items))

    def test_the_proposals_have_their_own_copy_block(self):
        html = rendered(geo_state(self.payload))
        self.assertIn("Weitere lokale DK-Vorschläge", html)
        self.assertEqual(first_text(html, "alima-result__geo-proposals"), "\n".join(PROPOSAL_LINES))
        self.assertIn('data-alima-copy=".alima-result__geo-proposals"', html)
        self.assertEqual(list_items(html, "alima-result__geo-proposal-entries"),
                         ["6700 DK 999.21 DK 999.21 Lagerstaetten (Test) über Lagerstätte"])

    def test_the_proposals_are_collapsed_by_default(self):
        html = rendered(geo_state(self.payload))
        start = html.index('<details class="alima-result__more alima-result__geo-more"')
        tag = html[start:html.index(">", start)]
        self.assertNotIn("open", tag)
        self.assertIn(f"Weitere lokale DK-Vorschläge ({len(PROPOSAL_LINES)})</summary>", html)
        # the whole proposal list sits inside that element
        self.assertLess(start, html.index("alima-result__geo-proposals"))
        self.assertLess(html.index("alima-result__geo-proposals"), html.index("</details>", start))
        # the winibw entry and the hits stay outside it
        self.assertLess(html.index("alima-result__winibw"), start)

    def test_the_proposals_are_not_in_the_winibw_copy_block(self):
        html = rendered(geo_state(self.payload))
        self.assertEqual(first_text(html, "alima-result__winibw"), "\n".join(EXPECTED_RECORD))
        self.assertLess(html.index("alima-result__winibw"), html.index("alima-result__geo-proposals"))

    def test_the_misses_follow(self):
        html = rendered(geo_state(self.payload))
        start = html.index("Ohne lokales Schlagwort")
        self.assertIn("Unbekanntwort", html[start:start + 300])

    def test_the_rule_snippet_button_still_copies_the_rule(self):
        state = geo_state(self.payload)
        state.rule_output = "5553 Regelzeile"
        html = rendered(state)
        self.assertEqual(first_text(html, "alima-result__pre"), "5553 Regelzeile")
        self.assertEqual(first_text(html, "alima-result__winibw"), "\n".join(EXPECTED_RECORD))

    def test_the_plain_text_carries_the_hits_and_proposals(self):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        _html, plain = PipelineResultFormatter.format_final_result_card_html(geo_state(self.payload))
        lines = plain.splitlines()
        start = lines.index(self.TITLE + ":")
        self.assertIn("  Testgestein → Testgestein · Sachschlagwort · PPN 111111111", lines[start:])
        p = lines.index("Weitere lokale DK-Vorschläge (nicht im WinIBW-Eintrag):")
        self.assertEqual(lines[p + 1], "6700 DK 999.21")
        self.assertIn("Ohne lokales Schlagwort: Unbekanntwort", lines)

    def test_only_misses_yield_a_block_without_copy_buttons(self):
        state = geo_state({"status": "ok", "not_found": ["Unbekanntwort"]}, keywords=[],
                          notations=[], chains=[])
        html = rendered(state)
        self.assertIn(self.TITLE, html)
        self.assertIn("Unbekanntwort", html)
        self.assertNotIn("alima-result__geo-proposals", html)
        self.assertNotIn("alima-result__geo-hits", html)

    def test_without_data_nothing_is_shown(self):
        html = rendered(geo_state(None))
        self.assertNotIn("geo_sachdb", html)
        self.assertNotIn("Ohne lokales Schlagwort", html)

    def test_the_cli_prints_hits_proposals_and_misses(self):
        out = cli(geo_state(self.payload))
        start = out.index(self.TITLE + ":")
        block = out[start:]
        self.assertIn("  Testgestein → Testgestein · Sachschlagwort · PPN 111111111", block)
        self.assertIn("  Geschichte → mehrdeutig, als Text: Geschichte · Sachschlagwort · PPN 100000001; "
                      "Geschichte · Sachschlagwort · PPN 100000002", block)
        p = block.index("  Weitere lokale DK-Vorschläge (nicht im WinIBW-Eintrag):")
        self.assertEqual(block[p + 1], "6700 DK 999.21")  # unindented, to copy
        self.assertIn("    6700 DK 999.21 = DK 999.21 Lagerstaetten (Test) · über Lagerstätte", block)
        self.assertIn("  ohne lokales Schlagwort: Unbekanntwort", block)

    def test_the_cli_prints_them_without_classifications_too(self):
        out = cli(geo_state(self.payload, notations=[]))
        self.assertIn(self.TITLE + ":", out)
        self.assertIn("5551 !111111111!", out)

    def test_the_cli_prints_nothing_without_data(self):
        for notations in (NOTATIONS, []):
            self.assertNotIn("geo_sachdb", "\n".join(cli(geo_state(None, notations=notations))))

    def test_the_webapp_payload_carries_it(self):
        from src.webapp.result_serialization import extract_results_from_analysis_state

        results = extract_results_from_analysis_state(geo_state(self.payload))
        self.assertEqual(results["geo_sachdb"]["terms"][0]["ppn"], "111111111")
        self.assertEqual(results["winibw"]["lines"], EXPECTED_RECORD)
        json.dumps(results["geo_sachdb"])
        json.dumps(results["winibw"])
        self.assertEqual(extract_results_from_analysis_state(geo_state(None))["geo_sachdb"], {})


class GeoFromBeforeTheTerms(GeoPayload):
    """Results saved before ``terms``/``notations``: entry without local PPNs,
    the stored (old-numbered) lines are no longer shown."""

    def _old_state(self):
        state = geo_state(None)
        state.geo_sachdb = _old_geo(self.payload)
        return state

    def test_the_record_is_built_without_geo(self):
        from src.utils.winibw_record import build_winibw_record

        self.assertEqual(build_winibw_record(self._old_state())["lines"], EXPECTED_RECORD_WITHOUT_GEO)

    def test_the_card_shows_proposals_and_misses_but_not_the_stored_lines(self):
        html = rendered(self._old_state())
        self.assertEqual(first_text(html, "alima-result__winibw"), "\n".join(EXPECTED_RECORD_WITHOUT_GEO))
        self.assertEqual(first_text(html, "alima-result__geo-proposals"), "\n".join(PROPOSAL_LINES))
        self.assertIn("Unbekanntwort", html)
        self.assertNotIn("5552 !333333333!", html)
        self.assertNotIn("alima-result__geo-hits", html)

    def test_the_cli_shows_the_same(self):
        out = cli(self._old_state())
        self.assertIn("6700 DK 999.21", out)
        self.assertNotIn("5552 !333333333!", out)
        self.assertIn("  ohne lokales Schlagwort: Unbekanntwort", out)


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
        from src.utils.winibw_record import build_winibw_record

        state = geo_state(self.payload)
        PipelineJsonManager.save_analysis_state(state, self.path)
        loaded = self._load()
        self.assertEqual(loaded.geo_sachdb, state.geo_sachdb)
        self.assertEqual(build_winibw_record(loaded)["lines"], EXPECTED_RECORD)

    def test_export_file_roundtrip(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file
        from src.utils.winibw_record import build_winibw_record

        state = geo_state(self.payload)
        export_analysis_state_to_file(state, self.path, validate_rvk=False)
        loaded = self._load()
        self.assertEqual(loaded.geo_sachdb, state.geo_sachdb)
        self.assertEqual(build_winibw_record(loaded)["lines"], EXPECTED_RECORD)

    def test_the_cli_protocol_shows_a_saved_result(self):
        from src.cli.formatters.protocol_formatters import display_protocol
        from src.utils.pipeline_persistence import PipelineJsonManager

        PipelineJsonManager.save_analysis_state(geo_state(self.payload), self.path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            display_protocol(self.path, ["dk_classification"])
        out = buffer.getvalue().splitlines()
        start = out.index("WinIBW-Eintrag:")
        self.assertEqual(out[start + 1:start + 1 + len(EXPECTED_RECORD)], EXPECTED_RECORD)
        self.assertIn("  Testgestein → Testgestein · Sachschlagwort · PPN 111111111", out)

    def test_an_old_state_file_without_the_field_loads(self):
        data = dataclasses.asdict(geo_state(None))
        data.pop("geo_sachdb")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, {})
        self.assertNotIn("geo_sachdb", rendered(state))

    def test_an_old_export_file_without_the_field_loads(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(geo_state(None), self.path, validate_rvk=False)
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["results"].pop("geo_sachdb")
        data["results"].pop("winibw")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, {})
        self.assertEqual(rendered(state).count("geo_sachdb"), 0)
        self.assertEqual(first_text(rendered(state), "alima-result__winibw"),
                         "\n".join(EXPECTED_RECORD_WITHOUT_GEO))

    def test_a_state_file_from_before_the_terms_loads_and_renders(self):
        data = dataclasses.asdict(geo_state(None))
        data["geo_sachdb"] = _old_geo(self.payload)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, _old_geo(self.payload))
        html = rendered(state)
        self.assertEqual(first_text(html, "alima-result__winibw"), "\n".join(EXPECTED_RECORD_WITHOUT_GEO))
        self.assertEqual(first_text(html, "alima-result__geo-proposals"), "\n".join(PROPOSAL_LINES))

    def test_an_export_file_from_before_the_terms_loads_and_renders(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        export_analysis_state_to_file(geo_state(None), self.path, validate_rvk=False)
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["results"]["geo_sachdb"] = _old_geo(self.payload)
        data["results"].pop("winibw")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        state = self._load()
        self.assertEqual(state.geo_sachdb, _old_geo(self.payload))
        out = cli(state)
        self.assertIn("6700 DK 999.21", out)
        start = out.index("WinIBW-Eintrag:")
        self.assertEqual(out[start + 1:start + 1 + len(EXPECTED_RECORD_WITHOUT_GEO)],
                         EXPECTED_RECORD_WITHOUT_GEO)


if __name__ == "__main__":
    unittest.main()
