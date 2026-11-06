"""WinIBW-Eintrag: ein Aufbau für alle Ausgaben - Claude Generated.

``src/utils/winibw_record.build_winibw_record`` baut den Eintrag aus Ketten,
finalen Schlagwörtern, Formschlagwörtern und Notationen (Vorgabe des
Anwenders): Ketten ab 5550, eine Feldnummer je Kette für alle Glieder, Abschluss
``<nr> $ADE-105``, Leerzeile zwischen den Ketten; Einzelschlagwörter ohne
``$ADE-105`` mit eigener, weiterzählender Nummer; Formschlagwörter in 1131; DK
als ``6700 DK <Notation>``, RVK als ``6700 <Notation>``. Die Obergrenze ist eine
Konstante; darüber gibt es keine Zeile, sondern einen Vermerk. Geprüft wird
auch, dass Abschlusskarte, CLI (``detailed`` und ``--format k10plus``),
Review-Reiter, ``marc``-Export und ``results.winibw`` denselben Text liefern.
Alle Daten erfunden.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils import winibw_record as wr
from src.utils.winibw_record import build_winibw_record, winibw_entry_text

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


def _payload(**results) -> dict:
    return {"results": results}


def _lines(**results) -> list:
    return build_winibw_record(_payload(**results))["lines"]


def _state(**fields):
    """A state object as the card and the CLI see it (only the fields read)."""
    final = SimpleNamespace(
        extracted_gnd_keywords=list(fields.pop("keywords", [])),
        keyword_chains=list(fields.pop("chains", [])),
        response_full_text="",
    )
    base = dict(working_title="", core_keywords=[], form_keywords=[], keyword_chains=[],
                final_llm_analysis=final, classification_entries=[], dk_classifications=[],
                shelf={}, rule_output="", geo_sachdb={})
    base.update(fields)
    return SimpleNamespace(**base)


class Chains(unittest.TestCase):
    def test_without_chains_there_is_no_ade_and_singles_start_at_5550(self):
        lines = _lines(final_keywords=["Mikroplastik (4127527-9)", "Meeresökologie (4038492-7)"])
        self.assertEqual(lines, ["5550 Mikroplastik", "5551 Meeresökologie"])

    def test_one_chain_takes_one_number_and_ends_with_ade(self):
        lines = _lines(keyword_chains=[{"chain": ["Cadmium", "Bodenverschmutzung"]}])
        self.assertEqual(lines, ["5550 Cadmium", "5550 Bodenverschmutzung", "5550 $ADE-105"])

    def test_several_chains_keep_their_order_with_a_blank_line_between(self):
        lines = _lines(keyword_chains=[
            {"chain": ["SG1-K1", "SG2-K1"]}, ["SG1-K2", "SG2-K2"], {"chain": ["SG1-K3"]},
        ])
        self.assertEqual(lines, [
            "5550 SG1-K1", "5550 SG2-K1", "5550 $ADE-105", "",
            "5551 SG1-K2", "5551 SG2-K2", "5551 $ADE-105", "",
            "5552 SG1-K3", "5552 $ADE-105",
        ])

    def test_chain_terms_lose_their_gnd_id_and_empty_or_scalar_chains_are_dropped(self):
        lines = _lines(keyword_chains=[{"chain": ["Term (GND-ID: 123-4)", " "]}, {"chain": []},
                                       "kein Kettenformat", {"reason": "ohne"}])
        self.assertEqual(lines, ["5550 Term", "5550 $ADE-105"])

    def test_chains_fall_back_to_the_llm_details(self):
        lines = _lines(final_llm_call_details={"keyword_chains": [["X", "Y"]]})
        self.assertEqual(lines, ["5550 X", "5550 Y", "5550 $ADE-105"])


class SingleKeywords(unittest.TestCase):
    def test_singles_follow_the_chains_without_ade(self):
        lines = _lines(keyword_chains=[["A", "B"], ["C"]],
                       final_keywords=["A (GND-ID: 1)", "D (GND-ID: 4)", "E"])
        self.assertEqual(lines, [
            "5550 A", "5550 B", "5550 $ADE-105", "",
            "5551 C", "5551 $ADE-105", "",
            "5552 D", "5553 E",
        ])

    def test_a_keyword_in_a_chain_is_not_repeated_and_duplicates_appear_once(self):
        lines = _lines(keyword_chains=[["Mikroplastik"]],
                       final_keywords=["mikroplastik", "Meer", "Meer (GND-ID: 2)", "MEER"])
        self.assertEqual(lines, ["5550 Mikroplastik", "5550 $ADE-105", "", "5551 Meer"])

    def test_the_final_keywords_fall_back_to_the_llm_details(self):
        lines = _lines(final_llm_call_details={"extracted_gnd_keywords": ["Wort (GND-ID: 1)"]})
        self.assertEqual(lines, ["5550 Wort"])


class FormKeywords(unittest.TestCase):
    def test_form_keywords_go_to_1131_not_to_555x(self):
        lines = _lines(final_keywords=["Werkstoffkunde", "Lehrbuch (GND-ID: 4123623-3)"],
                       form_keywords=["Lehrbuch (GND-ID: 4123623-3)"])
        self.assertEqual(lines, ["5550 Werkstoffkunde", "", "1131 Lehrbuch"])

    def test_a_form_keyword_is_taken_out_of_a_chain(self):
        record = build_winibw_record(_payload(
            keyword_chains=[["Werkstoffkunde", "Lehrbuch"], ["Lehrbuch"], ["Metall"]],
            form_keywords=["Lehrbuch"]))
        self.assertEqual(record["lines"], [
            "5550 Werkstoffkunde", "5550 $ADE-105", "",
            "5551 Metall", "5551 $ADE-105", "",
            "1131 Lehrbuch",
        ])
        self.assertEqual(record["entries"][-1]["note"], "aus Kette 1, 2 genommen")

    def test_form_keywords_appear_once(self):
        self.assertEqual(_lines(form_keywords=["Lehrbuch", "lehrbuch", "Bildband"]),
                         ["1131 Lehrbuch", "1131 Bildband"])


class Notations(unittest.TestCase):
    def test_dk_with_prefix_and_rvk_without(self):
        lines = _lines(classifications=[
            {"system": "DK", "code": "504.054", "display": "DK 504.054"},
            {"system": "RVK", "code": "AR 18100", "display": "RVK AR 18100"},
        ])
        self.assertEqual(lines, ["6700 DK 504.054", "6700 AR 18100"])

    def test_strings_and_unprefixed_codes(self):
        lines = _lines(dk_classifications=["DK 628.5", "RVK WI 4400", "QP 925", "553.62 - Label",
                                           "DK 628.5"])
        self.assertEqual(lines, ["6700 DK 628.5", "6700 WI 4400", "6700 QP 925", "6700 DK 553.62"])

    def test_an_unknown_system_from_the_export_schema_is_inferred(self):
        # ``parse_classification_entry`` marks an unprefixed code as UNKNOWN.
        lines = _lines(classifications=[{"system": "UNKNOWN", "code": "553.62"},
                                        {"system": "UNKNOWN", "code": "AR 18100"}])
        self.assertEqual(lines, ["6700 DK 553.62", "6700 AR 18100"])

    def test_other_systems_get_no_line_and_are_named(self):
        record = build_winibw_record(_payload(classifications=[
            {"system": "DDC", "code": "550"}, {"system": "DK", "code": "55"}]))
        self.assertEqual(record["lines"], ["6700 DK 55"])
        self.assertEqual(record["skipped"], [{"system": "DDC", "code": "550",
                                              "reason": "kein WinIBW-Feld festgelegt"}])
        self.assertEqual(wr.winibw_notes(record),
                         [("Nicht übernommen", "DDC 550 (kein WinIBW-Feld festgelegt)")])


class WholeEntry(unittest.TestCase):
    def test_sections_in_order_with_one_blank_line_between(self):
        record = build_winibw_record(_payload(
            keyword_chains=[["A", "B"]], final_keywords=["A", "C", "Lehrbuch"],
            form_keywords=["Lehrbuch"],
            classifications=[{"system": "DK", "code": "1"}, {"system": "RVK", "code": "AB 100"}]))
        self.assertEqual(record["text"], "5550 A\n5550 B\n5550 $ADE-105\n\n5551 C\n\n1131 Lehrbuch\n\n"
                                         "6700 DK 1\n6700 AB 100")
        self.assertEqual([e["line"] for e in record["entries"]], [l for l in record["lines"] if l])
        self.assertEqual([e["origin"] for e in record["entries"]],
                         ["chain", "chain", "chain_end", "single", "form", "DK", "RVK"])

    def test_an_empty_result_gives_an_empty_entry(self):
        record = build_winibw_record({})
        self.assertEqual((record["lines"], record["text"], record["entries"], record["overflow"]),
                         ([], "", [], []))

    def test_state_object_and_payload_give_the_same_entry(self):
        state = _state(chains=[{"chain": ["A (GND-ID: 1)", "B"]}], keywords=["A (GND-ID: 1)", "C"],
                       form_keywords=["Lehrbuch (GND-ID: 9)"],
                       classification_entries=[{"system": "RVK", "code": "AB 100"}])
        payload = _payload(keyword_chains=[{"chain": ["A (GND-ID: 1)", "B"]}],
                           final_keywords=["A (GND-ID: 1)", "C"], form_keywords=["Lehrbuch (GND-ID: 9)"],
                           classifications=[{"system": "RVK", "code": "AB 100"}])
        self.assertEqual(build_winibw_record(state), build_winibw_record(payload))

    def test_each_line_says_what_it_stands_for(self):
        record = build_winibw_record(_payload(
            keyword_chains=[["A"]], final_keywords=["C"], form_keywords=["Lehrbuch"],
            classifications=[{"system": "DK", "code": "1"}, {"system": "RVK", "code": "AB 100"}]))
        self.assertEqual([winibw_entry_text(e) for e in record["entries"]], [
            "5550 A · Kette 1",
            "5550 $ADE-105 · Abschluss Kette 1",
            "5551 C · Einzelschlagwort",
            "1131 Lehrbuch · Formschlagwort",
            "6700 DK 1",
            "6700 AB 100 = RVK AB 100",
        ])


class UpperBound(unittest.TestCase):
    def test_the_default_bound_is_5559(self):
        # Pinned: the operator still decides between 5559 and 5599.
        self.assertEqual(wr.WINIBW_SUBJECT_LAST_TAG, 5559)
        record = build_winibw_record(_payload(keyword_chains=[[f"K{i}"] for i in range(11)]))
        tags = {line.split(" ")[0] for line in record["lines"] if line}
        self.assertEqual(tags, {str(n) for n in range(5550, 5560)})
        self.assertEqual(record["overflow"], [{"origin": "chain", "term": "K10", "chain": 11}])

    def test_beyond_the_bound_nothing_is_merged_onto_the_last_number(self):
        with patch.object(wr, "WINIBW_SUBJECT_LAST_TAG", 5552):
            record = build_winibw_record(_payload(
                keyword_chains=[["A", "B"], ["C"]], final_keywords=["D", "E", "F"]))
        self.assertEqual(record["lines"], [
            "5550 A", "5550 B", "5550 $ADE-105", "", "5551 C", "5551 $ADE-105", "", "5552 D",
        ])
        self.assertEqual(record["overflow"], [{"origin": "single", "term": "E"},
                                              {"origin": "single", "term": "F"}])
        self.assertEqual(record["last_tag"], 5552)
        self.assertEqual(wr.winibw_notes(record), [("Ohne Feldnummer (5552 erreicht)", "E; F")])

    def test_a_chain_beyond_the_bound_is_named_whole(self):
        with patch.object(wr, "WINIBW_SUBJECT_LAST_TAG", 5550):
            record = build_winibw_record(_payload(keyword_chains=[["A"], ["B", "C"]], final_keywords=["B"]))
        self.assertEqual(record["lines"], ["5550 A", "5550 $ADE-105"])
        # B belongs to the chain without a number, it does not reappear as a single.
        self.assertEqual(record["overflow"], [{"origin": "chain", "term": "B → C", "chain": 2}])
        self.assertEqual(wr.winibw_notes(record), [("Ohne Feldnummer (5550 erreicht)", "Kette 2: B → C")])


def _geo(terms=(), notations=()) -> dict:
    return {"terms": list(terms), "notations": list(notations), "not_found": [], "candidates": []}


def _term(keyword, status, ppn="", records=None):
    records = records or [{"label": keyword, "kind": "Sachschlagwort", "ppn": ppn}]
    return {"keyword": keyword, "records": records, "status": status, "ppn": ppn if status == "unique" else "",
            "record": None if status == "ambiguous" else records[0]}


class LocalPpns(unittest.TestCase):
    def test_unique_ppn_replaces_the_text_in_chains_and_singles(self):
        geo = _geo([_term("Granit", "unique", "100000001"), _term("Erzgebirge", "unique", "100000002")])
        lines = _lines(keyword_chains=[["Granit", "Sachsen"]], final_keywords=["Erzgebirge"], geo_sachdb=geo)
        self.assertEqual(lines, ["5550 !100000001!", "5550 Sachsen", "5550 $ADE-105", "", "5551 !100000002!"])

    def test_ambiguous_and_ppn_less_hits_stay_text_with_a_note(self):
        geo = _geo([_term("Geschichte", "ambiguous", records=[
            {"label": "Geschichte", "kind": "Sachschlagwort", "ppn": "1"},
            {"label": "Geschichte", "kind": "Sachschlagwort", "ppn": "2"}]),
            _term("Gestein", "no_ppn")])
        record = build_winibw_record(_payload(final_keywords=["Geschichte", "Gestein"], geo_sachdb=geo))
        self.assertEqual(record["lines"], ["5550 Geschichte", "5551 Gestein"])
        self.assertEqual([e["note"] for e in record["entries"]],
                         ["lokal mehrdeutig (2 Sätze), als Text", "lokaler Satz ohne PPN, als Text"])

    def test_the_match_ignores_case_and_unicode_form(self):
        geo = _geo([_term("Lagerstätte", "unique", "333333333")])
        decomposed = "Lagerstätte"
        self.assertEqual(_lines(final_keywords=[decomposed.upper()], geo_sachdb=geo), ["5550 !333333333!"])

    def test_two_terms_with_the_same_local_ppn_give_one_line(self):
        geo = _geo([_term("Bergbau", "unique", "200000001"), _term("Montanwesen", "unique", "200000001")])
        record = build_winibw_record(_payload(final_keywords=["Bergbau", "Montanwesen", "Kohle"], geo_sachdb=geo))
        self.assertEqual(record["lines"], ["5550 !200000001!", "5551 Kohle"])
        self.assertEqual(record["entries"][0]["also"], ["Montanwesen"])

    def test_dk_with_one_local_ppn_becomes_a_ppn_line(self):
        geo = _geo(notations=[
            {"system": "DK", "notation": "999.3", "status": "unique", "ppn": "666666666",
             "local": [{"notation": "999.3", "label": "Musterland", "ppn": "666666666"}]},
            {"system": "DK", "notation": "999.1", "status": "ambiguous", "ppn": "",
             "local": [{"notation": "999.1", "label": "A", "ppn": "1"}, {"notation": "999.1", "label": "B", "ppn": "2"}]},
            {"system": "DK", "notation": "888.8", "status": "not_local", "ppn": "", "local": []},
        ])
        lines = _lines(classifications=[{"system": "DK", "code": "999.3"}, {"system": "DK", "code": "999.1"},
                                        {"system": "DK", "code": "888.8"}, {"system": "RVK", "code": "QX 200"}],
                       geo_sachdb=geo)
        self.assertEqual(lines, ["6700 !666666666!", "6700 DK 999.1", "6700 DK 888.8", "6700 QX 200"])

    def test_geo_without_the_new_fields_changes_nothing(self):
        old = {"lines": ["5551 !111111111!"], "entries": [], "not_found": ["X"], "candidates": []}
        self.assertEqual(_lines(final_keywords=["Testgestein"], geo_sachdb=old), ["5550 Testgestein"])


class Card(unittest.TestCase):
    """The WinIBW block of the result card (GUI log and webapp log)."""

    STATE = dict(chains=[{"chain": ["Cadmium", "Boden <Geologie>"]}], keywords=["Cadmium", "Blei"],
                 classification_entries=[{"system": "DK", "code": "504.064"}])

    def _card(self, **fields):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        return PipelineResultFormatter.format_final_result_card_html(_state(**fields))

    def test_the_copy_block_holds_exactly_the_entry(self):
        _geo_mod = _load_geo_result_module()
        html, _plain = self._card(**self.STATE)
        text = _geo_mod.first_text(html, "alima-result__winibw")
        self.assertEqual(text, build_winibw_record(_state(**self.STATE))["text"])
        self.assertEqual(text, "5550 Cadmium\n5550 Boden <Geologie>\n5550 $ADE-105\n\n5551 Blei\n\n"
                               "6700 DK 504.064")
        for word in ("Kette", "Einzelschlagwort", "Abschluss"):
            self.assertNotIn(word, text)
        self.assertIn('data-alima-copy=".alima-result__winibw"', html)
        self.assertIn("&lt;Geologie&gt;", html)

    def test_the_explanation_list_follows_the_copy_block(self):
        _geo_mod = _load_geo_result_module()
        html, _plain = self._card(**self.STATE)
        self.assertEqual(_geo_mod.list_items(html, "alima-result__winibw-entries"), [
            "5550 Cadmium Kette 1", "5550 Boden <Geologie> Kette 1", "5550 $ADE-105 Abschluss Kette 1",
            "5551 Blei Einzelschlagwort", "6700 DK 504.064",
        ])
        self.assertLess(html.index('class="alima-result__winibw"'),
                        html.index('class="alima-result__winibw-entries"'))

    def test_the_rule_snippet_keeps_its_own_target(self):
        _geo_mod = _load_geo_result_module()
        html, _plain = self._card(rule_output="5553 Regel", **self.STATE)
        self.assertEqual(_geo_mod.first_text(html, "alima-result__pre"), "5553 Regel")
        self.assertEqual(html.count('class="alima-copy"'), 3)  # whole card, entry, rule

    def test_the_plain_text_carries_the_entry_verbatim(self):
        _html, plain = self._card(**self.STATE)
        lines = plain.splitlines()
        start = lines.index("WinIBW-Eintrag:")
        record = build_winibw_record(_state(**self.STATE))
        self.assertEqual(lines[start + 1:start + 1 + len(record["lines"])], record["lines"])
        self.assertIn("  5551 Blei · Einzelschlagwort", lines)

    def test_overflow_and_skipped_notations_are_named(self):
        with patch.object(wr, "WINIBW_SUBJECT_LAST_TAG", 5550):
            html, _plain = self._card(keywords=["A", "B"],
                                      classification_entries=[{"system": "DDC", "code": "550"}])
        self.assertIn("Ohne Feldnummer (5550 erreicht)", html)
        self.assertIn("Nicht übernommen", html)
        self.assertIn("DDC 550 (kein WinIBW-Feld festgelegt)", html)

    def test_without_entry_lines_there_is_no_block(self):
        html, _plain = self._card(core_keywords=["X"])
        self.assertNotIn("WinIBW-Eintrag", html)


_GEO_MODULE = {}


def _load_geo_result_module():
    """The geo fixture and HTML helpers of ``test_geo_sachdb_result.py``."""
    if "module" not in _GEO_MODULE:
        spec = importlib.util.spec_from_file_location(
            "geo_result_fixture_winibw", Path(__file__).resolve().parent / "test_geo_sachdb_result.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _GEO_MODULE["module"] = module
    return _GEO_MODULE["module"]


class OutputPaths(unittest.TestCase):
    """Every output writes the same text: function, export, CLI, card, review tab."""

    @classmethod
    def setUpClass(cls):
        cls.geo = _load_geo_result_module()
        cls.state = cls.geo.geo_state(cls.geo.payload(), form_keywords=["Lehrbuch"])
        cls.expected = build_winibw_record(cls.state)["text"]
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_fixture_uses_geo_chains_singles_forms_and_notations(self):
        lines = self.expected.splitlines()
        self.assertIn("5550 !333333333!", lines)
        self.assertIn("5550 $ADE-105", lines)
        self.assertIn("5551 !111111111!", lines)
        self.assertIn("1131 Lehrbuch", lines)
        self.assertIn("6700 !666666666!", lines)
        self.assertIn("6700 QX 200", lines)

    def _export_payload(self):
        from src.webapp.result_serialization import build_export_payload, extract_results_from_analysis_state

        return build_export_payload(session_id="s", created_at=None, status="completed",
                                    current_step="classification", input_data={},
                                    results=extract_results_from_analysis_state(self.state),
                                    validate_rvk=False)

    def test_export_payload_marc_and_results_winibw(self):
        from src.utils import exporters

        payload = self._export_payload()
        self.assertEqual(payload["results"]["winibw"]["text"], self.expected)
        self.assertEqual(build_winibw_record(payload)["text"], self.expected)
        self.assertEqual("\n".join(exporters.generate_k10plus_lines(payload)), self.expected)
        out = os.path.join(self.tmp, "entry.txt")
        exporters.export(payload, "marc", out)
        self.assertEqual(Path(out).read_text(encoding="utf-8"), self.expected + "\n")

    def test_cli_k10plus_from_a_state_file_and_an_export_file(self):
        from src.cli.formatters.protocol_formatters import display_protocol_k10plus
        from src.utils.pipeline_persistence import PipelineJsonManager, export_analysis_state_to_file

        state_file = os.path.join(self.tmp, "state.json")
        export_file = os.path.join(self.tmp, "export.json")
        PipelineJsonManager.save_analysis_state(self.state, state_file)
        export_analysis_state_to_file(self.state, export_file, validate_rvk=False)
        for path in (state_file, export_file):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                display_protocol_k10plus(path)
            self.assertEqual(buffer.getvalue(), self.expected + "\n", path)

    def test_cli_detailed_prints_the_same_block(self):
        out = self.geo.cli(self.state)
        start = out.index("WinIBW-Eintrag:") + 1
        n = len(self.expected.splitlines())
        self.assertEqual("\n".join(out[start:start + n]), self.expected)

    def test_card_copy_block_and_plain_text(self):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        html, plain = PipelineResultFormatter.format_final_result_card_html(self.state)
        self.assertEqual(self.geo.first_text(html, "alima-result__winibw"), self.expected)
        lines = plain.splitlines()
        start = lines.index("WinIBW-Eintrag:") + 1
        n = len(self.expected.splitlines())
        self.assertEqual("\n".join(lines[start:start + n]), self.expected)

    def test_review_tab(self):
        from src.ui.analysis_review_tab import AnalysisReviewTab

        fake = SimpleNamespace(current_analysis=self.state)
        self.assertEqual("\n".join(AnalysisReviewTab._generate_k10plus_format(fake)), self.expected)
        self.assertEqual(AnalysisReviewTab._generate_k10plus_format(SimpleNamespace(current_analysis=None)), [])

    def test_the_rule_output_stays_independent(self):
        state = self.geo.geo_state(self.geo.payload())
        state.rule_output = "5550 Vom Modell"
        self.assertNotIn("Vom Modell", build_winibw_record(state)["text"])
        json.dumps(build_winibw_record(state))


if __name__ == "__main__":
    unittest.main()
