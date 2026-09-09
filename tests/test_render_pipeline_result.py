"""Tests for the shared pipeline-result emission (GUI + webapp). Claude Generated.

``render_pipeline_result`` is the one producer of the final result blocks
(completion line, GND keywords, Schlagwortketten, DK cards, workflow report).
The webapp bug it fixes: these blocks were only emitted by the GUI mixin, so
webapp runs ended with nothing but step collapsibles.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.render_events import MockTransport
from src.utils.pipeline_formatters import render_pipeline_result


class _MockCheckBox:
    def __init__(self):
        self.toggled = MagicMock()

    def isChecked(self):
        return True


def _make_renderer():
    from src.ui.unified_message_renderer import UnifiedMessageRenderer

    transport = MockTransport()
    return UnifiedMessageRenderer(transport, _MockCheckBox()), transport


class TestRenderPipelineResult(unittest.TestCase):

    def test_full_state_emits_all_sections(self):
        renderer, transport = _make_renderer()
        state = SimpleNamespace(
            final_llm_analysis=SimpleNamespace(
                extracted_gnd_keywords=["Chemie", "Informatik"],
                response_full_text=(
                    "Schlagwortketten:\nChemie → Informatik\nohne Pfeil\n"
                ),
            ),
            dk_classifications=None,
            report_markdown="| a | b |\n|---|---|\n| 1 | 2 |",
        )
        render_pipeline_result(renderer, state, "12.3s")
        blocks = transport.of_type("block")
        joined = "\n".join(b["html"] for b in blocks)
        self.assertIn("Pipeline vollständig abgeschlossen in 12.3s", joined)
        self.assertIn("GND-Schlagworte ausgewählt (2)", joined)
        self.assertIn("Chemie", joined)
        self.assertIn("Schlagwortketten", joined)
        self.assertIn("→", joined)
        # report_markdown rendered as an actual HTML table block.
        self.assertIn("<table>", joined)

    def test_empty_state_emits_only_completion_line(self):
        renderer, transport = _make_renderer()
        render_pipeline_result(renderer, None)
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)
        self.assertIn("Pipeline vollständig abgeschlossen", blocks[0]["html"])
        # No duration suffix without a duration.
        self.assertNotIn(" in ", blocks[0]["html"])

    def test_state_without_result_data_emits_no_empty_cards(self):
        renderer, transport = _make_renderer()
        state = SimpleNamespace(
            final_llm_analysis=None,
            dk_classifications=[],
            dk_statistics={},
            rvk_provenance={},
            report_markdown="",
        )
        render_pipeline_result(renderer, state)
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)  # only the completion line


class TestFinalResultCard(unittest.TestCase):
    """Die Abschlusskarte trägt, was ein Erschließer mitnimmt.

    Sie ersetzt fünf Logzeilen, die auseinanderscrollten. Jede Zeile, die dabei
    aufgegangen ist, wird hier festgehalten — sonst verschwindet sie beim
    nächsten Umbau unbemerkt. - Claude Generated
    """

    def _card(self, **fields) -> str:
        from src.utils.pipeline_formatters import PipelineResultFormatter

        base = dict(
            working_title="", core_keywords=[], form_keywords=[], keyword_chains=[],
            final_llm_analysis=None, classification_entries=[], dk_classifications=[],
            shelf={}, rule_output="",
        )
        base.update(fields)
        html, _plain = PipelineResultFormatter.format_final_result_card_html(
            SimpleNamespace(**base)
        )
        return html

    def test_a_run_without_results_yields_no_card(self):
        self.assertEqual(self._card(), "")
        from src.utils.pipeline_formatters import PipelineResultFormatter

        self.assertEqual(
            PipelineResultFormatter.format_final_result_card_html(None), ("", "")
        )

    def test_the_working_title_names_the_card(self):
        self.assertIn("Ergebnis · Cadmium", self._card(
            working_title="Cadmium", core_keywords=["Cadmiumbelastung"]))

    def test_the_core_comes_before_the_full_list(self):
        html = self._card(
            core_keywords=["Cadmiumbelastung"],
            final_llm_analysis=SimpleNamespace(
                extracted_gnd_keywords=["Cadmium", "Cadmiumbelastung"],
                response_full_text="", keyword_chains=[]),
        )
        self.assertLess(html.index("Kernschlagworte"), html.index("GND-Schlagworte"))

    def test_form_headings_travel_with_the_core(self):
        html = self._card(core_keywords=["Werkstoffkunde"], form_keywords=["Lehrbuch"])
        self.assertIn("Lehrbuch", html)

    def test_a_long_keyword_list_is_folded(self):
        html = self._card(final_llm_analysis=SimpleNamespace(
            extracted_gnd_keywords=[f"K{i}" for i in range(12)],
            response_full_text="", keyword_chains=[]))
        self.assertIn("<details", html)
        self.assertIn("… 4 weitere", html)
        self.assertIn("K11", html)  # nothing is dropped, only folded

    def test_chains_come_from_the_structured_field(self):
        html = self._card(keyword_chains=[{"chain": ["Cadmium", "Nahrungskette"]}])
        self.assertIn("Cadmium → Nahrungskette", html)

    def test_chains_fall_back_to_the_llm_analysis_field(self):
        html = self._card(final_llm_analysis=SimpleNamespace(
            extracted_gnd_keywords=[], response_full_text="",
            keyword_chains=[{"chain": ["A", "B"]}]))
        self.assertIn("A → B", html)

    def test_chains_fall_back_to_the_prose_scrape(self):
        # States that carry the chains only as text — the path the GUI showed
        # before the structured field existed.
        html = self._card(final_llm_analysis=SimpleNamespace(
            extracted_gnd_keywords=[], keyword_chains=[],
            response_full_text="Schlagwortketten:\nChemie → Informatik\nohne Pfeil"))
        self.assertIn("Chemie → Informatik", html)
        self.assertNotIn("ohne Pfeil", html)

    def test_the_core_notation_leads_and_the_rest_opens(self):
        html = self._card(classification_entries=[
            {"display": "DK 504.064", "rank": "core"},
            {"display": "DK 615.9", "rank": "additional"},
            {"display": "DK 546.48", "rank": "additional"}])
        self.assertIn("DK 504.064", html)
        self.assertIn("… 2 weitere", html)
        # No dead end: the folded codes have to be in the card, not just counted.
        self.assertIn("<details", html)
        self.assertIn("DK 615.9", html)
        self.assertIn("DK 546.48", html)

    def test_plain_codes_still_show_without_ranks(self):
        html = self._card(dk_classifications=["DK 504.064", "DK 615.9"])
        self.assertIn("DK 504.064", html)
        self.assertIn("DK 615.9", html)

    def test_the_shelf_candidates_are_reachable_too(self):
        html = self._card(shelf={"candidates": [
            {"notation": "622.34", "label": "Erze", "location": "OG"},
            {"notation": "622.3", "label": "Bergbauzweige", "location": "OG"},
            {"notation": "622...", "label": "Bergbau", "location": "OG"}]})
        self.assertIn("622.34", html)
        self.assertIn("… 2 weitere Vorschläge", html)
        self.assertIn("622.3", html)
        self.assertIn("Bergbauzweige", html)

    def test_nothing_in_the_card_is_truncated_without_a_way_to_open_it(self):
        # Every "… N weitere" must sit on a <details>; a counted-away list the
        # reader cannot open is worse than a long one. - Claude Generated
        html = self._card(
            final_llm_analysis=SimpleNamespace(
                extracted_gnd_keywords=[f"K{i}" for i in range(12)],
                response_full_text="", keyword_chains=[]),
            classification_entries=[
                {"display": "DK 1", "rank": "core"},
                {"display": "DK 2", "rank": "additional"}],
            shelf={"candidates": [{"notation": "1"}, {"notation": "2"}]},
        )
        self.assertEqual(html.count("… "), html.count("<summary>… "))

    def test_the_shelf_place_carries_its_reason(self):
        html = self._card(shelf={"placement": {
            "notation": "504.05/.06", "label": "Umweltqualität | darunter: X",
            "location": "OG", "selected_reason": "Cadmiumbelastung des Bodens"}})
        self.assertIn("504.05/.06", html)
        self.assertIn("Standort OG", html)
        self.assertIn("Cadmiumbelastung des Bodens", html)
        # The vocabulary's "darunter" tail belongs in the tooltip, not the line.
        self.assertIn('title="Umweltqualität | darunter: X"', html)
        self.assertNotIn(">Umweltqualität | darunter: X<", html)

    def test_without_a_pick_the_candidates_are_named(self):
        html = self._card(shelf={"candidates": [
            {"notation": "622.34"}, {"notation": "622.3"}]})
        self.assertIn("622.34", html)
        self.assertIn("keine Auswahl", html)

    def test_notations_without_a_shelf_place_are_named(self):
        html = self._card(shelf={
            "placement": {"notation": "622.34", "label": "Erze", "location": "OG"},
            "unplaced": ["004.42"]})
        self.assertIn("Ohne eigene Stelle", html)
        self.assertIn("004.42", html)

    def test_the_whole_result_is_copyable_not_only_the_rule_snippet(self):
        """Kopieren hängt nicht an der Zusatzregel.

        Der Knopf saß zuerst nur am Regelblock — ohne passende Regel gab es
        nichts zu kopieren, obwohl die Karte voller Ergebnis ist.
        - Claude Generated
        """
        html = self._card(core_keywords=["Cadmiumbelastung"])
        self.assertIn("alima-copy-scope", html)
        self.assertIn("Alles kopieren", html)
        # No target selector on the header button: it copies the whole card.
        self.assertIn("data-alima-copy ", html)

    def test_the_snippet_keeps_its_own_button_scoped_to_itself(self):
        html = self._card(core_keywords=["X"], rule_output="5550 Y")
        self.assertIn('data-alima-copy=".alima-result__pre"', html)
        self.assertEqual(html.count('class="alima-copy"'), 2)  # whole card + snippet

    def test_the_rule_snippet_is_preformatted_not_markdown(self):
        html = self._card(rule_output="5553 Toleranz <Biologie>\n5553 $ADE-105")
        self.assertIn("<pre", html)
        # The RSWK qualifier must survive — it names a different GND record.
        self.assertIn("&lt;Biologie&gt;", html)
        self.assertIn("data-alima-copy", html)

    def test_the_plain_text_keeps_the_snippet_verbatim(self):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        snippet = "5553 Toleranz <Biologie>\n5553 $ADE-105"
        _html, plain = PipelineResultFormatter.format_final_result_card_html(
            SimpleNamespace(
                working_title="", core_keywords=[], form_keywords=[],
                keyword_chains=[], final_llm_analysis=None,
                classification_entries=[], dk_classifications=[], shelf={},
                rule_output=snippet)
        )
        self.assertIn(snippet, plain)


if __name__ == "__main__":
    unittest.main()
