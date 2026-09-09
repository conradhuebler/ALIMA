"""Aufstellungssystematik: Vokabular, Zuordnung, Workflow-Steps. Claude Generated.

Die Sacherschließungs-Notation und die Aufstellungsnotation sind nicht
deckungsgleich: die Kernnotation liegt typischerweise nahe der Freihandstelle,
aber es gibt Fälle, in denen die Stelle thematisch woanders sitzt. Der Matcher
bietet deshalb zwei unabhängige Signale (Notationsnähe, Label-Treffer) und
entscheidet nichts. Diese Tests pinnen beide Signale, die Notationsformen der
Freiberger Systematik (Bereiche, offene Stellen, Anhängezahlen) und die
Verdrahtung im 105er-Workflow.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.lookups.local_scheme import (
    LocalSchemeLookup,
    _in_range,
    _split_range,
    digits_of,
)

# A small scheme with every notation form the real page uses.
SYNTHETIC = """
version: 1
scheme:
  id: test_scheme
  label: "Testsystematik"
  base_system: DK
  source: "https://example.invalid/systematik"
  retrieved: 2026-09-09
entries:
  - notation: "622"
    label: "Bergbau"
    location: "OG"
  - notation: "622.34"
    label: "Bergbauzweige: Erze"
    location: "OG"
  - notation: "622..."
    label: "Bergbau (Allgemeines)"
    location: "OG"
  - notation: "122/129"
    label: "Einzelzweige der Metaphysik"
    location: "Mag"
  - notation: "622.012/.03"
    label: "Bergbaubetrieb, Rohstoffwirtschaft"
    location: "OG"
  - notation: "069(036)"
    label: "Einzelne Museen, Museumsfuehrer"
    location: "Mag"
  - notation: "069"
    label: "Museen, Sammlungen"
    location: "Mag"
  - notation: "349.2"
    label: "Arbeitsrecht"
    location: "EG"
    see_also: "QV 700 Arbeitsrecht"
  - notation: "551(98/99)"
    label: "Arktis, Antarktis"
    location: "OG"
  - notation: "21/29"
    label: "Theologie, Religionen, Kirche"
    location: "Mag"
"""


class _SchemeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "test_scheme.yaml"
        self.path.write_text(SYNTHETIC, encoding="utf-8")
        self.lookup = LocalSchemeLookup(vocabulary=str(self.path))

    def tearDown(self):
        self._tmp.cleanup()

    def notations(self, result) -> list:
        return [c["notation"] for c in result["candidates"]]


class NotationArithmetic(unittest.TestCase):
    """DK ist eine Dezimalhierarchie — die Punkte sind Lesehilfe."""

    def test_digits_ignore_punctuation_and_tails(self):
        self.assertEqual(digits_of("622.83"), "62283")
        self.assertEqual(digits_of("32..."), "32")
        self.assertEqual(digits_of("069(036)"), "069")
        self.assertEqual(digits_of("622:943.21"), "622")

    def test_range_right_side_continues_the_left(self):
        self.assertEqual(_split_range("622.012/.03"), ("622012", "62203"))
        self.assertEqual(_split_range("122/129"), ("122", "129"))
        self.assertEqual(_split_range("347.77/.78"), ("34777", "34778"))
        self.assertIsNone(_split_range("622.34"))

    def test_a_slash_inside_the_auxiliary_is_not_a_range(self):
        # 551(98/99) is "Arktis, Antarktis", not the span 551.98 to 99.
        self.assertIsNone(_split_range("551(98/99)"))
        self.assertIsNone(_split_range("622(4/5)"))
        self.assertIsNone(_split_range("62:95/99"))

    def test_range_containment_compares_as_decimal_fractions(self):
        self.assertTrue(_in_range(digits_of("622.02"), "622012", "62203"))
        self.assertFalse(_in_range(digits_of("622.05"), "622012", "62203"))
        self.assertTrue(_in_range(digits_of("125.3"), "122", "129"))
        self.assertFalse(_in_range(digits_of("13"), "122", "129"))


class NotationProximity(_SchemeTestCase):
    def test_exact_notation_wins(self):
        result = self.lookup.lookup_shelf(notations=["622.34"])
        self.assertEqual(self.notations(result)[0], "622.34")
        self.assertIn("identisch", result["candidates"][0]["reason"])

    def test_deeper_code_lands_on_its_shelf_ancestor(self):
        result = self.lookup.lookup_shelf(notations=["DK 622.341"])
        self.assertEqual(self.notations(result)[0], "622.34")

    def test_the_general_catch_all_ranks_below_a_specific_place(self):
        result = self.lookup.lookup_shelf(notations=["622.341"])
        places = self.notations(result)
        self.assertLess(places.index("622.34"), places.index("622..."))

    def test_a_plain_entry_outranks_its_own_catch_all_twin(self):
        # "622" and "622..." describe the same place; the plain one is meant.
        result = self.lookup.lookup_shelf(notations=["622.99"])
        places = self.notations(result)
        self.assertLess(places.index("622"), places.index("622..."))

    def test_a_range_entry_contains_its_codes(self):
        result = self.lookup.lookup_shelf(notations=["622.02"])
        self.assertIn("622.012/.03", self.notations(result))

    def test_a_code_outside_the_range_is_not_placed_there(self):
        result = self.lookup.lookup_shelf(notations=["622.05"])
        self.assertNotIn("622.012/.03", self.notations(result))

    def test_a_code_above_a_range_still_reaches_it(self):
        # DK 2 (Religion) belongs to the shelf range 21/29 — measured against
        # real call numbers this is a placement, just a broad one.
        result = self.lookup.lookup_shelf(notations=["2"])
        self.assertIn("21/29", self.notations(result))

    def test_a_code_beside_a_range_does_not_reach_it(self):
        result = self.lookup.lookup_shelf(notations=["3"])
        self.assertNotIn("21/29", self.notations(result))

    def test_an_auxiliary_slash_does_not_swallow_unrelated_codes(self):
        # Read as a range, 551(98/99) spanned half the schedule.
        result = self.lookup.lookup_shelf(notations=["622.341"], max_results=20)
        self.assertNotIn("551(98/99)", self.notations(result))

    def test_auxiliary_matches_exactly_and_outranks_its_base(self):
        result = self.lookup.lookup_shelf(notations=["069(036)"])
        places = self.notations(result)
        self.assertEqual(places[0], "069(036)")
        self.assertIn("069", places)

    def test_a_code_without_the_auxiliary_prefers_the_plain_place(self):
        result = self.lookup.lookup_shelf(notations=["069"])
        self.assertEqual(self.notations(result)[0], "069")

    def test_a_notation_with_no_shelf_place_is_reported(self):
        result = self.lookup.lookup_shelf(notations=["004.42"])
        self.assertEqual(result["unplaced"], ["004.42"])

    def test_a_deeper_code_under_a_shelf_place_is_not_unplaced(self):
        # 622.99 has no entry of its own but sits under "622".
        result = self.lookup.lookup_shelf(notations=["622.99"])
        self.assertEqual(result["unplaced"], [])
        self.assertEqual(self.notations(result)[0], "622")

    def test_an_unplaced_notation_still_offers_neighbours(self):
        result = self.lookup.lookup_shelf(notations=["62.5"])
        self.assertEqual(result["unplaced"], ["62.5"])
        self.assertTrue(result["candidates"], "keine Nachbarstelle angeboten")
        self.assertIn("Nachbarstelle", result["candidates"][0]["reason"])


class LabelSignal(_SchemeTestCase):
    """Der Fall, den die Notationsnähe nicht trägt."""

    def test_keywords_alone_find_a_place(self):
        result = self.lookup.lookup_shelf(keywords=["Arbeitsrecht"])
        self.assertEqual(self.notations(result), ["349.2"])

    def test_a_thematically_distant_notation_is_corrected_by_the_keywords(self):
        # Sacherschließung sagt Informatik, das Werk handelt von Bergbau.
        result = self.lookup.lookup_shelf(
            notations=["004.42"], keywords=["Bergbau", "Erze"]
        )
        self.assertIn("622", self.notations(result))

    def test_generic_words_do_not_match(self):
        result = self.lookup.lookup_shelf(keywords=["Allgemeines", "und", "der"])
        self.assertEqual(result["candidates"], [])

    def test_two_signals_outrank_one(self):
        both = self.lookup.lookup_shelf(notations=["622.34"], keywords=["Erze"])
        top = both["candidates"][0]
        self.assertEqual(top["notation"], "622.34")
        self.assertIn("Label-Treffer", top["reason"])


class LabelSignalSwitch(_SchemeTestCase):
    """Das Label-Signal ist abschaltbar — es ist etwa einen Punkt Recall wert."""

    def test_off_leaves_only_the_notation_signal(self):
        lookup = LocalSchemeLookup(vocabulary=str(self.path), use_labels=False)
        result = lookup.lookup_shelf(notations=["622.34"], keywords=["Arbeitsrecht"])
        self.assertNotIn("349.2", [c["notation"] for c in result["candidates"]])

    def test_on_by_default(self):
        result = self.lookup.lookup_shelf(notations=["622.34"], keywords=["Arbeitsrecht"])
        self.assertIn("349.2", self.notations(result))


class VocabularyHandling(unittest.TestCase):
    def test_a_missing_vocabulary_is_an_empty_scheme_not_a_crash(self):
        lookup = LocalSchemeLookup(vocabulary="gibt_es_nicht_12345")
        result = lookup.lookup_shelf(notations=["622"])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["unplaced"], ["622"])

    def test_scheme_info_is_json_serialisable(self):
        # YAML parses `retrieved: 2026-09-09` as a date, which json cannot take.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.yaml"
            path.write_text(SYNTHETIC, encoding="utf-8")
            info = LocalSchemeLookup(vocabulary=str(path)).scheme_info()
            json.dumps(info)
            self.assertEqual(info["retrieved"], "2026-09-09")


class ShippedFreibergVocabulary(unittest.TestCase):
    """Die ausgelieferte Datei muss lesbar und vollständig sein."""

    def setUp(self):
        self.lookup = LocalSchemeLookup(vocabulary="freihand_fg")

    def test_it_is_found_and_complete(self):
        self.assertIsNotNone(self.lookup.vocabulary_path(), "freihand_fg nicht gefunden")
        self.assertGreaterEqual(len(self.lookup.entries()), 600)

    def test_every_entry_has_a_notation_and_a_label(self):
        bad = [e.notation for e in self.lookup.entries() if not e.notation or not e.label]
        self.assertEqual(bad, [])

    def test_provenance_is_recorded(self):
        info = self.lookup.scheme_info()
        self.assertTrue(info.get("source"))
        self.assertTrue(info.get("retrieved"))

    def test_a_real_freiberg_notation_places(self):
        result = self.lookup.lookup_shelf(notations=["DK 622.341"], max_results=3)
        self.assertEqual(result["candidates"][0]["notation"], "622.34")
        self.assertEqual(result["candidates"][0]["location"], "OG")

    def test_the_shelf_places_come_in_order_of_specificity(self):
        # 622.341 (Erzbergbau): the deepest place first, then its next level.
        result = self.lookup.lookup_shelf(notations=["DK 622.341"], max_results=6)
        places = [c["notation"] for c in result["candidates"]]
        self.assertEqual(places[0], "622.34")
        self.assertIn("622.3...", places)

    def test_far_ancestors_do_not_crowd_out_the_second_signal(self):
        # "62..." (Technik allgemein) is four levels up from 622.341 and adds
        # nothing a cataloguer needs.
        result = self.lookup.lookup_shelf(notations=["DK 622.341"], max_results=6)
        self.assertNotIn("62...", [c["notation"] for c in result["candidates"]])


class ShelfCandidatesStep(_SchemeTestCase):
    """Der deterministische Step: Rangfilter, Eingaben, leere Fälle."""

    def _call(self, **kwargs):
        from src.core.agents.deterministic_functions import shelf_candidates

        with patch("src.utils.lookups.build_lookup", return_value=self.lookup):
            return shelf_candidates(**kwargs)

    def _context(self, classifications, keywords=None):
        return SimpleNamespace(
            dk_classifications=classifications,
            extra={"final_keywords": keywords or []},
        )

    def test_only_the_core_notation_drives_the_placement(self):
        context = self._context([
            {"code": "DK 622.34", "type": "DK", "rank": "core"},
            {"code": "DK 069", "type": "DK", "rank": "additional"},
        ])
        result = self._call(context=context, config={"max_results": 8})
        self.assertNotIn("069", [c["notation"] for c in result["candidates"]])

    def test_without_a_core_the_additional_notations_are_used(self):
        context = self._context([{"code": "DK 069", "type": "DK", "rank": "additional"}])
        result = self._call(context=context)
        self.assertIn("069", [c["notation"] for c in result["candidates"]])

    def test_core_only_can_be_switched_off(self):
        context = self._context([
            {"code": "DK 622.34", "rank": "core"},
            {"code": "DK 069", "rank": "additional"},
        ])
        result = self._call(context=context, config={"core_only": False, "max_results": 8})
        self.assertIn("069", [c["notation"] for c in result["candidates"]])

    def test_keyword_dicts_are_unwrapped(self):
        context = self._context([], [{"keyword": "Arbeitsrecht", "gnd_id": "4002641-7"}])
        result = self._call(context=context)
        self.assertEqual([c["notation"] for c in result["candidates"]], ["349.2"])

    def test_a_disabled_plugin_yields_an_empty_result(self):
        from src.core.agents.deterministic_functions import shelf_candidates

        context = self._context([{"code": "DK 622.34", "rank": "core"}])
        with patch("src.utils.lookups.build_lookup", return_value=None):
            result = shelf_candidates(context=context)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["count"], 0)

    def test_nothing_to_place_is_not_an_error(self):
        result = self._call(context=self._context([], []))
        self.assertEqual(result["count"], 0)

    def test_prompt_text_names_the_scheme_and_the_places(self):
        context = self._context([{"code": "DK 622.34", "rank": "core"}])
        text = self._call(context=context)["prompt_text"]
        self.assertIn("Testsystematik", text)
        self.assertIn("622.34", text)
        self.assertIn("Standort: OG", text)

    def test_prompt_text_is_empty_without_candidates(self):
        # The workflow's `when:` guard hangs off exactly this.
        result = self._call(context=self._context([], []))
        self.assertEqual(result["prompt_text"], "")


class ShelfGuard(unittest.TestCase):
    """Die Auswahl darf nur aus der deterministischen Kandidatenliste stammen."""

    def _call(self, extra):
        from src.core.agents.deterministic_functions import filter_unauthorized_shelf

        return filter_unauthorized_shelf(context=SimpleNamespace(extra=extra))

    def _extra(self, selection):
        return {
            "shelf": {"candidates": [
                {"notation": "622.34", "label": "Bergbauzweige: Erze", "location": "OG"},
                {"notation": "622", "label": "Bergbau", "location": "OG"},
            ]},
            "shelf_selection": selection,
        }

    def test_a_candidate_is_kept_with_the_authority_data(self):
        result = self._call(self._extra({"notation": "622.34", "reason": "Erzgewinnung"}))
        self.assertEqual(result["shelf"]["notation"], "622.34")
        self.assertEqual(result["shelf"]["label"], "Bergbauzweige: Erze")
        self.assertEqual(result["shelf"]["selected_reason"], "Erzgewinnung")
        self.assertEqual(result["dropped"], [])

    def test_an_invented_notation_is_dropped(self):
        result = self._call(self._extra({"notation": "999.9", "reason": "erfunden"}))
        self.assertIsNone(result["shelf"])
        self.assertEqual(result["dropped"], ["999.9"])

    def test_a_bare_string_selection_is_accepted(self):
        result = self._call(self._extra("622"))
        self.assertEqual(result["shelf"]["notation"], "622")

    def test_no_selection_leaves_the_candidates_alone(self):
        # shelf_select switched off, or the model answered `null`.
        result = self._call({"shelf": {"candidates": [{"notation": "622"}]}})
        self.assertIsNone(result["shelf"])
        self.assertEqual(result["checked"], 0)

    def test_a_selection_without_candidates_cannot_pass(self):
        result = self._call({"shelf_selection": {"notation": "622.34"}})
        self.assertIsNone(result["shelf"])
        self.assertEqual(result["dropped"], ["622.34"])


class ShelfReachesTheResult(unittest.TestCase):
    """Der Standort muss aus dem Kontext bis in die Ausgaben durchkommen.

    Er lag zunächst nur in ``extra.shelf_final`` und tauchte nirgends auf: ein
    Ergebnis, das der Lauf erzeugt, aber niemand sieht. - Claude Generated
    """

    PAYLOAD = {
        "candidates": [
            {"notation": "622.34", "label": "Bergbauzweige: Erze", "location": "OG"},
            {"notation": "622.3", "label": "Einzelne Bergbauzweige", "location": "OG"},
        ],
        "unplaced": ["004.42"],
        "scheme": {"id": "freihand_fg", "label": "Freihandsystematik UB Freiberg"},
    }
    PLACEMENT = {
        "notation": "622.34", "label": "Bergbauzweige: Erze", "location": "OG",
        "selected_reason": "Erzgewinnung", "selected": True,
    }

    def _state(self, *, with_placement: bool):
        from src.core.agents.shared_context import SharedContext

        context = SharedContext()
        context.abstract = "Erzbergbau im Erzgebirge."
        context.extra["shelf"] = dict(self.PAYLOAD)
        if with_placement:
            context.extra["shelf_final"] = dict(self.PLACEMENT)
        return context.to_keyword_analysis_state()

    def _rendered(self, state) -> str:
        from src.utils.pipeline_formatters import render_pipeline_result

        lines = []

        class Renderer:
            """Collects everything the renderer emits.

            The result moved from log lines into an HTML card; a fake that
            only captured ``render_pipeline_log`` would have reported the
            placement as missing while it was there. - Claude Generated
            """

            def render_pipeline_log(self, text, kind="info"):
                lines.append(text)

            def render_html_block(self, html, *, kind=None, plain_text=""):
                lines.append(html)

            def render_markdown_block(self, markdown_text, *, kind=None):
                lines.append(markdown_text)

        render_pipeline_result(Renderer(), state)
        return "\n".join(lines)

    def test_the_context_carries_it_into_the_analysis_state(self):
        state = self._state(with_placement=True)
        self.assertEqual(state.shelf["placement"]["notation"], "622.34")
        self.assertEqual(state.shelf["unplaced"], ["004.42"])
        self.assertEqual(state.shelf["scheme"]["id"], "freihand_fg")

    def test_a_run_without_shelf_steps_keeps_the_field_empty(self):
        from src.core.agents.shared_context import SharedContext

        self.assertEqual(SharedContext().to_keyword_analysis_state().shelf, {})

    def test_the_shared_renderer_shows_the_placement(self):
        text = self._rendered(self._state(with_placement=True))
        self.assertIn("Aufstellung", text)
        self.assertIn("622.34", text)
        self.assertIn("Standort OG", text)
        self.assertIn("Erzgewinnung", text)

    def test_without_a_pick_the_renderer_shows_the_candidates(self):
        # This is the whole output when shelf_select is disabled.
        text = self._rendered(self._state(with_placement=False))
        self.assertIn("Vorschläge", text)
        self.assertIn("622.3", text)

    def test_the_unplaced_notations_are_named(self):
        self.assertIn("004.42", self._rendered(self._state(with_placement=True)))

    def test_a_run_without_shelf_renders_nothing_about_it(self):
        from src.core.agents.shared_context import SharedContext

        text = self._rendered(SharedContext().to_keyword_analysis_state())
        self.assertNotIn("Aufstellung", text)

    def test_the_webapp_payload_carries_it(self):
        from src.webapp.result_serialization import extract_results_from_analysis_state

        payload = extract_results_from_analysis_state(self._state(with_placement=True))
        self.assertEqual(payload["shelf"]["placement"]["notation"], "622.34")
        import json

        json.dumps(payload["shelf"])  # must stay serialisable

    def test_the_cli_prints_it_after_the_classifications(self):
        import io
        from contextlib import redirect_stdout

        from src.cli.formatters.protocol_formatters import format_step_dk_classification

        state = self._state(with_placement=True)
        state.dk_classifications = ["DK 622.341"]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            format_step_dk_classification(state)
        out = buffer.getvalue()
        self.assertIn("Aufstellung", out)
        self.assertIn("622.34", out)
        self.assertLess(out.index("DK 622.341"), out.index("Aufstellung"))


class WorkflowWiring(unittest.TestCase):
    """Die Steps hängen im 105er und nur dort."""

    def _workflow(self, name):
        from src.core.agents.workflow_loader import find_workflow_file, load_workflow

        return load_workflow(find_workflow_file(name), strict=True)

    def test_the_freiberg_workflow_ends_with_the_three_shelf_steps(self):
        steps = [s.id for s in self._workflow("alima_v51_105").steps]
        self.assertEqual(
            steps[-3:], ["shelf_candidates", "shelf_select", "shelf_guard"]
        )

    def test_the_steps_are_chained_and_functions_exist(self):
        from src.core.agents.registry import TOOL_FN_REGISTRY

        by_id = {s.id: s for s in self._workflow("alima_v51_105").steps}
        self.assertEqual(by_id["shelf_candidates"].depends_on, ["dk_postprocess"])
        self.assertEqual(by_id["shelf_select"].depends_on, ["shelf_candidates"])
        self.assertEqual(by_id["shelf_guard"].depends_on, ["shelf_select"])
        for step_id in ("shelf_candidates", "shelf_guard"):
            self.assertIn(by_id[step_id].raw.get("function"), TOOL_FN_REGISTRY)

    def test_the_llm_selection_is_gated_on_having_candidates(self):
        by_id = {s.id: s for s in self._workflow("alima_v51_105").steps}
        self.assertIn("extra.shelf.prompt_text", str(by_id["shelf_select"].condition))

    def test_the_cycle_budget_covers_every_step(self):
        workflow = self._workflow("alima_v51_105")
        enabled = [s for s in workflow.steps if s.enabled]
        budget = (workflow.meta_agent or {}).get("max_cycles", 0)
        self.assertGreater(budget, len(enabled), "max_cycles unter der Step-Zahl")

    def test_the_general_workflow_stays_free_of_the_house_system(self):
        steps = [s.id for s in self._workflow("alima_v51").steps]
        self.assertNotIn("shelf_candidates", steps)


if __name__ == "__main__":
    unittest.main()
