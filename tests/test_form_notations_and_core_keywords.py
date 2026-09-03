"""Form/provenance notations get marked; the RSWK core survives the pipeline.

Two defects observed on the 2026-09-03 materials-chemistry runs:

* The classification prompt ranks catalog candidates by frequency and calls
  frequency a relevance indicator. DK 378.245 (Hochschulschrift) leads that
  ranking in a holdings stock full of dissertations — it was the top candidate
  in 18 and among the top three in 36 of the 103 recorded runs. The prompt now
  says form notations are not subject notations; an assignment that happens
  anyway is marked so a cataloguer sees it.
* Both runs enumerated every material class of a survey work as a separate
  heading. The selection step now names a 2-5 heading RSWK core alongside the
  full retrieval list, and the form headings separately.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.agents.deterministic_functions import verify_final_keywords
from src.utils.classification_systems import FORM_NOTATIONS, form_notation_label
from src.utils.pipeline_formatters import PipelineResultFormatter
from src.webapp.result_serialization import build_structured_classifications

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ["alima_v51.yaml", "alima_v51_105.yaml"]


class TestFormNotationRegistry(unittest.TestCase):
    def test_known_form_notations(self):
        self.assertEqual(
            form_notation_label("DK", "378.245"), "Hochschulschrift, Dissertation"
        )
        self.assertEqual(form_notation_label("DK", "061.3"), "Kongress-, Tagungsband")

    def test_system_is_normalized(self):
        self.assertEqual(
            form_notation_label("dk", "378.245"), "Hochschulschrift, Dissertation"
        )

    def test_subject_notations_are_not_form(self):
        """Neighbours of the form codes carry real subject content."""
        for code in ("378", "620.1", "004", "006", "548"):
            self.assertIsNone(form_notation_label("DK", code), code)

    def test_match_is_exact_not_prefix(self):
        self.assertIsNone(form_notation_label("DK", "378.2451"))

    def test_only_dk_is_populated(self):
        """DDC/RVK form codes are not claimed until they are verified."""
        self.assertEqual(list(FORM_NOTATIONS), ["DK"])
        self.assertIsNone(form_notation_label("DDC", "378.245"))


class TestFormNotationMarking(unittest.TestCase):
    def test_marked_in_structured_classifications(self):
        entries = build_structured_classifications(
            ["DK 378.245", "DK 620.1", {"system": "DK", "code": "061.3", "display": "DK 061.3"}]
        )
        self.assertEqual(entries[0]["form_notation"], "Hochschulschrift, Dissertation")
        self.assertNotIn("form_notation", entries[1])
        self.assertEqual(entries[2]["form_notation"], "Kongress-, Tagungsband")

    def test_marked_for_plain_strings_in_the_render_layer(self):
        """A string that never passed the serializer is still resolved."""
        entries = PipelineResultFormatter.normalize_classifications(
            ["DK 378.245", "DK 620.1"]
        )
        self.assertEqual(entries[0]["form_notation"], "Hochschulschrift, Dissertation")
        self.assertIsNone(entries[1]["form_notation"])

    def test_badge_card_shows_the_mark(self):
        html = PipelineResultFormatter.format_classification_badge_card_html(
            PipelineResultFormatter.normalize_classifications(["DK 378.245", "DK 620.1"])
        )
        self.assertIn("Formnotation: Hochschulschrift, Dissertation", html)
        self.assertEqual(html.count("Formnotation:"), 1)

    def test_rvk_validation_is_untouched(self):
        entries = build_structured_classifications([{"system": "RVK", "code": "ZM 3000"}])
        self.assertEqual(entries[0]["validation_status"], "not_checked")
        self.assertNotIn("form_notation", entries[0])


class TestCoreKeywordVerification(unittest.TestCase):
    """verify_final_keywords aligns the core with the verified GND pool."""

    def _context(self):
        ctx = MagicMock()
        ctx.extra = {
            "final_keywords": [
                {"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"},
                {"keyword": "Recycling", "gnd_id": "4076573-8"},
            ],
            "core_keywords": [
                {"keyword": "Werkstoffkunde", "gnd_id": "WRONG-ID"},
                {"keyword": "Nicht im Pool", "gnd_id": "9999999-9"},
            ],
            "form_keywords": [{"keyword": "Recycling", "gnd_id": ""}],
        }
        ctx.selected_keywords = []
        ctx.gnd_entries = [
            {"title": "Werkstoffkunde", "gnd_ids": ["4079184-1"]},
            {"title": "Recycling", "gnd_ids": ["4076573-8"]},
        ]
        return ctx

    def test_core_is_filtered_and_ids_are_reattached(self):
        ctx = self._context()
        verify_final_keywords(context=ctx)
        self.assertEqual(
            ctx.extra["core_keywords"],
            [{"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"}],
        )
        self.assertEqual(
            ctx.extra["form_keywords"],
            [{"keyword": "Recycling", "gnd_id": "4076573-8"}],
        )

    def test_absent_core_is_left_absent(self):
        ctx = self._context()
        ctx.extra.pop("core_keywords")
        ctx.extra.pop("form_keywords")
        verify_final_keywords(context=ctx)
        self.assertNotIn("core_keywords", ctx.extra)


class TestCoreKeywordsReachTheResult(unittest.TestCase):
    """From SharedContext.extra through the state to the rendered output."""

    def test_shared_context_carries_core_into_the_analysis_state(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="Ein Überblick zur Materialchemie.")
        ctx.extra = {
            "final_keywords": [{"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"}],
            "core_keywords": [{"keyword": "Werkstoffkunde", "gnd_id": "4079184-1"}],
            "form_keywords": [{"keyword": "Lehrbuch", "gnd_id": "4123623-3"}],
        }
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.core_keywords, ["Werkstoffkunde (GND-ID: 4079184-1)"])
        self.assertEqual(state.form_keywords, ["Lehrbuch (GND-ID: 4123623-3)"])

    def test_render_shows_the_core_before_the_full_list(self):
        from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
        from src.utils.pipeline_formatters import render_pipeline_result

        state = KeywordAnalysisState(
            original_abstract="x", initial_keywords=[], search_suggesters_used=[]
        )
        state.core_keywords = ["Werkstoffkunde (GND-ID: 4079184-1)"]
        state.form_keywords = ["Lehrbuch (GND-ID: 4123623-3)"]
        state.final_llm_analysis = LlmKeywordAnalysis(
            task_name="selection",
            model_used="m",
            provider_used="p",
            prompt_template="",
            filled_prompt="",
            temperature=0.0,
            seed=0,
            response_full_text="",
            extracted_gnd_keywords=["Werkstoffkunde (GND-ID: 4079184-1)", "Keramik"],
        )

        logs = []
        renderer = MagicMock()
        renderer.render_pipeline_log.side_effect = lambda msg, kind=None: logs.append(msg)
        render_pipeline_result(renderer, state)

        core_idx = next(i for i, m in enumerate(logs) if "Kernschlagworte" in m)
        full_idx = next(i for i, m in enumerate(logs) if "GND-Schlagworte gesamt" in m)
        self.assertLess(core_idx, full_idx)
        self.assertIn("Lehrbuch", logs[core_idx])

    def test_render_without_a_core_keeps_the_old_wording(self):
        from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
        from src.utils.pipeline_formatters import render_pipeline_result

        state = KeywordAnalysisState(
            original_abstract="x", initial_keywords=[], search_suggesters_used=[]
        )
        state.final_llm_analysis = LlmKeywordAnalysis(
            task_name="selection",
            model_used="m",
            provider_used="p",
            prompt_template="",
            filled_prompt="",
            temperature=0.0,
            seed=0,
            response_full_text="",
            extracted_gnd_keywords=["Keramik"],
        )
        logs = []
        renderer = MagicMock()
        renderer.render_pipeline_log.side_effect = lambda msg, kind=None: logs.append(msg)
        render_pipeline_result(renderer, state)

        self.assertFalse(any("Kernschlagworte" in m for m in logs))
        self.assertTrue(any("GND-Schlagworte ausgewählt" in m for m in logs))


class TestPromptRules(unittest.TestCase):
    """The rules live in the workflow YAML — pin them so they are not lost."""

    def _text(self, name: str) -> str:
        return (REPO_ROOT / "workflows" / name).read_text(encoding="utf-8")

    def test_both_workflows_carry_the_form_notation_rule(self):
        for name in WORKFLOWS:
            text = self._text(name)
            self.assertIn("Formnotationen sind keine Sachnotationen", text, name)
            self.assertIn("378.245", text, name)

    def test_both_workflows_carry_the_survey_rule(self):
        for name in WORKFLOWS:
            self.assertIn("Gesamtdarstellung vs. Spezialwerk", self._text(name), name)

    def test_ten_is_a_ceiling_not_a_target(self):
        for name in WORKFLOWS:
            text = self._text(name)
            self.assertIn("Obergrenze, nicht das Ziel", text, name)
            # The old imperative "10 passende … ermitteln" made both models
            # deliver exactly ten.
            self.assertNotIn("um **10 passende", text, name)

    def test_selection_declares_core_and_form_outputs(self):
        from src.core.agents.workflow_loader import load_workflow

        for name in WORKFLOWS:
            wf = load_workflow(REPO_ROOT / "workflows" / name, strict=False)
            selection = next(s for s in wf.steps if s.id == "selection")
            self.assertEqual(
                selection.outputs.get("extra.core_keywords"), "response.core_keywords", name
            )
            self.assertEqual(
                selection.outputs.get("extra.form_keywords"), "response.form_keywords", name
            )
            classification = next(s for s in wf.steps if s.id == "classification")
            self.assertEqual(
                classification.inputs.get("core_keywords"), "${extra.core_keywords}", name
            )

    def test_absent_core_renders_as_an_empty_block(self):
        """A run whose model named no core must not print "None" into the prompt."""
        from src.core.agents.context_path import resolve_mapping
        from src.core.agents.prompt_resolver import _render
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.extra = {"final_keywords": [{"keyword": "Werkstoffkunde", "gnd_id": "1"}]}
        values = resolve_mapping(
            {
                "core_keywords": "${extra.core_keywords}",
                "final_keywords": "${extra.final_keywords}",
            },
            ctx,
        )
        rendered = _render("Kern:{core_keywords}|Alle:{final_keywords}", values)
        self.assertEqual(rendered.split("|")[0], "Kern:")
        self.assertIn("Werkstoffkunde", rendered)

    def test_rvk_lookup_gets_subject_headings_only(self):
        for name in WORKFLOWS:
            self.assertIn("nur Sachschlagwörter", self._text(name), name)


if __name__ == "__main__":
    unittest.main()
