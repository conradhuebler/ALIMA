"""Tests for KeywordAnalysisState mutation API (P-δ.1). Claude Generated."""

from __future__ import annotations

import unittest

from PyQt6.QtCore import QCoreApplication

from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
from src.core.state_bus import AlimaStateBus, reset


_qapp = QCoreApplication.instance() or QCoreApplication([])


def _empty_state(**overrides) -> KeywordAnalysisState:
    base = dict(
        original_abstract="abstract",
        initial_keywords=[],
        search_suggesters_used=[],
    )
    base.update(overrides)
    return KeywordAnalysisState(**base)


def _llm(task: str) -> LlmKeywordAnalysis:
    return LlmKeywordAnalysis(
        task_name=task,
        model_used="m",
        provider_used="p",
        prompt_template="t",
        filled_prompt="",
        temperature=0.0,
        seed=None,
        response_full_text="",
    )


class TestKasMutations(unittest.TestCase):
    def setUp(self):
        reset()
        self.events = []
        AlimaStateBus().subscribe(
            "state.changed", lambda d: self.events.append(d)
        )

    def tearDown(self):
        reset()

    def test_keyword_addition_appends_with_gnd(self):
        state = _empty_state(initial_keywords=["Boden"])
        ok = state.apply_keyword_addition("Cadmium", gnd_id="4007249-3")
        self.assertTrue(ok)
        self.assertEqual(
            state.initial_keywords,
            ["Boden", "Cadmium (GND-ID: 4007249-3)"],
        )
        self.assertEqual(self.events[-1]["op"], "keyword_addition")

    def test_keyword_addition_dedupes(self):
        state = _empty_state(
            initial_keywords=["Cadmium (GND-ID: 4007249-3)"]
        )
        ok = state.apply_keyword_addition("Cadmium", gnd_id="4007249-3")
        self.assertFalse(ok)

    def test_keyword_replacement_matches_canonical(self):
        state = _empty_state(initial_keywords=["Cd (GND-ID: 4007249-3)"])
        ok = state.apply_keyword_replacement("Cd", "Cadmium", gnd_id="4007249-3")
        self.assertTrue(ok)
        self.assertEqual(
            state.initial_keywords, ["Cadmium (GND-ID: 4007249-3)"]
        )
        self.assertEqual(self.events[-1]["op"], "keyword_replacement")

    def test_keyword_removal_finds_canonical(self):
        state = _empty_state(initial_keywords=["Cadmium (GND-ID: 4007249-3)"])
        ok = state.apply_keyword_removal("Cadmium")
        self.assertTrue(ok)
        self.assertEqual(state.initial_keywords, [])

    def test_classification_add_remove(self):
        state = _empty_state()
        state.dk_classifications = ["DK 577"]
        self.assertTrue(state.apply_classification_update("DK 631.4", "add"))
        self.assertEqual(state.dk_classifications, ["DK 577", "DK 631.4"])
        self.assertFalse(state.apply_classification_update("DK 577", "add"))
        self.assertTrue(state.apply_classification_update("DK 577", "remove"))
        self.assertEqual(state.dk_classifications, ["DK 631.4"])

    def test_step_result_override_initial(self):
        state = _empty_state()
        state.initial_llm_call_details = _llm("initialisation")
        ok = state.apply_step_result_override("initial", "model_used", "new-m")
        self.assertTrue(ok)
        self.assertEqual(state.initial_llm_call_details.model_used, "new-m")

    def test_step_result_override_rejects_unknown_step(self):
        state = _empty_state()
        ok = state.apply_step_result_override("nope", "model_used", "x")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
