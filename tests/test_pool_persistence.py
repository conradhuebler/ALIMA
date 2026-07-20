"""Regression tests for persisting the canonical GND-pool payload - Claude Generated.

Two defects lived here, both found while making the classic↔agentic convergence
fixture realistic (the previous fixture carried empty ``classifications``, which
hid both):

1. **Batch saving crashed.** ``task_state_to_dict`` returned raw ``asdict``
   output, so the sets in the nested per-term view ("gnd_ids", and the code
   containers inside "classifications") reached ``json.dump`` unconverted:
   ``TypeError: Object of type set is not JSON serializable`` on every state
   that had a GND hit carrying classifications. ``BatchProcessor._save_result``
   swallowed it per item, so batch runs reported failures instead of results.

2. **``rvk`` was not an equal-rank system on reload.** ``convert_lists_to_sets``
   hardcoded ``{"ddc", "dk"}``, so after a JSON round-trip DK/DDC came back as
   sets while RVK stayed a list — contradicting the WP-D1 decision that the
   classification systems are equal-rank keys.
"""

from __future__ import annotations

import json
import unittest

from src.core.data_models import KeywordAnalysisState, SearchResult
from src.utils.classification_systems import SYSTEM_KEYS
from src.utils.pipeline_persistence import PipelineJsonManager


def _state_with_sets() -> KeywordAnalysisState:
    """A state shaped like the classic search step's real output.

    ``nested_from_aggregate`` builds ``gnd_ids`` and every ``classifications``
    code container as a fresh ``set`` — reproduced verbatim here so the test
    fails if the save path stops converting them.
    """
    return KeywordAnalysisState(
        original_abstract="Limnologische Studien.",
        initial_keywords=["Limnologie"],
        search_suggesters_used=["lobid", "swb"],
        search_results=[
            SearchResult(
                search_term="Limnologie",
                results={
                    "Limnologie": {
                        "count": 1,
                        "display_count": 17,
                        "gnd_ids": {"4035769-7"},
                        "classifications": {
                            "DK": {"556.55"},
                            "DDC": {"551.48"},
                            "RVK": {"WI 5000"},
                        },
                    }
                },
            )
        ],
    )


class TestPoolStateIsJsonDumpable(unittest.TestCase):
    def test_task_state_to_dict_output_is_json_dumpable(self):
        """The exact BatchProcessor._save_result sequence must not raise."""
        out = PipelineJsonManager.task_state_to_dict(_state_with_sets())
        dumped = json.dumps(out, ensure_ascii=False)  # raised TypeError before
        payload = json.loads(dumped)["search_results"][0]["results"]["Limnologie"]

        self.assertEqual(sorted(payload["gnd_ids"]), ["4035769-7"])
        self.assertEqual(
            {sys: sorted(codes) for sys, codes in payload["classifications"].items()},
            {"DK": ["556.55"], "DDC": ["551.48"], "RVK": ["WI 5000"]},
        )
        # The count pair must survive: ``count`` is the ranking placeholder,
        # ``display_count`` the real Häufigkeit (the 038738e convention).
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["display_count"], 17)

    def test_dumpable_even_with_bare_sets_anywhere(self):
        """Conversion is recursive, not a special case for known field names."""
        state = _state_with_sets()
        state.initial_gnd_classes = ["31"]
        out = PipelineJsonManager.task_state_to_dict(state)
        json.dumps(out)  # must not raise


class TestClassificationSystemsRoundTripEqually(unittest.TestCase):
    def test_every_system_key_becomes_a_set_again(self):
        """DK/DDC/RVK must be equal-rank on reload — RVK used to stay a list."""
        original = _state_with_sets()
        reloaded = PipelineJsonManager.convert_lists_to_sets(
            json.loads(json.dumps(PipelineJsonManager.task_state_to_dict(original)))
        )
        payload = reloaded["search_results"][0]["results"]["Limnologie"]

        for system in ("DK", "DDC", "RVK"):
            with self.subTest(system=system):
                self.assertIsInstance(
                    payload["classifications"][system],
                    set,
                    f"{system} did not round-trip back to a set",
                )

    def test_set_fields_are_derived_from_the_shared_registry(self):
        """Adding a system to the registry must not need an edit here.

        Guards the fix itself: a hardcoded pair is what let RVK drift.
        """
        for system in SYSTEM_KEYS:
            with self.subTest(system=system):
                converted = PipelineJsonManager.convert_lists_to_sets(
                    {"classifications": {system: ["x"]}}
                )
                self.assertIsInstance(converted["classifications"][system], set)

    def test_gnd_ids_stays_a_list(self):
        """The display-order contract: gnd_ids must NOT become a set on reload."""
        reloaded = PipelineJsonManager.convert_lists_to_sets(
            {"gnd_ids": ["4035769-7", "4127654-7"]}
        )
        self.assertEqual(reloaded["gnd_ids"], ["4035769-7", "4127654-7"])


if __name__ == "__main__":
    unittest.main()
