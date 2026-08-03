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
                            "DK": [{"code": "556.55", "origin": "cooccurrence",
                                    "count": 7}],
                            "DDC": [{"code": "551.48", "origin": "authority"}],
                            "RVK": [
                                {"code": "WI 4700", "origin": "cooccurrence",
                                 "count": 13},
                                {"code": "WI 4800", "origin": "cooccurrence",
                                 "count": 6},
                            ],
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
            {
                system: [e["code"] for e in entries]
                for system, entries in payload["classifications"].items()
            },
            {"DK": ["556.55"], "DDC": ["551.48"], "RVK": ["WI 4700", "WI 4800"]},
        )
        # origin/count survive too — they are what distinguishes an authority
        # statement from statistical evidence.
        self.assertEqual(payload["classifications"]["DDC"][0]["origin"], "authority")
        self.assertEqual(payload["classifications"]["RVK"][0]["count"], 13)
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


class TestClassificationEntriesSurviveRoundTrip(unittest.TestCase):
    """Since the WP-D2 entry shape the system keys hold LISTS OF DICTS.

    They are deliberately no longer in ``SET_FIELDS``: entry dicts are
    unhashable (converting would raise) and their order carries the ranking
    (authority first, then descending co-occurrence), which a set destroys.
    """

    def test_entries_round_trip_unchanged_including_order(self):
        original = _state_with_sets()
        dumped = json.dumps(PipelineJsonManager.task_state_to_dict(original))
        reloaded = PipelineJsonManager.convert_lists_to_sets(json.loads(dumped))
        payload = reloaded["search_results"][0]["results"]["Limnologie"]

        for system in ("DK", "DDC", "RVK"):
            with self.subTest(system=system):
                entries = payload["classifications"][system]
                self.assertIsInstance(entries, list)
                self.assertTrue(all(isinstance(e, dict) for e in entries))

        self.assertEqual(
            [e["code"] for e in payload["classifications"]["RVK"]],
            [e["code"] for e in original.search_results[0].results["Limnologie"]
             ["classifications"]["RVK"]],
            "entry order (= the ranking) changed across the round trip",
        )

    def test_system_keys_are_not_set_converted(self):
        """A set-conversion here would raise on unhashable entry dicts."""
        converted = PipelineJsonManager.convert_lists_to_sets(
            {"classifications": {system: [{"code": "x", "origin": "authority"}]
                                 for system in SYSTEM_KEYS}}
        )
        for system in SYSTEM_KEYS:
            with self.subTest(system=system):
                self.assertIsInstance(converted["classifications"][system], list)

    def test_input_record_classifications_survive_save_and_load(self):
        """WP-D1 P2: the record-prior field round-trips through a real file;
        pre-P2 saves (key absent) load with the default empty dict."""
        import os
        import tempfile

        state = _state_with_sets()
        state.input_record_classifications = {
            "DK": [{"code": "556.55", "origin": "authority"}]
        }
        state.input_record_gnd_subjects = [{"term": "Limnologie", "gnd_id": "4074296-3"}]
        path = os.path.join(tempfile.mkdtemp(), "state.json")
        PipelineJsonManager.save_analysis_state(state, path)
        reloaded = PipelineJsonManager.load_analysis_state(path)
        self.assertEqual(
            reloaded.input_record_classifications,
            {"DK": [{"code": "556.55", "origin": "authority"}]},
        )
        self.assertEqual(
            reloaded.input_record_gnd_subjects,
            [{"term": "Limnologie", "gnd_id": "4074296-3"}],
        )

        # legacy save without the field
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        del data["input_record_classifications"]
        del data["input_record_gnd_subjects"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        legacy = PipelineJsonManager.load_analysis_state(path)
        self.assertEqual(legacy.input_record_classifications, {})
        self.assertEqual(legacy.input_record_gnd_subjects, [])

    def test_missing_concepts_still_becomes_a_set(self):
        """The one remaining set field must not have been lost in the change."""
        converted = PipelineJsonManager.convert_lists_to_sets(
            {"missing_concepts": ["a", "b"]}
        )
        self.assertEqual(converted["missing_concepts"], {"a", "b"})

    def test_gnd_ids_stays_a_list(self):
        """The display-order contract: gnd_ids must NOT become a set on reload."""
        reloaded = PipelineJsonManager.convert_lists_to_sets(
            {"gnd_ids": ["4035769-7", "4127654-7"]}
        )
        self.assertEqual(reloaded["gnd_ids"], ["4035769-7", "4127654-7"])


if __name__ == "__main__":
    unittest.main()
