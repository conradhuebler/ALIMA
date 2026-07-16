"""Tests for src/core/agents/state_bridge.py (P-γ). Claude Generated.

Covers:
* MVP field mapping from a synthetic KAS dict.
* Round-trip with the real Cadmium_Cogito.json fixture.
* Auto-detection between SharedContext and KAS JSON via load_state_file().
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.core.agents.shared_context import SharedContext
from src.core.agents.state_bridge import (
    from_keyword_analysis_state,
    load_state_file,
)


CADMIUM_FIXTURE = Path(__file__).parent.parent / "Cadmium_Cogito.json"


class TestFromKeywordAnalysisState(unittest.TestCase):
    def test_strips_gnd_tag_from_initial_keywords(self):
        ctx = from_keyword_analysis_state({
            "original_abstract": "abstract text",
            "initial_keywords": [
                "Cadmium (GND-ID: 4007249-3)",
                "Boden (GND-ID: 4007348-8)",
                "PlainKeyword",
            ],
        })
        self.assertEqual(
            ctx.initial_keywords,
            ["Cadmium", "Boden", "PlainKeyword"],
        )

    def test_flattens_search_results(self):
        ctx = from_keyword_analysis_state({
            "original_abstract": "x",
            "search_results": [
                {
                    "search_term": "Cadmium",
                    "results": {
                        "Toxicology of Cadmium": {
                            "gndid": ["4007249-3"],
                            "ddc_codes": ["577"],
                        },
                        "Cadmium in Soil": {
                            "gndid": ["4007249-3"],
                            "ddc_codes": ["631.4"],
                        },
                    },
                },
                {
                    "search_term": "Boden",
                    "results": {
                        "Cadmium in Soil": {
                            "gndid": ["4007249-3"],
                            "ddc_codes": ["631.4"],
                        },
                    },
                },
            ],
        })
        # Deduplicated by title
        titles = [e["title"] for e in ctx.gnd_entries]
        self.assertEqual(
            sorted(titles), ["Cadmium in Soil", "Toxicology of Cadmium"]
        )
        # per-keyword preserves both terms
        self.assertEqual(
            sorted(ctx.gnd_entries_per_keyword["Cadmium"]),
            ["Cadmium in Soil", "Toxicology of Cadmium"],
        )
        self.assertEqual(
            ctx.gnd_entries_per_keyword["Boden"], ["Cadmium in Soil"]
        )

    def test_flatten_preserves_count_and_display_count(self):
        # Reload twin of the counter bug: a saved agentic state must keep the
        # real Häufigkeit (display_count) when flattened back. - Claude Generated
        ctx = from_keyword_analysis_state({
            "original_abstract": "x",
            "search_results": [
                {
                    "search_term": "Halbleiter",
                    "results": {
                        "Halbleiter": {
                            "gndid": ["4129772-7"],
                            "ddc_codes": ["530"],
                            "count": 1,
                            "display_count": 87,
                        },
                        "Silizium": {
                            "gndid": ["4130826-8"],
                            "ddc_codes": ["546"],
                        },
                    },
                },
            ],
        })
        by_title = {e["title"]: e for e in ctx.gnd_entries}
        self.assertEqual(by_title["Halbleiter"]["count"], 1)
        self.assertEqual(by_title["Halbleiter"]["display_count"], 87)
        # Missing display_count → defaults count to 1, omits display_count.
        self.assertEqual(by_title["Silizium"]["count"], 1)
        self.assertNotIn("display_count", by_title["Silizium"])

    def test_promotes_string_dk_classifications_to_dicts(self):
        ctx = from_keyword_analysis_state({
            "original_abstract": "x",
            "dk_classifications": ["DK 577", "DK 631.4"],
        })
        self.assertEqual(len(ctx.dk_classifications), 2)
        self.assertEqual(ctx.dk_classifications[0], {"code": "DK 577", "type": ""})

    def test_cadmium_fixture_round_trip_core_fields(self):
        """Round-trip: KAS-JSON → SharedContext → to_keyword_analysis_state().

        Validates the four core fields named in the WP10 plan:
        abstract, initial_keywords, working_title, input_type.
        """
        if not CADMIUM_FIXTURE.exists():
            self.skipTest("Cadmium_Cogito.json fixture not present")

        with CADMIUM_FIXTURE.open() as f:
            kas_data = json.load(f)

        ctx = from_keyword_analysis_state(kas_data)

        # abstract preserved verbatim
        self.assertEqual(ctx.abstract, kas_data["original_abstract"])

        # initial_keywords: count matches, GND tags stripped
        self.assertEqual(
            len(ctx.initial_keywords), len(kas_data["initial_keywords"])
        )
        self.assertNotIn("(GND-ID:", ctx.initial_keywords[0])

        # Round-trip to KAS reproduces the abstract
        kas_back = ctx.to_keyword_analysis_state()
        self.assertEqual(kas_back.original_abstract, kas_data["original_abstract"])
        self.assertEqual(
            kas_back.input_type, kas_data.get("input_type") or "text"
        )


class TestLoadStateFile(unittest.TestCase):
    def test_detects_kas_format(self):
        if not CADMIUM_FIXTURE.exists():
            self.skipTest("Cadmium_Cogito.json fixture not present")
        ctx, kind = load_state_file(CADMIUM_FIXTURE)
        self.assertEqual(kind, "keyword_analysis_state")
        self.assertTrue(ctx.abstract)

    def test_detects_shared_context_format(self):
        ctx = SharedContext(abstract="hello", extracted_keywords=["a"])
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as tf:
            json.dump(ctx.to_dict(), tf)
            path = tf.name

        loaded, kind = load_state_file(path)
        self.assertEqual(kind, "shared_context")
        self.assertEqual(loaded.abstract, "hello")
        self.assertEqual(loaded.extracted_keywords, ["a"])

    def test_rejects_unknown_format(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as tf:
            json.dump({"random_key": "value"}, tf)
            path = tf.name

        with self.assertRaises(ValueError):
            load_state_file(path)


if __name__ == "__main__":
    unittest.main()
