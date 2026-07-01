"""Regression tests for classic/agentic dk_search_results shape convergence.

Contract (src/core/data_models.py):
  * ``dk_search_results``            = keyword-centric [{keyword, classifications:[...]}]
  * ``dk_search_results_flattened``  = DK-centric flat  [{dk, titles, count, ...}]

The agentic dk_postprocess step (build_dk_search_results) used to overwrite the
keyword-centric ``dk_search_results`` with its flat merged list, diverging from
the classic pipeline and making the webapp DK-Suche render
"unbekannt: 0 Klassifikationen". It must write ``dk_search_results_flattened``
instead. - Claude Generated
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestBuildDkSearchResultsTarget(unittest.TestCase):
    def test_writes_flattened_not_keyword_centric(self):
        from src.core.agents.deterministic_functions import build_dk_search_results
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.dk_search_results = [
            {"keyword": "Cadmium", "classifications": [{"dk": "615.9", "titles": ["Tox"]}]}
        ]  # keyword-centric, set by dk_collect

        out = build_dk_search_results(
            dk_entries=[{"dk": "615.9", "titles": ["Heavy Metal Toxicity"], "count": 3}],
            dk_classifications=[{"code": "504.064", "title": "Env", "confidence": 0.8}],
            context=ctx,
        )

        # keyword-centric field must be untouched
        self.assertEqual(ctx.dk_search_results[0]["keyword"], "Cadmium")
        self.assertIn("classifications", ctx.dk_search_results[0])

        # flattened field must now hold the flat merged (dk-centric) list
        self.assertTrue(ctx.dk_search_results_flattened)
        first = ctx.dk_search_results_flattened[0]
        self.assertIn("dk", first)
        self.assertNotIn("classifications", first)
        self.assertEqual(out["count"], len(ctx.dk_search_results_flattened))


class TestToAnalysisStatePrefersRichFlattened(unittest.TestCase):
    def test_prefers_populated_flattened_over_derived(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.dk_search_results = [
            {"keyword": "Cadmium", "classifications": [{"dk": "615.9", "titles": ["Tox"]}]}
        ]
        # rich flattened from dk_postprocess (real catalog titles)
        ctx.dk_search_results_flattened = [
            {"dk": "615.9", "titles": ["Heavy Metal Toxicity", "Fluorides"], "count": 3}
        ]
        # a classification that WOULD produce a thinner derived entry if used
        ctx.dk_classifications = [{"code": "615.9", "title": "615.9", "confidence": 0.5}]

        state = ctx.to_keyword_analysis_state()

        # keyword-centric passes through
        self.assertIn("classifications", state.dk_search_results[0])
        # flattened is the rich one (2 real titles), not the derived thin one
        self.assertEqual(state.dk_search_results_flattened[0]["titles"],
                         ["Heavy Metal Toxicity", "Fluorides"])

    def test_falls_back_to_derived_when_flattened_empty(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="x")
        ctx.dk_classifications = [{"code": "615.9", "title": "Toxikologie", "confidence": 0.7}]
        # dk_search_results_flattened intentionally left empty (dk_postprocess skipped)

        state = ctx.to_keyword_analysis_state()

        self.assertTrue(state.dk_search_results_flattened)
        self.assertEqual(state.dk_search_results_flattened[0]["dk"], "615.9")


if __name__ == "__main__":
    unittest.main()
