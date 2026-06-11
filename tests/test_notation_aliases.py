#!/usr/bin/env python3
"""Claude Generated - Tests for the notation_* aliases (WS2).

The dk_* fields stay canonical (saved-state JSON back-compat); notation_* are
preferred read/write aliases for new, system-agnostic code.
"""

import unittest
from dataclasses import asdict

from src.core.data_models import KeywordAnalysisState
from src.core.agents.shared_context import SharedContext


class TestNotationAliases(unittest.TestCase):

    def _kas(self):
        return KeywordAnalysisState(
            original_abstract="a", initial_keywords=[], search_suggesters_used=[]
        )

    def test_keyword_analysis_state_aliases_round_trip(self):
        s = self._kas()
        # write via notation_*, read via dk_* (and the existing classifications alias)
        s.notation_codes = ["DDC 530", "DK 53"]
        self.assertEqual(s.dk_classifications, ["DDC 530", "DK 53"])
        self.assertEqual(s.classifications, ["DDC 530", "DK 53"])
        # write via dk_*, read via notation_*
        s.dk_search_results = [{"keyword": "x"}]
        self.assertEqual(s.notation_search_results, [{"keyword": "x"}])
        s.notation_search_results_flattened = [{"dk": "004"}]
        self.assertEqual(s.dk_search_results_flattened, [{"dk": "004"}])
        s.notation_statistics = {"n": 1}
        self.assertEqual(s.dk_statistics, {"n": 1})

    def test_serialization_keeps_dk_keys(self):
        # asdict() serializes the canonical dk_* fields (back-compat); notation_*
        # are properties and must NOT appear as separate keys. - Claude Generated
        s = self._kas()
        s.notation_codes = ["DDC 530"]
        d = asdict(s)
        self.assertIn("dk_classifications", d)
        self.assertEqual(d["dk_classifications"], ["DDC 530"])
        self.assertNotIn("notation_codes", d)
        self.assertNotIn("notation_search_results", d)

    def test_shared_context_aliases_round_trip(self):
        c = SharedContext()
        c.notation_codes = [{"code": "DDC 530"}]
        self.assertEqual(c.dk_classifications, [{"code": "DDC 530"}])
        c.dk_search_results = [{"keyword": "x"}]
        self.assertEqual(c.notation_search_results, [{"keyword": "x"}])
        c.notation_catalog_stats = {"total": 3}
        self.assertEqual(c.dk_catalog_stats, {"total": 3})


if __name__ == "__main__":
    unittest.main()
