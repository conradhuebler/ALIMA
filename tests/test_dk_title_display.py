#!/usr/bin/env python3
"""Claude Generated - Regression tests for the DK title-list display.

Two prior bugs:
  1. The review tab's title lookup only understood the FLAT dk_search_results
     shape, so the agentic pipeline's keyword-centric shape yielded no titles
     (the "DK overview has no title list" report).
  2. Title lists were truncated with no way to see the rest — now rendered in a
     collapsible <details> by DkTableRenderer.

These tests exercise the renderer directly and the lookup via the unbound
method (no QApplication needed).
"""

import types
import unittest

from src.ui.renderers.dk_table import DkTableRenderer
from src.ui.analysis_review_tab import AnalysisReviewTab


class _Stub:
    """Bind the two methods under test onto a bare object (no Qt widget)."""
    _split_classification_code = AnalysisReviewTab.__dict__["_split_classification_code"]
    _get_titles_for_classification = AnalysisReviewTab.__dict__["_get_titles_for_classification"]


def _stub(results):
    s = _Stub()
    s.current_analysis = types.SimpleNamespace(dk_search_results=results)
    return s


class TestGetTitlesForClassification(unittest.TestCase):

    def test_keyword_centric_shape_yields_titles(self):
        # Agentic pipeline shape: [{keyword, classifications:[{dk, titles}]}]
        results = [{
            "keyword": "Quantenmechanik", "source": "finc",
            "classifications": [
                {"dk": "530.145", "classification_type": "DK", "titles": ["Buch A", "Buch B"]},
                {"dk": "530.1", "classification_type": "DK", "titles": ["Buch C"]},
            ],
        }]
        titles, count = _stub(results)._get_titles_for_classification("DK 530.145")
        self.assertEqual(titles, ["Buch A", "Buch B"])
        self.assertEqual(count, 2)

    def test_flat_shape_still_works(self):
        results = [{"dk": "504.53", "classification_type": "DK", "titles": ["X", "Y"]}]
        titles, count = _stub(results)._get_titles_for_classification("DK 504.53")
        self.assertEqual(titles, ["X", "Y"])
        self.assertEqual(count, 2)

    def test_titles_aggregated_and_deduped_across_keywords(self):
        # Same DK appears under two keywords -> titles merged, duplicates dropped
        results = [
            {"keyword": "A", "classifications": [{"dk": "54", "type": "DK", "titles": ["T1", "T2"]}]},
            {"keyword": "B", "classifications": [{"dk": "54", "type": "DK", "titles": ["T2", "T3"]}]},
        ]
        titles, count = _stub(results)._get_titles_for_classification("DK 54")
        self.assertEqual(titles, ["T1", "T2", "T3"])
        self.assertEqual(count, 3)

    def test_rvk_type_is_respected(self):
        results = [{"keyword": "A", "classifications": [
            {"dk": "1000", "classification_type": "RVK", "titles": ["R1"]},
            {"dk": "1000", "classification_type": "DK", "titles": ["D1"]},
        ]}]
        self.assertEqual(_stub(results)._get_titles_for_classification("RVK 1000")[0], ["R1"])
        self.assertEqual(_stub(results)._get_titles_for_classification("DK 1000")[0], ["D1"])

    def test_no_match_returns_empty(self):
        results = [{"keyword": "A", "classifications": [{"dk": "99", "type": "DK", "titles": ["x"]}]}]
        self.assertEqual(_stub(results)._get_titles_for_classification("DK 530.145"), ([], 0))

    def test_max_titles_caps_returned_list_but_not_count(self):
        results = [{"keyword": "A", "classifications": [
            {"dk": "54", "type": "DK", "titles": [f"T{i}" for i in range(10)]}]}]
        titles, count = _stub(results)._get_titles_for_classification("DK 54", max_titles=3)
        self.assertEqual(len(titles), 3)
        self.assertEqual(count, 10)


class TestDkTableCollapsibleRender(unittest.TestCase):

    def test_titles_rendered_in_collapsible_details(self):
        html = DkTableRenderer().render_html(
            [{"dk": "DK 530.145", "titles": ["T1", "T2", "T3", "T4"], "count": 4}]
        )
        self.assertIn("<details>", html)
        self.assertIn("<summary", html)
        # Full list present (no 3/5-title truncation inside the fold)
        for t in ("T1", "T2", "T3", "T4"):
            self.assertIn(t, html)

    def test_no_titles_shows_placeholder_not_details(self):
        html = DkTableRenderer().render_html([{"dk": "DK 1", "titles": [], "count": 0}])
        self.assertIn("Keine Titel gefunden", html)
        self.assertNotIn("<details>", html)


if __name__ == "__main__":
    unittest.main()
