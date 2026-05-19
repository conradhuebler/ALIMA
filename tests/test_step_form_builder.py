"""Tests for src/ui/forms/step_form_builder.py (P-γ). Claude Generated.

Qt-free. Loads workflows/alima_classic.yaml and exercises the form builder
+ cascade-prerequisite resolver against an empty and a partially-filled
SharedContext.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.core.agents.shared_context import SharedContext
from src.core.agents.workflow_loader import load_workflow
from src.ui.forms.step_form_builder import (
    build_step_form,
    find_missing_prerequisites,
)


WORKFLOW_PATH = (
    Path(__file__).parent.parent / "workflows" / "alima_classic.yaml"
)


class TestBuildStepForm(unittest.TestCase):
    def setUp(self):
        self.workflow = load_workflow(WORKFLOW_PATH)

    def test_user_fill_root_field(self):
        """`extraction.inputs.abstract = ${abstract}` is user_fill when context empty."""
        ctx = SharedContext()
        fields = build_step_form(self.workflow, "extraction", ctx)
        abstract = next(f for f in fields if f.name == "abstract")
        self.assertEqual(abstract.kind, "user_fill")
        self.assertEqual(abstract.widget, "text")
        self.assertTrue(abstract.value is None or abstract.value == "")

    def test_derived_root_field(self):
        """`selection_chunks.inputs.keywords = ${gnd_entries}` is derived from `search`."""
        ctx = SharedContext()
        fields = build_step_form(self.workflow, "selection_chunks", ctx)
        keywords = next(f for f in fields if f.name == "keywords")
        self.assertEqual(keywords.kind, "derived")
        self.assertEqual(keywords.widget, "readonly")
        self.assertTrue(keywords.missing)
        self.assertEqual(keywords.writer_step_id, "search")

    def test_derived_extra_field(self):
        """`classification.inputs.keywords = ${extra.dk_prompt_text}` is derived from dk_collect."""
        ctx = SharedContext()
        fields = build_step_form(self.workflow, "classification", ctx)
        keywords = next(f for f in fields if f.name == "keywords")
        self.assertEqual(keywords.kind, "derived")
        self.assertTrue(keywords.missing)
        self.assertEqual(keywords.writer_step_id, "dk_collect")

    def test_derived_steps_ref(self):
        """`dk_postprocess.inputs.dk_classifications = ${steps.classification.dk_classifications}`."""
        ctx = SharedContext()
        fields = build_step_form(self.workflow, "dk_postprocess", ctx)
        dk = next(f for f in fields if f.name == "dk_classifications")
        self.assertEqual(dk.kind, "derived")
        self.assertEqual(dk.writer_step_id, "classification")

    def test_derived_present_when_value_set(self):
        """When upstream value exists, missing=False and value populated."""
        ctx = SharedContext(
            abstract="An abstract about cadmium.",
            gnd_entries=[{"title": "Cadmium", "gnd_id": "4007249-3"}],
        )
        fields = build_step_form(self.workflow, "selection_chunks", ctx)
        keywords = next(f for f in fields if f.name == "keywords")
        self.assertFalse(keywords.missing)
        self.assertEqual(len(keywords.value), 1)


class TestFindMissingPrerequisites(unittest.TestCase):
    def setUp(self):
        self.workflow = load_workflow(WORKFLOW_PATH)

    def test_classification_with_only_abstract(self):
        """Only abstract present → full upstream chain required."""
        ctx = SharedContext(abstract="Cadmium toxicology in aquatic systems.")
        chain = find_missing_prerequisites(self.workflow, "classification", ctx)
        # Expected order from alima_classic.yaml: extraction → search →
        # selection_chunks → selection → dk_collect → classification
        self.assertEqual(
            chain,
            [
                "extraction",
                "search",
                "selection_chunks",
                "selection",
                "dk_collect",
                "classification",
            ],
        )

    def test_no_prereqs_when_upstream_filled(self):
        """Fully-populated context → chain is just [target]."""
        ctx = SharedContext(abstract="x")
        ctx.extra = {"dk_prompt_text": "non-empty"}
        chain = find_missing_prerequisites(self.workflow, "classification", ctx)
        self.assertEqual(chain, ["classification"])

    def test_partial_fill_only_missing_writers(self):
        """selected_keywords filled but extra.dk_prompt_text missing → only
        dk_collect (+ classification) needed."""
        ctx = SharedContext(
            abstract="x",
            selected_keywords=[{"keyword": "Cadmium", "gnd_id": "4007249-3"}],
        )
        chain = find_missing_prerequisites(self.workflow, "classification", ctx)
        # dk_collect declares depends_on=[selection]; selection writes
        # extra.final_keywords + keyword_chains + missing_concepts. Those
        # are not yet present, so selection is pulled in via depends_on too.
        self.assertEqual(chain[-1], "classification")
        self.assertIn("dk_collect", chain)
        # extraction + search should NOT be in chain (selected_keywords is set,
        # and selection_chunks writes selected_keywords).
        self.assertNotIn("extraction", chain)
        self.assertNotIn("search", chain)


if __name__ == "__main__":
    unittest.main()
