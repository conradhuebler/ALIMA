"""What a task really runs on, and which settings never get consulted.

The priority list of a task reads like a fallback chain, and the interface
presented it as one ("von oben nach unten, bis eines antwortet"). It is not:

* ``PipelineConfig.create_from_provider_preferences`` copies only
  ``model_priority[0]`` onto a step, and only where the task name *is* a step id;
* at run time ``_resolve_provider_smart`` returns an explicit step provider/model
  without testing reachability, so a dead first entry fails the step instead of
  moving on;
* ``SmartProviderSelector`` — the fallback that does run — does not read task
  preferences at all (explicit override → pipeline default → first available).

So on a real installation, three of six configured task preferences reach a run
and ten priority entries never do. The table states that instead of promising a
chain. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.ui.settings.effective_models import TASK_PREFERENCE_STEPS, effective_rows
from src.utils.config_models import (
    TaskPreference,
    TaskType,
    UnifiedProvider,
    UnifiedProviderConfig,
)


def _config(**prefs) -> UnifiedProviderConfig:
    cfg = UnifiedProviderConfig(providers=[
        UnifiedProvider(name="A", provider_type="ollama", enabled=True,
                        preferred_model="a1"),
        UnifiedProvider(name="B", provider_type="ollama", enabled=True,
                        preferred_model="b1"),
    ])
    cfg.pipeline_default_provider = "A"
    cfg.pipeline_default_model = "a1"
    for task, entries in prefs.items():
        cfg.task_preferences[task] = TaskPreference(
            task_type=TaskType(task),
            model_priority=[{"provider_name": p, "model_name": m} for p, m in entries],
        )
    return cfg


def _row(rows, needle):
    return next(r for r in rows if needle.lower() in r.task.lower())


class TheStepIdSetIsPinnedAgainstTheRealBuilderTest(unittest.TestCase):
    """If the builder gains or loses a step, this must fail, not drift silently."""

    def test_the_constant_matches_what_the_builder_produces(self):
        from unittest.mock import MagicMock

        from src.core.pipeline_manager import PipelineConfig

        cm = MagicMock()
        cm.get_unified_config.return_value = _config()
        built = PipelineConfig.create_from_provider_preferences(cm)

        llm_steps = {
            step_id for step_id, cfg in built.step_configs.items()
            if cfg.provider is not None
        }
        self.assertEqual(
            set(TASK_PREFERENCE_STEPS) & llm_steps, set(TASK_PREFERENCE_STEPS),
            "TASK_PREFERENCE_STEPS nennt einen Schritt, den der Erbauer nicht baut",
        )
        unknown = {"classification", "vision", "chunked"} & llm_steps
        self.assertEqual(
            unknown, set(),
            f"Diese Aufgaben haben jetzt doch einen Schritt: {unknown}",
        )


class APreferenceThatReachesAStepIsShownAsEffectiveTest(unittest.TestCase):
    def test_the_first_entry_wins(self):
        rows = effective_rows(_config(initialisation=[("B", "b1"), ("A", "a1")]))
        row = _row(rows, "Initialisation")
        self.assertEqual((row.provider, row.model), ("B", "b1"))
        self.assertEqual(row.source, "task_preference")

    def test_the_remaining_entries_are_named_as_unused(self):
        rows = effective_rows(_config(initialisation=[("B", "b1"), ("A", "a1")]))
        self.assertEqual(_row(rows, "Initialisation").unused, [("A", "a1")])

    def test_a_single_entry_leaves_nothing_unused(self):
        rows = effective_rows(_config(keywords=[("B", "b1")]))
        self.assertEqual(_row(rows, "Keywords").unused, [])


class APreferenceWithoutAStepIsNotShownAsEffectiveTest(unittest.TestCase):
    """The mistake this class exists to prevent: claiming a model that never runs."""

    def test_classification_falls_back_to_the_pipeline_default(self):
        rows = effective_rows(_config(classification=[("B", "b1"), ("A", "a1")]))
        row = _row(rows, "Classification")
        self.assertEqual((row.provider, row.model), ("A", "a1"))
        self.assertEqual(row.source, "pipeline_default_no_step")

    def test_the_whole_list_is_named_as_unused(self):
        rows = effective_rows(_config(vision=[("B", "b1")]))
        self.assertEqual(_row(rows, "Vision").unused, [("B", "b1")])

    def test_chunked_too(self):
        rows = effective_rows(_config(chunked=[("B", "b1")]))
        self.assertEqual(_row(rows, "Chunked").source, "pipeline_default_no_step")


class WithoutAnyPreferenceTest(unittest.TestCase):
    def test_every_task_shows_the_pipeline_default(self):
        rows = effective_rows(_config())
        for row in rows[:-1]:
            self.assertEqual((row.provider, row.model), ("A", "a1"), row.task)
            self.assertEqual(row.unused, [])

    def test_there_is_a_row_for_chat_and_agentic_runs(self):
        rows = effective_rows(_config())
        self.assertIn("agentisch", rows[-1].task.lower())

    def test_an_auto_entry_is_not_treated_as_a_choice(self):
        rows = effective_rows(_config(initialisation=[("auto", ""), ("B", "b1")]))
        row = _row(rows, "Initialisation")
        self.assertEqual(row.provider, "B")

    def test_every_source_key_has_a_label(self):
        from src.ui.settings.effective_models import SOURCE_LABELS

        for row in effective_rows(_config(initialisation=[("B", "b1")],
                                          vision=[("A", "a1")])):
            self.assertIn(row.source, SOURCE_LABELS, row.task)


if __name__ == "__main__":
    unittest.main()
