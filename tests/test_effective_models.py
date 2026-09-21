"""Which model runs where, and where that comes from.

Three axes are left after the Sep-2026 rework: the general default, the pipeline
default, and the vision default. Each is one row.

Per-task model choices are gone. They were a per-step choice for the classic
pipeline and only ever the *first* entry of a list that read like a fallback
chain and was not one: nothing consulted entries 2..n, a dead first entry failed
the step instead of moving on, and three of six configured tasks had no step to
land on at all. An agentic workflow names its model per step in its own YAML,
which is unaffected. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.ui.settings.effective_models import SOURCE_LABELS, effective_rows
from src.utils.config_models import (
    ProviderScope,
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
    for task, entries in prefs.items():
        cfg.task_preferences[task] = TaskPreference(
            task_type=TaskType(task),
            model_priority=[{"provider_name": p, "model_name": m} for p, m in entries],
        )
    return cfg


def _row(rows, needle):
    return next(r for r in rows if needle.lower() in r.task.lower())


class OneRowPerAxisTest(unittest.TestCase):
    def test_there_are_four_rows(self):
        rows = effective_rows(_config())
        self.assertEqual(
            [r.scope for r in rows], ["general", "pipeline", "agentic", "vision"]
        )

    def test_every_source_key_has_a_label(self):
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        cfg.vision_default_provider, cfg.vision_default_model = "B", "b1"
        for row in effective_rows(cfg):
            self.assertIn(row.source, SOURCE_LABELS, row.task)

    def test_the_agentic_row_names_the_workflow_override(self):
        """A workflow step may still carry its own model; the table must say so."""
        row = _row(effective_rows(_config()), "agentisch")
        self.assertIn("YAML", row.note)

    def test_an_inherited_row_is_marked_as_such(self):
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        rows = effective_rows(cfg)
        self.assertFalse(_row(rows, "Allgemein").is_fallback)
        self.assertTrue(_row(rows, "Pipeline").is_fallback)

    def test_an_own_choice_is_not_marked_as_inherited(self):
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        cfg.pipeline_default_provider, cfg.pipeline_default_model = "B", "b1"
        self.assertFalse(_row(effective_rows(cfg), "Pipeline").is_fallback)

    def test_nothing_configured_reports_no_provider(self):
        rows = effective_rows(UnifiedProviderConfig(providers=[]))
        for row in rows:
            self.assertFalse(row.configured)
            self.assertEqual(row.source, "none")


class TaskPreferencesNoLongerReachARunTest(unittest.TestCase):
    """The removal, stated as behaviour rather than as absence of code."""

    def test_the_step_config_uses_the_pipeline_default(self):
        from unittest.mock import MagicMock

        from src.core.pipeline_manager import PipelineConfig

        cfg = _config(initialisation=[("B", "b1")], keywords=[("B", "b1")])
        cfg.pipeline_default_provider, cfg.pipeline_default_model = "A", "a1"
        cm = MagicMock()
        cm.get_unified_config.return_value = cfg

        built = PipelineConfig.create_from_provider_preferences(cm)
        for step_id, step in built.step_configs.items():
            if step.provider is None:
                continue  # dk_search runs no LLM
            self.assertEqual(
                (step.provider, step.model), ("A", "a1"),
                f"{step_id} folgt noch einer Aufgaben-Voreinstellung",
            )

    def test_the_table_reports_no_task_preference_source(self):
        cfg = _config(initialisation=[("B", "b1")])
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        for row in effective_rows(cfg):
            self.assertNotEqual(row.source, "task_preference")


class VisionHasItsOwnDefaultTest(unittest.TestCase):
    """Image/OCR analysis is a capability question, not a pipeline step.

    It used to sit in ``task_preferences["vision"]`` and never reached a run:
    the vision path asked ``SmartProviderSelector``, which does not read task
    preferences, so image analysis quietly used the pipeline default — while the
    gate in front of it *demanded* a vision preference and refused to start
    without one. - Claude Generated
    """

    def test_the_vision_default_wins(self):
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        cfg.vision_default_provider, cfg.vision_default_model = "B", "b-vision"
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.VISION),
            ("B", "b-vision", "vision_default"),
        )

    def test_it_skips_the_pipeline_stage(self):
        """Vision is not a pipeline step, so a pipeline default must not apply."""
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        cfg.pipeline_default_provider, cfg.pipeline_default_model = "B", "b1"
        self.assertEqual(
            cfg.explain_default_provider_model(scope=ProviderScope.VISION),
            ("A", "a1", "preferred"),
        )

    def test_it_does_not_leak_into_the_pipeline_scope(self):
        cfg = _config()
        cfg.preferred_provider, cfg.preferred_model = "A", "a1"
        cfg.vision_default_provider, cfg.vision_default_model = "B", "b-vision"
        _p, _m, source = cfg.explain_default_provider_model(scope=ProviderScope.PIPELINE)
        self.assertNotEqual(source, "vision_default")


class TheLegacyVisionPreferenceIsAdoptedTest(unittest.TestCase):
    def test_it_is_carried_over(self):
        cfg = _config(vision=[("B", "b-vision")])
        self.assertTrue(cfg.adopt_vision_task_preference())
        self.assertEqual(
            (cfg.vision_default_provider, cfg.vision_default_model), ("B", "b-vision")
        )

    def test_an_explicit_vision_default_is_not_overwritten(self):
        cfg = _config(vision=[("B", "b-vision")])
        cfg.vision_default_provider = "A"
        self.assertFalse(cfg.adopt_vision_task_preference())
        self.assertEqual(cfg.vision_default_provider, "A")

    def test_it_is_idempotent(self):
        cfg = _config(vision=[("B", "b-vision")])
        self.assertTrue(cfg.adopt_vision_task_preference())
        self.assertFalse(cfg.adopt_vision_task_preference())

    def test_an_auto_entry_is_skipped(self):
        cfg = _config(vision=[("auto", ""), ("B", "b-vision")])
        cfg.adopt_vision_task_preference()
        self.assertEqual(cfg.vision_default_provider, "B")


class TheVisionGateChecksWhatItUsesTest(unittest.TestCase):
    def test_both_go_through_the_vision_scope(self):
        import inspect

        from src.utils import pipeline_input

        lines = [
            ln.strip()
            for ln in inspect.getsource(pipeline_input).splitlines()
            if ("task_preferences" in ln or "ProviderScope.VISION" in ln)
            and not ln.lstrip().startswith("#")
        ]
        self.assertTrue(
            any("ProviderScope.VISION" in ln for ln in lines),
            f"Vision-Scope wird nicht benutzt: {lines}",
        )
        leftovers = [ln for ln in lines if "task_preferences" in ln]
        self.assertEqual(leftovers, [], f"liest noch Task Preferences: {leftovers}")


if __name__ == "__main__":
    unittest.main()
