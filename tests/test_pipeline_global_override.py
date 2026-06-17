"""Regression: the global LLM override must win over per-step pipeline defaults.

Bug (reported via the pipeline tab): selecting a model in the toolbar's "🤖 LLM"
combo set ``global_provider_override`` / ``global_model_override`` on the config
but never propagated it into the per-step configs — ``apply_global_override()``
only ran in ``__post_init__``, so every LLM step kept its per-step default
(e.g. ``gemma``). The classic executor reads ``step_config.provider/model``
directly, so the selection was silently ignored.

The GUI now calls ``apply_global_override()`` after setting the override; these
tests lock in the propagation it relies on. Claude Generated.
"""
from __future__ import annotations

import unittest

from src.core.pipeline_manager import PipelineConfig
from src.utils.config_models import PipelineStepConfig

_LLM_STEPS = ("initialisation", "keywords", "dk_classification")


def _baseline_config() -> PipelineConfig:
    """A config whose LLM steps all use the per-step default 'gemma'."""
    return PipelineConfig(
        step_configs={
            "initialisation": PipelineStepConfig(
                step_id="initialisation", provider="ollama", model="gemma"),
            "keywords": PipelineStepConfig(
                step_id="keywords", provider="ollama", model="gemma"),
            "dk_search": PipelineStepConfig(step_id="dk_search"),  # non-LLM
            "dk_classification": PipelineStepConfig(
                step_id="dk_classification", provider="ollama", model="gemma"),
        }
    )


class TestGlobalOverride(unittest.TestCase):
    def test_override_set_after_construction_propagates_to_all_llm_steps(self):
        cfg = _baseline_config()
        # Mirror the GUI path: the override is chosen at runtime, after the config
        # already exists, then apply_global_override() is invoked.
        cfg.global_provider_override = "GWDG"
        cfg.global_model_override = "glm-4.7"
        cfg.apply_global_override()

        for step in _LLM_STEPS:
            self.assertEqual(cfg.step_configs[step].provider, "GWDG",
                             f"{step} provider must follow the global override")
            self.assertEqual(cfg.step_configs[step].model, "glm-4.7",
                             f"{step} model must follow the global override")

    def test_non_llm_step_is_untouched(self):
        cfg = _baseline_config()
        cfg.global_provider_override = "GWDG"
        cfg.global_model_override = "glm-4.7"
        cfg.apply_global_override()
        # dk_search has no LLM → must keep its empty provider/model.
        self.assertIsNone(cfg.step_configs["dk_search"].provider)
        self.assertIsNone(cfg.step_configs["dk_search"].model)

    def test_construction_with_override_applies_in_post_init(self):
        cfg = PipelineConfig(
            step_configs={
                "keywords": PipelineStepConfig(
                    step_id="keywords", provider="ollama", model="gemma"),
            },
            global_provider_override="GWDG",
            global_model_override="glm-4.7",
        )
        self.assertEqual(cfg.step_configs["keywords"].provider, "GWDG")
        self.assertEqual(cfg.step_configs["keywords"].model, "glm-4.7")


if __name__ == "__main__":
    unittest.main()
