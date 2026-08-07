"""Session prompt override: custom_params → step-executor kwargs - Claude Generated

The GUI Abstract tab (and CLI overrides via ``PipelineConfigBuilder``) place
``prompt_template``/``system_prompt`` in ``PipelineStepConfig.custom_params``.
These tests drive the real ``execute_single_step`` → ``execute_step`` →
``_execute_*_step`` path on a real ``PipelineManager`` (``pipeline_executor``
mocked) and pin that the override reaches the executor kwargs — and stays
absent when no override is set.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.utils.config_models import AlimaConfig, PipelineStepConfig

logging.disable(logging.CRITICAL)


def _make_pm() -> PipelineManager:
    pm = PipelineManager(
        alima_manager=MagicMock(),
        cache_manager=MagicMock(),
        logger=logging.getLogger("test_prompt_override_flow"),
    )
    pm.pipeline_executor = MagicMock()
    pm.pipeline_executor.execute_final_keyword_analysis.return_value = (
        ["KW"],
        [],
        MagicMock(),
    )
    pm.pipeline_executor.execute_initial_keyword_extraction.return_value = (
        "KW",
        ["004"],
        MagicMock(),
        "Titel",
    )
    return pm


def _config_with(step_id: str, custom_params: dict) -> PipelineConfig:
    config = PipelineConfig()
    config.step_configs[step_id] = PipelineStepConfig(
        step_id=step_id,
        provider="test-provider",
        model="test-model",
        task=step_id,
        custom_params=custom_params,
    )
    return config


# The keywords step reads TaskPreferences via ConfigManager; pin a default
# config so the test is independent of the operator's real config file.
@patch(
    "src.utils.config_manager.ConfigManager.load_config",
    return_value=AlimaConfig(),
)
class TestPromptOverrideFlow(unittest.TestCase):
    def test_keywords_step_forwards_prompt_override(self, _mock_cfg):
        pm = _make_pm()
        config = _config_with(
            "keywords",
            {"prompt_template": "T {abstract}", "system_prompt": "S"},
        )
        step = pm.execute_single_step("keywords", config, "Ein Abstract.")
        self.assertEqual(step.status, "completed")
        kwargs = pm.pipeline_executor.execute_final_keyword_analysis.call_args.kwargs
        self.assertEqual(kwargs["prompt_template"], "T {abstract}")
        self.assertEqual(kwargs["system"], "S")

    def test_keywords_step_without_override_stays_clean(self, _mock_cfg):
        pm = _make_pm()
        config = _config_with("keywords", {})
        step = pm.execute_single_step("keywords", config, "Ein Abstract.")
        self.assertEqual(step.status, "completed")
        kwargs = pm.pipeline_executor.execute_final_keyword_analysis.call_args.kwargs
        self.assertNotIn("prompt_template", kwargs)
        self.assertNotIn("system", kwargs)

    def test_initialisation_step_forwards_prompt_override(self, _mock_cfg):
        pm = _make_pm()
        config = _config_with(
            "initialisation",
            {"prompt_template": "T {abstract}", "system_prompt": "S"},
        )
        step = pm.execute_single_step("initialisation", config, "Ein Abstract.")
        self.assertEqual(step.status, "completed")
        kwargs = pm.pipeline_executor.execute_initial_keyword_extraction.call_args.kwargs
        self.assertEqual(kwargs["prompt_template"], "T {abstract}")
        self.assertEqual(kwargs["system"], "S")

    def test_initialisation_step_without_override_stays_clean(self, _mock_cfg):
        pm = _make_pm()
        config = _config_with("initialisation", {})
        step = pm.execute_single_step("initialisation", config, "Ein Abstract.")
        self.assertEqual(step.status, "completed")
        kwargs = pm.pipeline_executor.execute_initial_keyword_extraction.call_args.kwargs
        self.assertNotIn("prompt_template", kwargs)
        self.assertNotIn("system", kwargs)


if __name__ == "__main__":
    unittest.main()
