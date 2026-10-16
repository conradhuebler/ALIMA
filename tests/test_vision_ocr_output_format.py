"""Vision/OCR calls used to get JSON-mode forced on them.

``output_format=None`` defaults to JSON-mode on Gemini/OpenAI-compatible/
Ollama (``llm_service.py``'s three text generators). ``image_text_extraction``
wants raw OCR text (its own prompt says so: "Gib nur den erkannten Text
zurück"), but two independent gaps meant it never got that: (1)
``AlimaManager.execute_task``'s ``llm_params`` dict omitted ``output_format``
entirely, unlike every sibling call site in the same file; (2) even with
that fixed, the task's entry in ``prompts.json``/``prompts.yaml`` had no
``output_format`` key (JSON had none at all, YAML explicitly said ``json``),
so ``PromptService.get_prompt_config`` would still hand back ``"json"``.
Both fixed together: the prompt config now says ``"xml"``, and
``execute_task`` passes it through. - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.core.alima_manager import AlimaManager
from src.llm.prompt_service import PromptService
from src.utils.path_utils import get_project_root


class PromptConfigRequestsPlainTextForImageExtractionTest(unittest.TestCase):
    """Pins the prompts.json/prompts.yaml content fix directly - a revert of
    either file's output_format key must fail this."""

    def setUp(self):
        prompts_path = str(get_project_root() / "prompts.json")
        self.service = PromptService(prompts_path)

    def test_get_prompt_config_returns_xml_not_json(self):
        cfg = self.service.get_prompt_config("image_text_extraction", "gemini-2.0-flash")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.output_format, "xml")

    def test_a_fuzzy_or_default_model_match_still_gets_xml(self):
        # Any model, matched or falling back to "default" - the output_format
        # is task-level, not per-model.
        cfg = self.service.get_prompt_config("image_text_extraction", "some-unknown-model")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.output_format, "xml")


def _manager_with_prompt_config(output_format):
    llm_service = Mock()
    llm_service.generate_response.return_value = "extracted text"

    prompt_config = Mock()
    prompt_config.prompt.format.return_value = "prompt text"
    prompt_config.system = ""
    prompt_config.temp = 0.1
    prompt_config.p_value = 0.1
    prompt_config.seed = None
    prompt_config.output_format = output_format
    prompt_service = Mock()
    prompt_service.get_prompt_config.return_value = prompt_config

    from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig

    cfg = UnifiedProviderConfig(
        providers=[UnifiedProvider(name="claude", provider_type="ollama", enabled=True)],
        vision_default_provider="claude",
        vision_default_model="vision-model",
    )
    config_manager = Mock()
    config_manager.get_unified_config.return_value = cfg
    config_manager.load_config.return_value = Mock(repetition_config=None)

    manager = AlimaManager(
        llm_service=llm_service, prompt_service=prompt_service, config_manager=config_manager
    )
    return manager, llm_service


class ExecuteTaskPassesOutputFormatThroughTest(unittest.TestCase):
    def test_the_prompt_configs_output_format_reaches_generate_response(self):
        manager, llm_service = _manager_with_prompt_config(output_format="xml")
        manager.execute_task(task_name="image_text_extraction", context={"image_data": "x"})

        _, kwargs = llm_service.generate_response.call_args
        self.assertEqual(kwargs["output_format"], "xml")

    def test_a_json_prompt_config_also_reaches_generate_response_unchanged(self):
        """Not hardcoded to "xml" - this is a passthrough, not a forced override."""
        manager, llm_service = _manager_with_prompt_config(output_format="json")
        manager.execute_task(task_name="image_text_extraction", context={"image_data": "x"})

        _, kwargs = llm_service.generate_response.call_args
        self.assertEqual(kwargs["output_format"], "json")


if __name__ == "__main__":
    unittest.main()
