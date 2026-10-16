"""``task_preferences`` is retired as a live config surface (Oct 2026).

It used to feed six per-task wizard questions, a per-step override dialog
(removed separately, see ``tests/test_workflow_list.py``), an
``AbstractTab`` default-selection read, and ``AlimaManager.execute_task``'s
provider/model resolution for ``image_text_extraction`` - none of which the
real resolution hierarchy (``resolve_default_provider_model`` /
``explain_default_provider_model``) ever consulted. The field and
``TaskPreference`` dataclass survive only as input to
``adopt_vision_task_preference()``, a guarded one-shot migration that turns
a legacy ``task_preferences["vision"]`` entry into ``vision_default_provider``/
``vision_default_model`` on load. This file pins:

* the wizard-facing builder (``ConfigurationBuilder.create_initial_config``)
  now takes direct ``pipeline_model``/``vision_model`` params and writes the
  resolved-default fields directly, no ``TaskPreference`` objects built;
* ``save_config`` actively drops ``task_preferences`` from what it writes
  back to disk, rather than preserving whatever was already there forever;
* both config parse paths run the vision migration, so neither leaves a
  pre-Oct-2026 vision choice stranded;
* ``AlimaManager.execute_task`` resolves provider/model via the same
  scope-based hierarchy as every other caller, instead of reading
  ``task_preferences`` with an ad hoc vision/initialisation/keywords
  fallback chain;
* ``InstitutionPresets.pipeline_model()``/``vision_model()`` collapse an
  on-disk preset's per-task dict to the two scopes that are actually asked
  about now. - Claude Generated
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.alima_manager import AlimaManager
from src.ui.abstract_tab import AbstractTab
from src.utils.config_manager import ConfigManager
from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig
from src.utils.preset_loader import InstitutionPresets
from src.utils.setup_utils import ConfigurationBuilder


class ConfigurationBuilderPipelineVisionModelsTest(unittest.TestCase):
    """``create_initial_config`` builds the two resolved defaults directly."""

    def test_pipeline_and_vision_models_set_from_the_wizard_answers(self):
        config = ConfigurationBuilder.create_initial_config(
            provider_type="ollama",
            provider_name="My Ollama",
            base_url="http://localhost:11434",
            models=["m1", "m2"],
            pipeline_model="m2",
            vision_model="m1",
        )
        unified = config.unified_config
        self.assertEqual(unified.pipeline_default_provider, "ollama")
        self.assertEqual(unified.pipeline_default_model, "m2")
        self.assertEqual(unified.vision_default_provider, "ollama")
        self.assertEqual(unified.vision_default_model, "m1")

    def test_falls_back_to_the_first_available_model_when_unanswered(self):
        config = ConfigurationBuilder.create_initial_config(
            provider_type="ollama",
            provider_name="My Ollama",
            models=["first", "second"],
        )
        unified = config.unified_config
        self.assertEqual(unified.pipeline_default_model, "first")
        self.assertEqual(unified.vision_default_model, "first")

    def test_rejects_an_auto_select_placeholder_for_pipeline(self):
        with self.assertRaises(ValueError):
            ConfigurationBuilder.create_initial_config(
                provider_type="ollama",
                provider_name="My Ollama",
                models=["m1"],
                pipeline_model="auto-select",
            )

    def test_rejects_an_auto_select_placeholder_for_vision(self):
        with self.assertRaises(ValueError):
            ConfigurationBuilder.create_initial_config(
                provider_type="ollama",
                provider_name="My Ollama",
                models=["m1"],
                vision_model="Default",
            )

    def test_no_task_preferences_are_built_any_more(self):
        config = ConfigurationBuilder.create_initial_config(
            provider_type="ollama",
            provider_name="My Ollama",
            models=["m1"],
            pipeline_model="m1",
            vision_model="m1",
        )
        self.assertEqual(config.unified_config.task_preferences, {})


class _TempConfig(unittest.TestCase):
    """Real temp config file, same harness as
    tests/test_provider_reset_does_not_persist.py."""

    def setUp(self):
        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        ConfigManager.reset()

    def _write(self, payload: dict) -> None:
        self.tmp.write_text(json.dumps(payload), encoding="utf-8")

    def _manager(self) -> ConfigManager:
        ConfigManager.reset()
        cm = ConfigManager()
        cm.config_file = self.tmp
        return cm


def _provider(name="claude", provider_type="ollama", enabled=True):
    return {"name": name, "provider_type": provider_type, "enabled": enabled, "preferred_model": "m"}


class SaveConfigDropsTaskPreferencesTest(_TempConfig):
    """save_config no longer re-writes task_preferences to disk."""

    def test_task_preferences_disappears_from_disk_after_a_save(self):
        self._write({"unified_config": {
            "providers": [_provider(name="claude")],
            "task_preferences": {
                "vision": {
                    "task_type": "vision",
                    "model_priority": [{"provider_name": "claude", "model_name": "claude-vision"}],
                }
            },
        }})
        cm = self._manager()
        config = cm.load_config()
        # The migration already ran at load time (in memory).
        self.assertEqual(config.unified_config.vision_default_provider, "claude")
        self.assertEqual(config.unified_config.vision_default_model, "claude-vision")

        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertNotIn("task_preferences", on_disk["unified_config"])
        # The migration's target survives the same save - nothing is lost.
        self.assertEqual(on_disk["unified_config"]["vision_default_provider"], "claude")
        self.assertEqual(on_disk["unified_config"]["vision_default_model"], "claude-vision")

    def test_an_unrelated_save_does_not_resurrect_task_preferences(self):
        """Confirms the drop is active, not just "don't add it back":
        a config saved once already (task_preferences gone) stays gone on the
        next save too, even though the merge starts from the on-disk copy."""
        self._write({"unified_config": {"providers": [_provider(name="claude")]}})
        cm = self._manager()
        config = cm.load_config()
        cm.save_config(config)
        config.system_config.debug = True
        cm.save_config(config)

        on_disk = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertNotIn("task_preferences", on_disk["unified_config"])


class BothParsePathsAdoptTheVisionMigrationTest(_TempConfig):
    """adopt_vision_task_preference() must run regardless of which parser a
    config takes - a gap in _parse_modern_config meant an old-format config's
    vision choice never reached vision_default_provider/model at all."""

    def test_the_unified_format_parser_adopts_it(self):
        self._write({"unified_config": {
            "providers": [_provider(name="claude")],
            "task_preferences": {
                "vision": {
                    "task_type": "vision",
                    "model_priority": [{"provider_name": "claude", "model_name": "v-model"}],
                }
            },
        }})
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(config.unified_config.vision_default_provider, "claude")
        self.assertEqual(config.unified_config.vision_default_model, "v-model")

    def test_the_legacy_modern_format_parser_also_adopts_it(self):
        self._write({
            "llm": {
                "openai_compatible_providers": [
                    {"name": "claude", "base_url": "https://example.test"}
                ],
            },
            "task_preferences": {
                "vision": {
                    "task_type": "vision",
                    "model_priority": [{"provider_name": "claude", "model_name": "v-model"}],
                }
            },
        })
        cm = self._manager()
        config = cm.load_config()
        self.assertEqual(config.unified_config.vision_default_provider, "claude")
        self.assertEqual(config.unified_config.vision_default_model, "v-model")


def _manager_with_config(unified_config: UnifiedProviderConfig):
    llm_service = Mock()
    llm_service.generate_response.return_value = "extracted text"

    prompt_config = Mock()
    prompt_config.prompt.format.return_value = "prompt text"
    prompt_config.system = ""
    prompt_config.temp = 0.1
    prompt_config.p_value = 0.1
    prompt_config.seed = None
    prompt_service = Mock()
    prompt_service.get_prompt_config.return_value = prompt_config

    config_manager = Mock()
    config_manager.get_unified_config.return_value = unified_config
    config_manager.load_config.return_value = Mock(repetition_config=None)

    manager = AlimaManager(
        llm_service=llm_service, prompt_service=prompt_service, config_manager=config_manager
    )
    return manager, llm_service


class ExecuteTaskResolvesViaDefaultHierarchyTest(unittest.TestCase):
    """execute_task used to read task_preferences via
    get_model_priority_for_task(), with a vision/initialisation/keywords
    fallback chain for image_text_extraction specifically. Both now resolve
    through resolve_default_provider_model, the same hierarchy every other
    caller uses."""

    def _config(self) -> UnifiedProviderConfig:
        return UnifiedProviderConfig(
            providers=[
                UnifiedProvider(name="claude", provider_type="ollama", enabled=True),
                UnifiedProvider(name="ollama", provider_type="ollama", enabled=True),
            ],
            vision_default_provider="claude",
            vision_default_model="vision-model",
            pipeline_default_provider="ollama",
            pipeline_default_model="pipeline-model",
        )

    def test_image_text_extraction_uses_the_vision_default(self):
        manager, llm_service = _manager_with_config(self._config())
        manager.execute_task(task_name="image_text_extraction", context={"image_data": "x"})

        _, kwargs = llm_service.generate_response.call_args
        self.assertEqual(kwargs["provider"], "claude")
        self.assertEqual(kwargs["model"], "vision-model")

    def test_a_non_vision_task_uses_the_pipeline_default(self):
        manager, llm_service = _manager_with_config(self._config())
        manager.execute_task(task_name="keywords", context={})

        _, kwargs = llm_service.generate_response.call_args
        self.assertEqual(kwargs["provider"], "ollama")
        self.assertEqual(kwargs["model"], "pipeline-model")


class InstitutionPresetsPipelineVisionModelTest(unittest.TestCase):
    """pipeline_model()/vision_model() collapse an on-disk preset's per-task
    dict to the two scopes the wizards ask about now."""

    def test_pipeline_model_picks_any_non_vision_entry(self):
        # "vision" deliberately comes FIRST in the dict: a pipeline_model()
        # that forgot to skip it would return the vision model here, since a
        # naive "first value" scan would hit it before dk_classification.
        presets = InstitutionPresets(llm_task_models={
            "vision": "gemma-3-27b-it", "dk_classification": "glm-4.7",
        })
        self.assertEqual(presets.pipeline_model(), "glm-4.7")

    def test_vision_model_reads_the_vision_key(self):
        presets = InstitutionPresets(llm_task_models={
            "keywords": "glm-4.7", "vision": "gemma-3-27b-it",
        })
        self.assertEqual(presets.vision_model(), "gemma-3-27b-it")

    def test_both_fall_back_to_the_default_model_when_task_models_is_empty(self):
        presets = InstitutionPresets(llm_default_model="fallback-model")
        self.assertEqual(presets.pipeline_model(), "fallback-model")
        self.assertEqual(presets.vision_model(), "fallback-model")


class _FakeCombo:
    """Stands in for the two QComboBox attributes _apply_pipeline_default
    touches (findText/setCurrentIndex/currentText) - a real QComboBox is not
    needed for this method's logic and repeated raw construction across
    tests is a known PyQt6 lifetime hazard (segfaults without a QWidget
    parent)."""

    def __init__(self, items=()):
        self._items = list(items)
        self._current = -1

    def addItems(self, items):
        self._items = list(items)
        self._current = -1

    def findText(self, text):
        return self._items.index(text) if text in self._items else -1

    def setCurrentIndex(self, idx):
        self._current = idx

    def currentText(self):
        return self._items[self._current] if 0 <= self._current < len(self._items) else ""


class _FakeAbstractTabForPipelineDefault:
    """Exposes exactly what AbstractTab._apply_pipeline_default touches,
    without the real constructor's heavy UI/worker setup."""

    def __init__(self, task, alima_manager, available_models):
        self.task = task
        self.alima_manager = alima_manager
        self.available_models = available_models
        self.user_interaction_mode = True
        self.logger = logging.getLogger("test_apply_pipeline_default")
        self.provider_combo = _FakeCombo(available_models.keys())
        self.model_combo = _FakeCombo()
        self.update_models = Mock(side_effect=self._update_models)

    def _update_models(self, provider):
        self.model_combo.addItems(self.available_models.get(provider, []))


_apply_pipeline_default = AbstractTab._apply_pipeline_default


class AbstractTabAppliesThePipelineDefaultTest(unittest.TestCase):
    """Used to read task_preferences.get(self.task) - a per-step choice
    nothing else resolved through. Every task on this tab (initialisation/
    keywords/dk_classification) is a classic-pipeline step, so the pipeline
    scope answers all of them the same way now."""

    def _alima_manager(self, pipeline_provider, pipeline_model):
        providers = (
            [UnifiedProvider(name=pipeline_provider, provider_type="ollama", enabled=True)]
            if pipeline_provider else []
        )
        cfg = UnifiedProviderConfig(
            providers=providers,
            pipeline_default_provider=pipeline_provider,
            pipeline_default_model=pipeline_model,
        )
        manager = Mock()
        manager.config_manager.load_config.return_value = Mock(unified_config=cfg)
        return manager

    def test_selects_the_pipeline_default_when_available(self):
        tab = _FakeAbstractTabForPipelineDefault(
            task="keywords",
            alima_manager=self._alima_manager("claude", "opus"),
            available_models={"claude": ["opus", "haiku"], "ollama": ["cogito"]},
        )
        _apply_pipeline_default(tab)

        self.assertEqual(tab.provider_combo.currentText(), "claude")
        self.assertEqual(tab.model_combo.currentText(), "opus")

    def test_leaves_the_selection_alone_when_nothing_is_configured(self):
        tab = _FakeAbstractTabForPipelineDefault(
            task="keywords",
            alima_manager=self._alima_manager("", ""),
            available_models={"claude": ["opus"]},
        )
        tab.provider_combo.setCurrentIndex(0)
        _apply_pipeline_default(tab)

        tab.update_models.assert_not_called()

    def test_does_nothing_without_a_task(self):
        tab = _FakeAbstractTabForPipelineDefault(
            task=None,
            alima_manager=self._alima_manager("claude", "opus"),
            available_models={"claude": ["opus"]},
        )
        _apply_pipeline_default(tab)

        tab.update_models.assert_not_called()


class CLIWizardCollectsTwoModelQuestionsTest(unittest.TestCase):
    """_collect_task_model_selections used to ask once per LLM_TASK_DISPLAY_INFO
    entry (six questions) and store answers under task_type.name - a key
    ConfigurationBuilder never looked up (it reads .value), so every CLI
    wizard run's per-task choice was already silently discarded before this
    change. Now two questions, stored as plain pipeline_model/vision_model."""

    def _wizard(self, available_models, presets=None):
        from src.utils.cli_setup_wizard import CLISetupWizard

        wiz = CLISetupWizard.__new__(CLISetupWizard)
        wiz.available_models = available_models
        wiz.pipeline_model = ""
        wiz.vision_model = ""
        wiz._presets = presets if presets is not None else InstitutionPresets()
        return wiz

    def test_enter_accepts_the_first_model_for_both_questions(self):
        wiz = self._wizard(["m1", "m2"])
        with patch("builtins.input", return_value=""):
            wiz._collect_task_model_selections()

        self.assertEqual(wiz.pipeline_model, "m1")
        self.assertEqual(wiz.vision_model, "m1")

    def test_a_numeric_choice_picks_that_model(self):
        wiz = self._wizard(["m1", "m2"])
        with patch("builtins.input", side_effect=["2", ""]):
            wiz._collect_task_model_selections()

        self.assertEqual(wiz.pipeline_model, "m2")
        self.assertEqual(wiz.vision_model, "m1")

    def test_a_preset_model_becomes_the_shown_default(self):
        presets = InstitutionPresets(llm_task_models={"vision": "m2"})
        wiz = self._wizard(["m1", "m2"], presets=presets)
        with patch("builtins.input", return_value=""):
            wiz._collect_task_model_selections()

        self.assertEqual(wiz.pipeline_model, "m1")
        self.assertEqual(wiz.vision_model, "m2")

    def test_no_available_models_skips_without_asking(self):
        wiz = self._wizard([])
        with patch("builtins.input") as mock_input:
            wiz._collect_task_model_selections()
        mock_input.assert_not_called()


if __name__ == "__main__":
    unittest.main()
