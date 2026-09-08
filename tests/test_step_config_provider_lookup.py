"""Provider/model lookup of the pipeline config dialog's step widgets.

Claude Generated. Three calls in ``HybridStepConfigWidget`` addressed APIs that
do not exist any more (or never did), each behind a broad ``except`` that turned
the ``AttributeError`` into a log line:

* ``unified_config.openai_compatible_providers`` / ``.ollama_providers`` — gone
  since the provider unification, so **no** provider ever contributed its
  preferred model ("Error getting preferred model for X" on every call);
* ``SmartTaskType.from_pipeline_step(...).to_unified_task_type()`` — neither
  method exists, and ``SmartTaskType`` *is* the unified ``TaskType``, imported
  under a second name; the legacy task-preference fallback and the smart preview
  therefore never ran;
* ``SmartProviderSelector._get_preferred_model_from_config`` — no such method.

The methods are exercised as unbound functions on a stand-in, so no QWidget is
built and nothing reaches the network.
"""
from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from src.ui.step_config_widgets import HybridStepConfigWidget
from src.utils.config_models import (
    AlimaConfig,
    TaskType,
    UnifiedProvider,
    UnifiedProviderConfig,
)


def _widget(unified: UnifiedProviderConfig, step_id: str = "keywords",
            task_pref=("", "", "none")):
    """Stand-in carrying only what the two lookups touch."""
    config = AlimaConfig()
    config.unified_config = unified
    stub = SimpleNamespace(
        step_id=step_id,
        logger=logging.getLogger("test"),
        config_manager=SimpleNamespace(load_config=lambda force_reload=False: config),
    )
    stub._load_task_preferences_direct = lambda: task_pref
    for name in ("_get_preferred_model_for_provider", "_provider_names_match",
                 "_task_type_for_step"):
        setattr(stub, name, getattr(HybridStepConfigWidget, name).__get__(stub))
    return stub


def _unified(*providers, **fields) -> UnifiedProviderConfig:
    return UnifiedProviderConfig(providers=list(providers), **fields)


def _provider(name, model="", ptype="openai_compatible") -> UnifiedProvider:
    return UnifiedProvider(name=name, provider_type=ptype, preferred_model=model)


class PreferredModelLookupTest(unittest.TestCase):
    def test_the_providers_own_preferred_model_is_found(self):
        w = _widget(_unified(_provider("GWDG", "gemma-4-31b-it")))
        self.assertEqual(w._get_preferred_model_for_provider("GWDG"), "gemma-4-31b-it")

    def test_an_ollama_provider_is_found_too(self):
        w = _widget(_unified(_provider("LLMachine", "cogito:32b", "ollama")))
        self.assertEqual(w._get_preferred_model_for_provider("LLMachine"), "cogito:32b")

    def test_a_name_variant_still_matches(self):
        # "LLMachine/Ollama" in the config, "ollama" asked for.
        w = _widget(_unified(_provider("LLMachine/Ollama", "cogito:32b", "ollama")))
        self.assertEqual(w._get_preferred_model_for_provider("ollama"), "cogito:32b")

    def test_a_provider_without_a_preferred_model_yields_none(self):
        w = _widget(_unified(_provider("GWDG", "")))
        self.assertIsNone(w._get_preferred_model_for_provider("GWDG"))

    def test_an_unknown_provider_yields_none(self):
        w = _widget(_unified(_provider("GWDG", "m")))
        self.assertIsNone(w._get_preferred_model_for_provider("nvidia"))

    def test_the_legacy_gemini_field_is_still_read(self):
        w = _widget(_unified(gemini_preferred_model="gemini-3-pro"))
        self.assertEqual(w._get_preferred_model_for_provider("gemini"), "gemini-3-pro")

    def test_a_task_preference_outranks_the_provider_default(self):
        w = _widget(
            _unified(_provider("GWDG", "gemma-4-31b-it")),
            task_pref=("GWDG", "apertus-70b", "task preference"),
        )
        self.assertEqual(w._get_preferred_model_for_provider("GWDG"), "apertus-70b")

    def test_a_task_preference_for_another_provider_is_ignored(self):
        w = _widget(
            _unified(_provider("GWDG", "gemma-4-31b-it")),
            task_pref=("Mistral", "mistral-small", "task preference"),
        )
        self.assertEqual(w._get_preferred_model_for_provider("GWDG"), "gemma-4-31b-it")


class TaskTypeForStepTest(unittest.TestCase):
    def test_a_step_id_maps_to_its_task_type(self):
        for step_id, expected in (
            ("initialisation", TaskType.INITIALISATION),
            ("keywords", TaskType.KEYWORDS),
            ("dk_classification", TaskType.DK_CLASSIFICATION),
        ):
            with self.subTest(step_id=step_id):
                self.assertIs(_widget(_unified(), step_id)._task_type_for_step(), expected)

    def test_an_unknown_step_id_falls_back_to_general(self):
        self.assertIs(
            _widget(_unified(), "rvk_guard")._task_type_for_step(), TaskType.GENERAL
        )


class DeadApiTest(unittest.TestCase):
    """The three names the widget used to call must stay gone from its source.

    Not a style check: each of them was reached only inside a broad ``except``,
    so a reintroduction would be invisible again.
    """

    def test_no_attribute_access_to_a_name_that_does_not_exist(self):
        # Over the AST, not the text: the prose in this module names them too.
        import ast
        import pathlib

        import src.ui.step_config_widgets as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for name in ("openai_compatible_providers", "ollama_providers",
                     "from_pipeline_step", "to_unified_task_type",
                     "_get_preferred_model_from_config"):
            with self.subTest(name=name):
                self.assertNotIn(name, used)

    def test_the_attributes_really_are_absent(self):
        # If they ever come back on the config class, the check above may relax.
        unified = UnifiedProviderConfig()
        for name in ("openai_compatible_providers", "ollama_providers"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(unified, name))
        from src.utils.smart_provider_selector import SmartProviderSelector

        self.assertFalse(hasattr(SmartProviderSelector, "_get_preferred_model_from_config"))
        self.assertFalse(hasattr(TaskType, "from_pipeline_step"))


if __name__ == "__main__":
    unittest.main()
