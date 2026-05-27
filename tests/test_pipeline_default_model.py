"""Regression: a provider-only pipeline default must yield a complete model.

Claude Generated (P-ι follow-up).

Logic gap reported via the GUI chat ("Kein LLM-Provider konfiguriert"): the
settings UI lets you pick ``pipeline_default_provider`` but the model dropdown
can be empty (models not fetched) → ``pipeline_default_model`` saved as "". The
chat resolver requires BOTH provider and model, so the default never applied.

``PipelineConfig.create_from_provider_preferences`` now fills an empty default
model from the provider's preferred / first available model.
"""
from __future__ import annotations

import unittest

from src.core.pipeline_manager import PipelineConfig


class _FakeProvider:
    def __init__(self, name, preferred_model="", available_models=None):
        self.name = name
        self.preferred_model = preferred_model
        self.available_models = available_models or []


class _FakeUnified:
    def __init__(self, default_provider, default_model, providers,
                 preferred_provider="", preferred_model=""):
        self.pipeline_default_provider = default_provider
        self.pipeline_default_model = default_model
        self.preferred_provider = preferred_provider
        self.preferred_model = preferred_model
        self._providers = providers
        self.task_preferences = {}

    def get_enabled_providers(self):
        return self._providers


class _FakeCM:
    def __init__(self, unified):
        self._u = unified

    def get_unified_config(self):
        return self._u


def _build(default_provider, default_model, providers,
           preferred_provider="", preferred_model=""):
    cm = _FakeCM(_FakeUnified(default_provider, default_model, providers,
                              preferred_provider, preferred_model))
    return PipelineConfig.create_from_provider_preferences(cm)


class TestPipelineDefaultModel(unittest.TestCase):
    def test_provider_set_empty_model_fills_preferred(self):
        cfg = _build("GWDG", "", [_FakeProvider("GWDG", preferred_model="gemma-4-31b-it")])
        self.assertEqual(cfg.step_configs["keywords"].provider, "GWDG")
        self.assertEqual(cfg.step_configs["keywords"].model, "gemma-4-31b-it")

    def test_provider_set_empty_model_falls_back_to_first_available(self):
        cfg = _build("ollama", "", [
            _FakeProvider("ollama", preferred_model="", available_models=["cogito:14b", "x"])])
        self.assertEqual(cfg.step_configs["keywords"].model, "cogito:14b")

    def test_no_default_uses_first_provider(self):
        cfg = _build("", "", [_FakeProvider("ollama", preferred_model="cogito:14b")])
        self.assertEqual(cfg.step_configs["keywords"].provider, "ollama")
        self.assertEqual(cfg.step_configs["keywords"].model, "cogito:14b")

    def test_explicit_model_preserved(self):
        cfg = _build("GWDG", "explicit-model",
                     [_FakeProvider("GWDG", preferred_model="gemma-4-31b-it")])
        self.assertEqual(cfg.step_configs["keywords"].model, "explicit-model")

    def test_no_providers_returns_empty_config(self):
        cfg = _build("", "", [])
        # No enabled providers → bare PipelineConfig (no step_configs populated).
        self.assertEqual(cfg.step_configs, {})

    # --- central general default (preferred_provider/preferred_model) ---

    def test_general_default_used_when_no_pipeline_default(self):
        cfg = _build("", "", [
            _FakeProvider("ollama", preferred_model="cogito:14b"),
            _FakeProvider("GWDG", preferred_model="gemma-4-31b-it")],
            preferred_provider="GWDG", preferred_model="gemma-4-31b-it")
        # General default wins over the first-enabled-provider fallback.
        self.assertEqual(cfg.step_configs["keywords"].provider, "GWDG")
        self.assertEqual(cfg.step_configs["keywords"].model, "gemma-4-31b-it")

    def test_general_default_provider_only_fills_model(self):
        cfg = _build("", "", [_FakeProvider("GWDG", preferred_model="gemma-4-31b-it")],
                     preferred_provider="GWDG", preferred_model="")
        # preferred_model empty → filled from provider preferred.
        self.assertEqual(cfg.step_configs["keywords"].model, "gemma-4-31b-it")

    def test_pipeline_default_overrides_general(self):
        cfg = _build("ollama", "cogito:32b",
                     [_FakeProvider("ollama"), _FakeProvider("GWDG")],
                     preferred_provider="GWDG", preferred_model="gemma-4-31b-it")
        # pipeline-specific default takes precedence over the general default.
        self.assertEqual(cfg.step_configs["keywords"].provider, "ollama")
        self.assertEqual(cfg.step_configs["keywords"].model, "cogito:32b")


if __name__ == "__main__":
    unittest.main()
