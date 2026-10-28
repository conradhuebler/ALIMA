"""A chosen default provider used to "win" even when disabled or deleted.

``explain_default_provider_model`` checked only ``if provider:`` (a
non-empty name) for every candidate - ``preferred_provider``,
``pipeline_default_provider``,
``vision_default_provider``. Only the LAST-resort fallback branch
("first_enabled") checked ``is_usable``. Disable the provider a default
names, or delete it outright, and the resolution table still showed it as
"configured" - the actual run then failed downstream in llm_service with
"No client found", a confusing failure since the settings UI claimed it
was fine. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.utils.config_models import ProviderScope, UnifiedProvider, UnifiedProviderConfig


def _config(**kw) -> UnifiedProviderConfig:
    cfg = UnifiedProviderConfig(providers=[
        UnifiedProvider(name="A", provider_type="ollama", enabled=True, preferred_model="a1"),
        UnifiedProvider(name="B", provider_type="ollama", enabled=True, preferred_model="b1"),
    ])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class ADisabledNamedDefaultFallsThroughTest(unittest.TestCase):
    def test_a_disabled_preferred_provider_falls_through_to_first_enabled(self):
        cfg = _config(preferred_provider="A", preferred_model="a1")
        cfg.get_provider_by_name("A").enabled = False
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual(provider, "B")
        self.assertEqual(source, "first_enabled")

    def test_a_deleted_preferred_provider_falls_through(self):
        cfg = _config(preferred_provider="gone", preferred_model="x")
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual(provider, "A")
        self.assertEqual(source, "first_enabled")

    def test_a_disabled_pipeline_default_falls_through_to_preferred(self):
        cfg = _config(
            pipeline_default_provider="A", pipeline_default_model="a1",
            preferred_provider="B", preferred_model="b1",
        )
        cfg.get_provider_by_name("A").enabled = False
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.PIPELINE)
        self.assertEqual(provider, "B")
        self.assertEqual(source, "preferred")

    def test_a_disabled_general_default_falls_through_for_the_agentic_scope_too(self):
        cfg = _config(preferred_provider="A", preferred_model="a1")
        cfg.get_provider_by_name("A").enabled = False
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.AGENTIC)
        self.assertEqual(provider, "B")
        self.assertEqual(source, "first_enabled")

    def test_a_disabled_vision_default_falls_through_to_preferred(self):
        cfg = _config(
            vision_default_provider="A", vision_default_model="a1",
            preferred_provider="B", preferred_model="b1",
        )
        cfg.get_provider_by_name("A").enabled = False
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.VISION)
        self.assertEqual(provider, "B")
        self.assertEqual(source, "preferred")

    def test_an_unusable_keyless_cloud_provider_falls_through_too(self):
        """A keyless gemini/anthropic stub is enabled but not is_usable - the
        same rule the first_enabled fallback already applies."""
        cfg = _config(preferred_provider="A", preferred_model="a1")
        stub = cfg.get_provider_by_name("A")
        stub.provider_type = "anthropic"
        stub.api_key = ""
        self.assertFalse(stub.is_usable)
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual(provider, "B")

    def test_an_enabled_usable_named_default_still_wins_normally(self):
        """No regression on the common case."""
        cfg = _config(preferred_provider="A", preferred_model="a1")
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual(provider, "A")
        self.assertEqual(source, "preferred")

    def test_everything_unusable_resolves_to_none(self):
        cfg = _config(preferred_provider="A", preferred_model="a1")
        cfg.get_provider_by_name("A").enabled = False
        cfg.get_provider_by_name("B").enabled = False
        provider, model, source = cfg.explain_default_provider_model(scope=ProviderScope.GENERAL)
        self.assertEqual((provider, model, source), ("", "", "none"))


if __name__ == "__main__":
    unittest.main()
