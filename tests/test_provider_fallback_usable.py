"""Regression: the first-enabled default fallback must skip unusable providers.

An enabled-but-keyless cloud stub (e.g. a leftover ``gemini`` entry sitting at
index 0 of the providers list) must not be picked as the implicit default. With
the old "first enabled" fallback it was, then failed at runtime with
"No client found for provider 'gemini'". The resolver now skips providers that
cannot plausibly serve (``is_usable``). Claude Generated.
"""
from __future__ import annotations

import unittest

from src.utils.config_models import UnifiedProviderConfig, UnifiedProvider


class TestProviderIsUsable(unittest.TestCase):
    def test_keyless_cloud_provider_is_not_usable(self):
        self.assertFalse(UnifiedProvider(name="gemini", provider_type="gemini").is_usable)
        self.assertFalse(UnifiedProvider(name="anthropic", provider_type="anthropic").is_usable)

    def test_keyed_or_local_providers_are_usable(self):
        self.assertTrue(
            UnifiedProvider(name="gemini", provider_type="gemini", api_key="k").is_usable)
        self.assertTrue(UnifiedProvider(name="Localhost", provider_type="ollama").is_usable)
        self.assertTrue(
            UnifiedProvider(name="GWDG", provider_type="openai_compatible", api_key="k").is_usable)


class TestUsableFallback(unittest.TestCase):
    def _cfg(self, providers, **defaults):
        uc = UnifiedProviderConfig()
        uc.providers = providers
        for key, value in defaults.items():
            setattr(uc, key, value)
        return uc

    def test_fallback_skips_keyless_gemini_stub(self):
        uc = self._cfg([
            UnifiedProvider(name="gemini", provider_type="gemini"),  # keyless stub @ index 0
            UnifiedProvider(name="Localhost", provider_type="ollama",
                            preferred_model="cogito:32b"),
        ])
        provider, model = uc.resolve_default_provider_model(scope="general")
        self.assertEqual(provider, "Localhost",
                         "keyless gemini stub must be skipped for a usable provider")
        self.assertEqual(model, "cogito:32b")

    def test_explicit_preferred_still_wins(self):
        uc = self._cfg([
            UnifiedProvider(name="gemini", provider_type="gemini", api_key="k",
                            preferred_model="gemini-2.0"),
            UnifiedProvider(name="Localhost", provider_type="ollama",
                            preferred_model="cogito:32b"),
        ], preferred_provider="gemini", preferred_model="gemini-2.0")
        # A real, key-configured choice must be honoured regardless of list order.
        self.assertEqual(uc.resolve_default_provider_model(scope="general"),
                         ("gemini", "gemini-2.0"))

    def test_all_unusable_returns_empty(self):
        uc = self._cfg([UnifiedProvider(name="gemini", provider_type="gemini")])
        self.assertEqual(uc.resolve_default_provider_model(scope="general"), ("", ""))


if __name__ == "__main__":
    unittest.main()
