"""Regression: a configured provider literally named "ollama" must not be
swallowed by the legacy-"ollama"-alias handling.

Found while VM-testing (Aug 2026): the first-run config created a provider
named "ollama" with provider_type "openai_compatible" (Ollama via /v1 API).
Registration treated the name as the legacy alias, gated it on
get_enabled_ollama_providers() (which filters provider_type == "ollama"),
found none, and skipped it -> "Provider 'ollama' not available for
tool-calling" although Ollama was running.

Claude Generated.
"""
from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_service(supported_providers, ollama_type_providers):
    """Service stub with the real registration/mapping methods bound on. -
    Claude Generated"""
    from src.llm.llm_service import LlmService

    svc = MagicMock(spec=LlmService)
    svc._register_providers_lazy = LlmService._register_providers_lazy.__get__(
        svc, LlmService
    )
    svc._map_provider_name = LlmService._map_provider_name.__get__(svc, LlmService)
    svc.supported_providers = supported_providers
    svc.clients = {}
    svc.logger = logging.getLogger("test_ollama_alias")
    svc.alima_config = SimpleNamespace(
        unified_config=SimpleNamespace(
            get_enabled_ollama_providers=lambda: ollama_type_providers
        )
    )
    return svc


def _provider_entry(provider_type):
    return {"config": SimpleNamespace(provider_type=provider_type)}


class TestOllamaNamedProviderShadowsLegacyAlias(unittest.TestCase):
    """A real provider named "ollama" (openai_compatible) is registered and
    mapped to itself."""

    def setUp(self):
        self.svc = _make_service(
            supported_providers={"ollama": _provider_entry("openai_compatible")},
            ollama_type_providers=[],
        )

    def test_registration_includes_named_ollama_provider(self):
        self.svc._register_providers_lazy()
        self.assertIn("ollama", self.svc.clients)

    def test_mapping_returns_name_unchanged(self):
        self.assertEqual(self.svc._map_provider_name("ollama"), "ollama")


class TestLegacyOllamaAliasStillWorks(unittest.TestCase):
    """Without a provider named "ollama", the legacy alias keeps resolving to
    the first enabled ollama-type provider."""

    def setUp(self):
        self.svc = _make_service(
            supported_providers={"localhost": _provider_entry("ollama")},
            ollama_type_providers=[SimpleNamespace(name="localhost")],
        )

    def test_legacy_alias_maps_to_configured_ollama_provider(self):
        self.svc.clients = {"localhost": object()}
        self.assertEqual(self.svc._map_provider_name("ollama"), "localhost")

    def test_legacy_alias_not_registered_without_ollama_type_providers(self):
        svc = _make_service(
            supported_providers={"localhost": _provider_entry("ollama")},
            ollama_type_providers=[],
        )
        svc._register_providers_lazy(["ollama"])
        self.assertNotIn("ollama", svc.clients)


if __name__ == "__main__":
    unittest.main()
