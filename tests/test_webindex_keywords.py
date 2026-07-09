"""Tests for the keyword-extraction glue (model resolution + workflow extractor) -
Claude Generated.

Covers ``resolve_crawl_model`` (CLI → instance → global default) and
``build_keyword_extractor`` (guards + a workflow run driven by a fake LlmService,
mirroring the ``test_agent_loop_hooks`` pattern).
"""

import unittest
from unittest import mock
from unittest.mock import MagicMock

try:
    from src.utils.lookups.webindex.keywords import (
        build_keyword_extractor, resolve_crawl_model,
    )
    from src.core.data_models import AgentResponse
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _config(*, agentic=("", ""), pipeline=("", ""), preferred=("", ""),
            enabled_providers=()):
    """Minimal config with a UnifiedProviderConfig-like stub."""
    prov, model = agentic
    pprov, pmodel = pipeline
    pref_p, pref_m = preferred

    class _Prov:
        def __init__(self, name, usable=True, preferred_model="", available_models=()):
            self.name = name
            self.is_usable = usable
            self.preferred_model = preferred_model
            self.available_models = list(available_models)

    provs = list(enabled_providers)
    unified = MagicMock()
    unified.agentic_default_provider = prov
    unified.agentic_default_model = model
    unified.pipeline_default_provider = pprov
    unified.pipeline_default_model = pmodel
    unified.preferred_provider = pref_p
    unified.preferred_model = pref_m
    unified.resolve_default_provider_model.side_effect = lambda *, scope="agentic", fallback_to_first_enabled=True: (
        (prov, model) if prov else (pprov, pmodel) if pprov else (pref_p, pref_m)
    )
    unified.get_enabled_providers.return_value = provs
    unified.resolve_provider_model.side_effect = lambda name: (name, "model-for-" + name)
    unified.get_provider_by_name.side_effect = lambda name: next((p for p in provs if p.name == name), None)
    cfg = MagicMock()
    cfg.unified_config = unified
    return cfg


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class ResolveCrawlModelTest(unittest.TestCase):
    def test_cli_flag_wins_over_instance_and_default(self):
        cfg = _config(agentic=("gemini", "g"))
        inst = {"llm_provider": "ollama", "llm_model": "llama3"}
        self.assertEqual(
            resolve_crawl_model(cfg, inst, cli_provider="openai", cli_model="gpt4"),
            ("openai", "gpt4"),
        )

    def test_instance_wins_over_global_default(self):
        cfg = _config(agentic=("gemini", "g"))
        inst = {"llm_provider": "ollama", "llm_model": "llama3"}
        self.assertEqual(resolve_crawl_model(cfg, inst), ("ollama", "llama3"))

    def test_global_default_when_instance_empty(self):
        cfg = _config(agentic=("gemini", "gemini-flash"))
        self.assertEqual(resolve_crawl_model(cfg, {}), ("gemini", "gemini-flash"))

    def test_all_empty_returns_empty(self):
        cfg = _config()  # no defaults, no enabled providers
        self.assertEqual(resolve_crawl_model(cfg, {}), ("", ""))

    def test_cli_provider_without_model_still_wins(self):
        cfg = _config(agentic=("gemini", "g"))
        # CLI provider set, model empty → returns (provider, "") (caller resolves model).
        self.assertEqual(resolve_crawl_model(cfg, {}, cli_provider="openai"), ("openai", ""))


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class BuildKeywordExtractorTest(unittest.TestCase):
    def test_none_when_llm_service_missing(self):
        self.assertIsNone(build_keyword_extractor(None, "openai", "gpt4"))

    def test_none_when_provider_empty(self):
        self.assertIsNone(build_keyword_extractor(MagicMock(), "", ""))

    def test_none_when_workflow_missing(self):
        with mock.patch("src.core.agents.workflow_loader.find_workflow_file", return_value=None):
            self.assertIsNone(build_keyword_extractor(MagicMock(), "openai", "gpt4"))

    def test_extractor_runs_workflow_and_returns_keywords(self):
        """The extractor drives the webindex_keywords workflow with a fake
        LlmService and returns the JSON keyword list."""
        llm = MagicMock()
        llm.generate_with_tools.side_effect = lambda **kw: AgentResponse(
            content='{"keywords": ["Fernleihe", "Katalog", "Öffnungszeiten"]}',
            tool_calls=[],
        )
        extractor = build_keyword_extractor(llm, "openai", "gpt-4o-mini")
        self.assertIsNotNone(extractor)
        kws = extractor("Die Bibliothek bietet Fernleihe und einen Katalog. Oeffnungszeiten Mo-Fr.", 15)
        self.assertIn("Fernleihe", kws)
        self.assertIn("Katalog", kws)

    def test_extractor_falls_back_to_response_text_when_no_json(self):
        """If the model ignores the JSON rule, the extractor parses the raw text."""
        llm = MagicMock()
        llm.generate_with_tools.side_effect = lambda **kw: AgentResponse(
            content="Fernleihe, Katalog, Oeffnungszeiten",  # no JSON braces
            tool_calls=[],
        )
        extractor = build_keyword_extractor(llm, "openai", "gpt-4o-mini")
        kws = extractor("Die Bibliothek bietet Fernleihe und einen Katalog.", 15)
        self.assertTrue(any("fernleihe" == k.lower() for k in kws))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()