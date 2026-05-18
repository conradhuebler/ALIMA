"""P-η — Tests for the YAML-driven capability registry (WP11 Sek 3).

Covers load + 3-tier lookup (exact, wildcard, default). Operator can swap the
YAML without touching code; tests must keep the loader honest.

Claude Generated.
"""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.model_capabilities import (
    get_capability,
    load_capabilities_yaml,
    reset_capability_cache,
)


SAMPLE_YAML = textwrap.dedent(
    """
    providers:
      openai_compatible:
        "gpt-4*":
          seed_support: true
          tool_use: parallel_native
          family: openai-chat
      ollama:
        "cogito:32b":
          seed_support: true
          family: thinking
          max_context_tokens: 128000
        "cogito:*":
          seed_support: true
          family: thinking
          max_context_tokens: 32000
    """
).strip()


class TestCapabilityYaml(unittest.TestCase):

    def setUp(self):
        reset_capability_cache()
        self._tmpdir = TemporaryDirectory()
        self.yaml_path = Path(self._tmpdir.name) / "caps.yaml"
        self.yaml_path.write_text(SAMPLE_YAML)

    def tearDown(self):
        self._tmpdir.cleanup()
        reset_capability_cache()

    def test_loader_returns_providers_dict(self):
        data = load_capabilities_yaml(self.yaml_path)
        self.assertIn("openai_compatible", data)
        self.assertIn("ollama", data)

    def test_exact_lookup_takes_precedence_over_wildcard(self):
        # cogito:32b has an exact entry with max_context_tokens=128000;
        # the wildcard "cogito:*" has 32000. Exact must win.
        val = get_capability("ollama", "cogito:32b", "max_context_tokens", yaml_path=self.yaml_path)
        self.assertEqual(val, 128000)

    def test_wildcard_lookup_when_no_exact_match(self):
        val = get_capability("ollama", "cogito:14b", "max_context_tokens", yaml_path=self.yaml_path)
        self.assertEqual(val, 32000)

    def test_wildcard_lookup_openai_family(self):
        val = get_capability("openai_compatible", "gpt-4o", "family", yaml_path=self.yaml_path)
        self.assertEqual(val, "openai-chat")
        val = get_capability("openai_compatible", "gpt-4o-mini", "tool_use", yaml_path=self.yaml_path)
        self.assertEqual(val, "parallel_native")

    def test_unknown_provider_returns_default(self):
        val = get_capability("nonexistent", "x", "seed_support", default="missing", yaml_path=self.yaml_path)
        self.assertEqual(val, "missing")

    def test_unknown_model_returns_default(self):
        val = get_capability("ollama", "phi3:mini", "seed_support", default=False, yaml_path=self.yaml_path)
        self.assertFalse(val)

    def test_unknown_flag_returns_default(self):
        val = get_capability("ollama", "cogito:32b", "made_up_flag", default="x", yaml_path=self.yaml_path)
        self.assertEqual(val, "x")

    def test_missing_yaml_file_returns_empty(self):
        bogus = Path(self._tmpdir.name) / "missing.yaml"
        reset_capability_cache()
        data = load_capabilities_yaml(bogus)
        self.assertEqual(data, {})
        # Lookup must still return default, not crash
        val = get_capability("ollama", "cogito:32b", "seed_support", default=None, yaml_path=bogus)
        self.assertIsNone(val)

    def test_real_project_yaml_loads(self):
        """The shipped config/model_capabilities.yaml must parse and expose 3 providers."""
        reset_capability_cache()
        data = load_capabilities_yaml()  # default path
        self.assertIn("openai_compatible", data)
        self.assertIn("ollama", data)
        self.assertIn("gemini", data)
        # Spot-check a few flags from the shipped file
        self.assertTrue(get_capability("ollama", "cogito:32b", "seed_support"))
        self.assertEqual(get_capability("ollama", "cogito:32b", "family"), "thinking")
        self.assertEqual(get_capability("openai_compatible", "gpt-4o", "tool_use"), "parallel_native")
        self.assertTrue(get_capability("gemini", "gemini-2.5-flash", "seed_support"))

    def test_anthropic_not_in_yaml(self):
        """Operator-decision: Anthropic out of scope; YAML must not list it."""
        reset_capability_cache()
        data = load_capabilities_yaml()
        self.assertNotIn("anthropic", data)


if __name__ == "__main__":
    unittest.main()
