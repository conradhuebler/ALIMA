"""A provider whose endpoint cannot work should say so at the point of entry.

A wrong base URL produces no error anywhere in ALIMA: the provider is stored,
``LlmService.is_provider_reachable`` fails, ``get_available_models`` returns an
empty list (``llm_service.py:941``), and the model combo simply stays empty. In a
real installation a Mistral provider sat like that with the API key in the URL
field and the URL in the (masked) key field — a swap that is invisible, because
a URL pasted into a password field shows up as dots.

The plugin settings tab already sanity-checks its URL fields via
``net_guard.check_operator_url``; the LLM provider dialog did not.

Deliberately quiet about plain ``http``: intranet endpoints are a legitimate and
common setup here, and a warning on every save of a working provider is noise.
- Claude Generated
"""

from __future__ import annotations

import unittest

from src.ui.provider_dialogs import _endpoint_warnings


class QuietForWorkingEndpointsTest(unittest.TestCase):
    def test_a_normal_https_endpoint_is_quiet(self):
        self.assertEqual(_endpoint_warnings("https://api.mistral.ai/v1", "sk-abc"), [])

    def test_an_intranet_http_endpoint_is_quiet(self):
        """Legitimate here; the unencrypted note would fire on every save."""
        self.assertEqual(
            _endpoint_warnings("http://139.20.140.163:11434/v1", "key"), []
        )

    def test_an_unconfigured_provider_is_quiet(self):
        self.assertEqual(_endpoint_warnings("", ""), [])

    def test_a_provider_without_a_key_is_quiet(self):
        self.assertEqual(_endpoint_warnings("http://localhost:11434/v1", ""), [])


class SwappedFieldsAreNamedTest(unittest.TestCase):
    def test_the_swap_is_reported_as_a_swap(self):
        warnings = _endpoint_warnings(
            "lqsllfINPdvISD9op5DG7pc5zW2OqYbz", "https://api.mistral.ai/v1"
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("vertauscht", warnings[0])

    def test_the_swap_message_explains_why_it_is_invisible(self):
        warnings = _endpoint_warnings("keyish", "https://api.example/v1")
        self.assertIn("maskiert", warnings[0])

    def test_a_correct_pair_is_not_reported_as_swapped(self):
        self.assertEqual(_endpoint_warnings("https://a.example/v1", "https://b"), [])


class BrokenEndpointsAreReportedWithoutLeakingTheValueTest(unittest.TestCase):
    def test_a_missing_scheme_is_reported(self):
        warnings = _endpoint_warnings("api.mistral.ai/v1", "sk-abc")
        self.assertTrue(warnings)
        self.assertIn("http/https", warnings[0])

    def test_a_non_url_value_is_not_echoed(self):
        """It is usually the key; a warning dialog must not print it."""
        secret = "lqsllfINPdvISD9op5DG7pc5zW2OqYbz"
        warnings = _endpoint_warnings(secret, "not-a-url")
        self.assertTrue(warnings)
        self.assertNotIn(secret, warnings[0])
        self.assertIn("(Feldinhalt)", warnings[0])

    def test_a_real_url_is_echoed_so_the_message_is_useful(self):
        warnings = _endpoint_warnings("https:///v1", "k")
        self.assertTrue(warnings)
        self.assertIn("https:///v1", warnings[0])


if __name__ == "__main__":
    unittest.main()
