"""APIKeyValidator.validate_gemini against the real google-genai SDK.

Used by the first-start wizard when a Gemini key is typed in. It imported
``google.generativeai`` (does not exist under that name - the pinned,
installed SDK is ``google-genai``/``google.genai``) and called
module-level ``configure()``/``GenerativeModel()``. Every attempt raised
``ModuleNotFoundError`` and was reported to the operator as "API-Schlüssel-
Validierung fehlgeschlagen" - indistinguishable from an actually wrong key.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.utils.setup_utils import APIKeyValidator


class ValidateGeminiUsesTheRealSdkTest(unittest.TestCase):
    def test_too_short_a_key_fails_the_format_check_before_any_sdk_call(self):
        result = APIKeyValidator.validate_gemini("short")
        self.assertFalse(result.success)
        self.assertIn("Format", result.message)

    def test_a_working_key_is_reported_valid(self):
        fake_client = MagicMock()
        with patch("google.genai.Client", return_value=fake_client):
            result = APIKeyValidator.validate_gemini("a-long-enough-fake-api-key")
        self.assertTrue(result.success)
        fake_client.models.generate_content.assert_called_once()

    def test_the_client_is_constructed_with_the_given_key(self):
        fake_client = MagicMock()
        with patch("google.genai.Client", return_value=fake_client) as ctor:
            APIKeyValidator.validate_gemini("a-long-enough-fake-api-key")
        ctor.assert_called_once_with(api_key="a-long-enough-fake-api-key")

    def test_an_sdk_error_is_reported_not_raised(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("401 invalid key")
        with patch("google.genai.Client", return_value=fake_client):
            result = APIKeyValidator.validate_gemini("a-long-enough-fake-api-key")
        self.assertFalse(result.success)
        self.assertIn("401 invalid key", result.message)

    def test_does_not_import_the_nonexistent_old_package(self):
        """Pin it structurally: the old import must not come back."""
        import inspect

        from src.utils import setup_utils

        source = inspect.getsource(setup_utils.APIKeyValidator.validate_gemini)
        self.assertNotIn("import google.generativeai", source)
        self.assertIn("from google import genai", source)


if __name__ == "__main__":
    unittest.main()
