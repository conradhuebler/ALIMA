import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.webapp.result_serialization import (
    build_export_payload,
    extract_results_from_analysis_state,
    prepare_results_for_export,
)


class TestWebappResultSerialization(unittest.TestCase):
    @patch("src.webapp.result_serialization.requests.get")
    def test_prepare_results_without_rvk_validation_skips_api_calls(self, mock_get):
        results = {
            "classifications": [
                {"system": "RVK", "code": "QZ 123", "display": "RVK QZ 123"},
                {"system": "DK", "code": "614.7", "display": "DK 614.7"},
            ],
            "rvk_provenance": {"rvk_api": 1},
        }

        exported = prepare_results_for_export(results, validate_rvk=False)

        mock_get.assert_not_called()
        self.assertEqual(exported["classification_validation"]["rvk_total"], 1)
        self.assertEqual(exported["classification_validation"]["rvk_standard"], 0)
        rvk_entry = exported["classifications"][0]
        self.assertEqual(rvk_entry["validation_status"], "not_checked")
        self.assertEqual(rvk_entry["canonical_code"], "QZ 123")

    @patch("src.webapp.result_serialization.requests.get")
    def test_prepare_results_with_rvk_validation_calls_api(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "node": {
                "notation": "QZ 123",
                "benennung": "Biologie",
            }
        }
        mock_get.return_value = response

        exported = prepare_results_for_export(
            {"dk_classifications": ["RVK QZ 123"]},
            validate_rvk=True,
        )

        mock_get.assert_called_once()
        self.assertEqual(exported["classification_validation"]["rvk_standard"], 1)
        self.assertEqual(exported["classifications"][0]["label"], "Biologie")
        self.assertEqual(exported["classifications"][0]["validation_status"], "standard")

    def test_extract_results_from_analysis_state_uses_model_used_and_provider_used(self):
        analysis_state = SimpleNamespace(
            original_abstract="Abstract",
            working_title="Title",
            initial_keywords=["one"],
            dk_classifications=["RVK QZ 123"],
            rvk_provenance={"rvk_api": 1},
            search_results=[SimpleNamespace(search_term="term", results={"kw": {"count": 1}})],
            dk_search_results=[],
            dk_search_results_flattened=[],
            dk_statistics=None,
            search_suggesters_used=["lobid"],
            initial_gnd_classes=["001"],
            initial_llm_call_details=SimpleNamespace(
                response_full_text="init response",
                provider_used="provider-a",
                model_used="model-a",
                extracted_keywords=[],
                extracted_gnd_keywords=["one"],
                token_count=12,
            ),
            final_llm_analysis=SimpleNamespace(
                response_full_text="final response",
                provider_used="provider-b",
                model_used="model-b",
                extracted_keywords=[],
                extracted_gnd_keywords=["two"],
                token_count=34,
                verification={"stats": {"verified_count": 1}},
            ),
        )

        extracted = extract_results_from_analysis_state(analysis_state)

        self.assertEqual(extracted["initial_llm_call_details"]["provider"], "provider-a")
        self.assertEqual(extracted["initial_llm_call_details"]["model"], "model-a")
        self.assertEqual(extracted["final_llm_call_details"]["provider"], "provider-b")
        self.assertEqual(extracted["final_llm_call_details"]["model"], "model-b")
        self.assertEqual(extracted["verification"]["stats"]["verified_count"], 1)

    def test_build_export_payload_wraps_results_in_web_schema(self):
        payload = build_export_payload(
            session_id="cli",
            created_at="2026-03-29T10:00:00",
            status="completed",
            current_step="classification",
            input_data={"type": "text", "text_preview": "Abstract"},
            results={"dk_classifications": ["RVK QZ 123"]},
            autosave_timestamp=None,
            exported_at="2026-03-29T10:01:00",
            validate_rvk=False,
        )

        self.assertEqual(payload["session_id"], "cli")
        self.assertTrue(payload["is_complete"])
        self.assertEqual(payload["input"]["type"], "text")
        self.assertIn("results", payload)
        self.assertEqual(payload["results"]["dk_classifications"], ["RVK QZ 123"])
        self.assertEqual(payload["results"]["classifications"][0]["system"], "RVK")


if __name__ == "__main__":
    unittest.main()
