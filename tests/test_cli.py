import json
import os
import tempfile
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from src.cli.commands import state_cmd


class TestCliStateCommands(unittest.TestCase):
    def setUp(self):
        self.logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        payload = {
            "abstract_data": {"abstract": "Another test abstract.", "keywords": ""},
            "analysis_result": {
                "full_text": "This is the full text response.",
                "matched_keywords": {"keyword1": None, "keyword2": "GND123"},
                "gnd_systematic": "1.2|3.4",
            },
            "prompt_config": {
                "prompt": "Dummy prompt",
                "system": "Dummy system",
                "temp": 0.5,
                "p_value": 0.9,
                "models": ["dummy"],
                "seed": 1,
            },
            "status": "completed",
            "task_name": "keywords",
            "model_used": "another-test-model",
            "provider_used": "ollama",
            "use_chunking_abstract": False,
            "abstract_chunk_size": 100,
            "use_chunking_keywords": False,
            "keyword_chunk_size": 500,
        }
        json.dump(payload, self.temp_file, ensure_ascii=False)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_handle_load_state_prints_loaded_analysis(self):
        args = SimpleNamespace(input_file=self.temp_file.name)

        with patch("src.cli.commands.state_cmd.print_result") as mock_print:
            state_cmd.handle_load_state(args, self.logger)

        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertIn("--- Loaded Analysis Result ---", printed)
        self.assertIn("This is the full text response.", printed)
        self.assertIn("--- Matched Keywords ---", printed)
        self.assertIn({"keyword1": None, "keyword2": "GND123"}, printed)
        self.assertIn("--- GND Systematic ---", printed)
        self.assertIn("1.2|3.4", printed)

    def test_handle_save_state_reports_deprecation(self):
        args = SimpleNamespace(output_file="unused.json")

        logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=unittest.mock.Mock())
        state_cmd.handle_save_state(args, logger)

        logger.error.assert_called_once()
        self.assertIn("not yet fully implemented", logger.error.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
