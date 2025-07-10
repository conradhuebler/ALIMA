import unittest
import json
import os
from unittest.mock import MagicMock, patch
from io import StringIO
from contextlib import redirect_stdout

from alima_cli import main # Assuming alima_cli.py is in the root directory
from dataclasses import asdict
from src.core.data_models import AbstractData, AnalysisResult, TaskState, PromptConfigData

class TestCli(unittest.TestCase):

    def setUp(self):
        # Mock LlmService and PromptService
        self.mock_llm_service = MagicMock()
        self.mock_prompt_service = MagicMock()

        # Mock the get_prompt_config method to return a dummy PromptConfigData
        self.mock_prompt_service.get_prompt_config.return_value = PromptConfigData(
            prompt="Test prompt {abstract} {keywords}",
            system="Test system prompt",
            temp=0.5,
            p_value=0.9,
            models=["test-model"],
            seed=42
        )

        # Mock the generate_response method to return a predefined response
        self.mock_llm_service.generate_response.return_value = iter([
            "Basierend auf dem Text und den vorgeschlagenen GND-Systematik-Kategorien schlage ich folgende Schlagworte vor:\n\nSchlagworte:\nKadmium\nBodenverschmutzung\nPflanzenkrankheit\nUmweltgifte\nToxizität\nMenschliche Gesundheit\nLandwirtschaft\nÖkologie\nAbfallwirtschaft\nGesundheitsrisiko\n\nZusätzlich zu den genannten Schlagworten wären folgende GND-Systematik-Kategorien passend:\n\n10.7 Umweltschutz\n27.9 Innere Medizin\n32.5 Phytomedizin\n22.2 Theoretische Chemie\n24.3 Spezielle Botanik\n25.3 Spezielle Zoologie\n31.4 Bergbau\n31.6 Maschinenbau\n32.1 Landwirtschaft\n32.7 Milchwirtschaft\n\n<final_list>Umweltverschmutzung|Bodenkontamination|Pflanzenkrankheit|Gesundheitsrisiko|Toxizität|Mülldeponie|Chemikalieneinsatz|Landwirtschaftliche Flächen|Gesundheitsgefährdung</final_list>\n\n<class>10.7|27.1|32.5|22.2|27.6</class>"
        ])

        # Patch the services in alima_cli.py
        self.patcher_llm_service = patch('alima_cli.LlmService', return_value=self.mock_llm_service)
        self.patcher_prompt_service = patch('alima_cli.PromptService', return_value=self.mock_prompt_service)
        self.mock_llm_service_instance = self.patcher_llm_service.start()
        self.mock_prompt_service_instance = self.patcher_prompt_service.start()

        # Define a dummy output JSON file path
        self.output_json_file = "test_output.json"
        if os.path.exists(self.output_json_file):
            os.remove(self.output_json_file)

    def tearDown(self):
        self.patcher_llm_service.stop()
        self.patcher_prompt_service.stop()
        if os.path.exists(self.output_json_file):
            os.remove(self.output_json_file)

    def test_run_command_and_output_json(self):
        test_abstract = "This is a test abstract about environmental pollution."
        test_model = "test-model"
        test_task = "abstract"

        # Capture stdout
        with StringIO() as buf, redirect_stdout(buf):
            with patch('sys.argv', [
                'alima_cli.py',
                'run',
                test_task,
                '--abstract',
                test_abstract,
                '--model',
                test_model,
                '--output-json',
                self.output_json_file
            ]):
                main()
            output = buf.getvalue()

        # Assertions for stdout output
        self.assertIn("--- Analysis Result ---", output)
        self.assertIn("--- Matched Keywords ---", output)
        self.assertIn("--- GND Systematic ---", output)
        self.assertIn("""{'Umweltverschmutzung': None, 'Bodenkontamination': None, 'Pflanzenkrankheit': None, 'Gesundheitsrisiko': None, 'Toxizität': None, 'Mülldeponie': None, 'Chemikalieneinsatz': None, 'Landwirtschaftliche Flächen': None, 'Gesundheitsgefährdung': None}""", output)
        self.assertIn("10.7 Umweltschutz", output)
        self.assertIn("27.9 Innere Medizin", output)
        self.assertIn("32.5 Phytomedizin", output)
        self.assertIn("22.2 Theoretische Chemie", output)
        self.assertIn("24.3 Spezielle Botanik", output)
        self.assertIn("25.3 Spezielle Zoologie", output)
        self.assertIn("31.4 Bergbau", output)
        self.assertIn("31.6 Maschinenbau", output)
        self.assertIn("32.1 Landwirtschaft", output)
        self.assertIn("32.7 Milchwirtschaft", output)

        # Assertions for JSON file content
        self.assertTrue(os.path.exists(self.output_json_file))
        with open(self.output_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertEqual(data['abstract_data']['abstract'], test_abstract)
        self.assertEqual(data['analysis_result']['matched_keywords']['Umweltverschmutzung'], None)
        self.assertEqual(data['analysis_result']['gnd_systematic'], "10.7|27.1|32.5|22.2|27.6")
        self.assertEqual(data['task_name'], test_task)
        self.assertEqual(data['model_used'], test_model)
        self.assertEqual(data['status'], "completed")

    def test_load_state_command(self):
        test_abstract = "Another test abstract."
        test_model = "another-test-model"
        test_task = "keywords"
        test_full_text = "This is the full text response."
        test_matched_keywords = {"keyword1": None, "keyword2": "GND123"}
        test_gnd_systematic = "1.2|3.4"

        # Create a dummy TaskState to save
        dummy_task_state = TaskState(
            abstract_data=AbstractData(abstract=test_abstract, keywords=""),
            analysis_result=AnalysisResult(
                full_text=test_full_text,
                matched_keywords=test_matched_keywords,
                gnd_systematic=test_gnd_systematic,
            ),
            prompt_config=PromptConfigData(
                prompt="Dummy prompt", system="Dummy system", temp=0.5, p_value=0.9, models=["dummy"], seed=1
            ),
            status="completed",
            task_name=test_task,
            model_used=test_model,
            provider_used="ollama",
            use_chunking_abstract=False,
            abstract_chunk_size=100,
            use_chunking_keywords=False,
            keyword_chunk_size=500,
        )

        with open(self.output_json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(dummy_task_state), f, ensure_ascii=False, indent=4)

        # Capture stdout
        with StringIO() as buf, redirect_stdout(buf):
            with patch('sys.argv', [
                'alima_cli.py',
                'load-state',
                self.output_json_file
            ]):
                main()
            output = buf.getvalue()

        # Assertions for stdout output
        self.assertIn("--- Loaded Analysis Result ---", output)
        self.assertIn(test_full_text, output)
        self.assertIn(str(test_matched_keywords), output)
        self.assertIn(test_gnd_systematic, output)

if __name__ == '__main__':
    unittest.main()
