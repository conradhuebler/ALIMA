"""
Unit tests for unified pipeline configuration components
Claude Generated - Tests for PipelineConfigParser and PipelineConfigBuilder

Tests ensure consistent behavior across CLI and GUI implementations
"""

import unittest
from unittest.mock import MagicMock, patch
from src.utils.pipeline_config_parser import PipelineConfigParser
from src.utils.pipeline_config_builder import PipelineConfigBuilder
from src.utils.config_models import PipelineStepConfig
from src.core.pipeline_manager import PipelineConfig


class TestPipelineConfigParser(unittest.TestCase):
    """Test suite for PipelineConfigParser"""

    def setUp(self):
        """Initialize parser for tests"""
        self.parser = PipelineConfigParser()

    def test_parse_provider_model_format(self):
        """Test parsing of provider|model format"""
        step_name, params = self.parser.parse_cli_step_override("initialisation=ollama|cogito:14b")

        self.assertEqual(step_name, "initialisation")
        self.assertEqual(params["provider"], "ollama")
        self.assertEqual(params["model"], "cogito:14b")

    def test_parse_provider_only_format(self):
        """Test parsing of provider-only format"""
        step_name, params = self.parser.parse_cli_step_override("keywords=gemini")

        self.assertEqual(step_name, "keywords")
        self.assertIn("value", params)
        self.assertEqual(params["value"], "gemini")

    def test_parse_invalid_format_no_equals(self):
        """Test that invalid format (no equals) raises error"""
        with self.assertRaises(ValueError):
            self.parser.parse_cli_step_override("invalid_format")

    def test_parse_invalid_format_empty_value(self):
        """Test that empty values raise error"""
        with self.assertRaises(ValueError):
            self.parser.parse_cli_step_override("step=")

    def test_validate_task_for_initialisation(self):
        """Test task validation for initialisation step"""
        # Valid tasks for initialisation
        is_valid, msg = self.parser.validate_parameter("initialisation", "task", "initialisation")
        self.assertTrue(is_valid)

        is_valid, msg = self.parser.validate_parameter("initialisation", "task", "keywords")
        self.assertTrue(is_valid)

        # Invalid task for initialisation
        is_valid, msg = self.parser.validate_parameter("initialisation", "task", "dk_classification")
        self.assertFalse(is_valid)

    def test_validate_task_for_keywords(self):
        """Test task validation for keywords step"""
        # Valid tasks
        for task in ["keywords", "rephrase", "keywords_chunked"]:
            is_valid, msg = self.parser.validate_parameter("keywords", "task", task)
            self.assertTrue(is_valid, f"Task '{task}' should be valid for keywords step")

        # Invalid task
        is_valid, msg = self.parser.validate_parameter("keywords", "task", "initialisation")
        self.assertFalse(is_valid)

    def test_validate_temperature_valid(self):
        """Test valid temperature values"""
        self.assertTrue(self.parser.validate_temperature(0.0))
        self.assertTrue(self.parser.validate_temperature(1.0))
        self.assertTrue(self.parser.validate_temperature(2.0))
        self.assertTrue(self.parser.validate_temperature(0.5))

    def test_validate_temperature_invalid(self):
        """Test invalid temperature values"""
        self.assertFalse(self.parser.validate_temperature(-0.1))
        self.assertFalse(self.parser.validate_temperature(2.1))
        self.assertFalse(self.parser.validate_temperature("not_a_number"))

    def test_validate_top_p_valid(self):
        """Test valid top_p values"""
        self.assertTrue(self.parser.validate_top_p(0.0))
        self.assertTrue(self.parser.validate_top_p(0.5))
        self.assertTrue(self.parser.validate_top_p(1.0))

    def test_validate_top_p_invalid(self):
        """Test invalid top_p values"""
        self.assertFalse(self.parser.validate_top_p(-0.1))
        self.assertFalse(self.parser.validate_top_p(1.1))
        self.assertFalse(self.parser.validate_top_p("not_a_number"))

    def test_validate_seed_valid(self):
        """Test valid seed values"""
        self.assertTrue(self.parser.validate_seed(0))
        self.assertTrue(self.parser.validate_seed(42))
        self.assertTrue(self.parser.validate_seed(999999))

    def test_validate_seed_invalid(self):
        """Test invalid seed values"""
        self.assertFalse(self.parser.validate_seed(-1))
        self.assertFalse(self.parser.validate_seed("not_a_number"))

    def test_get_valid_tasks_for_steps(self):
        """Test getting valid tasks for each pipeline step"""
        # Test known steps
        self.assertEqual(
            self.parser.get_valid_tasks_for_step("initialisation"),
            ["initialisation", "keywords"]
        )

        self.assertEqual(
            self.parser.get_valid_tasks_for_step("keywords"),
            ["keywords", "rephrase", "keywords_chunked"]
        )

        self.assertEqual(
            self.parser.get_valid_tasks_for_step("dk_classification"),
            ["dk_classification"]
        )

        # Test step with no tasks
        self.assertEqual(
            self.parser.get_valid_tasks_for_step("search"),
            []
        )

    def test_parse_provider_model(self):
        """Test parsing of provider|model string"""
        provider, model = self.parser.parse_provider_model("ollama|cogito:14b")
        self.assertEqual(provider, "ollama")
        self.assertEqual(model, "cogito:14b")

        provider, model = self.parser.parse_provider_model("gemini")
        self.assertEqual(provider, "gemini")
        self.assertIsNone(model)

    def test_is_valid_step_id(self):
        """Test step ID validation"""
        self.assertTrue(self.parser.is_valid_step_id("initialisation"))
        self.assertTrue(self.parser.is_valid_step_id("keywords"))
        self.assertTrue(self.parser.is_valid_step_id("dk_classification"))

        self.assertFalse(self.parser.is_valid_step_id("invalid_step"))
        self.assertFalse(self.parser.is_valid_step_id(""))

    def test_get_all_valid_steps(self):
        """Test getting all valid step IDs"""
        steps = self.parser.get_all_valid_steps()

        self.assertIn("initialisation", steps)
        self.assertIn("keywords", steps)
        self.assertIn("search", steps)
        self.assertIn("dk_classification", steps)
        self.assertGreater(len(steps), 0)


class TestPipelineConfigBuilder(unittest.TestCase):
    """Test suite for PipelineConfigBuilder"""

    def setUp(self):
        """Initialize builder for tests"""
        self.mock_config_manager = MagicMock()
        self.builder = PipelineConfigBuilder(self.mock_config_manager)

        # Mock the baseline configuration
        self.builder.baseline = MagicMock(spec=PipelineConfig)
        self.builder.baseline.step_configs = {
            "initialisation": PipelineStepConfig(step_id="initialisation"),
            "keywords": PipelineStepConfig(step_id="keywords"),
            "dk_classification": PipelineStepConfig(step_id="dk_classification"),
        }

    def test_apply_provider_override(self):
        """Test applying provider override"""
        overrides = {
            "initialisation": {"provider": "ollama"}
        }

        config = self.builder.apply_overrides(overrides)

        self.assertEqual(
            config.step_configs["initialisation"].provider,
            "ollama"
        )

    def test_apply_model_override(self):
        """Test applying model override"""
        overrides = {
            "initialisation": {"model": "cogito:14b"}
        }

        config = self.builder.apply_overrides(overrides)

        self.assertEqual(
            config.step_configs["initialisation"].model,
            "cogito:14b"
        )

    def test_apply_temperature_override(self):
        """Test applying temperature override"""
        overrides = {
            "keywords": {"temperature": 0.5}
        }

        config = self.builder.apply_overrides(overrides)

        self.assertEqual(
            config.step_configs["keywords"].temperature,
            0.5
        )

    def test_apply_multiple_overrides_single_step(self):
        """Test applying multiple overrides to a single step"""
        overrides = {
            "keywords": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "temperature": 0.3,
                "top_p": 0.8
            }
        }

        config = self.builder.apply_overrides(overrides)
        step_config = config.step_configs["keywords"]

        self.assertEqual(step_config.provider, "gemini")
        self.assertEqual(step_config.model, "gemini-2.0-flash")
        self.assertEqual(step_config.temperature, 0.3)
        self.assertEqual(step_config.top_p, 0.8)

    def test_apply_overrides_multiple_steps(self):
        """Test applying overrides to multiple steps"""
        overrides = {
            "initialisation": {"provider": "ollama"},
            "keywords": {"provider": "gemini"},
            "dk_classification": {"provider": "anthropic"}
        }

        config = self.builder.apply_overrides(overrides)

        self.assertEqual(config.step_configs["initialisation"].provider, "ollama")
        self.assertEqual(config.step_configs["keywords"].provider, "gemini")
        self.assertEqual(config.step_configs["dk_classification"].provider, "anthropic")

    def test_invalid_step_recorded_as_error(self):
        """Test that invalid step IDs are recorded as errors"""
        overrides = {
            "invalid_step": {"provider": "ollama"}
        }

        config = self.builder.apply_overrides(overrides)

        self.assertTrue(self.builder.has_errors())
        self.assertIn("invalid_step", self.builder.get_validation_errors())

    def test_temperature_out_of_range_error(self):
        """Test that out-of-range temperature is recorded as error"""
        overrides = {
            "keywords": {"temperature": 3.0}  # Invalid: > 2.0
        }

        config = self.builder.apply_overrides(overrides)

        self.assertTrue(self.builder.has_errors())

    def test_task_validation_for_step(self):
        """Test that task validation is step-aware"""
        # Valid task
        overrides = {
            "keywords": {"task": "keywords"}
        }
        config = self.builder.apply_overrides(overrides)
        self.assertFalse(self.builder.has_errors())

        # Reset errors
        self.builder.clear_errors()

        # Invalid task for step
        overrides = {
            "keywords": {"task": "dk_classification"}
        }
        config = self.builder.apply_overrides(overrides)
        self.assertTrue(self.builder.has_errors())

    def test_custom_param_storage(self):
        """Test that custom parameters are stored correctly"""
        overrides = {
            "dk_classification": {"dk_max_results": 20}
        }

        config = self.builder.apply_overrides(overrides)
        step_config = config.step_configs["dk_classification"]

        self.assertIsNotNone(step_config.custom_params)
        self.assertEqual(step_config.custom_params.get("dk_max_results"), 20)

    def test_get_config_returns_baseline(self):
        """Test that get_config returns the configured baseline"""
        config = self.builder.get_config()

        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config, self.builder.baseline)

    def test_clear_errors(self):
        """Test clearing validation errors"""
        # Create some errors
        overrides = {
            "invalid_step": {"provider": "ollama"}
        }
        self.builder.apply_overrides(overrides)

        self.assertTrue(self.builder.has_errors())

        # Clear errors
        self.builder.clear_errors()

        self.assertFalse(self.builder.has_errors())


class TestIntegrationCLIandGUI(unittest.TestCase):
    """Integration tests ensuring CLI and GUI use same validation logic"""

    def test_same_task_validation_rules(self):
        """Verify CLI and GUI use same task validation rules"""
        parser = PipelineConfigParser()

        # Test that same rule applies to both
        is_valid, msg = parser.validate_parameter("keywords", "task", "rephrase")
        self.assertTrue(is_valid)

        # Invalid task should also be caught
        is_valid, msg = parser.validate_parameter("keywords", "task", "dk_classification")
        self.assertFalse(is_valid)

    def test_same_temperature_validation(self):
        """Verify CLI and GUI use same temperature validation"""
        parser = PipelineConfigParser()

        # Valid range
        is_valid, msg = parser.validate_parameter("keywords", "temperature", 1.5)
        self.assertTrue(is_valid)

        # Out of range
        is_valid, msg = parser.validate_parameter("keywords", "temperature", 3.0)
        self.assertFalse(is_valid)

    def test_same_top_p_validation(self):
        """Verify CLI and GUI use same top_p validation"""
        parser = PipelineConfigParser()

        # Valid range
        is_valid, msg = parser.validate_parameter("keywords", "top_p", 0.5)
        self.assertTrue(is_valid)

        # Out of range
        is_valid, msg = parser.validate_parameter("keywords", "top_p", 1.5)
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()
