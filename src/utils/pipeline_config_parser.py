"""
Unified Pipeline Configuration Parser and Validator
Claude Generated - Consolidates parameter parsing and validation logic for CLI and GUI

This module provides a single source of truth for:
- Parsing CLI-format configuration strings (e.g., "step=provider|model")
- Validating parameter values for specific pipeline steps
- Step-aware task selection
- Consistent error messages across interfaces
"""

from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineConfigParser:
    """Unified parser for pipeline configuration from any source (CLI or GUI)

    This class consolidates parsing and validation logic that was previously
    duplicated between CLI (alima_cli.py) and GUI (pipeline_config_dialog.py).
    """

    # Step-to-valid-tasks mapping - defines which tasks are valid for each step
    STEP_TASK_MAPPING = {
        "initialisation": ["initialisation", "keywords"],
        "keywords": ["keywords", "rephrase", "keywords_chunked"],
        "dk_classification": ["dk_classification"],
        "search": [],  # Search step doesn't use tasks
        "dk_search": [],  # DK search step doesn't use tasks
    }

    # Parameter validation rules
    PARAMETER_RANGES = {
        "temperature": {"min": 0.0, "max": 2.0, "type": float},
        "top_p": {"min": 0.0, "max": 1.0, "type": float},
        "seed": {"min": 0, "max": None, "type": int},
        "dk_max_results": {"min": 1, "max": None, "type": int},
        "dk_frequency_threshold": {"min": 1, "max": None, "type": int},
        "keyword_chunking_threshold": {"min": 1, "max": None, "type": int},
    }

    @staticmethod
    def parse_cli_step_override(value: str) -> Tuple[str, Dict[str, Any]]:
        """Parse CLI format configuration string

        Handles both:
        - step=provider|model (e.g., "initialisation=ollama|cogito:14b")
        - step=value (e.g., "initialisation=0.5")

        Args:
            value: Configuration string in CLI format

        Returns:
            Tuple of (step_name, parsed_params_dict)

        Raises:
            ValueError: If format is invalid
        """
        try:
            if '=' not in value:
                raise ValueError(f"Missing '=' separator")

            step_name, config_value = value.split('=', 1)
            step_name = step_name.strip()
            config_value = config_value.strip()

            if not step_name:
                raise ValueError("Step name cannot be empty")
            if not config_value:
                raise ValueError("Configuration value cannot be empty")

            # Try to parse as provider|model format
            if '|' in config_value:
                provider, model = config_value.split('|', 1)
                provider = provider.strip() or None
                model = model.strip() or None
                return step_name, {"provider": provider, "model": model}

            # Otherwise return as-is (could be task, temperature, etc.)
            # The caller will determine what parameter this is
            return step_name, {"value": config_value}

        except ValueError as e:
            raise ValueError(
                f"Invalid configuration format: '{value}'. "
                f"Expected 'STEP=PROVIDER|MODEL' or 'STEP=VALUE'. Error: {e}"
            )

    @staticmethod
    def validate_parameter(
        step_id: str,
        param_name: str,
        value: Any
    ) -> Tuple[bool, str]:
        """Validate a parameter for a specific pipeline step

        Args:
            step_id: Pipeline step identifier (e.g., "initialisation", "keywords")
            param_name: Parameter name (e.g., "temperature", "task", "model")
            value: Parameter value to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty string.
        """

        # Validate parameter exists and has a value
        if value is None or (isinstance(value, str) and not value):
            return False, f"{param_name}: Value cannot be empty"

        # Task validation - step-aware
        if param_name == "task":
            valid_tasks = PipelineConfigParser.get_valid_tasks_for_step(step_id)
            if valid_tasks and value not in valid_tasks:
                return False, (
                    f"Invalid task '{value}' for step '{step_id}'. "
                    f"Valid tasks: {', '.join(valid_tasks)}"
                )
            return True, ""

        # Range-based parameter validation
        if param_name in PipelineConfigParser.PARAMETER_RANGES:
            rule = PipelineConfigParser.PARAMETER_RANGES[param_name]
            param_type = rule["type"]

            # Type validation
            try:
                typed_value = param_type(value)
            except (ValueError, TypeError):
                return False, f"{param_name}: Expected {param_type.__name__}, got {type(value).__name__}"

            # Range validation
            min_val = rule.get("min")
            max_val = rule.get("max")

            if min_val is not None and typed_value < min_val:
                return False, f"{param_name}: Value must be >= {min_val}, got {typed_value}"

            if max_val is not None and typed_value > max_val:
                return False, f"{param_name}: Value must be <= {max_val}, got {typed_value}"

            return True, ""

        # Provider and model are not validated here (would require config context)
        if param_name in ["provider", "model"]:
            return True, ""

        # Unknown parameter - allow it (could be custom_param)
        return True, ""

    @staticmethod
    def get_valid_tasks_for_step(step_id: str) -> List[str]:
        """Get valid tasks for a specific pipeline step

        Consolidates logic from GUI's _get_available_tasks_for_step() and
        provides step-aware validation for CLI.

        Args:
            step_id: Pipeline step identifier

        Returns:
            List of valid task names for this step
        """
        return PipelineConfigParser.STEP_TASK_MAPPING.get(step_id, [])

    @staticmethod
    def validate_temperature(value: float) -> bool:
        """Validate temperature parameter (0.0-2.0)

        Args:
            value: Temperature value

        Returns:
            True if valid, False otherwise
        """
        try:
            temp = float(value)
            return 0.0 <= temp <= 2.0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_top_p(value: float) -> bool:
        """Validate top_p parameter (0.0-1.0)

        Args:
            value: Top_p value

        Returns:
            True if valid, False otherwise
        """
        try:
            top_p = float(value)
            return 0.0 <= top_p <= 1.0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_seed(value: int) -> bool:
        """Validate seed parameter (non-negative integer)

        Args:
            value: Seed value

        Returns:
            True if valid, False otherwise
        """
        try:
            seed = int(value)
            return seed >= 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def parse_provider_model(value: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse provider|model format

        Args:
            value: String in format "provider|model"

        Returns:
            Tuple of (provider, model)
        """
        if '|' in value:
            provider, model = value.split('|', 1)
            return provider.strip() or None, model.strip() or None
        else:
            # Just provider
            return value.strip() or None, None

    @staticmethod
    def format_cli_parameter(step_name: str, param_name: str, value: Any) -> str:
        """Format a parameter as a CLI-style string

        Inverse of parse_cli_step_override - used for logging/testing

        Args:
            step_name: Pipeline step name
            param_name: Parameter name
            value: Parameter value

        Returns:
            CLI-format string like "step=value"
        """
        return f"{step_name}={value}"

    @staticmethod
    def is_valid_step_id(step_id: str) -> bool:
        """Check if a step ID is valid

        Args:
            step_id: Step identifier to validate

        Returns:
            True if valid, False otherwise
        """
        valid_steps = list(PipelineConfigParser.STEP_TASK_MAPPING.keys())
        return step_id in valid_steps

    @staticmethod
    def get_all_valid_steps() -> List[str]:
        """Get list of all valid pipeline step IDs

        Returns:
            List of valid step identifiers
        """
        return list(PipelineConfigParser.STEP_TASK_MAPPING.keys())
