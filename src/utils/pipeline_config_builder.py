"""
Unified Pipeline Configuration Builder
Claude Generated - Consolidates configuration building logic for CLI and GUI

This module provides a single builder class that:
- Constructs baseline pipeline configuration from provider preferences
- Applies parameter overrides with validation
- Ensures consistency between CLI and GUI configuration handling
"""

from typing import Dict, Any, Tuple, Optional, List
import logging

from .pipeline_config_parser import PipelineConfigParser
from .config_models import PipelineStepConfig
from ..core.pipeline_manager import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineConfigBuilder:
    """Unified builder for constructing validated pipeline configurations

    This class consolidates configuration building logic that was previously
    duplicated between CLI (alima_cli.py's apply_cli_overrides) and GUI
    (pipeline_config_dialog.py's configuration extraction).

    Usage:
        builder = PipelineConfigBuilder(config_manager)

        # Parse overrides from CLI
        overrides = {}
        for cli_arg in args.step_temperature:
            step_name, params = PipelineConfigParser.parse_cli_step_override(cli_arg)
            overrides.setdefault(step_name, {})['temperature'] = float(params['value'])

        # Apply with validation
        config = builder.apply_overrides(overrides)
    """

    def __init__(self, config_manager):
        """Initialize builder with baseline configuration

        Args:
            config_manager: ConfigManager instance for creating baseline config
        """
        self.config_manager = config_manager
        self.parser = PipelineConfigParser()

        # Create baseline configuration from provider preferences
        self.baseline = PipelineConfig.create_from_provider_preferences(config_manager)

        # Track validation errors
        self.validation_errors: Dict[str, List[str]] = {}

    def apply_overrides(self, overrides: Dict[str, Dict[str, Any]]) -> PipelineConfig:
        """Apply parameter overrides to baseline configuration with validation

        Args:
            overrides: Dictionary of overrides in format:
                {
                    "step_name": {
                        "provider": "ollama",
                        "model": "cogito:14b",
                        "temperature": 0.3,
                        "task": "keywords",
                        "top_p": 0.1,
                        "seed": 42,
                        ...
                    }
                }

        Returns:
            Updated PipelineConfig with validated overrides applied

        Raises:
            ValueError: If critical validation errors occur (can be configured)
        """
        self.validation_errors.clear()

        for step_name, step_params in overrides.items():
            if not self.parser.is_valid_step_id(step_name):
                self._record_error(step_name, f"Unknown pipeline step: '{step_name}'")
                continue

            # Get the step config from baseline
            if step_name not in self.baseline.step_configs:
                self._record_error(step_name, f"Step '{step_name}' not in pipeline configuration")
                continue

            step_config = self.baseline.step_configs[step_name]

            # Apply and validate each parameter
            for param_name, param_value in step_params.items():
                success, error_msg = self.validate_and_apply_single_override(
                    step_name, param_name, param_value
                )

                if not success:
                    self._record_error(step_name, error_msg)
                    # Continue processing other parameters even if one fails
                    # (allows partial configuration)

        # Log any validation errors
        if self.validation_errors:
            self._log_validation_errors()

        return self.baseline

    def validate_and_apply_single_override(
        self,
        step_name: str,
        param_name: str,
        value: Any
    ) -> Tuple[bool, str]:
        """Validate and apply a single parameter override

        This is the core override application logic, replacing the direct
        mutation logic from apply_cli_overrides().

        Args:
            step_name: Pipeline step identifier
            param_name: Parameter name to override
            value: New parameter value

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            if step_name not in self.baseline.step_configs:
                return False, f"Step '{step_name}' not found in configuration"

            step_config = self.baseline.step_configs[step_name]

            # Special handling for different parameter types
            if param_name == "provider":
                # Provider is set directly (no validation needed here)
                step_config.provider = str(value) if value else None
                return True, ""

            elif param_name == "model":
                # Model is set directly (no validation needed here)
                step_config.model = str(value) if value else None
                return True, ""

            elif param_name == "task":
                # Task requires validation against step type
                is_valid, error_msg = self.parser.validate_parameter(step_name, param_name, value)
                if not is_valid:
                    return False, error_msg
                step_config.task = str(value)
                return True, ""

            elif param_name == "temperature":
                # Temperature with range validation
                is_valid, error_msg = self.parser.validate_parameter(step_name, param_name, value)
                if not is_valid:
                    return False, error_msg
                step_config.temperature = float(value)
                return True, ""

            elif param_name == "top_p":
                # Top_p with range validation
                is_valid, error_msg = self.parser.validate_parameter(step_name, param_name, value)
                if not is_valid:
                    return False, error_msg
                step_config.top_p = float(value)
                return True, ""

            elif param_name == "seed":
                # Seed with integer validation
                is_valid, error_msg = self.parser.validate_parameter(step_name, param_name, value)
                if not is_valid:
                    return False, error_msg
                step_config.seed = int(value)
                return True, ""

            elif param_name == "max_tokens":
                # Max tokens validation
                try:
                    step_config.max_tokens = int(value)
                    return True, ""
                except (ValueError, TypeError):
                    return False, f"max_tokens: Expected integer, got {type(value).__name__}"

            elif param_name in ["dk_max_results", "dk_frequency_threshold", "keyword_chunking_threshold"]:
                # Custom parameters stored in custom_params dict
                is_valid, error_msg = self.parser.validate_parameter(step_name, param_name, value)
                if not is_valid:
                    return False, error_msg

                if not hasattr(step_config, 'custom_params') or step_config.custom_params is None:
                    step_config.custom_params = {}

                step_config.custom_params[param_name] = int(value)
                return True, ""

            elif param_name == "chunking_task":
                # Custom parameter for chunking task
                if not hasattr(step_config, 'custom_params') or step_config.custom_params is None:
                    step_config.custom_params = {}

                step_config.custom_params['chunking_task'] = str(value)
                return True, ""

            elif param_name == "enable_iterative_refinement":
                # Iterative refinement flag - Claude Generated
                step_config.enable_iterative_refinement = bool(value)
                return True, ""

            elif param_name == "max_refinement_iterations":
                # Max refinement iterations - Claude Generated
                try:
                    step_config.max_refinement_iterations = int(value)
                    return True, ""
                except (ValueError, TypeError):
                    return False, f"max_refinement_iterations: Expected integer, got {type(value).__name__}"

            elif param_name == "enabled":
                # Enable/disable step
                step_config.enabled = bool(value)
                return True, ""

            else:
                # Unknown parameter - could be a future addition, allow it
                # Store in custom_params as fallback
                if not hasattr(step_config, 'custom_params') or step_config.custom_params is None:
                    step_config.custom_params = {}

                step_config.custom_params[param_name] = value
                logger.debug(f"Unknown parameter '{param_name}' stored in custom_params")
                return True, ""

        except Exception as e:
            return False, f"Error applying override: {e}"

    def get_config(self) -> PipelineConfig:
        """Get the final configuration

        Returns:
            Configured PipelineConfig object ready for execution
        """
        return self.baseline

    def get_validation_errors(self) -> Dict[str, List[str]]:
        """Get accumulated validation errors

        Returns:
            Dictionary of step_name -> list of error messages
        """
        return self.validation_errors

    def has_errors(self) -> bool:
        """Check if any validation errors occurred

        Returns:
            True if validation errors exist
        """
        return bool(self.validation_errors)

    def clear_errors(self):
        """Clear accumulated validation errors"""
        self.validation_errors.clear()

    def _record_error(self, step_name: str, error_msg: str):
        """Record a validation error

        Args:
            step_name: Pipeline step name
            error_msg: Error message
        """
        if step_name not in self.validation_errors:
            self.validation_errors[step_name] = []
        self.validation_errors[step_name].append(error_msg)

    def _log_validation_errors(self):
        """Log all accumulated validation errors"""
        logger.warning("Pipeline configuration validation errors:")
        for step_name, errors in self.validation_errors.items():
            for error in errors:
                logger.warning(f"  [{step_name}] {error}")

    @staticmethod
    def parse_and_apply_cli_args(builder: 'PipelineConfigBuilder', args) -> PipelineConfig:
        """Convenience method to parse and apply all CLI arguments

        This replaces the build_step_configurations() + apply_cli_overrides()
        functions from alima_cli.py

        Args:
            builder: PipelineConfigBuilder instance
            args: Parsed command-line arguments

        Returns:
            Configured PipelineConfig
        """
        overrides: Dict[str, Dict[str, Any]] = {}
        parser = PipelineConfigParser()

        # Parse --step arguments (format: step=provider|model)
        if hasattr(args, 'step') and args.step:
            for step_override in args.step:
                step_name, parsed = parser.parse_cli_step_override(step_override)
                if step_name not in overrides:
                    overrides[step_name] = {}
                overrides[step_name].update(parsed)

        # Parse --step-task arguments (format: step=task)
        if hasattr(args, 'step_task') and args.step_task:
            for step_task in args.step_task:
                step_name, parsed = parser.parse_cli_step_override(step_task)
                if step_name not in overrides:
                    overrides[step_name] = {}
                # Extract task value from the parsed result
                if 'value' in parsed:
                    overrides[step_name]['task'] = parsed['value']

        # Parse --step-temperature arguments
        if hasattr(args, 'step_temperature') and args.step_temperature:
            for step_temp in args.step_temperature:
                step_name, parsed = parser.parse_cli_step_override(step_temp)
                if step_name not in overrides:
                    overrides[step_name] = {}
                if 'value' in parsed:
                    overrides[step_name]['temperature'] = float(parsed['value'])

        # Parse --step-top-p arguments
        if hasattr(args, 'step_top_p') and args.step_top_p:
            for step_top_p in args.step_top_p:
                step_name, parsed = parser.parse_cli_step_override(step_top_p)
                if step_name not in overrides:
                    overrides[step_name] = {}
                if 'value' in parsed:
                    overrides[step_name]['top_p'] = float(parsed['value'])

        # Parse --step-seed arguments
        if hasattr(args, 'step_seed') and args.step_seed:
            for step_seed in args.step_seed:
                step_name, parsed = parser.parse_cli_step_override(step_seed)
                if step_name not in overrides:
                    overrides[step_name] = {}
                if 'value' in parsed:
                    overrides[step_name]['seed'] = int(parsed['value'])

        # Parse --disable-dk-classification
        if hasattr(args, 'disable_dk_classification') and args.disable_dk_classification:
            if 'dk_classification' not in overrides:
                overrides['dk_classification'] = {}
            overrides['dk_classification']['enabled'] = False

        # Parse DK classification parameters
        if hasattr(args, 'dk_max_results') and args.dk_max_results is not None:
            if 'dk_classification' not in overrides:
                overrides['dk_classification'] = {}
            overrides['dk_classification']['dk_max_results'] = args.dk_max_results

        if hasattr(args, 'dk_frequency_threshold') and args.dk_frequency_threshold is not None:
            if 'dk_classification' not in overrides:
                overrides['dk_classification'] = {}
            overrides['dk_classification']['dk_frequency_threshold'] = args.dk_frequency_threshold

        # Parse keyword chunking parameters
        if hasattr(args, 'keyword_chunking_threshold') and args.keyword_chunking_threshold is not None:
            if 'keywords' not in overrides:
                overrides['keywords'] = {}
            overrides['keywords']['keyword_chunking_threshold'] = args.keyword_chunking_threshold

        if hasattr(args, 'chunking_task') and args.chunking_task is not None:
            if 'keywords' not in overrides:
                overrides['keywords'] = {}
            overrides['keywords']['chunking_task'] = args.chunking_task

        # Parse iterative refinement parameters - Claude Generated
        if hasattr(args, 'enable_iterative_search') and args.enable_iterative_search:
            if 'keywords' not in overrides:
                overrides['keywords'] = {}
            overrides['keywords']['enable_iterative_refinement'] = True

        if hasattr(args, 'max_iterations') and args.max_iterations is not None:
            if 'keywords' not in overrides:
                overrides['keywords'] = {}
            # Validate max_iterations range (1-5)
            max_iter = max(1, min(5, args.max_iterations))
            if max_iter != args.max_iterations:
                logger.warning(f"max_iterations {args.max_iterations} out of range, clamped to {max_iter}")
            overrides['keywords']['max_refinement_iterations'] = max_iter

        # Apply all overrides with validation
        return builder.apply_overrides(overrides)
