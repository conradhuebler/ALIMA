# State Management Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for state management commands:
    - load-state: Load and display analysis state from JSON
    - save-state: Save analysis state to JSON (deprecated)
"""

import json
import logging
from src.core.data_models import TaskState, AbstractData, AnalysisResult, PromptConfigData
from src.utils.logging_utils import print_result


def handle_load_state(args, logger: logging.Logger):
    """Handle 'load-state' command - Load and display analysis from JSON.

    Args:
        args: Parsed command-line arguments with:
            - input_file: Path to TaskState JSON file
        logger: Logger instance
    """
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            task_state_dict = json.load(f)

        # Reconstruct dataclass objects
        abstract_data = AbstractData(**task_state_dict["abstract_data"])
        analysis_result = AnalysisResult(**task_state_dict["analysis_result"])
        prompt_config = (
            PromptConfigData(**task_state_dict["prompt_config"])
            if task_state_dict["prompt_config"]
            else None
        )

        task_state = TaskState(
            abstract_data=abstract_data,
            analysis_result=analysis_result,
            prompt_config=prompt_config,
            status=task_state_dict["status"],
            task_name=task_state_dict["task_name"],
            model_used=task_state_dict["model_used"],
            provider_used=task_state_dict["provider_used"],
            use_chunking_abstract=task_state_dict["use_chunking_abstract"],
            abstract_chunk_size=task_state_dict["abstract_chunk_size"],
            use_chunking_keywords=task_state_dict["use_chunking_keywords"],
            keyword_chunk_size=task_state_dict["keyword_chunk_size"],
        )

        print_result("--- Loaded Analysis Result ---")
        print_result(task_state.analysis_result.full_text)
        print_result("--- Matched Keywords ---")
        print_result(task_state.analysis_result.matched_keywords)
        print_result("--- GND Systematic ---")
        print_result(task_state.analysis_result.gnd_systematic)
        logger.info(f"Task state loaded from {args.input_file}")

    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {args.input_file}")
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON in {args.input_file}")
    except KeyError as e:
        logger.error(f"Error: Missing key in JSON data: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading task state: {e}")


def handle_save_state(args, logger: logging.Logger):
    """Handle 'save-state' command - Save analysis state (deprecated).

    Args:
        args: Parsed command-line arguments
        logger: Logger instance
    """
    logger.error(
        "The 'save-state' command is not yet fully implemented as a standalone command. "
        "Use 'pipeline --output-json' instead."
    )
