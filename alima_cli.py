import argparse
import logging
import os
import json
from dataclasses import asdict

from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.data_models import AbstractData, TaskState, AnalysisResult, PromptConfigData

PROMPTS_FILE = "prompts.json"

def _task_state_to_dict(task_state: TaskState) -> dict:
    """Converts a TaskState dataclass instance to a dictionary, handling nested dataclasses."""
    task_state_dict = asdict(task_state)
    # Convert nested dataclasses to dicts if they are not None
    if task_state_dict.get('abstract_data'):
        task_state_dict['abstract_data'] = asdict(task_state.abstract_data)
    if task_state_dict.get('analysis_result'):
        task_state_dict['analysis_result'] = asdict(task_state.analysis_result)
    if task_state_dict.get('prompt_config'):
        task_state_dict['prompt_config'] = asdict(task_state.prompt_config)
    return task_state_dict

def main():
    """Main function for the ALIMA CLI."""
    parser = argparse.ArgumentParser(description="ALIMA CLI - AI-powered abstract analysis.")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an analysis task.")
    run_parser.add_argument("task", help="The analysis task to perform (e.g., 'abstract', 'keywords').")
    run_parser.add_argument("--abstract", required=True, help="The abstract or text to analyze.")
    run_parser.add_argument("--keywords", help="Optional keywords to include in the analysis.")
    run_parser.add_argument("--model", required=True, help="The model to use for the analysis.")
    run_parser.add_argument("--provider", default="ollama", help="The LLM provider to use (e.g., 'ollama', 'gemini').")
    run_parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL.")
    run_parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port.")
    run_parser.add_argument("--use-chunking-abstract", action="store_true", help="Enable chunking for the abstract.")
    run_parser.add_argument("--abstract-chunk-size", type=int, default=100, help="Chunk size for the abstract.")
    run_parser.add_argument("--use-chunking-keywords", action="store_true", help="Enable chunking for keywords.")
    run_parser.add_argument("--keyword-chunk-size", type=int, default=500, help="Chunk size for keywords.")
    run_parser.add_argument("--output-json", help="Path to save the TaskState JSON output.")

    # Save-state command (placeholder for now, will be implemented after run)
    save_parser = subparsers.add_parser("save-state", help="Save the last analysis state to a JSON file.")
    save_parser.add_argument("output_file", help="Path to save the TaskState JSON output.")

    # Load-state command (placeholder for now)
    load_parser = subparsers.add_parser("load-state", help="Load and resume an analysis from a JSON file.")
    load_parser.add_argument("input_file", help="Path to the TaskState JSON input file.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check if prompts file exists
    if not os.path.exists(PROMPTS_FILE):
        logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
        return

    if args.command == "run":
        # Setup services
        llm_service = LlmService(ollama_url=args.ollama_host, ollama_port=args.ollama_port)
        prompt_service = PromptService(PROMPTS_FILE, logger)
        alima_manager = AlimaManager(llm_service, prompt_service, logger)

        abstract_data = AbstractData(abstract=args.abstract, keywords=args.keywords)
        try:
            task_state = alima_manager.analyze_abstract(
                abstract_data,
                args.task,
                args.model,
                args.use_chunking_abstract,
                args.abstract_chunk_size,
                args.use_chunking_keywords,
                args.keyword_chunk_size,
                provider=args.provider
            )
            print("--- Analysis Result ---")
            print(task_state.analysis_result.full_text)
            print("--- Matched Keywords ---")
            print(task_state.analysis_result.matched_keywords)
            print("--- GND Systematic ---")
            print(task_state.analysis_result.gnd_systematic)

            if args.output_json:
                try:
                    with open(args.output_json, 'w', encoding='utf-8') as f:
                        json.dump(_task_state_to_dict(task_state), f, ensure_ascii=False, indent=4)
                    logger.info(f"Task state saved to {args.output_json}")
                except Exception as e:
                    logger.error(f"Error saving task state to JSON: {e}")

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
    elif args.command == "save-state":
        logger.error("The 'save-state' command is not yet fully implemented as a standalone command. Use 'run --output-json' instead.")
    elif args.command == "load-state":
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                task_state_dict = json.load(f)
            
            # Reconstruct dataclass objects
            abstract_data = AbstractData(**task_state_dict['abstract_data'])
            analysis_result = AnalysisResult(**task_state_dict['analysis_result'])
            prompt_config = PromptConfigData(**task_state_dict['prompt_config']) if task_state_dict['prompt_config'] else None

            task_state = TaskState(
                abstract_data=abstract_data,
                analysis_result=analysis_result,
                prompt_config=prompt_config,
                status=task_state_dict['status'],
                task_name=task_state_dict['task_name'],
                model_used=task_state_dict['model_used'],
                provider_used=task_state_dict['provider_used'],
                use_chunking_abstract=task_state_dict['use_chunking_abstract'],
                abstract_chunk_size=task_state_dict['abstract_chunk_size'],
                use_chunking_keywords=task_state_dict['use_chunking_keywords'],
                keyword_chunk_size=task_state_dict['keyword_chunk_size'],
            )

            print("--- Loaded Analysis Result ---")
            print(task_state.analysis_result.full_text)
            print("--- Matched Keywords ---")
            print(task_state.analysis_result.matched_keywords)
            print("--- GND Systematic ---")
            print(task_state.analysis_result.gnd_systematic)
            logger.info(f"Task state loaded from {args.input_file}")

        except FileNotFoundError:
            logger.error(f"Error: Input file not found at {args.input_file}")
        except json.JSONDecodeError:
            logger.error(f"Error: Invalid JSON in {args.input_file}")
        except KeyError as e:
            logger.error(f"Error: Missing key in JSON data: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading task state: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()