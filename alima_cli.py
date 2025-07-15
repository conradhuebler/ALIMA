import argparse
import logging
import os
import json
from dataclasses import asdict

from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.data_models import (AbstractData, TaskState, AnalysisResult, 
                                PromptConfigData, KeywordAnalysisState, SearchResult, LlmKeywordAnalysis)
from src.core.search_cli import SearchCLI
from src.core.cache_manager import CacheManager
from src.core.suggesters.meta_suggester import SuggesterType
from src.core.processing_utils import extract_keywords_from_response, extract_gnd_system_from_response
from typing import List, Tuple

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

def _convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _convert_sets_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_sets_to_lists(elem) for elem in obj]
    return obj

def main():
    """Main function for the ALIMA CLI.

    This function parses command-line arguments and executes the appropriate command.

    Commands:
        run: Run an analysis task.
        save-state: Save the last analysis state to a JSON file.
        load-state: Load and resume an analysis from a JSON file.
        list-models: List all available models from all providers.
    """
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
    run_parser.add_argument("--prompt-template", help="The prompt template to use for the analysis.")

    # Save-state command (placeholder for now, will be implemented after run)
    save_parser = subparsers.add_parser("save-state", help="Save the last analysis state to a JSON file.")
    save_parser.add_argument("output_file", help="Path to save the TaskState JSON output.")

    # Load-state command (placeholder for now)
    load_parser = subparsers.add_parser("load-state", help="Load and resume an analysis from a JSON file.")
    load_parser.add_argument("input_file", help="Path to the TaskState JSON input file.")

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List all available models from all providers.")
    list_parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL.")
    list_parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port.")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for keywords using various suggesters.")
    search_parser.add_argument("search_terms", nargs='+', help="The search terms to use.")
    search_parser.add_argument("--suggesters", nargs='+', default=["lobid"], help="The suggesters to use (e.g., 'lobid', 'swb', 'catalog').")

    # Analyze keywords command
    analyze_parser = subparsers.add_parser("analyze-keywords", help="Analyze keywords by searching and using an LLM.")
    analyze_parser.add_argument("keywords", nargs='*', help="The keywords to analyze.")
    analyze_parser.add_argument("--abstract", help="An abstract to generate keywords from if no keywords are provided.")
    analyze_parser.add_argument("--suggesters", nargs='+', default=["lobid"], help="The suggesters to use for the search.")
    analyze_parser.add_argument("--model", required=True, help="The model to use for the analysis.")
    analyze_parser.add_argument("--provider", default="ollama", help="The LLM provider to use.")
    analyze_parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL.")
    analyze_parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port.")
    analyze_parser.add_argument("--output-json", help="Path to save the KeywordAnalysisState JSON output.")
    analyze_parser.add_argument("--input-json", help="Path to a KeywordAnalysisState JSON file to resume analysis.")
    analyze_parser.add_argument("--final-llm-task", default="keywords", choices=["keywords", "rephrase"], help="The final LLM task to perform (keywords or rephrase).")
    analyze_parser.add_argument("--final-llm-prompt-template", help="The prompt template to use for the final LLM analysis.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def stream_callback(text_chunk):
        print(text_chunk, end="", flush=True)

    def _extract_keywords_from_descriptive_text(text: str, gnd_compliant_keywords: List[str]) -> Tuple[List[str], List[str]]:
        import re
        pattern = re.compile(r'\b([A-Za-zäöüÄÖÜß\s-]+?)\s*\((\d{7}-\d|\d{7}-\d{1,2})\)')
        matches = pattern.findall(text)
        
        all_extracted_keywords = []
        exact_matches = []

        # Convert gnd_compliant_keywords to a set for faster lookup
        gnd_compliant_set = set(gnd_compliant_keywords)

        for keyword_part, gnd_id_part in matches:
            formatted_keyword = f"{keyword_part.strip()} ({gnd_id_part})"
            all_extracted_keywords.append(formatted_keyword)

            # Check if the formatted keyword from LLM output is in the gnd_compliant_keywords list
            if formatted_keyword in gnd_compliant_set:
                exact_matches.append(formatted_keyword)

        return all_extracted_keywords, exact_matches

    def _extract_keywords_from_descriptive_text_simple(text: str, gnd_compliant_keywords: List[str]) -> List[str]:
        """
        Simplified version using basic string containment (faster but less precise).
        """
        if not text or not gnd_compliant_keywords:
            return []
        
        matched_keywords = []
        text_lower = text.lower()
        
        for gnd_keyword in gnd_compliant_keywords:
            if '(' in gnd_keyword and ')' in gnd_keyword:
                # Extract clean keyword
                clean_keyword = gnd_keyword.split('(')[0].strip().lower()
                
                # Simple containment check
                if clean_keyword in text_lower:
                    matched_keywords.append(gnd_keyword)
                    self.logger.info(f"Simple match found: '{clean_keyword}' -> {gnd_keyword}")
        
        return matched_keywords
    
    def _extract_classes_from_descriptive_text(text: str) -> List[str]:
        import re
        match = re.search(r'<class>(.*?)</class>', text)
        if match:
            classes_str = match.group(1)
            return [cls.strip() for cls in classes_str.split('|') if cls.strip()]
        return []

    # Check if prompts file exists
    if not os.path.exists(PROMPTS_FILE) and args.command not in ["list-models"]:
        logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
        return

    if args.command == "run":
        # Setup services
        llm_service = LlmService(providers=[args.provider], ollama_url=args.ollama_host, ollama_port=args.ollama_port)
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
                provider=args.provider,
                stream_callback=stream_callback,
                prompt_template=args.prompt_template
            )
            print(json.dumps(_task_state_to_dict(task_state), ensure_ascii=False, indent=4))

            if args.output_json:
                try:
                    with open(args.output_json, 'w', encoding='utf-8') as f:
                        json.dump(_task_state_to_dict(task_state), f, ensure_ascii=False, indent=4)
                    logger.info(f"Task state saved to {args.output_json}")
                except Exception as e:
                    logger.error(f"Error saving task state to JSON: {e}")

        except Exception as e:
            logger.error(f"An error occurred during analysis: {e}")
    elif args.command == "search":
        cache_manager = CacheManager()
        search_cli = SearchCLI(cache_manager)

        suggester_types = []
        for suggester in args.suggesters:
            try:
                suggester_types.append(SuggesterType[suggester.upper()])
            except KeyError:
                logger.warning(f"Unknown suggester: {suggester}")

        if not suggester_types:
            logger.error("No valid suggesters specified.")
            return

        results = search_cli.search(args.search_terms, suggester_types)

        for search_term, term_results in results.items():
            print(f"--- Results for: {search_term} ---")
            if cache_manager.gnd_keyword_exists(search_term):
                print("  (Results found in cache)")
            else:
                print("  (Results not found in cache)")

            for keyword, data in term_results.items():
                print(f"  - {keyword}:")
                print(f"    GND IDs: {data.get('gndid')}")
                print(f"    Count: {data.get('count')}")

    elif args.command == "analyze-keywords":
        llm_service = LlmService(providers=[args.provider], ollama_url=args.ollama_host, ollama_port=args.ollama_port)
        prompt_service = PromptService(PROMPTS_FILE, logger)
        alima_manager = AlimaManager(llm_service, prompt_service, logger)

        initial_keywords = []
        original_abstract_for_llm = None # Initialize outside the if/else
        initial_gnd_classes = [] # Initialize outside the if/else
        analysis_state = None # Initialize analysis_state here

        if args.input_json:
            with open(args.input_json, 'r') as f:
                analysis_state = KeywordAnalysisState(**json.load(f))
            initial_keywords = analysis_state.initial_keywords
            original_abstract_for_llm = analysis_state.original_abstract
            initial_gnd_classes = analysis_state.initial_gnd_classes
        else:
            if not args.keywords and not args.abstract:
                logger.error("Either keywords or an abstract must be provided for analysis.")
                return

            if args.abstract and not args.keywords:
                logger.info("Generating initial keywords from abstract using LLM...")
                abstract_data = AbstractData(abstract=args.abstract)
                original_abstract_for_llm = args.abstract # Store the original abstract

                # Get prompt config for initial keyword extraction
                initial_prompt_config = prompt_service.get_prompt_config("abstract", args.model)
                if not initial_prompt_config:
                    logger.error(f"Prompt configuration for 'abstract' not found for model {args.model}")
                    return

                # Use the 'abstract' task to extract keywords from the abstract
                task_state = alima_manager.analyze_abstract(
                    abstract_data,
                    "abstract", # Correct task name
                    args.model,
                    provider=args.provider,
                    stream_callback=stream_callback
                )
                
                # Extract keywords and GND classes from the LLM's response
                extracted_keywords_str = extract_keywords_from_response(task_state.analysis_result.full_text)
                initial_keywords = [kw.strip() for kw in extracted_keywords_str.split(',') if kw.strip()]
                
                extracted_gnd_classes_str = extract_gnd_system_from_response(task_state.analysis_result.full_text)
                initial_gnd_classes = [cls.strip() for cls in extracted_gnd_classes_str.split('|') if cls.strip()] if extracted_gnd_classes_str else []

                logger.info(f"Generated keywords: {initial_keywords}")
                logger.info(f"Generated GND classes: {initial_gnd_classes}")

                # Populate initial_llm_call_details
                analysis_state = KeywordAnalysisState(original_abstract=original_abstract_for_llm, initial_keywords=initial_keywords, search_suggesters_used=args.suggesters, initial_gnd_classes=initial_gnd_classes)
                analysis_state.initial_llm_call_details = LlmKeywordAnalysis(
                    task_name="abstract",
                    model_used=args.model,
                    provider_used=args.provider,
                    prompt_template=initial_prompt_config.prompt,
                    filled_prompt=initial_prompt_config.prompt.format(abstract=args.abstract, keywords=""), # Pass empty keywords for initial abstract task
                    temperature=initial_prompt_config.temp,
                    seed=initial_prompt_config.seed,
                    response_full_text=task_state.analysis_result.full_text,
                    extracted_gnd_keywords=initial_keywords,
                    extracted_gnd_classes=initial_gnd_classes
                )

                if not initial_keywords:
                    logger.error("Failed to generate keywords from abstract.")
                    return
            elif args.keywords:
                initial_keywords = args.keywords
                original_abstract_for_llm = None # No original abstract if keywords are provided directly
                initial_gnd_classes = [] # No initial GND classes if keywords are provided directly
                analysis_state = KeywordAnalysisState(original_abstract=original_abstract_for_llm, initial_keywords=initial_keywords, search_suggesters_used=args.suggesters, initial_gnd_classes=initial_gnd_classes)
            else:
                logger.error("Either keywords or an abstract must be provided for analysis.")
                return
            
            cache_manager = CacheManager()
            search_cli = SearchCLI(cache_manager)

            suggester_types = []
            for suggester in args.suggesters:
                try:
                    suggester_types.append(SuggesterType[suggester.upper()])
                except KeyError:
                    logger.warning(f"Unknown suggester: {suggester}")

            if suggester_types:
                search_results_dict = search_cli.search(initial_keywords, suggester_types)
                analysis_state.search_results = [SearchResult(search_term=k, results=v) for k, v in search_results_dict.items()]

        # Format search results for LLM and collect all GND-compliant keywords
        search_results_text = ""
        gnd_compliant_keywords_for_llm = [] # New list to collect keywords with GND IDs
        for search_result in analysis_state.search_results:
            search_results_text += f"Search Term: {search_result.search_term}\n"
            for keyword, data in search_result.results.items():
                gnd_ids = ', '.join(data.get('gndid')) if data.get('gndid') else ''
                formatted_keyword = f"{keyword} ({gnd_ids})" if gnd_ids else keyword
                search_results_text += f"  - {formatted_keyword}\n"
                gnd_compliant_keywords_for_llm.append(formatted_keyword) # Collect keyword with GND ID

        # Determine the abstract content for the final LLM call
        final_llm_abstract_content = original_abstract_for_llm if original_abstract_for_llm else search_results_text

        # LLM analysis (using the selected final LLM task)
        # Pass the collected GND-compliant keywords with IDs to the AbstractData
        abstract_data = AbstractData(abstract=final_llm_abstract_content, keywords=", ".join(gnd_compliant_keywords_for_llm))
        task_state = alima_manager.analyze_abstract(
            abstract_data,
            args.final_llm_task, # Use the selected final LLM task
            args.model,
            provider=args.provider,
            stream_callback=stream_callback,
            prompt_template=args.final_llm_prompt_template
        )

        # Extract keywords and classes from the full_text response
        extracted_keywords_all, extracted_keywords_exact = _extract_keywords_from_descriptive_text(task_state.analysis_result.full_text, gnd_compliant_keywords_for_llm)
        extracted_gnd_classes = _extract_classes_from_descriptive_text(task_state.analysis_result.full_text)

        final_prompt_config = prompt_service.get_prompt_config(args.final_llm_task, args.model)
        if not final_prompt_config:
            logger.error(f"Prompt configuration for '{args.final_llm_task}' not found for model {args.model}")
            return

        analysis_state.final_llm_analysis = LlmKeywordAnalysis(
            task_name=args.final_llm_task,
            model_used=args.model,
            provider_used=args.provider,
            prompt_template=final_prompt_config.prompt,
            filled_prompt=final_prompt_config.prompt.format(abstract=final_llm_abstract_content, keywords=", ".join(gnd_compliant_keywords_for_llm)),
            temperature=final_prompt_config.temp,
            seed=final_prompt_config.seed,
            response_full_text=task_state.analysis_result.full_text,
            extracted_gnd_keywords=extracted_keywords_exact, # Use exact matches for final output
            extracted_gnd_classes=extracted_gnd_classes
        )

        print("--- Extracted GND Keywords (Exact Matches) ---")
        print(analysis_state.final_llm_analysis.extracted_gnd_keywords)
        print("--- Extracted GND Classes ---")
        print(analysis_state.final_llm_analysis.extracted_gnd_classes)

        if args.output_json:
            try:
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(_convert_sets_to_lists(asdict(analysis_state)), f, ensure_ascii=False, indent=4)
                logger.info(f"Task state saved to {args.output_json}")
            except Exception as e:
                logger.error(f"Error saving task state to JSON: {e}")

    elif args.command == "list-models":
        # Setup services
        llm_service = LlmService(ollama_url=args.ollama_host, ollama_port=args.ollama_port)
        providers = llm_service.get_available_providers()
        for provider in providers:
            print(f"--- {provider} ---")
            models = llm_service.get_available_models(provider)
            if models:
                for model in models:
                    print(model)
            else:
                print("No models found.")
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
