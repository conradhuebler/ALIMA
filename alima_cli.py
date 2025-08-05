import argparse
import logging
import os
import json
from dataclasses import asdict

from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.data_models import (
    AbstractData,
    TaskState,
    AnalysisResult,
    PromptConfigData,
    KeywordAnalysisState,
    SearchResult,
    LlmKeywordAnalysis,
)
from src.core.search_cli import SearchCLI
from src.core.cache_manager import CacheManager
from src.core.suggesters.meta_suggester import SuggesterType
from src.core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
)
from src.utils.pipeline_utils import (
    execute_complete_pipeline,
    PipelineJsonManager,
    PipelineStepExecutor,
)
from src.utils.doi_resolver import resolve_input_to_text
from typing import List, Tuple

PROMPTS_FILE = "prompts.json"


# Use shared JSON utilities from pipeline_utils
_task_state_to_dict = PipelineJsonManager.task_state_to_dict
_convert_sets_to_lists = PipelineJsonManager.convert_sets_to_lists


def main():
    """Main function for the ALIMA CLI.

    This function parses command-line arguments and executes the appropriate command.

    Commands:
        pipeline: Execute complete ALIMA analysis pipeline.
        run: Run an analysis task.
        save-state: Save the last analysis state to a JSON file.
        load-state: Load and resume an analysis from a JSON file.
        list-models: List all available models from all providers.
        search: Search for keywords using various suggesters.
    """
    parser = argparse.ArgumentParser(
        description="ALIMA CLI - AI-powered abstract analysis."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command (new unified pipeline)
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Execute complete ALIMA analysis pipeline."
    )
    # Input options - either text or DOI (mutually exclusive)
    input_group = pipeline_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Input text for analysis.")
    input_group.add_argument(
        "--doi",
        help="DOI or URL to resolve and analyze (e.g., 10.1007/978-3-031-47390-6, https://link.springer.com/book/...).",
    )
    pipeline_parser.add_argument(
        "--initial-model", required=True, help="Model for initial keyword extraction."
    )
    pipeline_parser.add_argument(
        "--final-model", required=True, help="Model for final keyword analysis."
    )
    pipeline_parser.add_argument(
        "--provider", default="ollama", help="LLM provider to use."
    )
    pipeline_parser.add_argument(
        "--suggesters",
        nargs="+",
        default=["lobid", "swb"],
        help="Search suggesters to use.",
    )
    pipeline_parser.add_argument(
        "--output-json", help="Path to save the complete pipeline results."
    )
    pipeline_parser.add_argument(
        "--resume-from", help="Path to resume pipeline from previous state."
    )
    pipeline_parser.add_argument(
        "--ollama-host", default="http://localhost", help="Ollama host URL."
    )
    pipeline_parser.add_argument(
        "--ollama-port", type=int, default=11434, help="Ollama port."
    )
    pipeline_parser.add_argument(
        "--initial-task",
        default="initialisation",
        help="Task for initial keyword extraction (initialisation, keywords, rephrase).",
    )
    pipeline_parser.add_argument(
        "--final-task",
        default="keywords",
        help="Task for final keyword analysis (keywords, rephrase, keywords_chunked).",
    )
    pipeline_parser.add_argument(
        "--keyword-chunking-threshold",
        type=int,
        default=500,
        help="Threshold for keyword chunking (default: 500).",
    )
    pipeline_parser.add_argument(
        "--chunking-task",
        default="keywords_chunked",
        help="Task to use for chunked processing (keywords_chunked, rephrase).",
    )
    # Catalog configuration arguments - Claude Generated
    pipeline_parser.add_argument(
        "--catalog-token",
        help="Catalog authentication token for library API access.",
    )
    pipeline_parser.add_argument(
        "--catalog-search-url",
        help="SOAP endpoint URL for catalog search operations.",
    )
    pipeline_parser.add_argument(
        "--catalog-details-url",
        help="SOAP endpoint URL for catalog detail retrieval.",
    )
    pipeline_parser.add_argument(
        "--include-dk-classification",
        action="store_true",
        default=True,
        help="Include DK classification step in pipeline (default: True).",
    )
    pipeline_parser.add_argument(
        "--disable-dk-classification",
        action="store_true",
        help="Disable DK classification step in pipeline.",
    )
    pipeline_parser.add_argument(
        "--auto-save-path",
        help="Path for automatic intermediate state saves (default: temp file).",
    )
    pipeline_parser.add_argument(
        "--dk-max-results",
        type=int,
        default=20,
        help="Maximum results for DK classification search (default: 20).",
    )

    # Run command (individual task)
    run_parser = subparsers.add_parser("run", help="Run an analysis task.")
    run_parser.add_argument(
        "task",
        help="The analysis task to perform (e.g., 'initialisation', 'keywords').",
    )
    run_parser.add_argument(
        "--abstract", required=True, help="The abstract or text to analyze."
    )
    run_parser.add_argument(
        "--keywords", help="Optional keywords to include in the analysis."
    )
    run_parser.add_argument(
        "--model", required=True, help="The model to use for the analysis."
    )
    run_parser.add_argument(
        "--provider",
        default="ollama",
        help="The LLM provider to use (e.g., 'ollama', 'gemini').",
    )
    run_parser.add_argument(
        "--ollama-host", default="http://localhost", help="Ollama host URL."
    )
    run_parser.add_argument(
        "--ollama-port", type=int, default=11434, help="Ollama port."
    )
    run_parser.add_argument(
        "--use-chunking-abstract",
        action="store_true",
        help="Enable chunking for the abstract.",
    )
    run_parser.add_argument(
        "--abstract-chunk-size",
        type=int,
        default=100,
        help="Chunk size for the abstract.",
    )
    run_parser.add_argument(
        "--use-chunking-keywords",
        action="store_true",
        help="Enable chunking for keywords.",
    )
    run_parser.add_argument(
        "--keyword-chunk-size", type=int, default=500, help="Chunk size for keywords."
    )
    run_parser.add_argument(
        "--output-json", help="Path to save the TaskState JSON output."
    )
    run_parser.add_argument(
        "--prompt-template", help="The prompt template to use for the analysis."
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search for keywords using various suggesters."
    )
    search_parser.add_argument(
        "search_terms", nargs="+", help="The search terms to use."
    )
    search_parser.add_argument(
        "--suggesters",
        nargs="+",
        default=["lobid"],
        help="The suggesters to use (e.g., 'lobid', 'swb', 'catalog').",
    )

    # Save-state command
    save_parser = subparsers.add_parser(
        "save-state", help="Save the last analysis state to a JSON file."
    )
    save_parser.add_argument(
        "output_file", help="Path to save the TaskState JSON output."
    )

    # Load-state command
    load_parser = subparsers.add_parser(
        "load-state", help="Load and resume an analysis from a JSON file."
    )
    load_parser.add_argument(
        "input_file", help="Path to the TaskState JSON input file."
    )

    # List models command
    list_parser = subparsers.add_parser(
        "list-models", help="List all available models from all providers."
    )
    list_parser.add_argument(
        "--ollama-host", default="http://localhost", help="Ollama host URL."
    )
    list_parser.add_argument(
        "--ollama-port", type=int, default=11434, help="Ollama port."
    )

    # Test catalog command - Claude Generated
    test_catalog_parser = subparsers.add_parser(
        "test-catalog", help="Test catalog search functionality standalone."
    )
    test_catalog_parser.add_argument(
        "search_terms", 
        nargs="+", 
        help="Search terms to test catalog search with."
    )
    test_catalog_parser.add_argument(
        "--max-results", 
        type=int, 
        default=5, 
        help="Maximum number of results to process per term (default: 5)."
    )
    test_catalog_parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging for detailed output."
    )
    test_catalog_parser.add_argument(
        "--catalog-token", 
        help="Override catalog token (default: from config)."
    )
    test_catalog_parser.add_argument(
        "--catalog-search-url", 
        help="Override catalog search URL (default: from config)."
    )
    test_catalog_parser.add_argument(
        "--catalog-details-url", 
        help="Override catalog details URL (default: from config)."
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if prompts file exists
    if not os.path.exists(PROMPTS_FILE) and args.command not in ["list-models"]:
        logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
        return

    if args.command == "pipeline":
        # Setup services
        llm_service = LlmService(
            providers=[args.provider],
            ollama_url=args.ollama_host,
            ollama_port=args.ollama_port,
        )
        prompt_service = PromptService(PROMPTS_FILE, logger)
        alima_manager = AlimaManager(llm_service, prompt_service, logger)
        cache_manager = CacheManager()

        def stream_callback(token, step_id):
            print(token, end="", flush=True)

        try:
            if args.resume_from:
                # Resume from existing state
                logger.info(f"Resuming pipeline from {args.resume_from}")
                analysis_state = PipelineJsonManager.load_analysis_state(
                    args.resume_from
                )

                # Continue pipeline from where it was left off
                # For now, we'll just show the loaded state
                print("--- Resumed Analysis State ---")
                print(f"Original Abstract: {analysis_state.original_abstract[:200]}...")
                print(f"Initial Keywords: {analysis_state.initial_keywords}")
                if analysis_state.final_llm_analysis:
                    print(
                        f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}"
                    )
                else:
                    print("Final analysis not yet completed")
            else:
                # Resolve input text (either from --input-text or --doi)
                if args.doi:
                    logger.info(f"Resolving input: {args.doi}")
                    success, input_text, error_msg = resolve_input_to_text(
                        args.doi, logger
                    )
                    if not success:
                        logger.error(f"Failed to resolve input: {error_msg}")
                        return
                    logger.info(
                        f"Input resolved successfully, content length: {len(input_text)}"
                    )
                    print(
                        f"Resolved '{args.doi}' to text content ({len(input_text)} chars)"
                    )
                else:
                    input_text = args.input_text

                # Execute complete pipeline
                logger.info("Starting complete pipeline execution")
                # Handle DK classification flag
                include_dk = args.include_dk_classification and not args.disable_dk_classification
                
                analysis_state = execute_complete_pipeline(
                    alima_manager=alima_manager,
                    cache_manager=cache_manager,
                    input_text=input_text,
                    initial_model=args.initial_model,
                    final_model=args.final_model,
                    provider=args.provider,
                    suggesters=args.suggesters,
                    stream_callback=stream_callback,
                    logger=logger,
                    initial_task=args.initial_task,
                    final_task=args.final_task,
                    keyword_chunking_threshold=args.keyword_chunking_threshold,
                    chunking_task=args.chunking_task,
                    include_dk_classification=include_dk,
                    # Catalog configuration - Claude Generated
                    catalog_token=args.catalog_token,
                    catalog_search_url=args.catalog_search_url,
                    catalog_details_url=args.catalog_details_url,
                    # Recovery configuration - Claude Generated
                    auto_save_path=args.auto_save_path,
                    resume_from_path=args.resume_from,
                    dk_max_results=args.dk_max_results,
                )

                print("\n--- Pipeline Results ---")
                print(f"Initial Keywords: {analysis_state.initial_keywords}")
                print(
                    f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}"
                )
                print(
                    f"GND Classes: {analysis_state.final_llm_analysis.extracted_gnd_classes}"
                )

            # Save results if requested
            if args.output_json:
                try:
                    PipelineJsonManager.save_analysis_state(
                        analysis_state, args.output_json
                    )
                    logger.info(f"Pipeline results saved to {args.output_json}")
                except Exception as e:
                    logger.error(f"Error saving pipeline results: {e}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")

    elif args.command == "run":
        # Setup services
        llm_service = LlmService(
            providers=[args.provider],
            ollama_url=args.ollama_host,
            ollama_port=args.ollama_port,
        )
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
                prompt_template=args.prompt_template,
            )
            print(
                json.dumps(
                    _task_state_to_dict(task_state), ensure_ascii=False, indent=4
                )
            )

            if args.output_json:
                try:
                    with open(args.output_json, "w", encoding="utf-8") as f:
                        json.dump(
                            _task_state_to_dict(task_state),
                            f,
                            ensure_ascii=False,
                            indent=4,
                        )
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

    elif args.command == "list-models":
        # Setup services
        llm_service = LlmService(
            ollama_url=args.ollama_host, ollama_port=args.ollama_port
        )
        providers = llm_service.get_available_providers()
        for provider in providers:
            print(f"--- {provider} ---")
            models = llm_service.get_available_models(provider)
            if models:
                for model in models:
                    print(model)
            else:
                print("No models found.")

    elif args.command == "test-catalog":
        # Claude Generated - Test catalog search functionality
        from src.core.biblioextractor import BiblioExtractor
        
        # Setup logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("biblio_extractor").setLevel(logging.DEBUG)
        
        print(f"🔍 Testing catalog search for terms: {', '.join(args.search_terms)}")
        print(f"📊 Max results per term: {args.max_results}")
        print("-" * 60)
        
        try:
            # Get catalog configuration from arguments (all required)
            catalog_token = args.catalog_token or ""
            catalog_search_url = args.catalog_search_url or ""
            catalog_details_url = args.catalog_details_url or ""
            
            if not catalog_token:
                logger.error("❌ No catalog token provided. Use --catalog-token TOKEN")
                return
                
            if not catalog_search_url:
                logger.error("❌ No catalog search URL provided. Use --catalog-search-url URL")
                return
                
            if not catalog_details_url:
                logger.error("❌ No catalog details URL provided. Use --catalog-details-url URL")
                return
                
            print(f"🔑 Using catalog token: {catalog_token[:10]}..." if len(catalog_token) > 10 else catalog_token)
            if catalog_search_url:
                print(f"🌐 Search URL: {catalog_search_url}")
            if catalog_details_url:
                print(f"🌐 Details URL: {catalog_details_url}")
            print()
            
            # Initialize BiblioExtractor
            extractor = BiblioExtractor(
                token=catalog_token,
                debug=args.debug
            )
            
            if catalog_search_url:
                extractor.SEARCH_URL = catalog_search_url
            if catalog_details_url:
                extractor.DETAILS_URL = catalog_details_url
            
            # Test search_subjects method
            print("🚀 Starting catalog subject search...")
            results = extractor.search_subjects(
                search_terms=args.search_terms,
                max_results=args.max_results
            )
            
            print("=" * 60)
            print("📋 SEARCH RESULTS SUMMARY")
            print("=" * 60)
            
            total_subjects = 0
            for search_term, term_results in results.items():
                subject_count = len(term_results)
                total_subjects += subject_count
                
                print(f"\n🔸 Search term: '{search_term}'")
                print(f"   Found subjects: {subject_count}")
                
                if subject_count > 0:
                    print("   📝 Subjects found:")
                    for i, (subject, data) in enumerate(term_results.items(), 1):
                        print(f"      {i}. {subject}")
                        print(f"         Count: {data.get('count', 0)}")
                        dk_count = len(data.get('dk', set()))
                        if dk_count > 0:
                            print(f"         DK classifications: {dk_count}")
                else:
                    print("   ❌ No subjects found")
            
            print(f"\n🎯 TOTAL: {total_subjects} subjects found across {len(args.search_terms)} search terms")
            
            if total_subjects == 0:
                print("\n⚠️  TROUBLESHOOTING:")
                print("   1. Check if catalog token is valid")
                print("   2. Verify catalog URLs are correct") 
                print("   3. Try different search terms")
                print("   4. Run with --debug flag for detailed logs")
            
        except Exception as e:
            logger.error(f"❌ Catalog test failed: {str(e)}")
            if args.debug:
                raise

    elif args.command == "save-state":
        logger.error(
            "The 'save-state' command is not yet fully implemented as a standalone command. Use 'pipeline --output-json' instead."
        )

    elif args.command == "load-state":
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
