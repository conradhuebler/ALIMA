import argparse
import logging
import os
import json
import time
import tempfile
import gzip
import requests
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
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.suggesters.meta_suggester import SuggesterType
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
from src.utils.config_manager import ConfigManager, OpenAICompatibleProvider
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
        list-providers: List all configured LLM providers with details.
        test-providers: Test connection to all configured LLM providers.
        list-models-detailed: List models from all providers with detailed information.
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

    # Cache management commands - Claude Generated  
    cache_clear_parser = subparsers.add_parser(
        "clear-cache", help="Clear cache database(s)."
    )
    cache_clear_parser.add_argument(
        "--type",
        choices=["all", "gnd", "search", "classifications"],
        default="all",
        help="Type of cache to clear (default: all)."
    )
    cache_clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt and clear immediately."
    )

    # DNB import command - Claude Generated
    dnb_import_parser = subparsers.add_parser(
        "dnb-import", help="Import DNB/GND data with progress information."
    )
    dnb_import_parser.add_argument(
        "--force",
        action="store_true", 
        help="Force re-download even if data exists."
    )
    dnb_import_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output during import."
    )

    # LLM provider configuration commands - Claude Generated
    list_providers_parser = subparsers.add_parser(
        "list-providers", help="List all configured LLM providers with details."
    )
    list_providers_parser.add_argument(
        "--show-config", 
        action="store_true",
        help="Show detailed configuration for each provider."
    )
    list_providers_parser.add_argument(
        "--show-models", 
        action="store_true",
        help="Show available models for each reachable provider."
    )

    # Test LLM providers command - Claude Generated  
    test_providers_parser = subparsers.add_parser(
        "test-providers", help="Test connection to all configured LLM providers."
    )
    test_providers_parser.add_argument(
        "--timeout", 
        type=float,
        default=5.0,
        help="Connection timeout in seconds (default: 5.0)."
    )
    test_providers_parser.add_argument(
        "--show-models", 
        action="store_true",
        help="Show available models for reachable providers."
    )

    # Enhanced list models command - Claude Generated
    list_models_detailed_parser = subparsers.add_parser(
        "list-models-detailed", help="List models from all providers with detailed information."
    )

    # Database configuration commands - Claude Generated
    db_config_parser = subparsers.add_parser(
        "db-config", help="Database configuration management."
    )
    db_subparsers = db_config_parser.add_subparsers(dest="db_action", help="Database config actions")
    
    # Show current database config
    db_show_parser = db_subparsers.add_parser("show", help="Show current database configuration")
    
    # Show config paths
    db_paths_parser = db_subparsers.add_parser("paths", help="Show OS-specific configuration paths")
    
    # Test database connection
    db_test_parser = db_subparsers.add_parser("test", help="Test database connection")
    
    # Configure SQLite database
    db_sqlite_parser = db_subparsers.add_parser("set-sqlite", help="Configure SQLite database")
    db_sqlite_parser.add_argument("--path", default="alima_knowledge.db", help="SQLite database path")
    db_sqlite_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", 
                                 help="Configuration scope")
    
    # Configure MySQL/MariaDB database
    db_mysql_parser = db_subparsers.add_parser("set-mysql", help="Configure MySQL/MariaDB database")
    db_mysql_parser.add_argument("--host", required=True, help="Database host")
    db_mysql_parser.add_argument("--port", type=int, default=3306, help="Database port")
    db_mysql_parser.add_argument("--database", required=True, help="Database name")
    db_mysql_parser.add_argument("--username", required=True, help="Database username")
    db_mysql_parser.add_argument("--password", help="Database password (will prompt if not provided)")
    db_mysql_parser.add_argument("--charset", default="utf8mb4", help="Character set")
    db_mysql_parser.add_argument("--ssl-disabled", action="store_true", help="Disable SSL connection")
    db_mysql_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                 help="Configuration scope")
    
    # Provider management commands - Claude Generated  
    provider_parser = subparsers.add_parser(
        "provider", help="OpenAI-compatible provider management."
    )
    provider_subparsers = provider_parser.add_subparsers(dest="provider_action", help="Provider management actions")
    
    # List providers
    provider_list_parser = provider_subparsers.add_parser("list", help="List all configured providers")
    provider_list_parser.add_argument("--enabled-only", action="store_true", 
                                     help="Show only enabled providers")
    
    # Add provider
    provider_add_parser = provider_subparsers.add_parser("add", help="Add new OpenAI-compatible provider")
    provider_add_parser.add_argument("--name", required=True, help="Provider name")
    provider_add_parser.add_argument("--base-url", required=True, help="API base URL")
    provider_add_parser.add_argument("--api-key", required=True, help="API key")
    provider_add_parser.add_argument("--description", default="", help="Provider description")
    provider_add_parser.add_argument("--enabled", action="store_true", default=True, 
                                   help="Enable provider (default: True)")
    provider_add_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                   help="Configuration scope")
    
    # Remove provider
    provider_remove_parser = provider_subparsers.add_parser("remove", help="Remove provider")
    provider_remove_parser.add_argument("--name", required=True, help="Provider name to remove")
    provider_remove_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                       help="Configuration scope")
    
    # Edit provider
    provider_edit_parser = provider_subparsers.add_parser("edit", help="Edit existing provider")
    provider_edit_parser.add_argument("--name", required=True, help="Provider name to edit")
    provider_edit_parser.add_argument("--new-name", help="New provider name")
    provider_edit_parser.add_argument("--base-url", help="New API base URL")
    provider_edit_parser.add_argument("--api-key", help="New API key")
    provider_edit_parser.add_argument("--description", help="New provider description")
    provider_edit_parser.add_argument("--enable", action="store_true", help="Enable provider")
    provider_edit_parser.add_argument("--disable", action="store_true", help="Disable provider")
    provider_edit_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                     help="Configuration scope")
    
    # Test provider
    provider_test_parser = provider_subparsers.add_parser("test", help="Test provider connection")
    provider_test_parser.add_argument("--name", required=True, help="Provider name to test")
    
    # Multi-instance Ollama provider commands - Claude Generated
    ollama_parser = provider_subparsers.add_parser(
        "ollama", help="Manage Ollama provider instances."
    )
    ollama_subparsers = ollama_parser.add_subparsers(dest="ollama_action", help="Ollama provider management actions")
    
    # List Ollama providers
    ollama_list_parser = ollama_subparsers.add_parser("list", help="List all Ollama providers")
    ollama_list_parser.add_argument("--enabled-only", action="store_true", help="Show only enabled providers")
    
    # Add Ollama provider
    ollama_add_parser = ollama_subparsers.add_parser("add", help="Add new Ollama provider")
    ollama_add_parser.add_argument("--name", required=True, help="Provider name (alias)")
    ollama_add_parser.add_argument("--host", required=True, help="Ollama server host")
    ollama_add_parser.add_argument("--port", type=int, default=11434, help="Ollama server port (default: 11434)")
    ollama_add_parser.add_argument("--api-key", help="API key for authenticated access")
    ollama_add_parser.add_argument("--ssl", action="store_true", help="Use HTTPS instead of HTTP")
    ollama_add_parser.add_argument("--type", choices=["openai_compatible", "native_client"], 
                                  default="openai_compatible", help="Connection type (default: openai_compatible)")
    ollama_add_parser.add_argument("--description", help="Provider description")
    ollama_add_parser.add_argument("--enabled", action="store_true", default=True, help="Enable provider (default: True)")
    ollama_add_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                  help="Configuration scope")
    
    # Edit Ollama provider
    ollama_edit_parser = ollama_subparsers.add_parser("edit", help="Edit existing Ollama provider")
    ollama_edit_parser.add_argument("--name", required=True, help="Provider name to edit")
    ollama_edit_parser.add_argument("--new-name", help="New provider name")
    ollama_edit_parser.add_argument("--host", help="New host")
    ollama_edit_parser.add_argument("--port", type=int, help="New port")
    ollama_edit_parser.add_argument("--api-key", help="New API key")
    ollama_edit_parser.add_argument("--ssl", action="store_true", help="Enable HTTPS")
    ollama_edit_parser.add_argument("--no-ssl", dest="ssl", action="store_false", help="Disable HTTPS")
    ollama_edit_parser.add_argument("--type", choices=["openai_compatible", "native_client"], help="New connection type")
    ollama_edit_parser.add_argument("--description", help="New description")
    ollama_edit_parser.add_argument("--enable", action="store_true", help="Enable provider")
    ollama_edit_parser.add_argument("--disable", action="store_true", help="Disable provider")
    ollama_edit_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                   help="Configuration scope")
    
    # Delete Ollama provider
    ollama_delete_parser = ollama_subparsers.add_parser("delete", help="Delete Ollama provider")
    ollama_delete_parser.add_argument("--name", required=True, help="Provider name to delete")
    ollama_delete_parser.add_argument("--scope", choices=["user", "project", "system"], default="user",
                                     help="Configuration scope")
    
    # Legacy Ollama commands (for backward compatibility)
    ollama_legacy_group = ollama_subparsers.add_parser("legacy", help="Legacy Ollama configuration commands")
    ollama_legacy_subparsers = ollama_legacy_group.add_subparsers(dest="ollama_legacy_action")
    
    # Enable local Ollama (legacy)
    ollama_local_parser = ollama_legacy_subparsers.add_parser(
        "enable-local", help="Enable local Ollama connection (legacy)."
    )
    ollama_local_parser.add_argument("--host", default="localhost", help="Ollama host (default: localhost)")
    ollama_local_parser.add_argument("--port", type=int, default=11434, help="Ollama port (default: 11434)")
    
    # Enable official Ollama API (legacy)
    ollama_official_parser = ollama_legacy_subparsers.add_parser(
        "enable-official", help="Enable official Ollama API connection (legacy)."
    )
    ollama_official_parser.add_argument("--api-key", required=True, help="Official Ollama API key")
    ollama_official_parser.add_argument("--base-url", default="https://ollama.com", help="Base URL (default: https://ollama.com)")
    
    # Enable native Ollama client (legacy)
    ollama_native_parser = ollama_legacy_subparsers.add_parser(
        "enable-native", help="Enable native Ollama client connection (legacy)."
    )  
    ollama_native_parser.add_argument("--api-key", help="API key for authenticated access")
    ollama_native_parser.add_argument("--host", default="https://ollama.com", help="Ollama host (default: https://ollama.com)")
    
    # Show Ollama status
    ollama_subparsers.add_parser("status", help="Show current Ollama configuration.")
    
    # Test Ollama provider
    ollama_test_parser = ollama_subparsers.add_parser("test", help="Test Ollama provider connection")
    ollama_test_parser.add_argument("--name", required=True, help="Ollama provider name to test")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if prompts file exists
    if not os.path.exists(PROMPTS_FILE) and args.command not in ["list-models", "list-providers", "test-providers", "list-models-detailed", "dnb-import", "clear-cache"]:
        logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
        return

    if args.command == "pipeline":
        # Setup services - provider mapping handled in PipelineStepExecutor - Claude Generated  
        from src.utils.config_manager import ConfigManager as CM
        config_manager = CM()
        llm_service = LlmService(
            providers=None,  # Initialize without specific providers, they'll be resolved dynamically
            config_manager=config_manager,
            ollama_url=args.ollama_host,
            ollama_port=args.ollama_port,
        )
        prompt_service = PromptService(PROMPTS_FILE, logger)
        alima_manager = AlimaManager(llm_service, prompt_service, logger)
        cache_manager = UnifiedKnowledgeManager()
        
        # Get catalog configuration - use args if provided, otherwise from config - Claude Generated
        catalog_config = config_manager.get_catalog_config()
        catalog_token = args.catalog_token or catalog_config.get("catalog_token", "")
        catalog_search_url = args.catalog_search_url or catalog_config.get("catalog_search_url", "")
        catalog_details_url = args.catalog_details_url or catalog_config.get("catalog_details_url", "")

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
                    provider=args.provider,  # Raw provider name, resolved in execute_complete_pipeline - Claude Generated
                    suggesters=args.suggesters,
                    stream_callback=stream_callback,
                    logger=logger,
                    initial_task=args.initial_task,
                    final_task=args.final_task,
                    keyword_chunking_threshold=args.keyword_chunking_threshold,
                    chunking_task=args.chunking_task,
                    include_dk_classification=include_dk,
                    # Catalog configuration - Claude Generated
                    catalog_token=catalog_token,
                    catalog_search_url=catalog_search_url,
                    catalog_details_url=catalog_details_url,
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

    elif args.command == "search":
        cache_manager = UnifiedKnowledgeManager()
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

    elif args.command == "list-providers":
        # Claude Generated - List all configured LLM providers with details
        print("=== ALIMA LLM Provider Configuration ===\n")
        
        try:
            # Use config manager to get complete configuration
            from src.utils.config_manager import ConfigManager as CM
            config_manager = CM()
            config = config_manager.load_config()
            
            # Initialize LLM service for reachability testing
            llm_service = LlmService(lazy_initialization=True)
            
            provider_count = 0
            reachable_count = 0
            
            # Display Ollama providers
            if config.llm.ollama_providers:
                print("🚀 Ollama Providers:")
                for provider in config.llm.ollama_providers:
                    status_icon = "✅" if provider.enabled else "❌"
                    reachable = llm_service.is_provider_reachable(provider.name) if provider.enabled else False
                    reachable_icon = "🌐" if reachable else "📡"
                    
                    print(f"  {status_icon} {provider.name} ({provider.host}:{provider.port})")
                    print(f"    URL: {provider.base_url}")
                    print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                    print(f"    Reachable: {'Yes' if reachable else 'No'} {reachable_icon}")
                    print(f"    SSL: {'Yes' if provider.use_ssl else 'No'}")
                    if provider.api_key:
                        print(f"    API Key: {'*' * 8}...")
                    if provider.description:
                        print(f"    Description: {provider.description}")
                    
                    if args.show_config:
                        print(f"    Connection Type: {provider.connection_type}")
                    
                    if args.show_models and provider.enabled and reachable:
                        try:
                            models = llm_service.get_available_models(provider.name)
                            if models:
                                print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                            else:
                                print("    Models: None available")
                        except Exception as e:
                            print(f"    Models: Error loading ({e})")
                    
                    provider_count += 1
                    if provider.enabled and reachable:
                        reachable_count += 1
                    print()
            
            # Display OpenAI-compatible providers
            if config.llm.openai_compatible_providers:
                print("🤖 OpenAI-Compatible Providers:")
                for provider in config.llm.openai_compatible_providers:
                    status_icon = "✅" if provider.enabled else "❌"
                    reachable = llm_service.is_provider_reachable(provider.name) if provider.enabled else False
                    reachable_icon = "🌐" if reachable else "📡"
                    
                    print(f"  {status_icon} {provider.name}")
                    print(f"    URL: {provider.base_url}")
                    print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                    print(f"    Reachable: {'Yes' if reachable else 'No'} {reachable_icon}")
                    if provider.api_key:
                        print(f"    API Key: {'*' * 8}...")
                    if provider.description:
                        print(f"    Description: {provider.description}")
                    
                    if args.show_models and provider.enabled and reachable:
                        try:
                            models = llm_service.get_available_models(provider.name)
                            if models:
                                print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                            else:
                                print("    Models: None available")
                        except Exception as e:
                            print(f"    Models: Error loading ({e})")
                    
                    provider_count += 1
                    if provider.enabled and reachable:
                        reachable_count += 1
                    print()
            
            # Display API-only providers (Gemini, Anthropic)
            api_providers = []
            if config.llm.gemini:
                api_providers.append(("Gemini", "Google", config.llm.gemini))
            if config.llm.anthropic:
                api_providers.append(("Anthropic", "Anthropic", config.llm.anthropic))
            
            if api_providers:
                print("🎯 API-Only Providers:")
                for name, company, api_key in api_providers:
                    print(f"  ✅ {name} ({company})")
                    print(f"    API Key: {'*' * 8}...")
                    print(f"    Status: Configured")
                    print(f"    Reachable: Yes (API service) 🌐")
                    
                    if args.show_models:
                        try:
                            models = llm_service.get_available_models(name.lower())
                            if models:
                                print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                            else:
                                print("    Models: None available")
                        except Exception as e:
                            print(f"    Models: Error loading ({e})")
                    
                    provider_count += 1
                    reachable_count += 1
                    print()
            
            # Summary
            print(f"📊 Summary: {reachable_count}/{provider_count} providers reachable")
            
        except Exception as e:
            logger.error(f"Error listing providers: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == "test-providers":
        # Claude Generated - Test connection to all configured LLM providers
        print("=== ALIMA LLM Provider Connection Test ===\n")
        
        try:
            # Initialize LLM service with timeout
            llm_service = LlmService(lazy_initialization=True)
            
            # Get all provider status
            print("🔍 Testing provider connections...\n")
            status_results = llm_service.refresh_all_provider_status()
            
            passed_tests = 0
            total_tests = len(status_results)
            
            for provider_name, result in status_results.items():
                if isinstance(result, dict):
                    reachable = result.get('reachable', False)
                    error = result.get('error', '')
                    latency = result.get('latency_ms', 0)
                else:
                    reachable = result
                    error = '' if reachable else 'Connection failed'
                    latency = 0
                
                status_icon = "✅" if reachable else "❌"
                print(f"{status_icon} {provider_name}")
                
                if reachable:
                    print(f"   Status: Connected")
                    if latency > 0:
                        print(f"   Latency: {latency:.1f}ms")
                    passed_tests += 1
                    
                    if args.show_models:
                        try:
                            models = llm_service.get_available_models(provider_name)
                            if models:
                                print(f"   Models ({len(models)}): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
                            else:
                                print("   Models: None available")
                        except Exception as e:
                            print(f"   Models: Error loading ({e})")
                else:
                    print(f"   Status: Failed")
                    if error:
                        print(f"   Error: {error}")
                
                print()
            
            # Summary
            print(f"📊 Test Results: {passed_tests}/{total_tests} providers passed")
            if passed_tests == total_tests:
                print("🎉 All providers are working correctly!")
            elif passed_tests == 0:
                print("⚠️  No providers are currently reachable")
            else:
                print(f"⚠️  {total_tests - passed_tests} provider(s) need attention")
                
        except Exception as e:
            logger.error(f"Error testing providers: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == "list-models-detailed":
        # Claude Generated - List models from all providers with detailed information
        print("=== ALIMA Comprehensive Model List ===\n")
        
        try:
            # Initialize LLM service
            llm_service = LlmService(lazy_initialization=True)
            
            # Get configuration for provider details
            from src.utils.config_manager import ConfigManager as CM
            config_manager = CM()
            config = config_manager.load_config()
            
            total_models = 0
            provider_count = 0
            
            print("🔍 Scanning all providers for available models...\n")
            
            # Check all configured providers
            all_providers = []
            
            # Add Ollama providers
            for provider in config.llm.ollama_providers:
                if provider.enabled:
                    all_providers.append((provider.name, 'Ollama', provider.base_url))
            
            # Add OpenAI-compatible providers  
            for provider in config.llm.openai_compatible_providers:
                if provider.enabled:
                    all_providers.append((provider.name, 'OpenAI-Compatible', provider.base_url))
            
            # Add API providers
            if config.llm.gemini:
                all_providers.append(('gemini', 'Google API', 'https://api.google.com'))
            if config.llm.anthropic:
                all_providers.append(('anthropic', 'Anthropic API', 'https://api.anthropic.com'))
            
            for provider_name, provider_type, base_url in all_providers:
                print(f"🚀 {provider_name} ({provider_type})")
                print(f"   URL: {base_url}")
                
                # Test reachability first
                reachable = llm_service.is_provider_reachable(provider_name)
                if not reachable:
                    print("   Status: ❌ Not reachable")
                    print()
                    continue
                    
                print("   Status: ✅ Reachable")
                
                try:
                    models = llm_service.get_available_models(provider_name)
                    if models:
                        print(f"   Models ({len(models)}):")
                        for i, model in enumerate(models, 1):
                            print(f"     {i:2d}. {model}")
                        total_models += len(models)
                    else:
                        print("   Models: None available")
                        
                    provider_count += 1
                        
                except Exception as e:
                    print(f"   Models: ❌ Error loading ({e})")
                
                print()
            
            # Summary
            print(f"📊 Summary:")
            print(f"   Providers scanned: {len(all_providers)}")
            print(f"   Providers with models: {provider_count}")
            print(f"   Total models found: {total_models}")
            
        except Exception as e:
            logger.error(f"Error listing detailed models: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == "test-catalog":
        # Claude Generated - Test catalog search functionality
        from src.utils.clients.biblio_client import BiblioClient
        
        # Setup logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("biblio_extractor").setLevel(logging.DEBUG)
        
        print(f"🔍 Testing catalog search for terms: {', '.join(args.search_terms)}")
        print(f"📊 Max results per term: {args.max_results}")
        print("-" * 60)
        
        try:
            # Get catalog configuration - use args if provided, otherwise from config - Claude Generated
            from src.utils.config_manager import ConfigManager as CM
            config_manager = CM()
            catalog_config = config_manager.get_catalog_config()
            
            catalog_token = args.catalog_token or catalog_config.get("catalog_token", "")
            catalog_search_url = args.catalog_search_url or catalog_config.get("catalog_search_url", "")
            catalog_details_url = args.catalog_details_url or catalog_config.get("catalog_details_url", "")
            
            if not catalog_token:
                logger.error("❌ No catalog token found in config or arguments. Configure in settings or use --catalog-token TOKEN")
                return
                
            if not catalog_search_url:
                logger.error("❌ No catalog search URL found in config or arguments. Configure in settings or use --catalog-search-url URL")
                return
                
            if not catalog_details_url:
                logger.error("❌ No catalog details URL found in config or arguments. Configure in settings or use --catalog-details-url URL")
                return
                
            print(f"🔑 Using catalog token: {catalog_token[:10]}..." if len(catalog_token) > 10 else catalog_token)
            if catalog_search_url:
                print(f"🌐 Search URL: {catalog_search_url}")
            if catalog_details_url:
                print(f"🌐 Details URL: {catalog_details_url}")
            print()
            
            # Initialize BiblioExtractor
            extractor = BiblioClient(
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
            
    elif args.command == "clear-cache":
        # Cache clearing functionality - Claude Generated
        try:
            cache_manager = UnifiedKnowledgeManager()
            
            if not args.confirm:
                # Show current cache stats before clearing
                stats = cache_manager.get_cache_stats()
                print("📊 Current cache statistics:")
                print(f"   GND entries: {stats.get('gnd_entries', 0)}")
                print(f"   Classifications: {stats.get('classification_entries', 0)}")  
                print(f"   Search mappings: {stats.get('search_mappings', 0)}")
                print(f"   Database size: {stats.get('size_mb', 0)} MB")
                print()
                
                # Confirmation prompt
                if args.type == "all":
                    confirm_msg = "⚠️  Are you sure you want to clear ALL cache data? This cannot be undone. [y/N]: "
                else:
                    confirm_msg = f"⚠️  Are you sure you want to clear {args.type} cache data? This cannot be undone. [y/N]: "
                    
                response = input(confirm_msg).lower().strip()
                if response not in ['y', 'yes']:
                    print("❌ Cache clearing cancelled.")
                    return
            
            print(f"🗑️  Clearing {args.type} cache data...")
            
            if args.type == "all":
                cache_manager.clear_database()
                print("✅ All cache data cleared successfully.")
            else:
                # Selective clearing - Claude Generated
                import sqlite3
                with sqlite3.connect(cache_manager.db_path) as conn:
                    if args.type == "gnd":
                        conn.execute("DELETE FROM gnd_entries")
                        print("✅ GND entries cleared successfully.")
                    elif args.type == "search":
                        conn.execute("DELETE FROM search_mappings") 
                        print("✅ Search mappings cleared successfully.")
                    elif args.type == "classifications":
                        conn.execute("DELETE FROM classifications")
                        print("✅ Classifications cleared successfully.")
                        
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            
    elif args.command == "dnb-import":
        # DNB XML import functionality with progress (like GUI) - Claude Generated
        try:
            from src.core.gndparser import GNDParser
            
            url = "https://data.dnb.de/GND/authorities-gnd-sachbegriff_dnbmarc.mrc.xml.gz"
            
            print("🌐 Starte DNB-Download...")
            print(f"📡 URL: {url}")
            
            start_time = time.time()
            
            try:
                # Download file with progress
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # Get file size if available
                total_size = int(response.headers.get("content-length", 0))
                if total_size > 0:
                    print(f"📦 Dateigröße: {total_size / (1024*1024):.1f} MB")
                
                # Create temporary files
                temp_dir = tempfile.mkdtemp()
                temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
                temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")
                
                # Download with progress
                downloaded = 0
                last_console_percent = 0
                
                print("⬇️ Download läuft...")
                with open(temp_gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            # Console progress every 10%
                            console_percent = (downloaded / total_size) * 100
                            if console_percent - last_console_percent >= 10:
                                print(f"📊 Download: {console_percent:.0f}%")
                                last_console_percent = console_percent
                
                print("📦 Entpacke GZ-Datei...")
                
                # Extract gz file
                with gzip.open(temp_gz_path, "rb") as gz_file:
                    with open(temp_xml_path, "wb") as xml_file:
                        xml_file.write(gz_file.read())
                
                print("✅ Download und Entpackung abgeschlossen")
                
                # Import into cache using GNDParser
                print("🔄 Starte GND-Datenbank Import...")
                print(f"📁 Datei: {temp_xml_path}")
                
                cache_manager = UnifiedKnowledgeManager()
                parser = GNDParser(cache_manager)
                
                print("⚙️ Verarbeite XML-Daten...")
                
                # Process the file
                parser.process_file(temp_xml_path)
                
                # Clean up temp files
                os.remove(temp_gz_path)
                os.remove(temp_xml_path)
                os.rmdir(temp_dir)
                
                elapsed = time.time() - start_time
                print(f"✅ DNB-Import erfolgreich abgeschlossen in {elapsed:.2f} Sekunden")
                
                # Show cache statistics
                stats = cache_manager.get_cache_stats()
                print(f"📊 Cache-Statistiken:")
                print(f"   GND-Einträge: {stats.get('gnd_entries', 0):,}")
                print(f"   Klassifikationen: {stats.get('classification_entries', 0):,}")
                print(f"   Datenbank-Größe: {stats.get('size_mb', 0):.1f} MB")
                
            except requests.RequestException as e:
                logger.error(f"❌ Download-Fehler: {e}")
                if args.debug:
                    raise
            except Exception as e:
                logger.error(f"❌ Import-Fehler: {e}")
                if args.debug:
                    raise
                    
        except ImportError as e:
            logger.error(f"❌ Fehlende Module für DNB-Import: {e}")
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler beim DNB-Import: {e}")
            if args.debug:
                raise
                
    elif args.command == "provider":
        # OpenAI-compatible provider management commands - Claude Generated
        from src.utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        
        if args.provider_action == "list":
            try:
                config = config_manager.load_config()
                providers = config.llm.openai_compatible_providers
                
                if not providers:
                    print("🔍 No OpenAI-compatible providers configured.")
                    print("   Use 'alima_cli.py provider add' to add a new provider.")
                    return
                
                print(f"🤖 OpenAI-Compatible Providers ({len(providers)} configured):")
                print()
                
                for i, provider in enumerate(providers, 1):
                    status_icon = "✅" if provider.enabled else "❌"
                    api_key_display = provider.api_key[:8] + "..." if provider.api_key else "❌ Not set"
                    
                    print(f"{i}. {status_icon} {provider.name}")
                    print(f"   Base URL: {provider.base_url}")
                    print(f"   API Key:  {api_key_display}")
                    print(f"   Enabled:  {'Yes' if provider.enabled else 'No'}")
                    if provider.description:
                        print(f"   Description: {provider.description}")
                    if provider.models:
                        print(f"   Models: {', '.join(provider.models[:3])}{'...' if len(provider.models) > 3 else ''}")
                    print()
                    
            except Exception as e:
                logger.error(f"❌ Error listing providers: {e}")
                
        elif args.provider_action == "add":
            try:
                config = config_manager.load_config()
                
                # Check if provider already exists
                if config.llm.get_provider_by_name(args.name):
                    print(f"❌ Provider '{args.name}' already exists.")
                    print("   Use 'alima_cli.py provider edit' to modify existing providers.")
                    return
                
                # Create new provider
                new_provider = OpenAICompatibleProvider(
                    name=args.name,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    enabled=args.enabled,
                    description=args.description
                )
                
                # Add provider to configuration
                config.llm.add_provider(new_provider)
                
                # Save configuration
                success = config_manager.save_config(config, args.scope)
                if success:
                    print(f"✅ Provider '{args.name}' added successfully to {args.scope} scope")
                    print(f"   Base URL: {args.base_url}")
                    print(f"   Enabled: {'Yes' if args.enabled else 'No'}")
                    if args.description:
                        print(f"   Description: {args.description}")
                else:
                    print("❌ Failed to save provider configuration")
                    
            except ValueError as e:
                print(f"❌ Invalid provider configuration: {e}")
            except Exception as e:
                logger.error(f"❌ Error adding provider: {e}")
                
        elif args.provider_action == "remove":
            try:
                config = config_manager.load_config()
                
                # Check if provider exists
                provider = config.llm.get_provider_by_name(args.name)
                if not provider:
                    print(f"❌ Provider '{args.name}' not found.")
                    print("   Use 'alima_cli.py provider list' to see available providers.")
                    return
                
                # Remove provider
                success = config.llm.remove_provider(args.name)
                if success:
                    # Save configuration
                    config_saved = config_manager.save_config(config, args.scope)
                    if config_saved:
                        print(f"✅ Provider '{args.name}' removed successfully from {args.scope} scope")
                    else:
                        print("❌ Failed to save configuration after removal")
                else:
                    print(f"❌ Failed to remove provider '{args.name}'")
                    
            except Exception as e:
                logger.error(f"❌ Error removing provider: {e}")
                
        elif args.provider_action == "edit":
            try:
                config = config_manager.load_config()
                
                # Find provider to edit
                provider = config.llm.get_provider_by_name(args.name)
                if not provider:
                    print(f"❌ Provider '{args.name}' not found.")
                    print("   Use 'alima_cli.py provider list' to see available providers.")
                    return
                
                # Update provider fields if provided
                if args.base_url:
                    provider.base_url = args.base_url
                if args.api_key:
                    provider.api_key = args.api_key
                if args.description is not None:  # Allow empty string
                    provider.description = args.description
                if hasattr(args, 'enabled') and args.enabled is not None:
                    provider.enabled = args.enabled
                
                # Save configuration
                success = config_manager.save_config(config, args.scope)
                if success:
                    print(f"✅ Provider '{args.name}' updated successfully in {args.scope} scope")
                    print(f"   Base URL: {provider.base_url}")
                    print(f"   Enabled: {'Yes' if provider.enabled else 'No'}")
                    if provider.description:
                        print(f"   Description: {provider.description}")
                else:
                    print("❌ Failed to save provider configuration")
                    
            except ValueError as e:
                print(f"❌ Invalid provider configuration: {e}")
            except Exception as e:
                logger.error(f"❌ Error editing provider: {e}")
                
        elif args.provider_action == "test":
            try:
                config = config_manager.load_config()
                
                # Find provider to test
                provider = config.llm.get_provider_by_name(args.name)
                if not provider:
                    print(f"❌ Provider '{args.name}' not found.")
                    print("   Use 'alima_cli.py provider list' to see available providers.")
                    return
                
                if not provider.enabled:
                    print(f"⚠️  Provider '{args.name}' is disabled.")
                    print("   Enable it first or test anyway? (y/N): ", end="")
                    response = input().strip().lower()
                    if response != 'y':
                        return
                
                print(f"🔌 Testing connection to '{args.name}'...")
                print(f"   Base URL: {provider.base_url}")
                
                # Test provider configuration by attempting to initialize LLM service
                try:
                    llm_service = LlmService()
                    
                    # Initialize providers silently to test configuration
                    old_level = logging.getLogger().level
                    logging.getLogger().setLevel(logging.ERROR)  # Suppress info logs during test
                    
                    llm_service.initialize_providers()
                    
                    logging.getLogger().setLevel(old_level)  # Restore logging level
                    
                    # If we get here, basic initialization worked
                    print(f"✅ Provider '{args.name}' configuration is valid and loaded successfully")
                    print(f"   API Key: {'✅ Set' if provider.api_key else '❌ Not set'}")
                    print(f"   Base URL: {provider.base_url}")
                    
                    if not provider.api_key:
                        print("   ⚠️  No API key configured - actual LLM calls will fail")
                    else:
                        print("   ✅ Provider ready for use")
                    
                except Exception as e:
                    print(f"❌ Provider test failed: {e}")
                    print(f"   Check base URL and configuration")
                    
            except Exception as e:
                logger.error(f"❌ Error testing provider: {e}")
        
        elif args.provider_action == "ollama":
            # Handle Ollama configuration commands - Claude Generated
            if args.ollama_action == "status":
                try:
                    config = config_manager.load_config()
                    print("🔧 Current Ollama Configuration:")
                    print(f"   Local:    {'✅ Enabled' if config.llm.ollama.local_enabled else '❌ Disabled'} ({config.llm.ollama.local_host}:{config.llm.ollama.local_port})")
                    print(f"   Official: {'✅ Enabled' if config.llm.ollama.official_enabled else '❌ Disabled'} ({config.llm.ollama.official_base_url})")
                    print(f"   Native:   {'✅ Enabled' if config.llm.ollama.native_enabled else '❌ Disabled'} ({config.llm.ollama.native_host})")
                    print(f"   Active:   {config.llm.ollama.get_active_connection_type()}")
                except Exception as e:
                    print(f"❌ Error loading Ollama configuration: {str(e)}")
            
            elif args.ollama_action == "enable-local":
                try:
                    config = config_manager.load_config()
                    
                    # Update local Ollama settings
                    config.llm.ollama.local_enabled = True
                    config.llm.ollama.local_host = args.host
                    config.llm.ollama.local_port = args.port
                    
                    # Disable other types for clarity
                    config.llm.ollama.official_enabled = False
                    config.llm.ollama.native_enabled = False
                    
                    config_manager.save_config(config)
                    print(f"✅ Local Ollama enabled: {args.host}:{args.port}")
                    
                except Exception as e:
                    print(f"❌ Error enabling local Ollama: {str(e)}")
            
            elif args.ollama_action == "enable-official":
                try:
                    config = config_manager.load_config()
                    
                    # Update official Ollama settings
                    config.llm.ollama.official_enabled = True
                    config.llm.ollama.official_base_url = args.base_url
                    config.llm.ollama.official_api_key = args.api_key
                    
                    # Disable other types for clarity
                    config.llm.ollama.local_enabled = False
                    config.llm.ollama.native_enabled = False
                    
                    config_manager.save_config(config)
                    print(f"✅ Official Ollama API enabled: {args.base_url}")
                    print(f"   API Key: {args.api_key[:20]}...")
                    
                except Exception as e:
                    print(f"❌ Error enabling official Ollama: {str(e)}")
            
            elif args.ollama_action == "enable-native":
                try:
                    config = config_manager.load_config()
                    
                    # Update native Ollama settings
                    config.llm.ollama.native_enabled = True
                    config.llm.ollama.native_host = args.host
                    if args.api_key:
                        config.llm.ollama.native_api_key = args.api_key
                    
                    # Disable other types for clarity
                    config.llm.ollama.local_enabled = False
                    config.llm.ollama.official_enabled = False
                    
                    config_manager.save_config(config)
                    print(f"✅ Native Ollama client enabled: {args.host}")
                    if args.api_key:
                        print(f"   API Key: {args.api_key[:20]}...")
                    else:
                        print("   No API key configured (local access)")
                    
                except Exception as e:
                    print(f"❌ Error enabling native Ollama: {str(e)}")
        else:
            print("❌ No provider action specified.")
            print("   Use: list, add, remove, edit, test, or ollama")
                
    elif args.command == "db-config":
        # Database configuration commands - Claude Generated
        from src.utils.config_manager import ConfigManager, DatabaseConfig, AlimaConfig
        import getpass
        
        config_manager = ConfigManager()
        
        if args.db_action == "show":
            try:
                config = config_manager.load_config()
                print("📊 Current Database Configuration:")
                print(f"   Type: {config.database.db_type}")
                if config.database.db_type == 'sqlite':
                    print(f"   Path: {config.database.sqlite_path}")
                else:
                    print(f"   Host: {config.database.host}:{config.database.port}")
                    print(f"   Database: {config.database.database}")
                    print(f"   Username: {config.database.username}")
                    print(f"   Charset: {config.database.charset}")
                    print(f"   SSL Disabled: {config.database.ssl_disabled}")
                print(f"   Connection Timeout: {config.database.connection_timeout}s")
                print(f"   Auto Create Tables: {config.database.auto_create_tables}")
            except Exception as e:
                logger.error(f"❌ Error showing database config: {e}")
                
        elif args.db_action == "paths":
            try:
                from pathlib import Path
                config_info = config_manager.get_config_info()
                print(f"🖥️  Configuration Paths for {config_info['os']}:")
                print(f"   Project:  {config_info['project_config']}")
                print(f"   User:     {config_info['user_config']}")
                print(f"   System:   {config_info['system_config']}")
                print(f"   Legacy:   {config_info['legacy_config']}")
                print()
                
                # Show which files exist
                paths = [
                    ("Project", config_info['project_config']),
                    ("User", config_info['user_config']),
                    ("System", config_info['system_config']),
                    ("Legacy", config_info['legacy_config'])
                ]
                
                print("📁 File Status:")
                for name, path in paths:
                    exists = Path(path).exists()
                    status = "✅ EXISTS" if exists else "❌ NOT FOUND"
                    print(f"   {name:8}: {status}")
                    
            except Exception as e:
                logger.error(f"❌ Error showing config paths: {e}")
                
        elif args.db_action == "test":
            try:
                success, message = config_manager.test_database_connection()
                print(f"🔌 Database Connection Test: {'✅ SUCCESS' if success else '❌ FAILED'}")
                print(f"   {message}")
            except Exception as e:
                logger.error(f"❌ Error testing database connection: {e}")
                
        elif args.db_action == "set-sqlite":
            try:
                config = config_manager.load_config()
                
                # Update database configuration
                config.database.db_type = "sqlite"
                config.database.sqlite_path = args.path
                
                # Save configuration
                success = config_manager.save_config(config, args.scope)
                if success:
                    print(f"✅ SQLite database configuration saved to {args.scope} scope")
                    print(f"   Database path: {args.path}")
                else:
                    print("❌ Failed to save SQLite configuration")
            except Exception as e:
                logger.error(f"❌ Error configuring SQLite database: {e}")
                
        elif args.db_action == "set-mysql":
            try:
                config = config_manager.load_config()
                
                # Get password if not provided
                password = args.password
                if not password:
                    password = getpass.getpass("Enter database password: ")
                
                # Update database configuration
                config.database.db_type = "mysql"
                config.database.host = args.host
                config.database.port = args.port
                config.database.database = args.database
                config.database.username = args.username
                config.database.password = password
                config.database.charset = args.charset
                config.database.ssl_disabled = args.ssl_disabled
                
                # Test connection first
                print("🔌 Testing MySQL connection...")
                success, message = config_manager.test_database_connection()
                if not success:
                    print(f"❌ Connection test failed: {message}")
                    print("Configuration not saved.")
                    return
                
                print("✅ Connection test successful!")
                
                # Save configuration
                success = config_manager.save_config(config, args.scope)
                if success:
                    print(f"✅ MySQL database configuration saved to {args.scope} scope")
                    print(f"   Host: {args.host}:{args.port}")
                    print(f"   Database: {args.database}")
                    print(f"   Username: {args.username}")
                else:
                    print("❌ Failed to save MySQL configuration")
                    
            except Exception as e:
                logger.error(f"❌ Error configuring MySQL database: {e}")
        else:
            print("❌ No database action specified. Use 'show', 'test', 'set-sqlite', or 'set-mysql'")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
