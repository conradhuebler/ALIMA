import argparse
import logging
import os
import json
import time
import tempfile
import gzip
import requests
import sys
from dataclasses import asdict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    PipelineJsonManager,
    PipelineStepExecutor,
)
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.config_manager import ConfigManager, OpenAICompatibleProvider
from src.utils.config_models import PipelineMode, PipelineStepConfig
from typing import List, Tuple

PROMPTS_FILE = "prompts.json"


# Use shared JSON utilities from pipeline_utils
_task_state_to_dict = PipelineJsonManager.task_state_to_dict
_convert_sets_to_lists = PipelineJsonManager.convert_sets_to_lists


def parse_step_config(value: str) -> tuple:
    """Parse STEP=PROVIDER|MODEL format - Claude Generated"""
    try:
        step, provider_model = value.split('=', 1)

        if '|' in provider_model:
            provider, model = provider_model.split('|', 1)
            return step.strip(), provider.strip() or None, model.strip() or None
        else:
            # Fallback: treat as provider only
            return step.strip(), provider_model.strip(), None
    except ValueError:
        raise ValueError(f"Invalid step config format: {value}. Expected: STEP=PROVIDER|MODEL")


def parse_step_parameter(value: str) -> tuple:
    """Parse STEP=VALUE format for parameters - Claude Generated"""
    try:
        step, param_value = value.split('=', 1)
        return step.strip(), param_value.strip()
    except ValueError:
        raise ValueError(f"Invalid step parameter format: {value}. Expected: STEP=VALUE")


def build_step_configurations(args) -> dict:
    """Build step configurations from CLI arguments - Claude Generated"""
    step_configs = {}

    # Parse step provider|model configurations
    if args.step:
        for step_config in args.step:
            step, provider, model = parse_step_config(step_config)
            if step not in step_configs:
                step_configs[step] = {}
            if provider:
                step_configs[step]['provider'] = provider
            if model:
                step_configs[step]['model'] = model

    # Parse step task configurations
    if args.step_task:
        for task_config in args.step_task:
            step, task = parse_step_parameter(task_config)
            if step not in step_configs:
                step_configs[step] = {}
            step_configs[step]['task'] = task

    # Parse expert mode parameters
    if args.step_temperature:
        for temp_config in args.step_temperature:
            step, temperature = parse_step_parameter(temp_config)
            if step not in step_configs:
                step_configs[step] = {}
            step_configs[step]['temperature'] = float(temperature)

    if args.step_top_p:
        for top_p_config in args.step_top_p:
            step, top_p = parse_step_parameter(top_p_config)
            if step not in step_configs:
                step_configs[step] = {}
            step_configs[step]['top_p'] = float(top_p)

    if args.step_seed:
        for seed_config in args.step_seed:
            step, seed = parse_step_parameter(seed_config)
            if step not in step_configs:
                step_configs[step] = {}
            step_configs[step]['seed'] = int(seed)

    return step_configs


def create_config_from_cli_args(args, base_config):
    """Convert CLI arguments to PipelineConfig - Claude Generated"""
    from copy import deepcopy

    # Make a copy of base config to avoid modifying the original
    config = deepcopy(base_config)

    # Global settings
    if hasattr(args, 'suggesters') and args.suggesters:
        config.search_suggesters = args.suggesters

    if hasattr(args, 'disable_dk_classification') and args.disable_dk_classification:
        # Disable DK classification step
        if hasattr(config, 'step_configs_v2') and 'dk_classification' in config.step_configs_v2:
            config.step_configs_v2['dk_classification'].enabled = False

    # Process step-specific arguments
    step_configs = {}

    # Parse --step arguments (format: step=provider|model)
    if hasattr(args, 'step') and args.step:
        for step_config in args.step:
            step_name, provider_model = step_config.split('=', 1)
            if '|' in provider_model:
                provider, model = provider_model.split('|', 1)
                step_configs[step_name] = {'provider': provider, 'model': model}

    # Parse --step-task arguments (format: step=task)
    if hasattr(args, 'step_task') and args.step_task:
        for step_task in args.step_task:
            step_name, task = step_task.split('=', 1)
            if step_name not in step_configs:
                step_configs[step_name] = {}
            step_configs[step_name]['task'] = task

    # Parse expert mode parameters
    if hasattr(args, 'step_temperature') and args.step_temperature:
        for step_temp in args.step_temperature:
            step_name, temp = step_temp.split('=', 1)
            if step_name not in step_configs:
                step_configs[step_name] = {}
            step_configs[step_name]['temperature'] = float(temp)

    if hasattr(args, 'step_top_p') and args.step_top_p:
        for step_top_p in args.step_top_p:
            step_name, top_p = step_top_p.split('=', 1)
            if step_name not in step_configs:
                step_configs[step_name] = {}
            step_configs[step_name]['top_p'] = float(top_p)

    if hasattr(args, 'step_seed') and args.step_seed:
        for step_seed in args.step_seed:
            step_name, seed = step_seed.split('=', 1)
            if step_name not in step_configs:
                step_configs[step_name] = {}
            step_configs[step_name]['seed'] = int(seed)

    # Apply mode-specific settings to affected steps
    mode_mapping = {
        'smart': PipelineMode.SMART,
        'advanced': PipelineMode.ADVANCED,
        'expert': PipelineMode.EXPERT
    }

    target_mode = mode_mapping.get(args.mode, PipelineMode.SMART)

    # Apply step configurations to pipeline config
    for step_name, step_params in step_configs.items():
        if hasattr(config, 'step_configs_v2') and step_name in config.step_configs_v2:
            step_config = config.step_configs_v2[step_name]

            # Set mode
            step_config.mode = target_mode

            # Apply parameters
            if 'provider' in step_params:
                step_config.provider = step_params['provider']
            if 'model' in step_params:
                step_config.model = step_params['model']
            if 'task' in step_params:
                step_config.task = step_params['task']
            if 'temperature' in step_params:
                step_config.temperature = step_params['temperature']
            if 'top_p' in step_params:
                step_config.top_p = step_params['top_p']
            if 'seed' in step_params:
                step_config.seed = step_params['seed']

    return config


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

    # Pipeline command (new unified pipeline) - Claude Generated
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Execute complete ALIMA analysis pipeline with mode-based configuration.",
        description="""
ALIMA Pipeline Command - Three Configuration Modes:

SMART MODE (--mode smart, default):
  Automatically uses task preferences from config.json. No manual provider/model selection needed.
  Best for: Regular analysis workflow with pre-configured preferences.

ADVANCED MODE (--mode advanced):
  Manual provider/model override with prompt task selection using --step format.
  Format: --step STEP=PROVIDER|MODEL (e.g., --step initialisation=ollama|cogito:14b)
  Best for: Testing different models or overriding specific steps.

EXPERT MODE (--mode expert):
  Full parameter control including temperature, top-p, and seed values.
  Uses --step, --step-task, --step-temperature, --step-top-p, --step-seed arguments.
  Best for: Fine-tuning and experimentation.

PIPELINE STEPS:
  - input: Text input processing (no LLM)
  - initialisation: LLM keyword extraction from text
  - search: GND database search (no LLM)
  - keywords: LLM keyword verification with GND context
  - classification: LLM DDC/DK/RVK classification (optional)

EXAMPLES:
  # Smart mode (default)
  python alima_cli.py pipeline --input-text "Your text here"

  # Advanced mode with specific models
  python alima_cli.py pipeline --mode advanced --input-text "Text" \\
    --step initialisation=ollama|cogito:14b --step keywords=gemini|gemini-1.5-flash

  # Expert mode with full control
  python alima_cli.py pipeline --mode expert --input-text "Text" \\
    --step initialisation=ollama|cogito:32b --step-temperature initialisation=0.3 \\
    --step-top-p keywords=0.1 --step-seed keywords=42
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Input options - either text or DOI (mutually exclusive)
    input_group = pipeline_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Input text for analysis.")
    input_group.add_argument(
        "--doi",
        help="DOI or URL to resolve and analyze (e.g., 10.1007/978-3-031-47390-6, https://link.springer.com/book/...).",
    )
    # Configuration mode selection - Claude Generated
    pipeline_parser.add_argument(
        "--mode",
        choices=["smart", "advanced", "expert"],
        default="smart",
        help="Configuration mode: smart (uses task_preferences), advanced (manual provider|model), expert (full parameter control)"
    )

    # Step-specific configuration with provider|model format - Claude Generated
    pipeline_parser.add_argument(
        "--step",
        action="append",
        help="Set provider|model for specific step: STEP=PROVIDER|MODEL (e.g., initialisation=ollama|cogito:32b)"
    )

    # Task selection for LLM steps - Claude Generated
    pipeline_parser.add_argument(
        "--step-task",
        action="append",
        help="Set prompt task for specific step: STEP=TASK (e.g., keywords=rephrase)"
    )

    # Expert mode parameters - Claude Generated
    pipeline_parser.add_argument(
        "--step-temperature",
        action="append",
        help="Set temperature for specific step: STEP=VALUE (Expert mode only)"
    )
    pipeline_parser.add_argument(
        "--step-top-p",
        action="append",
        help="Set top-p for specific step: STEP=VALUE (Expert mode only)"
    )
    pipeline_parser.add_argument(
        "--step-seed",
        action="append",
        help="Set seed for specific step: STEP=VALUE (Expert mode only)"
    )

    # Step control - Claude Generated
    pipeline_parser.add_argument(
        "--enable-step",
        help="Comma-separated list of steps to enable"
    )
    pipeline_parser.add_argument(
        "--disable-step",
        help="Comma-separated list of steps to disable"
    )

    # Configuration display - Claude Generated
    pipeline_parser.add_argument(
        "--show-smart-config",
        action="store_true",
        help="Show what Smart mode would use and exit"
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
    pipeline_parser.add_argument(
        "--dk-frequency-threshold",
        type=int,
        default=10,
        help="Minimum occurrence count for DK classifications to be included in LLM analysis (default: 10). Only classifications appearing >= N times in catalog will be passed to LLM.",
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
        # Setup services with Provider Preferences integration - Claude Generated  
        from src.utils.config_manager import ConfigManager as CM
        from src.core.pipeline_manager import PipelineConfig
        config_manager = CM()
        llm_service = LlmService(
            providers=None,  # Initialize without specific providers, they'll be resolved dynamically
            config_manager=config_manager,
            ollama_url=args.ollama_host,
            ollama_port=args.ollama_port,
        )
        prompt_service = PromptService(PROMPTS_FILE, logger)
        alima_manager = AlimaManager(llm_service, prompt_service, config_manager, logger)
        cache_manager = UnifiedKnowledgeManager()

        # Initialize PipelineManager for unified CLI/GUI pipeline logic - Claude Generated
        from src.core.pipeline_manager import PipelineManager
        pipeline_manager = PipelineManager(
            alima_manager=alima_manager,
            cache_manager=cache_manager,
            logger=logger,
            config_manager=config_manager
        )

        # CLI Callbacks for pipeline events - Claude Generated
        def cli_step_started(step):
            provider_info = f"{step.provider}/{step.model}" if step.provider and step.model else "Smart Mode"
            print(f"▶ Starte Schritt: {step.name} ({provider_info})")

        def cli_step_completed(step):
            print(f"✅ Schritt abgeschlossen: {step.name}")

        def cli_step_error(step, error_message):
            print(f"❌ Fehler in Schritt {step.name}: {error_message}")

        def cli_pipeline_completed(analysis_state):
            print("\n🎉 Pipeline vollständig abgeschlossen!")

        def cli_stream_callback(token, step_id):
            print(token, end="", flush=True)
        
        # Create pipeline config from Provider Preferences as baseline - Claude Generated
        try:
            pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)
            logger.info("Pipeline configuration loaded from Provider Preferences")
        except Exception as e:
            logger.warning(f"Failed to load Provider Preferences, using defaults: {e}")
            pipeline_config = PipelineConfig()
        
        # CLI parameters are now handled through mode-based step configurations - Claude Generated
        
        # Show configuration if requested - Claude Generated
        if getattr(args, 'show_config', False):
            print("🔧 Pipeline Configuration:")
            print(f"  Mode: {args.mode}")
            print(f"  Task preferences enabled: {'✅ Yes' if args.mode == 'smart' else '⚠️ Mode-based override active'}")

            # Show step configurations from new mode-based system
            step_configs = build_step_configurations(args)
            if step_configs:
                print(f"  CLI step configurations:")
                for step_id, config in step_configs.items():
                    parts = []
                    if config.get('provider'):
                        parts.append(f"provider={config['provider']}")
                    if config.get('model'):
                        parts.append(f"model={config['model']}")
                    if config.get('task'):
                        parts.append(f"task={config['task']}")
                    if config.get('temperature'):
                        parts.append(f"temp={config['temperature']}")
                    if config.get('top_p'):
                        parts.append(f"top_p={config['top_p']}")
                    print(f"    {step_id}: {', '.join(parts) if parts else 'default'}")
            else:
                print(f"  Using default configuration from pipeline_config")
                for step_id, step_config in pipeline_config.step_configs_v2.items():
                    if step_config.enabled and step_config.provider:
                        provider = step_config.provider
                        model = step_config.model
                        print(f"    {step_id}: {provider}/{model}")

            print(f"  Save preferences: {'✅ Yes' if getattr(args, 'save_preferences', False) else '❌ No'}")
            print()
        
        # Get catalog configuration - use args if provided, otherwise from config - Claude Generated
        catalog_config = config_manager.get_catalog_config()
        catalog_token = args.catalog_token or getattr(catalog_config, "catalog_token", "")
        catalog_search_url = args.catalog_search_url or getattr(catalog_config, "catalog_search_url", "")
        catalog_details_url = args.catalog_details_url or getattr(catalog_config, "catalog_details_url", "")

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

                # Execute complete pipeline with new mode-based configuration - Claude Generated
                logger.info("Starting complete pipeline execution with mode-based configuration")

                # Handle DK classification flag
                include_dk = getattr(args, 'include_dk_classification', False) and not getattr(args, 'disable_dk_classification', False)

                # Build step configurations from CLI arguments
                step_configs = build_step_configurations(args)
                logger.debug(f"Step configurations: {step_configs}")

                # Unified Pipeline Execution via PipelineManager - Claude Generated
                logger.info(f"Starting pipeline execution in {args.mode} mode using PipelineManager")

                # Create pipeline configuration from CLI arguments
                try:
                    # Use the base pipeline_config as starting point
                    updated_pipeline_config = create_config_from_cli_args(args, pipeline_config)
                    logger.info(f"Pipeline configuration created from CLI args: mode={args.mode}")
                except Exception as e:
                    logger.error(f"Failed to create pipeline configuration: {e}")
                    return

                # Set pipeline configuration
                pipeline_manager.set_config(updated_pipeline_config)

                # Set callbacks for CLI output
                pipeline_manager.set_callbacks(
                    step_started=cli_step_started,
                    step_completed=cli_step_completed,
                    step_error=cli_step_error,
                    pipeline_completed=cli_pipeline_completed,
                    stream_callback=cli_stream_callback
                )

                # Execute pipeline
                print(f"🚀 Starting {args.mode} mode pipeline...")
                try:
                    pipeline_manager.start_pipeline(input_text=input_text)

                    # Wait for pipeline completion (synchronous mode for CLI)
                    import time
                    timeout = 300  # 5 minutes timeout
                    elapsed = 0
                    while pipeline_manager.is_running and elapsed < timeout:
                        time.sleep(0.1)
                        elapsed += 0.1

                    if pipeline_manager.is_running:
                        logger.error("Pipeline execution timed out")
                        return

                    # Get final analysis state
                    analysis_state = pipeline_manager.current_analysis_state
                    if not analysis_state:
                        logger.error("Pipeline completed but no analysis state available")
                        return

                except Exception as e:
                    logger.error(f"Pipeline execution failed: {e}")
                    return

                print("\n--- Pipeline Results ---")
                print(f"Initial Keywords: {analysis_state.initial_keywords}")
                print(
                    f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}"
                )
                print(
                    f"GND Classes: {analysis_state.final_llm_analysis.extracted_gnd_classes}"
                )
                
                # Save preferences if requested and pipeline was successful - Claude Generated
                if getattr(args, 'save_preferences', False):
                    try:
                        unified_config = config_manager.get_unified_config()
                        preferences_updated = False

                        # Update preferences based on successful execution from mode-based configuration
                        if args.mode == 'smart':
                            print(f"\n📋 Smart mode used - no preference updates needed (task preferences already active)")

                        elif args.mode in ['advanced', 'expert']:
                            # Extract used providers/models from step configurations
                            used_providers = set()
                            used_models = set()

                            for step, config in step_configs.items():
                                provider_used = config.get('provider')
                                model_used = config.get('model')

                                if provider_used:
                                    used_providers.add(provider_used)
                                if model_used:
                                    used_models.add(model_used)

                                    # TODO: Update preferred model for this provider in UnifiedProviderConfig
                                    if provider_used and model_used:
                                        # TODO: Implement preferred_models in UnifiedProviderConfig
                                        pass  # Disabled until proper implementation
                                        preferences_updated = True

                            # Set most used provider as preferred
                            if used_providers:
                                most_used_provider = list(used_providers)[0]  # Take first for simplicity
                                if not unified_config.preferred_provider or unified_config.preferred_provider == "ollama":
                                    unified_config.preferred_provider = most_used_provider
                                    preferences_updated = True

                                # Ensure all used providers are in priority list
                                for provider_used in used_providers:
                                    if provider_used not in unified_config.provider_priority:
                                        unified_config.provider_priority.insert(0, provider_used)
                                        preferences_updated = True

                        if preferences_updated:
                            config_manager.save_config()
                            print(f"\n✅ Provider preferences updated and saved:")
                            print(f"   Mode used: {args.mode}")
                            print(f"   Preferred provider: {unified_config.preferred_provider}")
                            # TODO: Show preferred models when implemented in UnifiedProviderConfig
                            # for provider, model in unified_config.preferred_models.items():
                            #     if provider in used_providers:
                            #         print(f"   Preferred model for {provider}: {model}")
                        else:
                            print(f"\n📋 No preference changes needed - current settings already optimal")

                    except Exception as e:
                        logger.warning(f"Failed to save provider preferences: {e}")
                        print(f"\n⚠️ Failed to save preferences: {e}")

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

            # Group providers by type for organized display
            providers_by_type = {
                'ollama': [],
                'openai_compatible': [],
                'gemini': [],
                'anthropic': []
            }

            for provider in config.unified_config.providers:
                providers_by_type[provider.provider_type].append(provider)

            # Display Ollama providers
            if providers_by_type['ollama']:
                print("🚀 Ollama Providers:")
                for provider in providers_by_type['ollama']:
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
            if providers_by_type['openai_compatible']:
                print("🤖 OpenAI-Compatible Providers:")
                for provider in providers_by_type['openai_compatible']:
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
            api_providers = providers_by_type['gemini'] + providers_by_type['anthropic']
            if api_providers:
                print("🎯 API-Only Providers:")
                for provider in api_providers:
                    print(f"  ✅ {provider.name} ({provider.provider_type.title()})")
                    if provider.api_key:
                        print(f"    API Key: {'*' * 8}...")
                    print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                    print(f"    Reachable: Yes (API service) 🌐")
                    if provider.description:
                        print(f"    Description: {provider.description}")

                    if args.show_models:
                        try:
                            models = llm_service.get_available_models(provider.name)
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
            # Get configuration for provider details
            from src.utils.config_manager import ConfigManager as CM
            config_manager = CM()
            config = config_manager.load_config()

            # Initialize LLM service with direct initialization
            llm_service = LlmService(config_manager=config_manager)
            
            total_models = 0
            provider_count = 0
            
            print("🔍 Scanning all providers for available models...\n")
            
            # Check all configured providers using unified configuration
            all_providers = []

            # Use unified configuration for all providers
            for provider in config.unified_config.providers:
                if provider.enabled:
                    # Map provider types to display names
                    provider_type_display = {
                        'ollama': 'Ollama',
                        'openai_compatible': 'OpenAI-Compatible',
                        'gemini': 'Google API',
                        'anthropic': 'Anthropic API'
                    }.get(provider.provider_type, provider.provider_type.title())

                    # Use provider's base_url, fallback to constructed URL for API providers
                    if provider.base_url:
                        base_url = provider.base_url
                    elif provider.provider_type == 'gemini':
                        base_url = 'https://api.google.com'
                    elif provider.provider_type == 'anthropic':
                        base_url = 'https://api.anthropic.com'
                    else:
                        base_url = f"https://api.{provider.provider_type}.com"

                    all_providers.append((provider.name, provider_type_display, base_url))
            
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
            
            catalog_token = args.catalog_token or getattr(catalog_config, "catalog_token", "")
            catalog_search_url = args.catalog_search_url or getattr(catalog_config, "catalog_search_url", "")
            catalog_details_url = args.catalog_details_url or getattr(catalog_config, "catalog_details_url", "")
            
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
                providers = config.unified_config.openai_compatible_providers
                
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
                if config.unified_config.get_provider_by_name(args.name):
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
                config.unified_config.add_provider(new_provider)
                
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
                provider = config.unified_config.get_provider_by_name(args.name)
                if not provider:
                    print(f"❌ Provider '{args.name}' not found.")
                    print("   Use 'alima_cli.py provider list' to see available providers.")
                    return
                
                # Remove provider
                success = config.unified_config.remove_provider(args.name)
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
                provider = config.unified_config.get_provider_by_name(args.name)
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
                provider = config.unified_config.get_provider_by_name(args.name)
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
                    print(f"   Local:    {'✅ Enabled' if config.unified_config.ollama.local_enabled else '❌ Disabled'} ({config.unified_config.ollama.local_host}:{config.unified_config.ollama.local_port})")
                    print(f"   Official: {'✅ Enabled' if config.unified_config.ollama.official_enabled else '❌ Disabled'} ({config.unified_config.ollama.official_base_url})")
                    print(f"   Native:   {'✅ Enabled' if config.unified_config.ollama.native_enabled else '❌ Disabled'} ({config.unified_config.ollama.native_host})")
                    print(f"   Active:   {config.unified_config.ollama.get_active_connection_type()}")
                except Exception as e:
                    print(f"❌ Error loading Ollama configuration: {str(e)}")
            
            elif args.ollama_action == "enable-local":
                try:
                    config = config_manager.load_config()
                    
                    # Update local Ollama settings
                    config.unified_config.ollama.local_enabled = True
                    config.unified_config.ollama.local_host = args.host
                    config.unified_config.ollama.local_port = args.port
                    
                    # Disable other types for clarity
                    config.unified_config.ollama.official_enabled = False
                    config.unified_config.ollama.native_enabled = False
                    
                    config_manager.save_config(config)
                    print(f"✅ Local Ollama enabled: {args.host}:{args.port}")
                    
                except Exception as e:
                    print(f"❌ Error enabling local Ollama: {str(e)}")
            
            elif args.ollama_action == "enable-official":
                try:
                    config = config_manager.load_config()
                    
                    # Update official Ollama settings
                    config.unified_config.ollama.official_enabled = True
                    config.unified_config.ollama.official_base_url = args.base_url
                    config.unified_config.ollama.official_api_key = args.api_key
                    
                    # Disable other types for clarity
                    config.unified_config.ollama.local_enabled = False
                    config.unified_config.ollama.native_enabled = False
                    
                    config_manager.save_config(config)
                    print(f"✅ Official Ollama API enabled: {args.base_url}")
                    print(f"   API Key: {args.api_key[:20]}...")
                    
                except Exception as e:
                    print(f"❌ Error enabling official Ollama: {str(e)}")
            
            elif args.ollama_action == "enable-native":
                try:
                    config = config_manager.load_config()
                    
                    # Update native Ollama settings
                    config.unified_config.ollama.native_enabled = True
                    config.unified_config.ollama.native_host = args.host
                    if args.api_key:
                        config.unified_config.ollama.native_api_key = args.api_key
                    
                    # Disable other types for clarity
                    config.unified_config.ollama.local_enabled = False
                    config.unified_config.ollama.official_enabled = False
                    
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
