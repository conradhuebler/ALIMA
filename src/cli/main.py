# ALIMA CLI Main Entry Point
# Claude Generated - Refactored from alima_cli.py
"""
Main entry point for ALIMA CLI with argument parsing and command dispatching.

This module provides the argument parser and dispatches commands to specialized handlers.
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.utils.config_manager import ConfigManager
from src.utils.logging_utils import setup_logging

# Import command handlers
from src.cli.commands import (
    pipeline_cmd,
    provider_cmd,
    database_cmd,
    search_cmd,
    state_cmd,
    protocol_cmd,
    setup_cmd,
    workflow_cmd,
    agent_cmd,
    bundle_cmd,
    webindex_cmd,
    rules_cmd,
)


# Commands that need neither first-run setup nor prompts.json (infra/inspection only)
_SETUP_EXEMPT_COMMANDS = {
    "setup", "list-models", "list-providers", "test-providers",
    "list-models-detailed", "dnb-import", "clear-cache", "migrate-db",
    "db-config", "workflows", "bundle", "webindex", "rules",
}


def _add_llm_args(parser, *, provider_help="Default LLM provider",
                  model_help="Default LLM model",
                  temperature_help="Default LLM temperature"):
    """Add the shared LLM provider/model/temperature options to a subparser. - Claude Generated"""
    parser.add_argument("--provider", help=provider_help)
    parser.add_argument("--model", help=model_help)
    parser.add_argument("--temperature", type=float, help=temperature_help)


def create_argument_parser():
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=(
            "ALIMA CLI - Automated Library Indexing and Metadata Assignment: "
            "classic and agentic pipelines, batch processing, chat agent, "
            "GND/DK/RVK search, and configuration (providers, database, plugins)."
        )
    )

    # Global logging level argument
    parser.add_argument(
        "--log-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Set logging verbosity: 0=Quiet, 1=Normal (default), 2=Debug, 3=Verbose"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Execute complete ALIMA analysis pipeline with mode-based configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = pipeline_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-text", help="Input text for analysis.")
    input_group.add_argument("--doi", help="DOI or URL to resolve and analyze.")
    input_group.add_argument("--input-image", help="Path to image file for OCR analysis.")

    # Configuration mode
    pipeline_parser.add_argument(
        "--mode",
        choices=["smart", "advanced", "expert"],
        default="smart",
        help="Configuration mode"
    )

    # Step configuration
    pipeline_parser.add_argument("--step", action="append", help="Set provider|model for step: STEP=PROVIDER|MODEL")
    pipeline_parser.add_argument("--step-task", action="append", help="Set prompt task for step: STEP=TASK")
    pipeline_parser.add_argument("--step-temperature", action="append", help="Set temperature for step")
    pipeline_parser.add_argument("--step-top-p", action="append", help="Set top-p for step")
    pipeline_parser.add_argument("--step-seed", action="append", help="Set seed for step")
    pipeline_parser.add_argument("--step-think", action="append", help="Enable/disable thinking mode for step: STEP=true|false")
    pipeline_parser.add_argument("--max-tokens", type=int, default=None,
                                 help="Token budget for all agentic LLM steps; outranks the workflow YAML")

    # Step control
    pipeline_parser.add_argument("--enable-step", help="Comma-separated list of steps to enable")
    pipeline_parser.add_argument("--disable-step", help="Comma-separated list of steps to disable")

    # Iterative refinement
    pipeline_parser.add_argument("--enable-iterative-search", action="store_true", help="Enable iterative GND search")
    pipeline_parser.add_argument("--max-iterations", type=int, default=2, help="Max refinement iterations")

    # Configuration display
    pipeline_parser.add_argument("--show-smart-config", action="store_true", help="Show Smart mode config and exit")
    pipeline_parser.add_argument("--suggesters", nargs="+", default=["lobid", "swb"], help="Search suggesters")
    pipeline_parser.add_argument("--output-json", help="Path to save pipeline results")
    pipeline_parser.add_argument("--title", help="Override generated work title")
    pipeline_parser.add_argument("--resume-from", help="Path to resume pipeline from previous state")
    pipeline_parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL")
    pipeline_parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port")

    pipeline_parser.add_argument("--include-dk-classification", action="store_true", default=True, help="Include DK classification")
    pipeline_parser.add_argument("--disable-dk-classification", action="store_true", help="Disable DK classification")
    pipeline_parser.add_argument("--auto-save-path", help="Path for automatic intermediate saves")

    # DK parameters
    from src.utils.pipeline_defaults import DEFAULT_DK_MAX_RESULTS, DEFAULT_DK_FREQUENCY_THRESHOLD
    pipeline_parser.add_argument("--dk-max-results", type=int, default=DEFAULT_DK_MAX_RESULTS, help=f"Max DK results (default: {DEFAULT_DK_MAX_RESULTS})")
    pipeline_parser.add_argument("--dk-frequency-threshold", type=int, default=DEFAULT_DK_FREQUENCY_THRESHOLD, help=f"Min DK frequency (default: {DEFAULT_DK_FREQUENCY_THRESHOLD})")
    pipeline_parser.add_argument("--force-update", action="store_true", help="Force catalog cache update")

    # Agentic mode - Claude Generated
    pipeline_parser.add_argument("--agentic", action="store_true", help="Use agentic mode with MCP tools instead of sequential pipeline")
    pipeline_parser.add_argument("--agentic-max-iterations", type=int, default=20, help="Max tool-calling iterations per agent (default: 20)")
    pipeline_parser.add_argument("--agentic-quality-threshold", type=float, default=0.6, help="Min quality score per agent (default: 0.6)")
    pipeline_parser.add_argument("--agentic-verbose", action="store_true", help="Log full system+user prompts to stream in agentic mode")

    # Workflow configuration - Claude Generated
    pipeline_parser.add_argument("--workflow", type=str, default=None, help="Workflow name to use (default: SystemConfig.default_workflow, falls back to alima_v51)")
    pipeline_parser.add_argument("--custom-workflow", type=str, default=None, help="Path to custom workflow YAML/JSON file")

    # Single-step execution - Claude Generated
    pipeline_parser.add_argument("--only-step", type=str, default=None,
                                 metavar="STEP_ID",
                                 help="Run only a single agentic step: extraction|search|selection|classification")

    # Global provider/model override - Claude Generated
    pipeline_parser.add_argument("--override", dest="global_override", metavar="PROVIDER/MODEL",
                                 help="Force ALL LLM steps: PROVIDER/MODEL or PROVIDER|MODEL (e.g. gemini/gemini-2.0-flash, openai_compatible/glm-4.6:cloud)")

    # Keyword chunking
    pipeline_parser.add_argument("--keyword-chunking-threshold", type=int, help="Keyword count threshold for chunking")
    pipeline_parser.add_argument("--chunking-task", type=str, help="Task for chunked processing")

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple sources through ALIMA pipeline in batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    batch_input_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input_group.add_argument("--batch-file", help="Path to batch file")
    batch_input_group.add_argument("--resume", help="Path to .batch_state.json to resume")
    batch_input_group.add_argument(
        "--siegel", metavar="SIEGEL_NAME",
        help="K10Plus Paketsigel (z.B. ZDB-2-CMS) – wird zu DOI-Liste expandiert"
    )

    batch_parser.add_argument(
        "--siegel-cache-dir", metavar="DIR",
        help="Cache-Verzeichnis für K10Plus XML-Dateien (default: kein Cache)"
    )

    batch_parser.add_argument("--output-dir", help="Directory for output JSON files")
    batch_parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error")
    batch_parser.add_argument("--continue-on-error", action="store_true", default=True, help="Continue on error (default)")
    batch_parser.add_argument("--mode", choices=["smart", "advanced", "expert"], default="smart", help="Configuration mode")
    batch_parser.add_argument("--step", action="append", help="Set provider|model for step")
    batch_parser.add_argument("--step-task", action="append", help="Set prompt task for step")
    batch_parser.add_argument("--step-temperature", action="append", help="Set temperature for step")
    batch_parser.add_argument("--step-top-p", action="append", help="Set top-p for step")
    batch_parser.add_argument("--step-seed", action="append", help="Set seed for step")
    batch_parser.add_argument("--step-think", action="append", help="Enable/disable thinking mode for step: STEP=true|false")
    batch_parser.add_argument("--max-tokens", type=int, default=None,
                              help="Token budget for all agentic LLM steps; outranks the workflow YAML")
    batch_parser.add_argument("--suggesters", nargs="+", default=["lobid", "swb"], help="Search suggesters")
    batch_parser.add_argument("--disable-dk-classification", action="store_true", help="Disable DK classification")
    batch_parser.add_argument("--override", dest="global_override", metavar="PROVIDER/MODEL",
                              help="Force ALL LLM steps: PROVIDER/MODEL or PROVIDER|MODEL")

    # Show-protocol command
    show_protocol_parser = subparsers.add_parser(
        "show-protocol",
        help="Display pipeline results from JSON protocol file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    show_protocol_parser.add_argument("json_file", help="Path to JSON protocol/results file")
    show_protocol_parser.add_argument("--steps", nargs="+", choices=["all", "input", "initialisation", "search", "keywords", "dk_search", "dk_classification"], default=["all"], help="Steps to display")
    show_protocol_parser.add_argument("--format", choices=["detailed", "compact", "k10plus"], default="detailed", help="Output format")
    show_protocol_parser.add_argument("--header", action="store_true", help="Print CSV header (compact format only)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for keywords using various suggesters.")
    search_parser.add_argument("search_terms", nargs="+", help="The search terms to use.")
    search_parser.add_argument("--suggesters", nargs="+", default=["lobid"], help="Suggesters to use")

    # Save-state command
    save_parser = subparsers.add_parser("save-state", help="Save the last analysis state to JSON (deprecated).")
    save_parser.add_argument("output_file", help="Path to save TaskState JSON")

    # Load-state command
    load_parser = subparsers.add_parser("load-state", help="Load and resume analysis from JSON.")
    load_parser.add_argument("input_file", help="Path to TaskState JSON input file")

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List all available models from all providers.")
    list_parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL")
    list_parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port")

    # Test catalog command
    test_catalog_parser = subparsers.add_parser("test-catalog", help="Test catalog search functionality.")
    test_catalog_parser.add_argument("search_terms", nargs="+", help="Search terms to test")
    test_catalog_parser.add_argument("--max-results", type=int, default=5, help="Max results per term")
    test_catalog_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    test_catalog_parser.add_argument("--catalog-token", help="Override catalog token")
    test_catalog_parser.add_argument("--catalog-search-url", help="Override search URL")
    test_catalog_parser.add_argument("--catalog-details-url", help="Override details URL")

    # Clear cache command
    cache_clear_parser = subparsers.add_parser("clear-cache", help="Clear cache database(s).")
    cache_clear_parser.add_argument("--type", choices=["all", "gnd", "search", "classifications"], default="all", help="Type of cache to clear")
    cache_clear_parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")

    # DNB import command
    dnb_import_parser = subparsers.add_parser("dnb-import", help="Import DNB/GND data with progress.")
    dnb_import_parser.add_argument("--force", action="store_true", help="Force re-download")
    dnb_import_parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # List providers command
    list_providers_parser = subparsers.add_parser("list-providers", help="List all configured LLM providers.")
    list_providers_parser.add_argument("--show-config", action="store_true", help="Show detailed configuration")
    list_providers_parser.add_argument("--show-models", action="store_true", help="Show available models")

    # Test providers command
    test_providers_parser = subparsers.add_parser("test-providers", help="Test connection to all providers.")
    test_providers_parser.add_argument("--timeout", type=float, default=5.0, help="Connection timeout (seconds)")
    test_providers_parser.add_argument("--show-models", action="store_true", help="Show models for reachable providers")

    # List models detailed command
    list_models_detailed_parser = subparsers.add_parser("list-models-detailed", help="List models with detailed info.")

    # Database config command
    db_config_parser = subparsers.add_parser("db-config", help="Database configuration management.")
    db_subparsers = db_config_parser.add_subparsers(dest="db_action", help="Database config actions")

    db_show_parser = db_subparsers.add_parser("show", help="Show current database configuration")
    db_paths_parser = db_subparsers.add_parser("paths", help="Show OS-specific configuration paths")
    db_test_parser = db_subparsers.add_parser("test", help="Test database connection")

    db_sqlite_parser = db_subparsers.add_parser("set-sqlite", help="Configure SQLite database")
    db_sqlite_parser.add_argument("--path", default="alima_knowledge.db", help="SQLite database path")
    db_sqlite_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", help="Configuration scope")

    db_mysql_parser = db_subparsers.add_parser("set-mysql", help="Configure MySQL/MariaDB database")
    db_mysql_parser.add_argument("--host", required=True, help="Database host")
    db_mysql_parser.add_argument("--port", type=int, default=3306, help="Database port")
    db_mysql_parser.add_argument("--database", required=True, help="Database name")
    db_mysql_parser.add_argument("--username", required=True, help="Database username")
    db_mysql_parser.add_argument("--password", help="Database password (prompted if not provided)")
    db_mysql_parser.add_argument("--charset", default="utf8mb4", help="Character set")
    db_mysql_parser.add_argument("--ssl-disabled", action="store_true", help="Disable SSL connection")
    db_mysql_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", help="Configuration scope")

    # Provider management command
    provider_parser = subparsers.add_parser("provider", help="OpenAI-compatible provider management.")
    provider_subparsers = provider_parser.add_subparsers(dest="provider_action", help="Provider management actions")

    provider_list_parser = provider_subparsers.add_parser("list", help="List all configured providers")
    provider_list_parser.add_argument("--enabled-only", action="store_true", help="Show only enabled providers")

    provider_add_parser = provider_subparsers.add_parser("add", help="Add new OpenAI-compatible provider")
    provider_add_parser.add_argument("--name", required=True, help="Provider name")
    provider_add_parser.add_argument("--base-url", required=True, help="API base URL")
    provider_add_parser.add_argument("--api-key", required=True, help="API key")
    provider_add_parser.add_argument("--description", default="", help="Provider description")
    provider_add_parser.add_argument("--enabled", action="store_true", default=True, help="Enable provider")
    provider_add_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", help="Configuration scope")

    provider_remove_parser = provider_subparsers.add_parser("remove", help="Remove provider")
    provider_remove_parser.add_argument("--name", required=True, help="Provider name to remove")
    provider_remove_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", help="Configuration scope")

    provider_edit_parser = provider_subparsers.add_parser("edit", help="Edit existing provider")
    provider_edit_parser.add_argument("--name", required=True, help="Provider name to edit")
    provider_edit_parser.add_argument("--new-name", help="New provider name")
    provider_edit_parser.add_argument("--base-url", help="New API base URL")
    provider_edit_parser.add_argument("--api-key", help="New API key")
    provider_edit_parser.add_argument("--description", help="New provider description")
    provider_edit_parser.add_argument("--enable", action="store_true", help="Enable provider")
    provider_edit_parser.add_argument("--disable", action="store_true", help="Disable provider")
    provider_edit_parser.add_argument("--scope", choices=["user", "project", "system"], default="user", help="Configuration scope")

    provider_test_parser = provider_subparsers.add_parser("test", help="Test provider connection")
    provider_test_parser.add_argument("--name", required=True, help="Provider name to test")

    # Ollama provider commands
    ollama_parser = provider_subparsers.add_parser("ollama", help="Manage Ollama provider instances.")
    ollama_subparsers = ollama_parser.add_subparsers(dest="ollama_action", help="Ollama provider actions")

    ollama_list_parser = ollama_subparsers.add_parser("list", help="List all Ollama providers")
    ollama_list_parser.add_argument("--enabled-only", action="store_true", help="Show only enabled providers")

    ollama_status_parser = ollama_subparsers.add_parser("status", help="Show current Ollama configuration")

    ollama_add_parser = ollama_subparsers.add_parser("add", help="Add a native Ollama provider")
    ollama_add_parser.add_argument("--name", required=True, help="Provider name (e.g. 'ollama_local')")
    ollama_add_parser.add_argument("--host", required=True, help="Ollama server host/URL (e.g. 'http://localhost' or 'http://192.168.1.10')")
    ollama_add_parser.add_argument("--port", type=int, default=11434, help="Ollama port (default: 11434)")
    ollama_add_parser.add_argument("--api-key", default="", help="Optional API key")
    ollama_add_parser.add_argument("--description", default="", help="Optional description")

    ollama_remove_parser = ollama_subparsers.add_parser("remove", help="Remove a native Ollama provider")
    ollama_remove_parser.add_argument("--name", required=True, help="Provider name to remove")

    # Migrate database command
    migrate_parser = subparsers.add_parser("migrate-db", help="Database migration and backup operations.")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_action", help="Migration actions")

    migrate_export_parser = migrate_subparsers.add_parser("export", help="Export database to JSON backup")
    migrate_export_parser.add_argument("--output", required=True, help="Output JSON file path")
    migrate_export_parser.add_argument("--show-info", action="store_true", help="Show export stats without exporting")

    migrate_import_parser = migrate_subparsers.add_parser("import", help="Import database from JSON backup")
    migrate_import_parser.add_argument("--input", required=True, help="Input JSON file path")
    migrate_import_parser.add_argument("--clear", action="store_true", help="Clear destination before import")
    migrate_import_parser.add_argument("--dry-run", action="store_true", help="Validate without importing")

    # Institutional bundle commands (plugins + advisory config profile) - Claude Generated
    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Deploy institutional bundles (plugins + advisory config profile).",
    )
    bundle_sub = bundle_parser.add_subparsers(dest="bundle_action", help="Bundle actions")
    bundle_build = bundle_sub.add_parser("build", help="Validate a bundle dir + write approvals.json (admin).")
    bundle_build.add_argument("source", help="Path to the bundle directory")
    bundle_build.add_argument("-o", "--output", help="Write a .zip artifact instead of building in place")
    bundle_install = bundle_sub.add_parser("install", help="Install a bundle (directory or .zip) into this config.")
    bundle_install.add_argument("path", help="Path to the bundle directory or .zip")
    bundle_sub.add_parser("list", help="List installed bundles.")
    bundle_remove = bundle_sub.add_parser("remove", help="Remove an installed bundle by id.")
    bundle_remove.add_argument("id", help="Installed bundle id")
    bundle_export = bundle_sub.add_parser("export", help="Export the current setup as a bundle (admin).")
    bundle_export.add_argument("id", help="Bundle id for the export (e.g. ub-freiberg)")
    bundle_export.add_argument("-o", "--output", required=True, help="Output directory or .zip path")
    bundle_export.add_argument("--version", default="1.0", help="Bundle version (default 1.0)")
    bundle_export.add_argument("--label", default="", help="Human-readable label")
    bundle_export.add_argument("--institution", default="", help="Institution name")
    bundle_export.add_argument("--plugin", action="append", dest="plugins", metavar="INSTANCE_ID",
                               help="Only export this instance id (repeatable). Default: all enabled.")

    # Workflow v4 commands (generic agents + deterministic steps) - Claude Generated
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Run a v4 workflow (YAML-driven agent pipeline) by name",
    )
    workflow_parser.add_argument("name", help="Workflow name (e.g. alima_classic) or path to YAML")
    wf_input = workflow_parser.add_mutually_exclusive_group()
    wf_input.add_argument("--input", help="JSON string with workflow input")
    wf_input.add_argument("--input-file", help="Path to JSON file with workflow input")
    workflow_parser.add_argument("--output", help="Path to write JSON report")
    _add_llm_args(workflow_parser)
    workflow_parser.add_argument("--only-step", help="Run only the named step")
    workflow_parser.add_argument("--quiet", action="store_true", help="Suppress streaming output")

    workflows_parser = subparsers.add_parser(
        "workflows",
        help="Inspect v4 workflows (`alima workflows list`)",
    )
    workflows_sub = workflows_parser.add_subparsers(dest="workflows_action")
    workflows_sub.add_parser("list", help="List all discovered workflow YAMLs")

    # Headless chat-agent command (P-ι) - Claude Generated
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run the chat-agent headless (same toolset as the GUI chat)",
    )
    agent_input = agent_parser.add_mutually_exclusive_group()
    agent_input.add_argument("--doi", help="DOI or URL to resolve into the work's text")
    agent_input.add_argument("--input", help="Inline abstract/text of the work")
    agent_input.add_argument("--input-file", help="Path to a text file with the work")
    agent_input.add_argument("--input-image", help="Path to an image to OCR via Vision-LLM")
    agent_parser.add_argument("--prompt", help="Instruction for the agent (default: analyse the work)")
    _add_llm_args(
        agent_parser,
        provider_help="LLM provider (falls back to ChatConfig default)",
        model_help="LLM model (falls back to ChatConfig default)",
        temperature_help="Sampling temperature",
    )
    agent_parser.add_argument("--max-iterations", type=int, dest="max_iterations",
                              help="Max tool-calling iterations")
    agent_parser.add_argument("--autonomous", action="store_true",
                              help="Skip y/N confirmation for mutations/pipeline starts")
    agent_parser.add_argument("--output", help="Path to write the JSON result (default: stdout)")
    agent_parser.add_argument("--quiet", action="store_true", help="Suppress streamed tokens on stdout")
    agent_parser.add_argument("--mode", choices=["verschlagwortung", "suche", "general", "auto"],
                              default="auto", help="Agent mode: verschlagwortung (cataloging), suche (search), general, auto (detect)")

    # Personal indexing rules (persönliche Zusatzregeln) - Claude Generated
    rules_parser = subparsers.add_parser(
        "rules",
        help="Manage personal indexing rules injected into agentic prompts",
    )
    rules_parser.add_argument(
        "--rules-file", dest="rules_file",
        help="Rules file to operate on (default: ~/.config/alima/rules.yaml)",
    )
    rules_sub = rules_parser.add_subparsers(dest="rules_action")

    rules_list = rules_sub.add_parser("list", help="List stored rules")
    rules_list.add_argument("--enabled-only", action="store_true", dest="enabled_only",
                            help="Only show active rules")

    rules_show = rules_sub.add_parser("show", help="Show one rule with full provenance")
    rules_show.add_argument("rule_id", help="Rule id, e.g. r-20260903-01")

    rules_add = rules_sub.add_parser("add", help="Add a rule")
    rules_add.add_argument("text", help="The rule as one clear instruction")
    rules_add.add_argument("--when", help="Prose condition, e.g. 'bei Überblickswerken'")
    rules_add.add_argument("--workflows", nargs="+",
                           help="Workflow name globs (default: all)")
    rules_add.add_argument("--steps", nargs="+",
                           help="Step id globs, e.g. selection* classification (default: all)")
    rules_add.add_argument("--note", help="Why this rule — stored as provenance")
    rules_add.add_argument("--author", help="Who formulated it — stored as provenance")
    rules_add.add_argument("--inactive", action="store_true",
                           help="Store the rule without activating it")

    rules_enable = rules_sub.add_parser("enable", help="Activate a rule")
    rules_enable.add_argument("rule_id")
    rules_disable = rules_sub.add_parser("disable", help="Silence a rule")
    rules_disable.add_argument("rule_id")

    rules_remove = rules_sub.add_parser("remove", help="Delete a rule permanently")
    rules_remove.add_argument("rule_id")
    rules_remove.add_argument("-y", "--yes", action="store_true",
                              help="Delete without asking back")

    rules_export = rules_sub.add_parser(
        "export", help="Write rules to a shareable file (provenance kept verbatim)")
    rules_export.add_argument("--out", help="Target file (default: alima_rules.yaml)")
    rules_export.add_argument("--ids", nargs="+", help="Only these rule ids")
    rules_export.add_argument("--enabled-only", action="store_true", dest="enabled_only",
                              help="Only export active rules")

    rules_import = rules_sub.add_parser(
        "import", help="Merge a rules file; imported rules land inactive")
    rules_import.add_argument("source", help="File to read")
    rules_import.add_argument("--activate", action="store_true",
                              help="Activate the imported rules immediately")

    # Setup wizard command
    setup_parser = subparsers.add_parser("setup", help="Run ALIMA first-start setup wizard")
    setup_parser.add_argument("--skip-gnd", action="store_true", help="Skip GND database download option")
    setup_parser.add_argument("--force", action="store_true", help="Force setup wizard even if config exists")

    # Webindex (website-RAG) commands - Claude Generated
    webindex_parser = subparsers.add_parser(
        "webindex",
        help="Crawl a website into the keyword index + retrieval (RAG chatbot source).",
    )
    webindex_sub = webindex_parser.add_subparsers(dest="webindex_action", help="Webindex actions")
    wi_crawl = webindex_sub.add_parser("crawl", help="Crawl a base URL into the index DB.")
    wi_crawl.add_argument("--base-url", help="Root URL to crawl (children only; overrides instance base_url)")
    wi_crawl.add_argument("--instance", help="webindex plugin instance id whose settings to use")
    wi_crawl.add_argument("--db-path", help="Override the webindex DB path")
    wi_crawl.add_argument("--max-depth", type=int, default=2, help="BFS depth (0 = root only)")
    wi_crawl.add_argument("--max-pages", type=int, default=50, help="Max pages to visit")
    wi_crawl.add_argument("--include", help="Regex a URL must match to be indexed")
    wi_crawl.add_argument("--exclude", help="Regex a URL must NOT match to be indexed")
    wi_crawl.add_argument("--timeout", type=int, default=20, help="Per-fetch timeout (s)")
    wi_crawl.add_argument("--user-agent", default="ALIMA-webindex", help="User-Agent string")
    wi_crawl.add_argument("--min-chars", type=int, default=50, help="Min extracted text chars to index a page")
    wi_crawl.add_argument("--max-keywords", type=int, default=15, help="Max LLM-extracted keywords per page")
    wi_crawl.add_argument("--provider", help="LLM provider for keyword extraction (needs --model)")
    wi_crawl.add_argument("--model", help="LLM model for keyword extraction (needs --provider)")
    wi_crawl.add_argument("--dry-run", action="store_true", help="Discover URLs without writing to the DB")
    wi_stats = webindex_sub.add_parser("stats", help="Show index stats (pages/keywords/last fetch).")
    wi_stats.add_argument("--instance", help="webindex plugin instance id")
    wi_stats.add_argument("--db-path", help="Override the webindex DB path")
    wi_kw = webindex_sub.add_parser("list-keywords", help="Browse the central keyword catalogue.")
    wi_kw.add_argument("--instance", help="webindex plugin instance id")
    wi_kw.add_argument("--db-path", help="Override the webindex DB path")
    wi_kw.add_argument("--limit", type=int, default=200, help="Max keywords to list")
    wi_kw.add_argument("--contains", default="", help="Case-insensitive substring filter")
    wi_search = webindex_sub.add_parser("search", help="Run a retrieval query against the index.")
    wi_search.add_argument("query", help="Natural-language question / keywords")
    wi_search.add_argument("--instance", help="webindex plugin instance id")
    wi_search.add_argument("--db-path", help="Override the webindex DB path")
    wi_search.add_argument("--max-results", type=int, default=0, help="Max hits (0 = instance default)")

    return parser


def main():
    """Main function for ALIMA CLI.

    Parses command-line arguments and dispatches to appropriate command handler.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup centralized logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Operational notices (e.g. an LLM rate-limit wait) are broadcast on the
    # state bus; the CLI has no log surface, so mirror them to stderr. No Qt
    # loop runs here → deliver synchronously. - Claude Generated
    try:
        from src.core.state_bus import AlimaStateBus, set_direct_dispatch

        set_direct_dispatch(True)
        AlimaStateBus().subscribe(
            "state.notice",
            lambda p: print((p or {}).get("text", ""), file=sys.stderr, flush=True),
        )
    except Exception:
        logger.debug("state.notice CLI subscription failed", exc_info=True)

    # First-run setup + prompts.json checks (skipped for infra/inspection commands).
    # Single config load shared by both checks. - Claude Generated
    if args.command not in _SETUP_EXEMPT_COMMANDS:
        config_manager = ConfigManager()
        config = config_manager.load_config()

        if not config.system_config.first_run_completed and not config.system_config.skip_first_run_check:
            logger.info("First-run setup required. Run: python alima_cli.py setup")
            print("\n❌ ALIMA requires setup before use.")
            print("   Run: python alima_cli.py setup")
            print("\nOr set 'skip_first_run_check: true' in config.json to disable this check.\n")
            return

        prompts_file_path = config.system_config.prompts_path
        if not os.path.exists(prompts_file_path):
            logger.error(f"Prompts file not found at: {prompts_file_path}")
            logger.error("Please check your config.json or create prompts.json in the project directory.")
            return

    # Initialize shared services for commands that need them
    config_manager = None
    llm_service = None
    prompt_service = None

    if args.command in ["pipeline", "batch", "workflow", "agent"]:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        prompts_path = config.system_config.prompts_path

        llm_service = LlmService(
            providers=None,
            config_manager=config_manager,
            ollama_url=getattr(args, 'ollama_host', 'http://localhost'),
            ollama_port=getattr(args, 'ollama_port', 11434),
        )
        prompt_service = PromptService(prompts_path, logger)

    # Dispatch to command handler
    if args.command == "setup":
        setup_cmd.handle_setup(args, logger)
    elif args.command == "pipeline":
        pipeline_cmd.handle_pipeline(args, config_manager, llm_service, prompt_service, logger)
    elif args.command == "batch":
        pipeline_cmd.handle_batch(args, config_manager, llm_service, prompt_service, logger)
    elif args.command == "show-protocol":
        protocol_cmd.handle_show_protocol(args)
    elif args.command == "search":
        search_cmd.handle_search(args, logger)
    elif args.command == "test-catalog":
        search_cmd.handle_test_catalog(args, logger)
    elif args.command == "load-state":
        state_cmd.handle_load_state(args, logger)
    elif args.command == "save-state":
        state_cmd.handle_save_state(args, logger)
    elif args.command == "list-models":
        provider_cmd.handle_list_models(args, logger)
    elif args.command == "list-providers":
        provider_cmd.handle_list_providers(args, logger)
    elif args.command == "test-providers":
        provider_cmd.handle_test_providers(args, logger)
    elif args.command == "list-models-detailed":
        provider_cmd.handle_list_models_detailed(args, logger)
    elif args.command == "provider":
        provider_cmd.handle_provider(args, logger)
    elif args.command == "db-config":
        database_cmd.handle_db_config(args, logger)
    elif args.command == "migrate-db":
        database_cmd.handle_migrate_db(args, logger)
    elif args.command == "clear-cache":
        database_cmd.handle_clear_cache(args, logger)
    elif args.command == "dnb-import":
        database_cmd.handle_dnb_import(args, logger)
    elif args.command == "agent":
        sys.exit(agent_cmd.handle_agent(args, config_manager, llm_service, prompt_service, logger))
    elif args.command == "rules":
        sys.exit(rules_cmd.handle_rules(args, logger))
    elif args.command == "workflow":
        sys.exit(workflow_cmd.handle_workflow(args, config_manager, llm_service, logger))
    elif args.command == "workflows":
        if args.workflows_action == "list":
            sys.exit(workflow_cmd.handle_workflows_list(args, logger))
        else:
            parser.print_help()
    elif args.command == "bundle":
        sys.exit(bundle_cmd.handle_bundle(args, logger))
    elif args.command == "webindex":
        sys.exit(webindex_cmd.handle_webindex(args, logger))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
