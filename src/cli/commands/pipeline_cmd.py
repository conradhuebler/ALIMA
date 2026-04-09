# Pipeline and Batch Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for pipeline-related commands:
    - pipeline: Execute complete ALIMA analysis pipeline
    - batch: Process multiple sources in batch mode
"""

import os
import logging
from typing import Any
from datetime import datetime

from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.utils.pipeline_utils import PipelineJsonManager, PipelineStepExecutor, execute_input_extraction
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.config_manager import ConfigManager
from src.utils.pipeline_config_builder import PipelineConfigBuilder
from src.utils.logging_utils import print_result
from src.utils.batch_processor import BatchProcessor
from src.webapp.result_serialization import (
    build_export_payload,
    extract_results_from_analysis_state,
)


def apply_cli_overrides(pipeline_config, args):
    """Apply CLI argument overrides to a baseline PipelineConfig.

    This function delegates to PipelineConfigBuilder for unified configuration logic.
    """
    builder = PipelineConfigBuilder(ConfigManager())
    builder.baseline = pipeline_config

    # Global settings
    if hasattr(args, 'suggesters') and args.suggesters:
        pipeline_config.search_suggesters = args.suggesters

    # Use the unified builder to apply all CLI overrides
    return PipelineConfigBuilder.parse_and_apply_cli_args(builder, args)


def handle_pipeline(args, config_manager: ConfigManager, llm_service: LlmService,
                   prompt_service: Any, logger: logging.Logger):
    """Handle 'pipeline' command - Execute complete ALIMA analysis pipeline.

    Args:
        args: Parsed command-line arguments
        config_manager: Configuration manager instance
        llm_service: LLM service instance
        prompt_service: Prompt service instance
        logger: Logger instance
    """
    # Setup managers
    alima_manager = AlimaManager(llm_service, prompt_service, config_manager, logger)
    cache_manager = UnifiedKnowledgeManager()

    # Initialize PipelineManager
    pipeline_manager = PipelineManager(
        alima_manager=alima_manager,
        cache_manager=cache_manager,
        logger=logger,
        config_manager=config_manager
    )

    # CLI Callbacks for pipeline events
    def cli_step_started(step):
        provider_info = f"{step.provider}/{step.model}" if step.provider and step.model else "Smart Mode"
        logger.info(f"▶ Starte Schritt: {step.name} ({provider_info})")

    def cli_step_completed(step):
        logger.info(f"✅ Schritt abgeschlossen: {step.name}")

    def cli_step_error(step, error_message):
        logger.error(f"❌ Fehler in Schritt {step.name}: {error_message}")

    def cli_pipeline_completed(analysis_state):
        logger.info("\n🎉 Pipeline vollständig abgeschlossen!")

    def cli_stream_callback(token, step_id):
        print(token, end="", flush=True)

    # Create pipeline config from Provider Preferences as baseline
    try:
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)
        logger.info("Pipeline configuration loaded from Provider Preferences")
    except Exception as e:
        logger.warning(f"Failed to load Provider Preferences, using defaults: {e}")
        pipeline_config = PipelineConfig()

    # Show configuration if requested
    if getattr(args, 'show_config', False):
        print("🔧 Pipeline Configuration:")
        print(f"  Mode: {args.mode}")
        print(f"  Task preferences enabled: {'✅ Yes' if args.mode == 'smart' else '⚠️ Mode-based override active'}")

        builder = PipelineConfigBuilder(config_manager)
        config_for_display = PipelineConfigBuilder.parse_and_apply_cli_args(builder, args)
        step_configs = config_for_display.step_configs
        if step_configs:
            print(f"  CLI step configurations:")
            for step_id, config in step_configs.items():
                parts = []
                if config.provider:
                    parts.append(f"provider={config.provider}")
                if config.model:
                    parts.append(f"model={config.model}")
                if config.task:
                    parts.append(f"task={config.task}")
                if config.temperature is not None:
                    parts.append(f"temp={config.temperature}")
                if config.top_p is not None:
                    parts.append(f"top_p={config.top_p}")
                print(f"    {step_id}: {', '.join(parts) if parts else 'default'}")
        else:
            print(f"  Using default configuration from pipeline_config")
            for step_id, step_config in pipeline_config.step_configs.items():
                if step_config.enabled and step_config.provider:
                    provider = step_config.provider
                    model = step_config.model
                    print(f"    {step_id}: {provider}/{model}")

        print(f"  Save preferences: {'✅ Yes' if getattr(args, 'save_preferences', False) else '❌ No'}")
        print()

    # Get catalog configuration
    catalog_config = config_manager.get_catalog_config()
    catalog_token = args.catalog_token or getattr(catalog_config, "catalog_token", "")
    catalog_search_url = args.catalog_search_url or getattr(catalog_config, "catalog_search_url", "")
    catalog_details_url = args.catalog_details_url or getattr(catalog_config, "catalog_details_url", "")

    try:
        input_type = "text"
        input_source = None

        if args.resume_from:
            # Resume from existing state
            logger.info(f"Resuming pipeline from {args.resume_from}")
            analysis_state = PipelineJsonManager.load_analysis_state(args.resume_from)

            print("--- Resumed Analysis State ---")
            print(f"Original Abstract: {analysis_state.original_abstract[:200]}...")
            print(f"Initial Keywords: {analysis_state.initial_keywords}")
            if analysis_state.final_llm_analysis:
                print(f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}")
            else:
                print("Final analysis not yet completed")
        else:
            # Resolve input text (from --input-text, --doi, or --input-image)
            if args.doi:
                input_type = "doi"
                input_source = args.doi
                logger.info(f"Resolving input: {args.doi}")
                success, input_text, error_msg = resolve_input_to_text(args.doi, logger)
                if not success:
                    logger.error(f"Failed to resolve input: {error_msg}")
                    return
                logger.info(f"Input resolved successfully, content length: {len(input_text)}")
                print(f"Resolved '{args.doi}' to text content ({len(input_text)} chars)")

            elif args.input_image:
                input_type = "img"
                input_source = args.input_image
                # Image OCR analysis
                logger.info(f"Analyzing image: {args.input_image}")

                if not os.path.exists(args.input_image):
                    logger.error(f"Image file not found: {args.input_image}")
                    return

                print(f"🖼️ Analyzing image: {args.input_image}")

                def image_stream_callback(text):
                    print(text, end="", flush=True)

                try:
                    input_text, source_info, extraction_method = execute_input_extraction(
                        llm_service=llm_service,
                        input_source=args.input_image,
                        input_type="image",
                        stream_callback=image_stream_callback,
                        logger=logger
                    )

                    logger.info(f"Image analysis completed: {extraction_method}")
                    print(f"✓ {source_info} ({len(input_text)} characters extracted)")

                    print("\n" + "="*60)
                    print("EXTRAHIERTER TEXT")
                    print("="*60)
                    print(input_text)
                    print("="*60 + "\n")

                except Exception as e:
                    logger.error(f"Image analysis failed: {e}")
                    print(f"❌ Error analyzing image: {e}")
                    return
            else:
                input_type = "text"
                input_source = args.input_text
                input_text = args.input_text

            # Execute complete pipeline
            logger.info("Starting complete pipeline execution with mode-based configuration")
            logger.info(f"Starting pipeline execution in {args.mode} mode using PipelineManager")

            # Apply CLI overrides to baseline configuration
            try:
                updated_pipeline_config = apply_cli_overrides(pipeline_config, args)
                logger.info(f"Pipeline configuration: baseline + CLI overrides applied (mode={args.mode})")
            except Exception as e:
                logger.error(f"Failed to apply CLI overrides: {e}")
                return

            # Apply agentic mode if requested - Claude Generated
            if getattr(args, 'agentic', False):
                updated_pipeline_config.enable_agentic_mode = True
                updated_pipeline_config.agentic_max_iterations = getattr(args, 'agentic_max_iterations', 20)
                updated_pipeline_config.agentic_quality_threshold = getattr(args, 'agentic_quality_threshold', 0.6)
                updated_pipeline_config.agentic_verbose = getattr(args, 'agentic_verbose', False)
                logger.info("🤖 Agentic mode enabled")

                # Workflow configuration - Claude Generated
                if getattr(args, 'workflow', None):
                    updated_pipeline_config.workflow_name = args.workflow
                    logger.info(f"📋 Workflow: {args.workflow}")
                if getattr(args, 'custom_workflow', None):
                    updated_pipeline_config.custom_workflow_path = args.custom_workflow
                    logger.info(f"📋 Custom workflow: {args.custom_workflow}")

                # Single-step execution - Claude Generated
                if getattr(args, 'step', None):
                    updated_pipeline_config.agentic_step_id = args.step
                    logger.info(f"🎯 Single-step mode: {args.step}")
                if getattr(args, 'resume_from', None):
                    updated_pipeline_config.agentic_input_context_path = args.resume_from
                    logger.info(f"📂 Warm-start from: {args.resume_from}")

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
            if hasattr(args, 'force_update') and args.force_update:
                print("⚠️ Force update enabled: catalog cache will be ignored")

            try:
                pipeline_manager.start_pipeline(
                    input_text=input_text,
                    force_update=getattr(args, 'force_update', False)
                )

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

            print_result("\n--- Pipeline Results ---")
            if analysis_state.working_title:
                print_result(f"Working Title: {analysis_state.working_title}")
            print_result(f"Initial Keywords: {analysis_state.initial_keywords}")
            print_result(f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}")
            print_result(f"GND Classes: {analysis_state.final_llm_analysis.extracted_gnd_classes}")

            # Save preferences if requested
            if getattr(args, 'save_preferences', False):
                try:
                    unified_config = config_manager.get_unified_config()
                    preferences_updated = False

                    cli_overrides_used = bool(args.step or args.step_task or args.step_temperature or args.step_top_p or args.step_seed)

                    if not cli_overrides_used:
                        print(f"\n📋 Smart mode baseline used - no preference updates needed")
                    else:
                        # Extract used providers/models from step configurations
                        used_providers = set()
                        used_models = set()

                        for step, config in updated_pipeline_config.step_configs.items():
                            if config.provider:
                                used_providers.add(config.provider)
                            if config.model:
                                used_models.add(config.model)

                        # Set most used provider as preferred
                        if used_providers:
                            most_used_provider = list(used_providers)[0]
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
                        print(f"   Configuration: baseline + CLI overrides")
                        print(f"   Preferred provider: {unified_config.preferred_provider}")
                    else:
                        print(f"\n📋 No preference changes needed - current settings already optimal")

                except Exception as e:
                    logger.warning(f"Failed to save provider preferences: {e}")
                    print(f"\n⚠️ Failed to save preferences: {e}")

        # Save results if requested
        output_file = args.output_json
        if not output_file and analysis_state.working_title:
            output_file = f"{analysis_state.working_title}.json"
            logger.info(f"Auto-generated output filename from working title: {output_file}")
        elif not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"analysis_state_{timestamp}.json"
            logger.info(f"Auto-generated output filename from timestamp: {output_file}")

        if output_file:
            try:
                results = extract_results_from_analysis_state(analysis_state)
                export_payload = build_export_payload(
                    session_id="cli",
                    created_at=getattr(analysis_state, "timestamp", None),
                    status="completed",
                    current_step="classification",
                    input_data={
                        "type": input_type,
                        "source": input_source,
                        "text_preview": (analysis_state.original_abstract or "")[:100],
                    },
                    results=results,
                    autosave_timestamp=None,
                    exported_at=datetime.now().isoformat(),
                    validate_rvk=True,
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    import json
                    json.dump(export_payload, f, ensure_ascii=False, indent=2)
                logger.info(f"Pipeline results saved to {output_file}")
                print(f"\n💾 Results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving pipeline results: {e}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    finally:
        try:
            if getattr(cache_manager, "db_manager", None):
                cache_manager.db_manager.close_connection()
        except Exception as cleanup_error:
            logger.debug(f"CLI cache cleanup failed: {cleanup_error}")


def handle_batch(args, config_manager: ConfigManager, llm_service: LlmService,
                prompt_service: Any, logger: logging.Logger):
    """Handle 'batch' command - Process multiple sources in batch mode.

    Args:
        args: Parsed command-line arguments
        config_manager: Configuration manager instance
        llm_service: LLM service instance
        prompt_service: Prompt service instance
        logger: Logger instance
    """
    # Validate arguments
    if args.batch_file and not args.output_dir and not args.resume:
        logger.error("--output-dir is required when using --batch-file")
        return

    # Siegel expansion: fetch DOIs and write temporary batch file
    if getattr(args, "siegel", None):
        if not args.output_dir:
            logger.error("--output-dir is required when using --siegel")
            return
        from src.utils.k10plus_resolver import fetch_dois_for_siegel
        import tempfile

        def _siegel_progress(current, total, msg):
            logger.info(f"  [{current}/{total}] {msg}")

        logger.info(f"Fetching DOIs for Paketsigel '{args.siegel}' ...")
        try:
            dois = fetch_dois_for_siegel(
                args.siegel,
                cache_dir=getattr(args, "siegel_cache_dir", None),
                progress_callback=_siegel_progress,
                logger=logger,
            )
        except Exception as exc:
            logger.error(f"Failed to fetch DOIs for siegel '{args.siegel}': {exc}")
            return

        logger.info(f"Fetched {len(dois)} DOIs from '{args.siegel}'")
        if not dois:
            logger.warning("No DOIs found for the given Paketsigel – nothing to process.")
            return

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        for doi in dois:
            tmp.write(f"DOI:{doi}\n")
        tmp.close()
        args.batch_file = tmp.name
        logger.info(f"Temporary batch file written: {tmp.name} ({len(dois)} entries)")

    # Setup managers
    alima_manager = AlimaManager(llm_service, prompt_service, config_manager, logger)
    cache_manager = UnifiedKnowledgeManager()

    # Create PipelineStepExecutor for batch processing
    executor = PipelineStepExecutor(
        alima_manager=alima_manager,
        cache_manager=cache_manager,
        logger=logger,
        config_manager=config_manager
    )

    # Determine output directory
    if args.resume:
        from src.utils.batch_processor import BatchState
        try:
            state = BatchState.load(args.resume)
            output_dir = state.output_dir
            logger.info(f"Resuming batch from: {args.resume}")
        except Exception as e:
            logger.error(f"Failed to load resume state: {e}")
            return
    else:
        output_dir = args.output_dir

    # Create BatchProcessor
    batch_processor = BatchProcessor(
        pipeline_executor=executor,
        cache_manager=cache_manager,
        output_dir=output_dir,
        logger=logger,
        continue_on_error=not args.stop_on_error
    )

    # Setup callbacks for progress reporting
    def on_source_start(source, current, total):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{current}/{total}] Processing: {source.source_type.value}")
        logger.info(f"Source: {source.source_value}")
        if source.custom_name:
            logger.info(f"Custom name: {source.custom_name}")
        logger.info(f"{'='*60}")

    def on_source_complete(result):
        if result.success:
            logger.info(f"✅ Completed: {result.output_file}")
        else:
            logger.error(f"❌ Failed: {result.error_message}")

    def on_batch_complete(results):
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Batch Processing Complete!")
        logger.info(f"{'='*60}")

        summary = batch_processor.get_batch_summary()
        logger.info(f"Total sources: {summary['total_sources']}")
        logger.info(f"Processed: {summary['processed']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Output directory: {summary['output_dir']}")

        if summary['failed'] > 0:
            logger.info("\n❌ Failed sources:")
            for fail in batch_processor.batch_state.failed_sources:
                logger.info(f"  - {fail['source']}: {fail['error']}")

    batch_processor.on_source_start = on_source_start
    batch_processor.on_source_complete = on_source_complete
    batch_processor.on_batch_complete = on_batch_complete

    # Build pipeline configuration
    try:
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)
        logger.info("Pipeline configuration loaded from Provider Preferences")
    except Exception as e:
        logger.warning(f"Failed to load Provider Preferences, using defaults: {e}")
        pipeline_config = PipelineConfig()

    # Apply CLI overrides
    pipeline_config = apply_cli_overrides(pipeline_config, args)

    # Convert pipeline_config to dict for batch processor
    pipeline_config_dict = {
        "step_configs": {
            step_id: {
                "provider": step_config.provider,
                "model": step_config.model,
                "task": step_config.task,
                "temperature": step_config.temperature,
                "top_p": step_config.top_p,
                "enabled": step_config.enabled,
            }
            for step_id, step_config in pipeline_config.step_configs.items()
        }
    }

    # Process batch
    try:
        logger.info(f"Starting batch processing...")
        if args.batch_file:
            logger.info(f"Batch file: {args.batch_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Continue on error: {not args.stop_on_error}")

        results = batch_processor.process_batch_file(
            batch_file=args.batch_file if args.batch_file else batch_processor.batch_state.batch_file,
            pipeline_config=pipeline_config_dict,
            resume_state=args.resume
        )

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
    finally:
        try:
            if getattr(cache_manager, "db_manager", None):
                cache_manager.db_manager.close_connection()
        except Exception as cleanup_error:
            logger.debug(f"Batch cache cleanup failed: {cleanup_error}")
        import traceback
        traceback.print_exc()
