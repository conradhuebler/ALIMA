"""Pipeline analysis + input-extraction endpoints for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. Holds the heavyweight
``POST /api/analyze/{id}`` and ``POST /api/input/{id}`` routes plus their
background workers ``run_analysis`` / ``run_input_extraction``. Tests patch
``src.webapp.routers.analysis.{PipelineManager, resolve_input_to_text,
AppContext, _autosave_session_state}`` (resolved in this module's namespace).
"""

import asyncio
import re
import tempfile
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.utils.doi_resolver import (
    UnifiedResolver,
    _get_doi_config,
    format_doi_metadata,
    resolve_input_to_text,
)
from src.utils.error_visibility import log_caught
from src.utils.pipeline_formatters import render_pipeline_result
from src.utils.pipeline_utils import PipelineResultFormatter
from src.webapp.render_bridge import _SessionBusSubscriber, _build_session_renderer
from src.webapp.result_serialization import (
    extract_results_from_analysis_state as _extract_results_from_analysis_state,
    prepare_results_for_export as _prepare_results_for_export,
)
from src.webapp.session_io import (
    _autosave_session_state,
    _parse_max_tokens_override,
    _parse_think_override,
)
from src.webapp.session_state import AppContext, Session, sessions

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/analyze/{session_id}")
async def start_analysis(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
    global_override: Optional[str] = Form(None),  # "provider|model" override - Claude Generated
    think_override: Optional[str] = Form(None),  # "default"|"on"|"off" thinking override - Claude Generated
    max_tokens_override: Optional[str] = Form(None),  # token budget for all agentic LLM steps - Claude Generated
    source_type: Optional[str] = Form(None),   # Original source type for filename metadata - Claude Generated
    source_value: Optional[str] = Form(None),  # DOI/URL/filename for working title - Claude Generated
    workflow: Optional[str] = Form(None),  # Workflow stem or __classic__ - Claude Generated
) -> dict:
    """Start pipeline analysis - Direct execution with LLM queueing - Claude Generated (2026-01-13)"""
    # Auto-create session if not exists (for /webapp route with injected sessionId) - Claude Generated (2026-01-13)
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /api/analyze")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.input_data = {"type": input_type, "content": content}

    # READ FILE CONTENTS IMMEDIATELY before creating background task - Claude Generated (Defensive)
    # This prevents "read of closed file" error that occurs when UploadFile is passed to background task
    file_contents = None
    filename = None
    if file:
        try:
            file_contents = await file.read()
            filename = file.filename
            if not file_contents:
                raise HTTPException(status_code=400, detail="File is empty")
            logger.info(f"File read successfully: {len(file_contents)} bytes")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    session.status = "running"
    if workflow:
        session.workflow_name = workflow
        logger.info(f"Session {session_id} workflow set to: {workflow}")

    # Start analysis in background with file contents, not the UploadFile object
    asyncio.create_task(run_analysis(session_id, input_type, content, file_contents, filename, global_override, source_type, source_value, workflow, think_override, max_tokens_override))

    return {"session_id": session_id, "status": "started"}


@router.post("/api/input/{session_id}")
async def process_input_only(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
) -> dict:
    """Process only the input step (text extraction/OCR) - Claude Generated"""

    # Auto-create session if not exists (for /webapp route with injected sessionId) - Claude Generated (2026-01-13)
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /api/input")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.input_data = {"type": input_type, "content": content}

    # READ FILE CONTENTS IMMEDIATELY before creating background task - Claude Generated (Defensive)
    file_contents = None
    if file:
        try:
            file_contents = await file.read()
            if not file_contents:
                raise HTTPException(status_code=400, detail="File is empty")
            logger.info(f"File read successfully: {len(file_contents)} bytes")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    session.status = "running"

    # Start input-only processing in background - Claude Generated
    asyncio.create_task(run_input_extraction(session_id, input_type, content, file_contents, file.filename if file else None))

    return {"session_id": session_id, "status": "started", "mode": "input_extraction"}


async def run_analysis(
    session_id: str,
    input_type: str,
    content: Optional[str],
    file_contents: Optional[bytes],
    filename: Optional[str],
    global_override: Optional[str] = None,
    source_type: Optional[str] = None,   # Original source type for working title - Claude Generated
    source_value: Optional[str] = None,  # DOI/URL/filename for working title - Claude Generated
    workflow: Optional[str] = None,  # Workflow stem or __classic__ - Claude Generated
    think_override: Optional[str] = None,  # "default"|"on"|"off" thinking override - Claude Generated
    max_tokens_override: Optional[str] = None,  # token budget for all agentic LLM steps - Claude Generated
):
    """Execute pipeline analysis with direct PipelineManager - Claude Generated"""

    session = sessions[session_id]
    session.status = "running"

    # WP12: single shared producer for the DK/GND chrome (same renderer the GUI
    # uses); events are buffered on the session and broadcast over the WebSocket.
    session_renderer = _build_session_renderer(session)
    # Phase 4: bridge AlimaStateBus events (tool calls, pipeline steps/prompts)
    # into the same render buffer. Subscriber is local to this run.
    bus_subscriber = _SessionBusSubscriber(session_renderer)

    try:
        bus_subscriber.subscribe()
        # Resolve input to text - Claude Generated
        input_text = None

        if input_type == "text" and content:
            input_text = content
        elif input_type == "doi" and content:
            logger.info(f"Resolving DOI: {content}")
            def _resolve_doi():
                cfg = _get_doi_config()
                resolver = UnifiedResolver(logger,
                    contact_email=cfg['contact_email'],
                    use_crossref=cfg['use_crossref'],
                    use_openalex=cfg['use_openalex'],
                    use_datacite=cfg['use_datacite'],
                )
                success, metadata, text_result = resolver.resolve(content)
                return format_doi_metadata(metadata, text_result or "") if success else None
            input_text = await asyncio.to_thread(_resolve_doi)
        elif input_type == "pdf" and file_contents:
            # Save and extract from PDF - Claude Generated (File contents already read)
            try:
                suffix = ".pdf" if filename and filename.endswith(".pdf") else ".pdf"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                logger.info(f"Extracting text from PDF: {temp_file.name} ({len(file_contents)} bytes)")
                input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
            except Exception as e:
                logger.error(f"PDF processing error: {e}")
                raise
        elif input_type == "img" and file_contents:
            # Save and extract from image - Claude Generated (File contents already read)
            try:
                # Determine extension from filename or default to jpg
                suffix = ""
                if filename:
                    if filename.lower().endswith(".png"):
                        suffix = ".png"
                    elif filename.lower().endswith(".jpeg"):
                        suffix = ".jpeg"
                    else:
                        suffix = ".jpg"
                else:
                    suffix = ".jpg"

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                logger.info(f"Analyzing image: {temp_file.name} ({len(file_contents)} bytes)")
                input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                raise
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if not input_text:
            raise ValueError("Could not extract text from input")

        logger.info(f"Input text extracted ({len(input_text)} chars)")
        session.input_data = {"type": input_type, "text_preview": input_text[:100]}

        # Get or initialize services (singleton pattern - Claude Generated)
        app_context = AppContext()
        services = app_context.get_services()

        config_manager = services['config_manager']

        # Create a NEW PipelineManager for this session to prevent cross-session contamination - Claude Generated (2026-01-13)
        # This ensures each concurrent analysis has its own isolated pipeline state
        pipeline_manager = PipelineManager(
            alima_manager=services['alima_manager'],
            cache_manager=services['cache_manager'],
            config_manager=config_manager
        )
        logger.info(f"Created new PipelineManager for session {session_id}")

        # Create pipeline config from preferences
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)

        # Apply global override if provided - Claude Generated
        think_val = _parse_think_override(think_override)
        if global_override or think_val is not None:
            if global_override:
                provider, model = PipelineConfig.parse_override_string(global_override)
                pipeline_config.global_provider_override = provider
                pipeline_config.global_model_override = model
            pipeline_config.global_think_override = think_val
            pipeline_config.apply_global_override()
            logger.info(
                f"🔬 Webapp global override applied: "
                f"{pipeline_config.global_provider_override}/{pipeline_config.global_model_override} "
                f"think={think_val}"
            )

        # Token budget: independent of the provider/think override — it travels
        # to the agentic steps through the config, not through
        # apply_global_override(). - Claude Generated
        budget_val = _parse_max_tokens_override(max_tokens_override)
        if budget_val:
            pipeline_config.global_max_tokens_override = budget_val
            logger.info(f"🔬 Webapp token budget: max_tokens={budget_val}")

        # Set config on pipeline manager (was missing - config was built but never applied)
        pipeline_manager.set_config(pipeline_config)

        # Remember the effective provider/model so the session chat agent can fall
        # back to the same credentials after the pipeline manager is discarded.
        eff_provider = pipeline_config.global_provider_override
        eff_model = pipeline_config.global_model_override
        if not eff_provider:
            init_cfg = pipeline_config.step_configs.get("initialisation")
            if init_cfg is not None:
                eff_provider = getattr(init_cfg, "provider", None) or ""
                eff_model = getattr(init_cfg, "model", None) or ""
        session.last_provider = eff_provider or None
        session.last_model = eff_model or None

        # Configure agentic mode when a non-classic workflow is requested - Claude Generated
        if workflow and workflow != "__classic__":
            pipeline_config.enable_agentic_mode = True
            pipeline_config.workflow_name = workflow
            logger.info(f"🧬 Agentic mode enabled for workflow: {workflow}")
        else:
            pipeline_config.enable_agentic_mode = False
            pipeline_config.workflow_name = None
            logger.info("🔒 Classic (non-agentic) pipeline mode selected")

        # Chat-UX 5/9: classic-pipeline LLM tokens stream into the shared #log
        # as stream blocks (identical chrome to the GUI). Agentic runs keep the
        # buffer-only path — their chrome comes from the bus bridge, and the
        # same stream_callback fires for both modes, so this MUST stay gated
        # or agentic runs would render every step twice. - Claude Generated
        is_classic = not pipeline_config.enable_agentic_mode
        stream_state = {"step": None}

        def _close_stream_block():
            if stream_state["step"] is not None:
                try:
                    session_renderer.end_streaming_line()
                except Exception:
                    logger.debug("end_streaming_line failed", exc_info=True)
                stream_state["step"] = None

        # Define callbacks for live updates - Claude Generated
        def on_step_started(step):
            session.current_step = step.step_id
            session.current_step_status = 'running'  # Claude Generated
            logger.info(f"Step started: {step.step_id}")

        def on_step_completed(step):
            _close_stream_block()
            session.current_step = step.step_id
            session.current_step_status = 'completed'  # Claude Generated
            logger.info(f"Step completed: {step.step_id}")

            # WP12: emit the DK/RVK catalog-research card as a render event so
            # the webapp shows the identical chrome the GUI does.
            if step.step_id == "dk_search" and step.output_data:
                try:
                    html, plain = PipelineResultFormatter.format_dk_search_card_html(
                        step.output_data
                    )
                    if html:
                        session_renderer.render_html_block(
                            html, kind="dk_search", plain_text=plain
                        )
                except Exception:
                    logger.exception("WP12: dk_search card emission failed")

            # Sync analysis state reference so autosave has access - Claude Generated
            # Must be set here because start_pipeline() hasn't returned yet when callbacks fire
            session.current_analysis_state = pipeline_manager.current_analysis_state

            # Update working title after initialisation - Claude Generated
            if step.step_id == "initialisation":
                if pipeline_manager.current_analysis_state and hasattr(pipeline_manager.current_analysis_state, 'working_title'):
                    wt = pipeline_manager.current_analysis_state.working_title
                    logger.debug(f"Working title from analysis state: '{wt}'")
                    session.working_title = wt
                    if not session.results:
                        session.results = {}
                    session.results['working_title'] = wt
                    logger.info(f"Session working title set: {wt}")
                else:
                    logger.warning("No working_title available after initialisation")

            # Add delay after LLM steps to allow WebSocket to fetch buffered tokens - Claude Generated
            llm_steps = ["initialisation", "keywords", "dk_classification"]
            if step.step_id in llm_steps:
                import time
                time.sleep(0.7)  # 700ms = 500ms poll + 200ms margin
                logger.debug(f"Waited 700ms for streaming token transmission after {step.step_id}")

            # Auto-save after each step completion - Claude Generated
            if session.autosave_enabled:
                try:
                    _autosave_session_state(session)
                except Exception as e:
                    logger.error(f"Auto-save error (continuing): {e}")

        def on_step_error(step, error_msg):
            _close_stream_block()
            session.current_step = step.step_id
            session.error_message = error_msg
            logger.error(f"Step error: {step.step_id}: {error_msg}")

        def on_agentic_context(step_id, snapshot):
            """Mirror agentic step progress into the session for the frontend
            pipeline-stepper. Agentic workflows don't use step_started_callback;
            they report per-step completion via context snapshots (running
            snapshots are skipped upstream). - Claude Generated"""
            try:
                sid = step_id or (snapshot or {}).get("_step_id")
                if not sid:
                    return
                session.current_step = sid
                snap_status = (snapshot or {}).get("_step_status") or "completed"
                session.current_step_status = (
                    "error" if snap_status == "error" else "completed"
                )
            except Exception:
                logger.debug("agentic context step update failed", exc_info=True)

        def on_pipeline_completed(analysis_state):
            _close_stream_block()
            logger.info(f"Pipeline completed, storing results")

            # Sync analysis state reference so autosave has access - Claude Generated
            session.current_analysis_state = analysis_state

            # Shared result emission (GUI parity): completion line, final GND
            # keywords, Schlagwortketten, DK/Auswertung cards, workflow
            # report_markdown — buffered before status flips to "completed",
            # so the WS picks it up in the final message. - Claude Generated
            try:
                render_pipeline_result(session_renderer, analysis_state)
            except Exception as e:
                log_caught(logger, e, "pipeline result emission (render events)")

            # Use shared extraction helper (DRY principle) - Claude Generated
            session.results = _prepare_results_for_export(
                _extract_results_from_analysis_state(analysis_state),
                validate_rvk=True,
            )

            # Synchronize session.working_title with session.results['working_title'] - Claude Generated
            if session.results.get('working_title'):
                session.working_title = session.results['working_title']
                logger.info(f"✅ Synchronized session.working_title from results: {session.working_title}")
            else:
                logger.warning(f"⚠️ No working_title in results, session.working_title remains: {session.working_title}")

            # Log summary
            final_keywords = session.results.get("final_keywords", [])
            dk_classifications = session.results.get("dk_classifications", [])
            initial_keywords = session.results.get("initial_keywords", [])
            logger.info(f"Extracted results - keywords: {len(final_keywords)}, classifications: {len(dk_classifications)}, initial: {len(initial_keywords)}")

            # Wait for WebSocket to send ALL remaining streaming tokens - Claude Generated
            # WebSocket sends updates every 500ms, so wait at least 600ms to ensure final tokens are sent
            import time
            time.sleep(0.6)

            total_tokens = sum(len(t) for t in session.streaming_buffer.values()) if session.streaming_buffer else 0
            logger.info(f"Waited 600ms for final streaming tokens to be sent (buffer has {total_tokens} total tokens)")

            session.status = "completed"
            session.current_step = "classification"

            # Final auto-save - Claude Generated
            if session.autosave_enabled:
                try:
                    _autosave_session_state(session)
                except Exception as e:
                    logger.error(f"Final auto-save error: {e}")

        def on_stream_token(token: str, step_id: str = ""):
            """Handle token streaming - buffer tokens for WebSocket - Claude Generated"""
            # Check for abort request - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            # Extract DK search progress if present - Claude Generated
            # Pattern: [N/M] (P%) Suche 'keyword'...
            if step_id == "dk_search":
                progress_match = re.match(r'\[(\d+)/(\d+)\]\s*\((\d+)%\)', token)
                if progress_match:
                    current = int(progress_match.group(1))
                    total = int(progress_match.group(2))
                    percent = int(progress_match.group(3))
                    session.dk_search_progress = {
                        "current": current,
                        "total": total,
                        "percent": percent
                    }

            # Classic runs: mirror the token into the shared #log stream block
            # (opened lazily on the first token of each step; the renderer has
            # a single open-stream slot, classic steps are sequential).
            if is_classic and step_id:
                try:
                    if stream_state["step"] != step_id:
                        _close_stream_block()
                        session_renderer.start_streaming_line(step_id)
                        stream_state["step"] = step_id
                    session_renderer.render_streaming_token(token, step_id)
                except Exception:
                    logger.debug("stream-block token render failed", exc_info=True)

            # Buffer tokens by step for periodic transmission via WebSocket
            # (polling API / external consumers; the browser renders the
            # stream events above, not these frames).
            if step_id:
                session.add_streaming_token(token, step_id)
            logger.debug(f"Token [{step_id}]: {token[:30] if len(token) > 30 else token}...")

        # Run pipeline in background thread - Claude Generated
        def execute_pipeline():
            try:
                # Check for abort before starting - Claude Generated
                if session.abort_requested:
                    raise Exception("Pipeline execution cancelled by user")

                # Set up callbacks
                pipeline_manager.set_callbacks(
                    step_started=on_step_started,
                    step_completed=on_step_completed,
                    step_error=on_step_error,
                    pipeline_completed=on_pipeline_completed,
                    stream_callback=on_stream_token,
                    agentic_context=on_agentic_context,
                )

                # Store reference and wire interrupt callback for step-abort - Claude Generated
                session.pipeline_manager_ref = pipeline_manager
                if hasattr(pipeline_manager, 'set_interrupt_flag'):
                    import threading
                    pipeline_manager.set_interrupt_flag(
                        threading.Lock(),
                        lambda: session.abort_requested
                    )

                # Determine effective source type/value for working title BEFORE start_pipeline runs.
                # start_pipeline executes the pipeline synchronously, so overriding state afterwards is too late.
                # source_type/source_value come from JS when the text was pre-extracted (DOI resolved in browser). - Claude Generated
                if source_type and source_type != 'text' and source_value:
                    effective_input_type = source_type      # e.g. 'doi'
                    effective_input_source = source_value   # e.g. '10.1007/...'
                elif input_type in ("doi", "url"):
                    effective_input_type = input_type
                    effective_input_source = content
                else:
                    effective_input_type = input_type
                    effective_input_source = filename or None

                logger.info(f"Starting pipeline: input_type={input_type}, effective_type={effective_input_type}, source={effective_input_source}")
                pipeline_id = pipeline_manager.start_pipeline(
                    input_text,
                    input_type=effective_input_type,
                    input_source=effective_input_source,
                )

                # Store analysis state reference for auto-save - Claude Generated
                session.current_analysis_state = pipeline_manager.current_analysis_state

                logger.info(f"Pipeline {pipeline_id} started with input_type={input_type}")

            except Exception as e:
                logger.error(f"Pipeline execution error: {str(e)}", exc_info=True)
                session.status = "error"
                session.error_message = str(e)
            finally:
                session.pipeline_manager_ref = None  # Clear reference after pipeline ends - Claude Generated

        # Run in executor to avoid blocking
        await asyncio.to_thread(execute_pipeline)

    except Exception as e:
        logger.error(f"Analysis setup error: {str(e)}", exc_info=True)
        session.status = "error"
        session.error_message = str(e)
    finally:
        # Phase 4: remove session-local bus handlers before cleanup.
        try:
            bus_subscriber.unsubscribe()
        except Exception:
            logger.exception("Failed to unsubscribe session bus subscriber")
        # Cleanup
        if session_id in sessions:
            session.cleanup()


async def run_input_extraction(
    session_id: str,
    input_type: str,
    content: Optional[str],
    file_contents: Optional[bytes],
    filename: Optional[str],
):
    """Execute only the input extraction step (text extraction/OCR) - Claude Generated"""

    session = sessions[session_id]
    session.status = "running"

    try:
        # Use execute_input_extraction from pipeline_utils (same as pipeline does) - Claude Generated
        from src.utils.pipeline_utils import execute_input_extraction

        # Chat-UX 5/9: extraction progress renders as a shared #log stream
        # block (the client no longer renders raw streaming_tokens frames).
        extraction_renderer = _build_session_renderer(session)
        extraction_stream = {"open": False}

        def stream_callback_wrapper(message: str):
            """Wrap stream callback for live progress - Claude Generated"""
            # Check for abort before updating - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            session.current_step = "input"
            try:
                if not extraction_stream["open"]:
                    extraction_renderer.start_streaming_line("input")
                    extraction_stream["open"] = True
                extraction_renderer.render_streaming_token(message, "input")
            except Exception:
                logger.debug("extraction stream render failed", exc_info=True)
            # Use existing add_streaming_token method - correct parameter order: (token, step_id) - Claude Generated
            session.add_streaming_token(message, "input")
            logger.info(f"[Stream] {message}")

        def execute_extraction():
            # Check for abort before starting - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            # Prepare input source and normalize input_type - Claude Generated
            input_source = None
            normalized_input_type = input_type  # Will change for doi->text after resolution

            if input_type == "text" and content:
                input_source = content
            elif input_type == "doi" and content:
                # Resolve DOI/URL to text first - Claude Generated
                logger.info(f"Resolving DOI/URL: {content}")
                cfg = _get_doi_config()
                resolver = UnifiedResolver(logger,
                    contact_email=cfg['contact_email'],
                    use_crossref=cfg['use_crossref'],
                    use_openalex=cfg['use_openalex'],
                    use_datacite=cfg['use_datacite'],
                )
                success, metadata, text_result = resolver.resolve(content)
                if not success:
                    raise ValueError(f"DOI resolution failed: {text_result}")
                text_content = format_doi_metadata(metadata, text_result or "")
                if not text_content:
                    raise ValueError("DOI resolution returned no content")
                input_source = text_content
                normalized_input_type = "text"  # Now treat as text
                logger.info(f"✅ DOI resolved to {len(text_content)} characters")
            elif input_type == "pdf" and file_contents:
                # Save PDF temporarily - Claude Generated
                suffix = ".pdf"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                input_source = temp_file.name
                logger.info(f"Saved PDF to {temp_file.name}")
            elif input_type == "img" and file_contents:
                # Save image temporarily - Claude Generated
                suffix = ""
                if filename:
                    if filename.lower().endswith(".png"):
                        suffix = ".png"
                    elif filename.lower().endswith(".jpeg"):
                        suffix = ".jpeg"
                    else:
                        suffix = ".jpg"
                else:
                    suffix = ".jpg"

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                input_source = temp_file.name
                logger.info(f"Saved image to {temp_file.name}")
            else:
                raise ValueError(f"Invalid input: type={input_type}")

            # Get LLM service via AppContext - Claude Generated
            app_context = AppContext()
            services = app_context.get_services()
            llm_service = services['llm_service']

            # Call execute_input_extraction with normalized input_type - Claude Generated
            logger.info(f"Executing input extraction with input_type={normalized_input_type} from {str(input_source)[:50]}...")
            extracted_text, source_info, extraction_method = execute_input_extraction(
                llm_service=llm_service,
                input_source=input_source,
                input_type=normalized_input_type if normalized_input_type != "img" else "image",  # pipeline uses "image" not "img"
                stream_callback=stream_callback_wrapper,
                logger=logger,
            )

            return extracted_text, source_info, extraction_method

        # Run extraction in executor to avoid blocking - Claude Generated
        try:
            extracted_text, source_info, extraction_method = await asyncio.to_thread(execute_extraction)
        finally:
            if extraction_stream["open"]:
                try:
                    extraction_renderer.end_streaming_line()
                except Exception:
                    logger.debug("extraction stream close failed", exc_info=True)

        # Store extracted text in results - Claude Generated
        session.results = {
            "original_abstract": extracted_text,
            "input_type": input_type,
            "input_mode": "extraction_only",
            "source_info": source_info,
            "extraction_method": extraction_method,
        }

        logger.info(f"✅ Input extraction completed: {extraction_method} - {len(extracted_text)} characters")
        session.current_step = "input"
        session.status = "completed"

    except Exception as e:
        logger.error(f"Input extraction error: {str(e)}", exc_info=True)
        session.status = "error"
        session.error_message = str(e)
    finally:
        # Cleanup
        if session_id in sessions:
            session.cleanup()


# DELETE /api/session/{id} now lives in routers/sessions.py. - Claude Generated

