from typing import List, Dict, Any, Optional, Callable
import logging
import threading
import time
import uuid

from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from .data_models import AbstractData, AnalysisResult, PromptConfigData, TaskState
from .processing_utils import (
    chunk_abstract_by_lines,
    chunk_keywords_by_comma,
    parse_keywords_from_list,
    extract_keywords_from_response,
    extract_gnd_system_from_response,
    extract_keyword_chains_from_response,
    match_keywords_against_text,
)
from .provider_status_service import ProviderStatusService
from ..utils.error_visibility import log_caught
from ..utils.repetition_detector import (
    EVENT_ABORT,
    EVENT_GRACE,
    EVENT_RESOLVED,
    EVENT_WARN,
    RepetitionDetector,
    RepetitionMonitor,
    RepetitionResult,
    get_parameter_variation_suggestions,
    resolve_repetition_config,
    retry_sampling_params,
)


class AlimaManager:
    def __init__(
        self,
        llm_service: LlmService,
        prompt_service: PromptService,
        config_manager: "ConfigManager",
        logger: logging.Logger = None,
    ):
        self.ollama_url = "http://localhost"  # Default Ollama URL, can be overridden
        self.ollama_port = 11434  # Default Ollama port, can be overridden

        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)

        # Initialize LLM Queue Management - Claude Generated (2026-01-13)
        self._llm_semaphore = threading.Semaphore(3)  # Max 3 concurrent LLM calls
        self._llm_queue_lock = threading.Lock()  # Protect stats dict
        self._llm_queue_stats = {
            "active": 0,
            "pending": 0,
            "total_completed": 0,
            "start_times": {}  # Track when each request started for duration calculation
        }
        self._llm_durations = []  # Keep last 10 durations for EMA calculation

        # Initialize Provider Status Service - Claude Generated
        try:
            self.provider_status_service = ProviderStatusService(self.llm_service)
            self.logger.info("ProviderStatusService initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ProviderStatusService: {e}")
            self.provider_status_service = None

        # Initialize Repetition Detector - Claude Generated
        self._init_repetition_detector()

        # Interrupt callback for streaming loop - Claude Generated
        self._interrupt_check_func: Optional[Callable[[], bool]] = None

    def reload_repetition_config(self):
        """Rebuild the detector from the current config - Claude Generated.

        The GUI creates one AlimaManager for the process, so without this a
        changed setting only reached the detector after a restart — and until
        Sep 2026 not even then, because the loader never restored the section.
        Called from MainWindow._refresh_components after a settings save.
        """
        self._init_repetition_detector()

    def _init_repetition_detector(self):
        """Read the repetition settings - Claude Generated

        ``resolve_repetition_config`` falls back to the defaults for anything
        that is not a real config (a test Mock, a failed load). Each
        ``_generate_response`` builds its own monitor from it: up to three
        generations run at once (``_llm_semaphore``), and one shared detector
        would mix their texts.
        """
        self.repetition_config = resolve_repetition_config(self.config_manager)
        cfg = self.repetition_config
        self.logger.debug(
            f"Repetition detection: enabled={cfg.enabled}, warn<{cfg.warn_threshold}, "
            f"abort<{cfg.abort_threshold}, action={cfg.classic_repetition_action}"
        )

    def _report_repetition_event(
        self,
        name: str,
        result: Optional[RepetitionResult],
        callback: Optional[Callable],
        temperature: float,
        top_p: float,
        repetition_penalty: Optional[float],
        note: str = "",
    ) -> None:
        """Monitor event → log + warning-bar callback - Claude Generated

        Callback contract (unchanged): ``(result, suggestions, grace_period,
        resolved, grace_seconds)``. ``warn`` shows the bar without countdown,
        ``grace`` with it, ``resolved`` turns it green; ``abort`` shows the
        result with ``note`` (what happens next) as its details.
        """
        grace = self.repetition_config.grace_period_seconds
        if name == EVENT_RESOLVED:
            self.logger.info("✅ REPETITION_RESOLVED: Informationsrate wieder über der Warnschwelle")
            if callback:
                callback(None, [], False, True, 0.0)
            return
        if result is None:
            return
        if name == EVENT_WARN:
            self.logger.warning(f"🔄 REPETITION_WARN: {result.details}")
        elif name == EVENT_GRACE:
            self.logger.warning(f"🔄 REPETITION_DETECTED: {result.details}")
            self.logger.info(f"⏳ GRACE_PERIOD_STARTED: {grace}s countdown")
        if not callback:
            return
        suggestions = get_parameter_variation_suggestions(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty or 1.0,
            detection_type=result.detection_type,
        )
        if name == EVENT_ABORT:
            from dataclasses import replace

            callback(replace(result, details=note or result.details), suggestions, False, False, 0.0)
        else:
            callback(result, suggestions, name == EVENT_GRACE, False, grace if name == EVENT_GRACE else 0.0)

    def _notify_repetition(self, text: str, level: str = "warning") -> None:
        """Repetition notice on the state bus (GUI log, webapp, CLI stderr) - Claude Generated"""
        try:
            from .state_bus import AlimaStateBus

            AlimaStateBus().emit_event("state.notice", {"text": text, "level": level})
        except Exception as e:  # display only
            log_caught(self.logger, e, "repetition notice bus emit")

    def set_ollama_url(self, url: str):
        """Set the Ollama URL for LLM requests."""
        self.ollama_url = url
        self.llm_service.set_ollama_url(url)  # Update the LLM service with the new URL
        self.logger.debug(f"Ollama URL set to: {self.ollama_url}")

    def set_ollama_port(self, port: int):
        """Set the Ollama port for LLM requests."""
        self.ollama_port = port
        self.llm_service.set_ollama_port(
            port
        )  # Update the LLM service with the new port
        self.logger.debug(f"Ollama port set to: {self.ollama_port}")

    def set_interrupt_callback(self, func: Optional[Callable[[], bool]]) -> None:
        """Set interrupt check function for streaming loop - Claude Generated

        Args:
            func: Callable returning True if streaming should stop immediately.
                  Pass None to clear.
        """
        self._interrupt_check_func = func

    def analyze_abstract(
        self,
        abstract_data: AbstractData,
        task: str,
        model: str,
        use_chunking_abstract: bool = False,
        abstract_chunk_size: int = 100,
        use_chunking_keywords: bool = False,
        keyword_chunk_size: int = 500,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: int = 0,
        system: Optional[str] = None,
        mode=None,  # <--- NEUER PARAMETER: Pipeline mode for PromptService
        repetition_penalty: Optional[float] = None,
        think: Optional[bool] = None,
        on_repetition_detected: Optional[Callable[[Optional[RepetitionResult], List[Dict], bool, bool, float], None]] = None,
    ) -> TaskState:
        # Log analysis start with workflow-relevant info - Claude Generated
        provider_display = provider if provider else "auto"
        self.logger.info(f"🚀 Starting analysis: task={task}, provider={provider_display}, model={model}")
        self.logger.debug(f"Parameters: temperature={temperature}, top_p={p_value}, seed={seed}")
        request_id = str(uuid.uuid4())

        if prompt_template:
            # If a prompt template is provided directly, use XML (legacy) since custom prompts
            # are not designed for JSON output - Claude Generated
            prompt_config = PromptConfigData(
                prompt=prompt_template,
                models=[model],
                temp=temperature,
                p_value=p_value,
                seed=seed,
                system=system,
                output_format="xml",
            )
        else:
            # Otherwise, get the prompt from the prompt service.
            prompt_config = self.prompt_service.get_prompt_config(task, model)

        if not prompt_config:
            error_msg = f"Error: No prompt configuration found for task '{task}' and model '{model}'"
            self.logger.error(error_msg)
            analysis_result = AnalysisResult(full_text=error_msg)
            return TaskState(
                abstract_data=abstract_data,
                analysis_result=analysis_result,
                status="failed",
            )

        # Log prompt template preview for workflow visibility - Claude Generated
        if hasattr(prompt_config, 'prompt') and prompt_config.prompt:
            prompt_preview = prompt_config.prompt[:200] + "..." if len(prompt_config.prompt) > 200 else prompt_config.prompt
            self.logger.info(f"📝 Using prompt template: {prompt_preview}")

        if use_chunking_abstract or use_chunking_keywords:
            analysis_result = self._perform_chunked_analysis(
                request_id,
                prompt_config,
                abstract_data,
                use_chunking_abstract,
                abstract_chunk_size,
                use_chunking_keywords,
                keyword_chunk_size,
                provider,
                task,
                model,
                stream_callback,
                temperature,
                p_value,
                seed,
                repetition_penalty,
                think,
                on_repetition_detected,
            )
        else:
            variables = {
                "abstract": abstract_data.abstract,
                "keywords": (
                    abstract_data.keywords
                    if abstract_data.keywords
                    else "Keine Keywords vorhanden"
                ),
            }

            # Prompt template already logged above for all tasks - Claude Generated
            analysis_result = self._perform_single_analysis(
                request_id,
                prompt_config,
                variables,
                task,
                model,
                provider,
                use_chunking_abstract,
                abstract_chunk_size,
                use_chunking_keywords,
                keyword_chunk_size,
                stream_callback,
                temperature,
                p_value,
                seed,
                repetition_penalty,
                think,
                on_repetition_detected,
            )

        status = "completed" if "Error" not in analysis_result.full_text else "failed"

        return TaskState(
            abstract_data=abstract_data,
            analysis_result=analysis_result,
            prompt_config=prompt_config,
            status=status,
            task_name=task,
            model_used=model,
            provider_used=provider,
            use_chunking_abstract=use_chunking_abstract,
            abstract_chunk_size=abstract_chunk_size,
            use_chunking_keywords=use_chunking_keywords,
            keyword_chunk_size=keyword_chunk_size,
        )

    def _perform_single_analysis(
        self,
        request_id: str,
        prompt_config: PromptConfigData,
        variables: Dict[str, str],
        task: str,
        model: str,
        provider: Optional[str] = None,
        use_chunking_abstract: bool = False,
        abstract_chunk_size: Optional[int] = None,
        use_chunking_keywords: bool = False,
        keyword_chunk_size: Optional[int] = None,
        stream_callback: Optional[callable] = None,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: int = 0,
        repetition_penalty: Optional[float] = None,
        think: Optional[bool] = None,
        on_repetition_detected: Optional[Callable[[Optional[RepetitionResult], List[Dict], bool, bool, float], None]] = None,
    ) -> AnalysisResult:
        formatted_prompt = prompt_config.prompt.format(**variables)

        self.logger.debug(
            f"Formatted prompt for request {request_id}: {formatted_prompt[:200]}..."
        )
        response_text = self._generate_response(
            request_id,
            prompt_config,
            formatted_prompt,
            provider,
            stream_callback,
            temperature,
            p_value,
            seed,
            repetition_penalty,
            think,
            on_repetition_detected,
        )
        if response_text is None:
            return AnalysisResult(full_text="Error: LLM call failed.")
        return self._create_analysis_result(
            response_text,
            variables.get("keywords", ""),
            task,
            model,
            provider,
            use_chunking_abstract,
            abstract_chunk_size,
            use_chunking_keywords,
            keyword_chunk_size,
            output_format=prompt_config.output_format,
        )

    def _perform_chunked_analysis(
        self,
        request_id: str,
        prompt_config: PromptConfigData,
        abstract_data: AbstractData,
        use_chunking_abstract: bool,
        abstract_chunk_size: int,
        use_chunking_keywords: bool,
        keyword_chunk_size: int,
        provider: Optional[str] = None,
        task: str = "",
        model: str = "",
        stream_callback: Optional[callable] = None,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: int = 0,
        repetition_penalty: Optional[float] = None,
        think: Optional[bool] = None,
        on_repetition_detected: Optional[Callable[[Optional[RepetitionResult], List[Dict], bool, bool, float], None]] = None,
    ) -> AnalysisResult:
        chunk_results = []

        abstract_chunks = (
            chunk_abstract_by_lines(abstract_data.abstract, abstract_chunk_size)
            if use_chunking_abstract
            else [abstract_data.abstract]
        )
        keyword_chunks = (
            chunk_keywords_by_comma(abstract_data.keywords, keyword_chunk_size)
            if use_chunking_keywords
            else [abstract_data.keywords]
        )

        for i, abstract_chunk in enumerate(abstract_chunks):
            for j, keyword_chunk in enumerate(keyword_chunks):
                self.logger.info(
                    f"Processing chunk {i+1}/{len(abstract_chunks)} (abstract) and {j+1}/{len(keyword_chunks)} (keywords)"
                )
                variables = {
                    "abstract": abstract_chunk,
                    "keywords": (
                        keyword_chunk if keyword_chunk else "Keine Keywords vorhanden"
                    ),
                }
                formatted_prompt = prompt_config.prompt.format(**variables)

                response_text = self._generate_response(
                    request_id,
                    prompt_config,
                    formatted_prompt,
                    provider,
                    stream_callback,
                    temperature,
                    p_value,
                    seed,
                    repetition_penalty,
                    think,
                    on_repetition_detected,
                )
                if response_text is None:
                    return AnalysisResult(
                        full_text=f"Error: LLM call failed for chunk {i+1}/{j+1}."
                    )
                chunk_results.append(response_text)

        combined_text = self._combine_chunk_results(
            request_id,
            chunk_results,
            prompt_config,
            provider,
            stream_callback,
            temperature,
            p_value,
            seed,
            repetition_penalty,
            on_repetition_detected,
        )
        if combined_text is None:
            return AnalysisResult(
                full_text="Error: LLM call failed during chunk combination."
            )
        return self._create_analysis_result(
            combined_text,
            abstract_data.keywords,
            task,
            model,
            provider,
            use_chunking_abstract,
            abstract_chunk_size,
            use_chunking_keywords,
            keyword_chunk_size,
            output_format=prompt_config.output_format,
        )

    def _combine_chunk_results(
        self,
        request_id: str,
        chunk_results: List[str],
        prompt_config: PromptConfigData,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: int = 0,
        repetition_penalty: Optional[float] = None,
        on_repetition_detected: Optional[Callable[[Optional[RepetitionResult], List[Dict], bool, bool, float], None]] = None,
    ) -> Optional[str]:
        if len(chunk_results) == 1:
            return chunk_results[0]

        self.logger.info("Combining chunk results.")
        combination_prompt_text = self.prompt_service.get_combination_prompt()
        if not combination_prompt_text:
            self.logger.warning(
                "No combination prompt found. Returning concatenated results."
            )
            return "\n\n---\n\n".join(chunk_results)

        combined_input = "\n\n---\n\n".join(chunk_results)
        formatted_prompt = combination_prompt_text.format(chunks=combined_input)

        return self._generate_response(
            request_id,
            prompt_config,
            formatted_prompt,
            provider,
            stream_callback,
            temperature,
            p_value,
            seed,
            repetition_penalty,
            think,
            on_repetition_detected,
        )

    def _generate_response(
        self,
        request_id: str,
        prompt_config: PromptConfigData,
        formatted_prompt: str,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: int = 0,
        repetition_penalty: Optional[float] = None,
        think: Optional[bool] = None,
        on_repetition_detected: Optional[Callable[[Optional[RepetitionResult], List[Dict], bool, bool, float], None]] = None,
    ) -> Optional[str]:
        # Acquire LLM semaphore (max 3 concurrent LLM calls) - Claude Generated (2026-01-13)
        semaphore_acquired_time = time.time()
        semaphore_acquired = self._llm_semaphore.acquire(blocking=True)

        try:
            # Update queue stats - now active after acquiring semaphore
            with self._llm_queue_lock:
                self._llm_queue_stats["active"] += 1
                self._llm_queue_stats["start_times"][request_id] = time.time()

            queue_wait_time = self._llm_queue_stats["start_times"][request_id] - semaphore_acquired_time
            if queue_wait_time > 0.1:  # Only log if waited > 100ms
                self.logger.info(f"⏳ LLM request {request_id} waited {queue_wait_time:.1f}s in queue")

            self.logger.debug(f"PROVIDER_RESOLUTION: input provider='{provider}', models={prompt_config.models}")

            # Provider must be explicitly provided - no more guessing - Claude Generated
            if not provider:
                raise ValueError(f"Provider must be explicitly provided to AlimaManager. "
                               f"Received: provider='{provider}', models={prompt_config.models}")

            actual_provider = provider

            self.logger.debug(f"PROVIDER_RESOLVED: actual_provider='{actual_provider}'")

            llm_model = prompt_config.models[0] if prompt_config.models else "default"
            self.logger.debug(f"LLM_SERVICE_CALL: provider='{actual_provider}', model='{llm_model}'")
            self.logger.debug(f"STREAM_CALLBACK: {'present' if stream_callback else 'none'}")

            rep_cfg = self.repetition_config
            attempt = 0
            cur_temperature, cur_top_p = temperature, p_value

            # One pass per attempt: a repetition abort with
            # classic_repetition_action="retry" regenerates with varied
            # sampling (retry_sampling_params), at most rep_cfg.max_retries
            # times. - Claude Generated
            while True:
                response_generator = self.llm_service.generate_response(
                    provider=actual_provider,
                    model=llm_model,
                    prompt=formatted_prompt,
                    request_id=request_id,
                    temperature=cur_temperature,
                    p_value=cur_top_p,
                    seed=seed,
                    system=prompt_config.system,
                    stream=True,  # Enable streaming
                    repetition_penalty=repetition_penalty,
                    think=think,
                    output_format=prompt_config.output_format,  # JSON-Mode - Claude Generated
                )

                self.logger.debug(f"LLM_SERVICE_RESULT: response_generator={response_generator is not None}")

                full_response_text = ""
                if response_generator is None:
                    self.logger.error(
                        f"💥 LLM service returned None for request_id: {request_id}, provider: {actual_provider}, model: {llm_model}"
                    )
                    return None

                # Initialize JSON stream filter for display - Claude Generated
                from ..utils.json_stream_filter import JsonStreamFilter
                json_filter = JsonStreamFilter(enabled=(prompt_config.output_format != "xml"))

                # A fresh monitor per generation (see _init_repetition_detector).
                monitor = RepetitionMonitor(
                    RepetitionDetector.from_config(rep_cfg),
                    auto_abort=rep_cfg.auto_abort,
                    grace_seconds=rep_cfg.grace_period_seconds,
                )
                aborted: Optional[RepetitionResult] = None
                interrupted = False

                try:
                    chunk_count = 0
                    for text_chunk in response_generator:
                        if text_chunk is None:
                            continue  # Skip None chunks
                        chunk_count += 1
                        full_response_text += text_chunk

                        # Check for external interrupt (e.g. user pressed Stop) - Claude Generated
                        if self._interrupt_check_func and self._interrupt_check_func():
                            self.logger.info("🛑 STREAM_INTERRUPTED: User requested stop during streaming")
                            interrupted = True
                            if hasattr(response_generator, 'close'):
                                try:
                                    response_generator.close()
                                except Exception:
                                    pass  # best-effort generator close on user abort - Claude Generated
                            break

                        event = monitor.feed(text_chunk)
                        if event:
                            name, res = event
                            if name == EVENT_ABORT:
                                aborted = res
                                break
                            self._report_repetition_event(
                                name, res, on_repetition_detected,
                                cur_temperature, cur_top_p, repetition_penalty,
                            )

                        if stream_callback:
                            self.logger.debug(f"📡 Streaming chunk #{chunk_count}: '{text_chunk[:50]}...'")
                            # Apply JSON filter for display (full_response_text remains unfiltered) - Claude Generated
                            filtered_chunk = json_filter.feed(text_chunk)
                            if filtered_chunk:
                                stream_callback(filtered_chunk)
                        else:
                            self.logger.debug(f"📡 Chunk #{chunk_count} (no callback): '{text_chunk[:50]}...'")

                    # A response that ends at abort level looped until it
                    # stopped by itself (often at the token limit).
                    if aborted is None and not interrupted:
                        event = monitor.finish()
                        if event and event[0] == EVENT_ABORT:
                            aborted = event[1]

                    # Close the generator to release the HTTP connection - Claude Generated
                    if aborted is not None and hasattr(response_generator, 'close'):
                        try:
                            response_generator.close()
                        except Exception as close_err:
                            self.logger.debug(f"Generator close error (ignored): {close_err}")

                    if aborted is not None:
                        self.logger.info(f"⚠️ LLM_STREAM_ABORTED: {chunk_count} chunks, {len(full_response_text)} chars (repetition detected)")
                    else:
                        self.logger.info(f"✅ LLM_STREAM_COMPLETE: {chunk_count} chunks, {len(full_response_text)} chars total")

                    if monitor.detections:
                        self.logger.info(
                            f"📊 REPETITION_SUMMARY: {monitor.detections} Wiederholung(en) erkannt, "
                            f"{monitor.resolutions} während der Gnadenfrist ({rep_cfg.grace_period_seconds}s) behoben"
                        )

                except Exception as e:
                    self.logger.error(
                        f"💥 Error during streaming LLM response for request_id {request_id}: {e}"
                    )
                    return f"Error: {e}"

                if aborted is None:
                    return full_response_text

                # Keep the text before the loop, drop the loop tail.
                cut = max(0, min(len(full_response_text), aborted.position))
                kept = full_response_text[:cut]
                self.logger.warning(
                    f"🔁 REPETITION_ABORT ({aborted.detection_type}): {aborted.details}; "
                    f"behalten {cut} von {len(full_response_text)} Zeichen"
                )
                if rep_cfg.classic_repetition_action == "retry" and attempt < rep_cfg.max_retries:
                    attempt += 1
                    params = retry_sampling_params(cur_temperature, cur_top_p, attempt)
                    cur_temperature, cur_top_p = params["temperature"], params["top_p"]
                    message = (
                        f"🔁 Wiederholungsschleife erkannt, neuer Versuch {attempt}/{rep_cfg.max_retries} "
                        f"mit temperature={cur_temperature}, top_p={cur_top_p}"
                    )
                    self._report_repetition_event(
                        EVENT_ABORT, aborted, on_repetition_detected,
                        cur_temperature, cur_top_p, repetition_penalty, note=message,
                    )
                    self._notify_repetition(message)
                    if stream_callback:
                        stream_callback(f"\n\n{message}\n\n")
                    continue

                message = (
                    f"🔁 Wiederholungsschleife erkannt, Generierung abgebrochen. "
                    f"Das Teilergebnis bis Zeichen {cut} von {len(full_response_text)} bleibt erhalten"
                )
                self._report_repetition_event(
                    EVENT_ABORT, aborted, on_repetition_detected,
                    cur_temperature, cur_top_p, repetition_penalty, note=message,
                )
                self._notify_repetition(message)
                return kept
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return f"Error: {e}"
        finally:
            # Release LLM semaphore and update stats - Claude Generated (2026-01-13)
            try:
                if semaphore_acquired:
                    self._llm_semaphore.release()

                with self._llm_queue_lock:
                    self._llm_queue_stats["active"] = max(0, self._llm_queue_stats["active"] - 1)
                    self._llm_queue_stats["total_completed"] += 1

                    # Calculate and track execution duration
                    if request_id in self._llm_queue_stats["start_times"]:
                        duration = time.time() - self._llm_queue_stats["start_times"][request_id]
                        self._llm_durations.append(duration)
                        # Keep only last 10 durations for EMA
                        if len(self._llm_durations) > 10:
                            self._llm_durations.pop(0)
                        del self._llm_queue_stats["start_times"][request_id]
            except Exception as e:
                self.logger.error(f"Error in LLM queue cleanup: {e}")

    def _create_analysis_result(
        self,
        response_text: Optional[str],
        keywords_input: str,
        task: str,
        model: str,
        provider: Optional[str],
        use_chunking_abstract: bool,
        abstract_chunk_size: Optional[int],
        use_chunking_keywords: bool,
        keyword_chunk_size: Optional[int],
        output_format: Optional[str] = None,
    ) -> AnalysisResult:
        if response_text is None:
            return AnalysisResult(full_text="Error: No response from LLM.")

        # Ensure response_text is a string before passing to extraction functions
        response_text_str = str(response_text)

        if keywords_input:
            matched_keywords = self._extract_and_match_keywords(
                response_text_str, keywords_input, output_format=output_format
            )
        else:
            extracted_keywords_str = extract_keywords_from_response(response_text_str, output_format=output_format)
            matched_keywords = parse_keywords_from_list(extracted_keywords_str)
        gnd_systematic = extract_gnd_system_from_response(response_text_str, output_format=output_format)
        keyword_chains = extract_keyword_chains_from_response(response_text_str, output_format=output_format)

        # Handle cases where extraction functions might return None or empty string
        if not matched_keywords:
            matched_keywords = {}
            # Visibility: empty extraction from a non-empty response is more
            # likely a parse failure than "no keywords" - Claude Generated
            if response_text_str.strip():
                self.logger.warning(
                    f"Keyword extraction returned nothing for task '{task}' "
                    f"(model={model}, response length {len(response_text_str)}) — "
                    "possible unparseable LLM output."
                )
        if gnd_systematic is None:
            gnd_systematic = ""
        if not keyword_chains:
            keyword_chains = []

        return AnalysisResult(
            full_text=response_text,
            matched_keywords=matched_keywords,
            gnd_systematic=gnd_systematic,
            keyword_chains=keyword_chains,
        )

    def _extract_and_match_keywords(
        self, response_text: str, keywords_input: str, output_format: Optional[str] = None
    ) -> Dict[str, str]:
        if keywords_input is None:
            self.logger.warning(
                "No keywords input provided, skipping keyword extraction."
            )
            return {}
        matched_keywords = {}

        keywords_dict = parse_keywords_from_list(keywords_input)

        final_list_extracted = extract_keywords_from_response(response_text, output_format=output_format)
        if final_list_extracted:
            final_list_keywords_dict = parse_keywords_from_list(final_list_extracted)
            matched_keywords.update(final_list_keywords_dict)

        matched_keywords.update(
            match_keywords_against_text(keywords_dict, response_text)
        )
        return matched_keywords

    def execute_task(
        self, 
        task_name: str, 
        context: Dict[str, Any],
        stream_callback: Optional[callable] = None
    ) -> str:
        """Execute a task using configured model priorities with fallback support - Claude Generated
        
        Args:
            task_name: Name of the task (e.g., 'image_text_extraction', 'keywords', 'keywords_chunked')
            context: Task context containing required data (e.g., 'image_data', 'abstract', 'keywords')
            stream_callback: Optional callback for streaming output
            
        Returns:
            str: Task execution result
            
        Raises:
            Exception: If all configured models fail to execute the task
        """
        self.logger.info(f"Executing task: {task_name}")
        
        # Use config_manager directly instead of bridge
        # unified_config_manager = UnifiedProviderConfigManager(self.config_manager) # Removed bridge

        # Handle chunked tasks: keywords_chunked -> keywords base task
        base_task_name = task_name
        if task_name.endswith('_chunked'):
            base_task_name = task_name[:-8]  # Remove '_chunked' suffix
        
        # Resolve the provider/model for this task via the same hierarchy every
        # other call site uses. Used to read task_preferences via
        # get_model_priority_for_task(), with a fallback chain through
        # "vision"/"initialisation"/"keywords" task preferences for
        # image_text_extraction specifically - task_preferences is retired as a
        # live config surface (Oct 2026); resolve_default_provider_model's
        # vision/pipeline scopes are the real resolution path now and were
        # already what every other vision caller (ImageAnalysisTab via
        # SmartProviderSelector) used. - Claude Generated
        model_priority = []
        try:
            from ..utils.config_models import ProviderScope

            unified_config = self.config_manager.get_unified_config()
            scope = (
                ProviderScope.VISION
                if base_task_name in ("image_text_extraction", "vision")
                else ProviderScope.PIPELINE
            )
            provider, model = unified_config.resolve_default_provider_model(scope=scope)
            if provider:
                model_priority = [{"provider_name": provider, "model_name": model}]
                self.logger.info(f"Using resolved default for task '{task_name}' ({scope}): {provider}/{model}")
        except Exception as e:
            log_caught(self.logger, e, "Default-Provider/Modell für Task auflösen (nutze Fallback)")
        
        # Fallback to available providers if no specific priority configured
        if not model_priority:
            try:
                # Use detection service to get real available providers
                detection_service = self.config_manager.get_provider_detection_service()
                available_providers = detection_service.get_available_providers()

                # Create fallback priority from available providers
                for provider in available_providers:
                    model_priority.append({"provider_name": provider, "model_name": "default"})
                
                # Final fallback if detection fails
                if not model_priority:
                    # Try to get first available ollama provider from config
                    try:
                        unified_config = self.config_manager.get_unified_config()
                        first_ollama = None
                        for provider in unified_config.providers:
                            if provider.provider_type == "ollama" and provider.enabled:
                                first_ollama = provider.name
                                break
                        fallback_provider = first_ollama or "localhost"
                        model_priority = [{"provider_name": fallback_provider, "model_name": "default"}]
                    except:
                        model_priority = [{"provider_name": "localhost", "model_name": "default"}]

            except Exception as e:
                self.logger.warning(f"Error getting available providers for fallback: {e}")
                # Emergency fallback - use first available ollama provider
                try:
                    unified_config = self.config_manager.get_unified_config()
                    first_ollama = None
                    for provider in unified_config.providers:
                        if provider.provider_type == "ollama" and provider.enabled:
                            first_ollama = provider.name
                            break
                    fallback_provider = first_ollama or "localhost"
                    model_priority = [{"provider_name": fallback_provider, "model_name": "default"}]
                except:
                    model_priority = [{"provider_name": "localhost", "model_name": "default"}]
        
        # Try each model in priority order
        last_error = None
        for i, model_config in enumerate(model_priority):
            provider_name = model_config["provider_name"]
            model_name = model_config["model_name"]

            # Get prompt configuration for the current model
            prompt_config = self.prompt_service.get_prompt_config(base_task_name, model_name)
            if not prompt_config:
                self.logger.warning(f"No prompt configuration found for task '{base_task_name}' and model '{model_name}', skipping.")
                continue
            
            # Handle 'default' model name by using first available model from provider
            if model_name == "default":
                try:
                    available_models = self.llm_service.get_available_models(provider_name)
                    if available_models:
                        model_name = available_models[0]
                    else:
                        self.logger.warning(f"No models available for provider {provider_name}")
                        continue
                except Exception as e:
                    self.logger.warning(f"Error getting models for {provider_name}: {e}")
                    continue
            
            try:
                self.logger.info(f"Attempting task execution with {provider_name}:{model_name} (attempt {i+1}/{len(model_priority)})")
                
                # Generate request ID
                request_id = str(uuid.uuid4())
                
                # Prepare LLM parameters
                llm_params = {
                    'provider': provider_name,
                    'model': model_name,
                    'prompt': prompt_config.prompt.format(**context),
                    'system': prompt_config.system or '',
                    'request_id': request_id,
                    'temperature': float(prompt_config.temp),
                    'p_value': float(prompt_config.p_value),
                    'seed': prompt_config.seed,
                    'stream': stream_callback is not None,  # Enable streaming if callback provided - Claude Generated
                    # Every sibling call site passes this through (see
                    # analyze_abstract above) - without it, output_format=None
                    # defaults to JSON-mode on Gemini/OpenAI-compatible/Ollama,
                    # which matters here because image_text_extraction wants
                    # raw OCR text, not JSON. - Claude Generated
                    'output_format': prompt_config.output_format,
                }
                
                # Add image if present in context
                if 'image_data' in context:
                    llm_params['image'] = context['image_data']
                
                # Execute LLM call
                response = self.llm_service.generate_response(**llm_params)

                # Handle response (generator or string) with streaming support - Claude Generated
                if hasattr(response, "__iter__") and not isinstance(response, str):
                    # Generator response - with streaming support
                    response_parts = []
                    for chunk in response:
                        if isinstance(chunk, str):
                            chunk_text = chunk
                        elif hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        elif hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        else:
                            chunk_text = str(chunk)

                        response_parts.append(chunk_text)

                        # Call stream callback if provided
                        if stream_callback and chunk_text:
                            stream_callback(chunk_text)

                    result = "".join(response_parts)
                else:
                    result = str(response)
                
                if result.strip():
                    self.logger.info(f"Task {task_name} completed successfully with {provider_name}:{model_name}")
                    return result
                else:
                    raise Exception("Empty response from LLM")
                    
            except Exception as e:
                last_error = e
                log_caught(self.logger, e, f"Task execution with {provider_name}:{model_name}")
                continue
        
        # All models failed
        error_msg = f"All configured models failed for task '{task_name}'. Last error: {last_error}"
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def get_llm_queue_status(self) -> Dict[str, Any]:
        """Get current LLM queue statistics for public status display - Claude Generated (2026-01-13)"""
        with self._llm_queue_lock:
            # Calculate average duration from last 10 requests
            avg_duration = 0.0
            if self._llm_durations:
                avg_duration = sum(self._llm_durations) / len(self._llm_durations)

            return {
                "active_llm_requests": self._llm_queue_stats["active"],
                "pending_llm_requests": self._llm_queue_stats["pending"],
                "max_concurrent": 3,
                "total_completed": self._llm_queue_stats["total_completed"],
                "avg_duration_seconds": avg_duration
            }
