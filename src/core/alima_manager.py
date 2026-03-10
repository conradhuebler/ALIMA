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
    match_keywords_against_text,
)
from .provider_status_service import ProviderStatusService
from ..utils.repetition_detector import (
    RepetitionDetector,
    RepetitionDetectorConfig,
    RepetitionResult,
    get_parameter_variation_suggestions,
    format_repetition_warning,
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

    def _init_repetition_detector(self):
        """Initialize repetition detector from config - Claude Generated"""
        try:
            config = self.config_manager.load_config()
            rep_config = getattr(config, 'repetition_config', None)

            if rep_config:
                detector_config = RepetitionDetectorConfig(
                    # Master switches
                    enabled=rep_config.enabled,
                    auto_abort=rep_config.auto_abort,
                    grace_period_seconds=getattr(rep_config, 'grace_period_seconds', 2.0),  # Claude Generated (2026-02-17)
                    # N-gram detection
                    ngram_size=rep_config.ngram_size,
                    ngram_threshold=rep_config.ngram_threshold,
                    # Window similarity
                    window_size=getattr(rep_config, 'window_size', 300),
                    similarity_threshold=rep_config.window_similarity_threshold,
                    min_windows=getattr(rep_config, 'min_windows', 4),
                    # Character patterns
                    char_repeat_threshold=rep_config.char_repeat_threshold,
                    # Processing control
                    min_text_length=getattr(rep_config, 'min_text_length', 1000),
                    check_interval=getattr(rep_config, 'check_interval', 200),
                )
            else:
                detector_config = RepetitionDetectorConfig()

            self.repetition_detector = RepetitionDetector(detector_config)
            self.logger.debug(f"RepetitionDetector initialized: ngram={detector_config.ngram_threshold}, min_text={detector_config.min_text_length}, enabled={detector_config.enabled}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize RepetitionDetector: {e}, using defaults")
            self.repetition_detector = RepetitionDetector()

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

        # Inject JSON-Instruktionen unless output_format == "xml" - Claude Generated
        if prompt_config.output_format != "xml":
            from .json_schemas import get_json_instruction_text
            json_instruction = get_json_instruction_text(task)
            formatted_prompt += "\n\n" + json_instruction
            self.logger.info(f"🔧 JSON-Instruktionen für Task '{task}' in Prompt injiziert")

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

                # Inject JSON-Instruktionen for chunked analysis - Claude Generated
                if prompt_config.output_format != "xml":
                    from .json_schemas import get_json_instruction_text
                    json_instruction = get_json_instruction_text(task)
                    formatted_prompt += "\n\n" + json_instruction

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

            response_generator = self.llm_service.generate_response(
                provider=actual_provider,
                model=llm_model,
                prompt=formatted_prompt,
                request_id=request_id,
                temperature=temperature,
                p_value=p_value,
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

            # Reset repetition detector for new generation - Claude Generated
            self.repetition_detector.reset()
            repetition_aborted = False

            # Grace period state tracking - Claude Generated (2026-02-17)
            repetition_grace_start: Optional[float] = None  # Timestamp when grace period started
            repetition_last_detected: Optional[RepetitionResult] = None  # Last detection result
            repetition_count = 0  # Count total detections
            repetition_resolved_count = 0  # Count successful resolutions

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
                        if hasattr(response_generator, 'close'):
                            try:
                                response_generator.close()
                            except Exception:
                                pass
                        break

                    # Check for repetition patterns - Claude Generated
                    rep_result = self.repetition_detector.add_chunk(text_chunk)
                    if rep_result and rep_result.is_repetitive:
                        # Repetition detected - handle grace period - Claude Generated (2026-02-17)
                        if repetition_grace_start is None:
                            # START grace period
                            repetition_grace_start = time.time()
                            repetition_last_detected = rep_result
                            repetition_count += 1  # Count first detection

                            self.logger.warning(f"🔄 REPETITION_DETECTED: {rep_result.detection_type} - {rep_result.details}")
                            self.logger.info(f"⏳ GRACE_PERIOD_STARTED: {self.repetition_detector.config.grace_period_seconds}s countdown")

                            # Generate parameter suggestions
                            suggestions = get_parameter_variation_suggestions(
                                temperature=temperature,
                                top_p=p_value,
                                repetition_penalty=repetition_penalty or 1.0,
                                detection_type=rep_result.detection_type,
                            )

                            # Call callback with grace_period=True
                            if on_repetition_detected:
                                on_repetition_detected(
                                    rep_result,
                                    suggestions,
                                    True,  # grace_period
                                    False,  # resolved
                                    self.repetition_detector.config.grace_period_seconds  # grace_seconds
                                )
                        else:
                            # CONTINUE grace period - repetition persists
                            repetition_last_detected = rep_result
                            elapsed = time.time() - repetition_grace_start

                            # Check if grace period expired
                            if elapsed >= self.repetition_detector.config.grace_period_seconds:
                                # ABORT after grace period
                                if self.repetition_detector.config.auto_abort:
                                    self.logger.warning(f"🛑 AUTO_ABORT: Grace period expired ({elapsed:.1f}s), cancelling generation")
                                    repetition_aborted = True
                                    # Try to cancel the LLM generation
                                    try:
                                        self.llm_service.cancel_generation(reason="repetition_persistent")
                                    except Exception as cancel_err:
                                        self.logger.debug(f"Cancel generation not supported: {cancel_err}")
                                    break
                    else:
                        # No repetition in this chunk - check for resolution - Claude Generated (2026-02-17)
                        if repetition_grace_start is not None:
                            # RESOLVED - repetition stopped during grace period
                            elapsed = time.time() - repetition_grace_start
                            repetition_resolved_count += 1  # Count resolution
                            self.logger.info(f"✅ REPETITION_RESOLVED: Stopped after {elapsed:.1f}s, continuing generation")
                            repetition_grace_start = None
                            repetition_last_detected = None

                            # Notify UI that repetition resolved
                            if on_repetition_detected:
                                on_repetition_detected(None, [], False, True, 0.0)  # result, suggestions, grace_period, resolved, grace_seconds

                    if stream_callback:
                        self.logger.debug(f"📡 Streaming chunk #{chunk_count}: '{text_chunk[:50]}...'")
                        # Apply JSON filter for display (full_response_text remains unfiltered) - Claude Generated
                        filtered_chunk = json_filter.feed(text_chunk)
                        if filtered_chunk:
                            stream_callback(filtered_chunk)
                    else:
                        self.logger.debug(f"📡 Chunk #{chunk_count} (no callback): '{text_chunk[:50]}...'")

                # Close the generator to release the HTTP connection - Claude Generated
                if repetition_aborted and hasattr(response_generator, 'close'):
                    try:
                        response_generator.close()
                    except Exception as close_err:
                        self.logger.debug(f"Generator close error (ignored): {close_err}")

                if repetition_aborted:
                    self.logger.info(f"⚠️ LLM_STREAM_ABORTED: {chunk_count} chunks, {len(full_response_text)} chars (repetition detected)")
                else:
                    self.logger.info(f"✅ LLM_STREAM_COMPLETE: {chunk_count} chunks, {len(full_response_text)} chars total")

                # Log repetition statistics summary - Claude Generated (2026-02-17)
                if repetition_count > 0:
                    self.logger.info(
                        f"📊 REPETITION_SUMMARY: {repetition_count} Wiederholung(en) erkannt, "
                        f"{repetition_resolved_count} erfolgreich behoben durch Grace Period ({self.repetition_detector.config.grace_period_seconds}s)"
                    )

            except Exception as e:
                self.logger.error(
                    f"💥 Error during streaming LLM response for request_id {request_id}: {e}"
                )
                return None

            return full_response_text
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return None
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

        # Handle cases where extraction functions might return None or empty string
        if not matched_keywords:
            matched_keywords = {}
        if gnd_systematic is None:
            gnd_systematic = ""

        return AnalysisResult(
            full_text=response_text,
            matched_keywords=matched_keywords,
            gnd_systematic=gnd_systematic,
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
        is_chunked = False
        if task_name.endswith('_chunked'):
            base_task_name = task_name[:-8]  # Remove '_chunked' suffix
            is_chunked = True
        
        # Get model priority for task
        model_priority = []
        try:
            unified_config = self.config_manager.get_unified_config()
            model_priority = unified_config.get_model_priority_for_task(task_name, is_chunked)
            
            # Fallback for image_text_extraction: use vision, initialisation, or keywords preference
            if not model_priority and task_name == "image_text_extraction":
                for fallback_task in ["vision", "initialisation", "keywords"]:
                    model_priority = unified_config.get_model_priority_for_task(fallback_task)
                    if model_priority:
                        self.logger.info(f"Using fallback task '{fallback_task}' for image_text_extraction")
                        break
            
            self.logger.info(f"Using task-specific model priority: {model_priority}")
        except Exception as e:
            self.logger.warning(f"Error getting task preferences: {e}, using fallback")
        
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
                    'stream': stream_callback is not None  # Enable streaming if callback provided - Claude Generated
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
                self.logger.warning(f"Task execution failed with {provider_name}:{model_name}: {e}")
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
