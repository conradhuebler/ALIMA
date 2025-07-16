from typing import List, Dict, Any, Optional
import logging
import uuid

from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from .data_models import AbstractData, AnalysisResult, PromptConfigData, TaskState
from .processing_utils import chunk_abstract_by_lines, chunk_keywords_by_comma, parse_keywords_from_list, extract_keywords_from_response, extract_gnd_system_from_response, match_keywords_against_text

class AlimaManager:
    def __init__(
        self,
        llm_service: LlmService,
        prompt_service: PromptService,
        logger: logging.Logger = None,
    ):   
        self.ollama_url = "http://localhost"  # Default Ollama URL, can be overridden
        self.ollama_port = 11434  # Default Ollama port, can be overridden

        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.logger = logger or logging.getLogger(__name__)


    def set_ollama_url(self, url: str):
        """Set the Ollama URL for LLM requests."""
        self.ollama_url = url
        self.llm_service.set_ollama_url(url)  # Update the LLM service with the new URL
        self.logger.info(f"Ollama URL set to: {self.ollama_url}")

    def set_ollama_port(self, port: int):
        """Set the Ollama port for LLM requests."""
        self.ollama_port = port
        self.llm_service.set_ollama_port(port)  # Update the LLM service with the new port
        self.logger.info(f"Ollama port set to: {self.ollama_port}")


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
    ) -> TaskState:
        self.logger.info(f"Starting analysis for task: {task}, model: {model}")
        request_id = str(uuid.uuid4())

        if prompt_template:
            # If a prompt template is provided directly, use it.
            prompt_config = PromptConfigData(
                prompt=prompt_template,
                models=[model],
                temp=0.5,  # Using a default temperature
                p_value=0.5, # Using a default p_value
                seed=None,
                system=None
            )
        else:
            # Otherwise, get the prompt from the prompt service.
            prompt_config = self.prompt_service.get_prompt_config(task, model)

        if not prompt_config:
            analysis_result = AnalysisResult(full_text=f"Error: No prompt configuration found for task '{task}' and model '{model}'")
            return TaskState(abstract_data=abstract_data, analysis_result=analysis_result, status="failed")

        if use_chunking_abstract or use_chunking_keywords:
            analysis_result = self._perform_chunked_analysis(
                request_id, prompt_config, abstract_data, use_chunking_abstract, abstract_chunk_size, use_chunking_keywords, keyword_chunk_size, provider, task, model, stream_callback
            )
        else:
            variables = {
                "abstract": abstract_data.abstract,
                "keywords": abstract_data.keywords if abstract_data.keywords else "Keine Keywords vorhanden",
            }
            analysis_result = self._perform_single_analysis(request_id, prompt_config, variables, task, model, provider, use_chunking_abstract, abstract_chunk_size, use_chunking_keywords, keyword_chunk_size, stream_callback)
        
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
            keyword_chunk_size=keyword_chunk_size
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
    ) -> AnalysisResult:
        formatted_prompt = prompt_config.prompt.format(**variables)
        response_text = self._generate_response(request_id, prompt_config, formatted_prompt, provider, stream_callback)
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
    ) -> AnalysisResult:
        chunk_results = []

        abstract_chunks = chunk_abstract_by_lines(abstract_data.abstract, abstract_chunk_size) if use_chunking_abstract else [abstract_data.abstract]
        keyword_chunks = chunk_keywords_by_comma(abstract_data.keywords, keyword_chunk_size) if use_chunking_keywords else [abstract_data.keywords]

        for i, abstract_chunk in enumerate(abstract_chunks):
            for j, keyword_chunk in enumerate(keyword_chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(abstract_chunks)} (abstract) and {j+1}/{len(keyword_chunks)} (keywords)")
                variables = {
                    "abstract": abstract_chunk,
                    "keywords": keyword_chunk if keyword_chunk else "Keine Keywords vorhanden",
                }
                formatted_prompt = prompt_config.prompt.format(**variables)
                response_text = self._generate_response(request_id, prompt_config, formatted_prompt, provider)
                if response_text is None:
                    return AnalysisResult(full_text=f"Error: LLM call failed for chunk {i+1}/{j+1}.")
                chunk_results.append(response_text)

        combined_text = self._combine_chunk_results(request_id, chunk_results, prompt_config, provider)
        if combined_text is None:
            return AnalysisResult(full_text="Error: LLM call failed during chunk combination.")
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
        )

    def _combine_chunk_results(self, request_id: str, chunk_results: List[str], prompt_config: PromptConfigData, provider: Optional[str] = None) -> Optional[str]:
        if len(chunk_results) == 1:
            return chunk_results[0]

        self.logger.info("Combining chunk results.")
        combination_prompt_text = self.prompt_service.get_combination_prompt()
        if not combination_prompt_text:
            self.logger.warning("No combination prompt found. Returning concatenated results.")
            return "\n\n---\n\n".join(chunk_results)

        combined_input = "\n\n---\n\n".join(chunk_results)
        formatted_prompt = combination_prompt_text.format(chunks=combined_input)
        
        return self._generate_response(request_id, prompt_config, formatted_prompt, provider)

    def _generate_response(self, request_id: str, prompt_config: PromptConfigData, formatted_prompt: str, provider: Optional[str] = None, stream_callback: Optional[callable] = None) -> Optional[str]:
        try:
            actual_provider = provider if provider else self._get_provider_for_model(prompt_config.models[0], provider) if prompt_config.models else "default"
            
            response_generator = self.llm_service.generate_response(
                provider=actual_provider,
                model=prompt_config.models[0] if prompt_config.models else "default",
                prompt=formatted_prompt,
                request_id=request_id,
                temperature=prompt_config.temp,
                seed=prompt_config.seed,
                system=prompt_config.system,
                stream=True, # Enable streaming
            )

            full_response_text = ""
            if response_generator is None:
                self.logger.error(f"LLM service returned None for request_id: {request_id}")
                return None

            try:
                for text_chunk in response_generator:
                    if text_chunk is None:
                        continue # Skip None chunks
                    full_response_text += text_chunk
                    if stream_callback:
                        stream_callback(text_chunk)
            except Exception as e:
                self.logger.error(f"Error during streaming LLM response for request_id {request_id}: {e}")
                return None

            return full_response_text
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return None

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
    ) -> AnalysisResult:
        if response_text is None:
            return AnalysisResult(full_text="Error: No response from LLM.")
        
        # Ensure response_text is a string before passing to extraction functions
        response_text_str = str(response_text)

        if keywords_input:
            matched_keywords = self._extract_and_match_keywords(response_text_str, keywords_input)
        else:
            extracted_keywords_str = extract_keywords_from_response(response_text_str)
            matched_keywords = parse_keywords_from_list(extracted_keywords_str)
        gnd_systematic = extract_gnd_system_from_response(response_text_str)
        
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

    def _extract_and_match_keywords(self, response_text: str, keywords_input: str) -> Dict[str, str]:
        if keywords_input is None:
            self.logger.warning("No keywords input provided, skipping keyword extraction.")
            return {}
        matched_keywords = {}

        keywords_dict = parse_keywords_from_list(keywords_input)

        final_list_extracted = extract_keywords_from_response(response_text)
        if final_list_extracted:
            final_list_keywords_dict = parse_keywords_from_list(final_list_extracted)
            matched_keywords.update(final_list_keywords_dict)

        matched_keywords.update(match_keywords_against_text(keywords_dict, response_text))
        return matched_keywords

    def _get_provider_for_model(self, model_name: str, explicit_provider: Optional[str] = None) -> str:
        if explicit_provider:
            return explicit_provider
        if "gemini" in model_name.lower():
            return "gemini"
        elif "gpt" in model_name.lower():
            return "openai"
        else:
            return "ollama"
