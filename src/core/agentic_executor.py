"""Agentic Pipeline Executor - Claude Generated

Bridges the MetaAgent system with PipelineManager's KeywordAnalysisState output format.
Produces identical output to the sequential pipeline, but using LLM-driven agents with MCP tools.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from src.core.data_models import (
    KeywordAnalysisState, LlmKeywordAnalysis, SearchResult,
)
from src.core.agents.base_agent import AgentConfig
from src.core.agents.meta_agent import MetaAgent
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgenticPipelineExecutor:
    """
    Executes the ALIMA pipeline using agents instead of sequential steps.

    Produces the same KeywordAnalysisState output as the sequential PipelineStepExecutor,
    ensuring backward compatibility with JSON export, GUI display, and CLI formatting.
    """

    def __init__(
        self,
        llm_service,
        config_manager=None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.llm_service = llm_service
        self.config_manager = config_manager
        self.stream_callback = stream_callback

        # Initialize tool registry
        self.tool_registry = ToolRegistry(config_manager=config_manager)
        self.tool_registry.register_all_tools()

    def execute(
        self,
        abstract: str,
        provider: str = "",
        model: str = "",
        temperature: float = 0.3,
        initial_keywords: Optional[List[str]] = None,
        enable_classification: bool = True,
        enable_validation: bool = True,
        max_iterations: int = 20,
        quality_threshold: float = 0.6,
        input_type: str = "text",
        source_value: Optional[str] = None,
    ) -> KeywordAnalysisState:
        """
        Execute full agentic pipeline, return KeywordAnalysisState.

        Args:
            abstract: Input text
            provider: LLM provider name
            model: Model name
            temperature: Sampling temperature
            initial_keywords: Pre-generated keywords (skip initialisation if provided)
            enable_classification: Run DK classification
            enable_validation: Run validation
            max_iterations: Max tool-calling iterations per agent
            quality_threshold: Minimum quality score for each agent
            input_type: Source type ('text', 'doi', 'pdf', etc.)
            source_value: Original source (DOI, file path, etc.)

        Returns:
            KeywordAnalysisState compatible with sequential pipeline output
        """
        logger.info(f"Starting agentic pipeline with {provider}/{model}")

        # Adapt stream callback from pipeline format (token, step_id) to agent format (text)
        def agent_stream(text: str):
            if self.stream_callback:
                self.stream_callback(text, "agentic")

        # Configure agents
        agent_config = AgentConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tool_iterations=max_iterations,
            quality_threshold=quality_threshold,
        )

        # Run MetaAgent
        meta = MetaAgent(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            config=agent_config,
            stream_callback=agent_stream,
        )

        results = meta.execute(
            abstract=abstract,
            initial_keywords=initial_keywords,
            enable_classification=enable_classification,
            enable_validation=enable_validation,
        )

        # Convert to KeywordAnalysisState
        state = self._convert_to_state(results, abstract, provider, model, temperature,
                                        input_type, source_value)
        return state

    def _convert_to_state(
        self,
        results: Dict[str, Any],
        abstract: str,
        provider: str,
        model: str,
        temperature: float,
        input_type: str,
        source_value: Optional[str],
    ) -> KeywordAnalysisState:
        """Convert MetaAgent results to KeywordAnalysisState."""

        # Extract keywords
        selected_kw = results.get("keywords", {}).get("selected_keywords", [])
        gnd_keywords = [kw.get("title", "") for kw in selected_kw if kw.get("title")]
        gnd_ids = [kw.get("gnd_id", "") for kw in selected_kw if kw.get("gnd_id")]

        # Extract classifications
        dk_raw = results.get("classification", {}).get("dk_classifications", [])
        dk_codes = [c.get("code", "") for c in dk_raw if c.get("code")]

        # Build search results
        search_data = results.get("search", {})
        search_results = []
        for entry in search_data.get("gnd_entries", []):
            search_results.append(SearchResult(
                search_term=entry.get("title", ""),
                results={entry.get("gnd_id", ""): {"title": entry.get("title", ""), "source": entry.get("source", "")}},
            ))

        # Build LLM analysis records
        search_agent_result = search_data.get("_agent_result")
        keyword_agent_result = results.get("keywords", {}).get("_agent_result")

        # Aggregate full response text from agents for LlmKeywordAnalysis
        initial_response = ""
        if search_agent_result:
            initial_response = search_agent_result.content

        final_response = ""
        if keyword_agent_result:
            final_response = keyword_agent_result.content

        initial_analysis = LlmKeywordAnalysis(
            task_name="agentic_search",
            model_used=model,
            provider_used=provider,
            prompt_template="[agentic: SearchAgent system prompt]",
            filled_prompt="[agentic: dynamic]",
            temperature=temperature,
            seed=None,
            response_full_text=initial_response,
            extracted_gnd_keywords=gnd_keywords,
            extracted_gnd_classes=gnd_ids,
            missing_concepts=results.get("keywords", {}).get("missing_concepts", []),
        )

        final_analysis = LlmKeywordAnalysis(
            task_name="agentic_keywords",
            model_used=model,
            provider_used=provider,
            prompt_template="[agentic: KeywordAgent system prompt]",
            filled_prompt="[agentic: dynamic]",
            temperature=temperature,
            seed=None,
            response_full_text=final_response,
            extracted_gnd_keywords=gnd_keywords,
            extracted_gnd_classes=gnd_ids,
        )

        # Build classification analysis if available
        dk_analysis = None
        class_result = results.get("classification", {})
        if class_result:
            class_agent_result = class_result.get("_agent_result")
            dk_analysis = LlmKeywordAnalysis(
                task_name="agentic_classification",
                model_used=model,
                provider_used=provider,
                prompt_template="[agentic: ClassificationAgent system prompt]",
                filled_prompt="[agentic: dynamic]",
                temperature=temperature,
                seed=None,
                response_full_text=class_agent_result.content if class_agent_result else "",
                extracted_gnd_keywords=[],
                extracted_gnd_classes=dk_codes,
            )

        state = KeywordAnalysisState(
            original_abstract=abstract,
            initial_keywords=results.get("initial_keywords", []),
            search_suggesters_used=["agentic"],
            working_title=None,
            input_type=input_type,
            source_value=source_value,
            initial_gnd_classes=gnd_ids,
            search_results=search_results[:50],  # Limit to avoid huge JSONs
            initial_llm_call_details=initial_analysis,
            final_llm_analysis=final_analysis,
            timestamp=datetime.now().isoformat(),
            pipeline_step_completed="classification" if dk_codes else "keywords",
            dk_llm_analysis=dk_analysis,
            dk_classifications=dk_codes,
            refinement_iterations=[{
                "type": "agentic",
                "agent_log": results.get("agent_log", []),
                "total_duration_s": results.get("total_duration_s", 0),
            }],
        )

        return state
