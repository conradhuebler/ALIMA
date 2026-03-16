"""MetaAgent: Orchestrator that coordinates sub-agents - Claude Generated

Analyzes text type, plans execution strategy, runs sub-agents in sequence,
and aggregates results into a final KeywordAnalysisState.
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable

from src.core.agents.base_agent import BaseAgent, AgentConfig, QualityMetrics
from src.core.agents.search_agent import SearchAgent
from src.core.agents.keyword_agent import KeywordAgent
from src.core.agents.classification_agent import ClassificationAgent
from src.core.agents.validation_agent import ValidationAgent
from src.core.data_models import AgentResult
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class MetaAgent:
    """
    Top-level orchestrator that runs specialized agents in sequence.

    Unlike other agents, MetaAgent does NOT use the AgentLoop directly.
    Instead it coordinates sub-agents, passing results between them.
    """

    def __init__(
        self,
        llm_service,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self.stream_callback = stream_callback
        self.logger = logging.getLogger(f"{__name__}.MetaAgent")

    def execute(self, abstract: str, initial_keywords: Optional[List[str]] = None,
                enable_classification: bool = True, enable_validation: bool = True) -> Dict[str, Any]:
        """
        Run the full agentic pipeline.

        Args:
            abstract: Input text to analyze
            initial_keywords: Pre-generated free keywords (from initialisation step)
            enable_classification: Whether to run ClassificationAgent
            enable_validation: Whether to run ValidationAgent

        Returns:
            Dict with all agent results, ready for conversion to KeywordAnalysisState
        """
        start_time = time.time()
        results = {
            "abstract": abstract,
            "initial_keywords": initial_keywords or [],
            "agent_log": [],
        }

        if self.stream_callback:
            self.stream_callback("\n" + "=" * 50)
            self.stream_callback("\n🤖 ALIMA Agentic Pipeline gestartet")
            self.stream_callback("\n" + "=" * 50)

        # Step 1: SearchAgent - Build GND pool
        self.logger.info("MetaAgent: Starting SearchAgent")
        if self.stream_callback:
            self.stream_callback("\n\n📍 Phase 1: GND-Suche")

        search_agent = SearchAgent(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            config=self._make_agent_config(),
            stream_callback=self.stream_callback,
        )

        search_input = {
            "abstract": abstract,
            "initial_keywords": initial_keywords or [],
        }

        search_result = search_agent.execute_with_self_reflection(search_input)
        results["search"] = search_result
        results["agent_log"].append({
            "agent": "SearchAgent",
            "duration_s": round(time.time() - start_time, 1),
            "gnd_entries_found": len(search_result.get("gnd_entries", [])),
            "quality": search_result.get("_quality", {}),
        })

        # Step 2: KeywordAgent - Select best keywords
        self.logger.info("MetaAgent: Starting KeywordAgent")
        if self.stream_callback:
            self.stream_callback("\n\n📍 Phase 2: Schlagwort-Auswahl")

        keyword_agent = KeywordAgent(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            config=self._make_agent_config(),
            stream_callback=self.stream_callback,
        )

        keyword_input = {
            "abstract": abstract,
            "gnd_entries": search_result.get("gnd_entries", []),
        }

        keyword_result = keyword_agent.execute_with_self_reflection(keyword_input)
        results["keywords"] = keyword_result
        results["agent_log"].append({
            "agent": "KeywordAgent",
            "duration_s": round(time.time() - start_time, 1),
            "keywords_selected": len(keyword_result.get("selected_keywords", [])),
            "quality": keyword_result.get("_quality", {}),
        })

        # Step 3: ClassificationAgent (optional)
        if enable_classification:
            self.logger.info("MetaAgent: Starting ClassificationAgent")
            if self.stream_callback:
                self.stream_callback("\n\n📍 Phase 3: Klassifikation")

            class_agent = ClassificationAgent(
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                config=self._make_agent_config(),
                stream_callback=self.stream_callback,
            )

            class_input = {
                "abstract": abstract,
                "selected_keywords": keyword_result.get("selected_keywords", []),
            }

            class_result = class_agent.execute_with_self_reflection(class_input)
            results["classification"] = class_result
            results["agent_log"].append({
                "agent": "ClassificationAgent",
                "duration_s": round(time.time() - start_time, 1),
                "dk_count": len(class_result.get("dk_classifications", [])),
                "quality": class_result.get("_quality", {}),
            })

        # Step 4: ValidationAgent (optional)
        if enable_validation:
            self.logger.info("MetaAgent: Starting ValidationAgent")
            if self.stream_callback:
                self.stream_callback("\n\n📍 Phase 4: Validierung")

            val_agent = ValidationAgent(
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                config=self._make_agent_config(),
                stream_callback=self.stream_callback,
            )

            val_input = {
                "abstract": abstract,
                "selected_keywords": keyword_result.get("selected_keywords", []),
                "dk_classifications": results.get("classification", {}).get("dk_classifications", []),
            }

            val_result = val_agent.execute_with_self_reflection(val_input)
            results["validation"] = val_result
            results["agent_log"].append({
                "agent": "ValidationAgent",
                "duration_s": round(time.time() - start_time, 1),
                "quality_score": val_result.get("quality_score", 0),
            })

        # Summary
        total_time = time.time() - start_time
        results["total_duration_s"] = round(total_time, 1)

        if self.stream_callback:
            self.stream_callback("\n\n" + "=" * 50)
            self.stream_callback(f"\n✅ Agentic Pipeline abgeschlossen ({total_time:.1f}s)")
            kw_count = len(keyword_result.get("selected_keywords", []))
            dk_count = len(results.get("classification", {}).get("dk_classifications", []))
            self.stream_callback(f"\n   Schlagwörter: {kw_count}, Klassifikationen: {dk_count}")
            self.stream_callback("\n" + "=" * 50 + "\n")

        self.logger.info(f"MetaAgent completed in {total_time:.1f}s")
        return results

    def _make_agent_config(self) -> AgentConfig:
        """Create agent config from MetaAgent config."""
        return AgentConfig(
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            max_self_reflection_iterations=self.config.max_self_reflection_iterations,
            max_tool_iterations=self.config.max_tool_iterations,
            timeout_seconds=self.config.timeout_seconds,
            quality_threshold=self.config.quality_threshold,
        )
