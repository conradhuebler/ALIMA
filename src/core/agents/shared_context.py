"""Shared Context for MetaAgent Orchestration - Claude Generated

Provides ToolResultCache for avoiding redundant tool calls across SubAgents,
and SharedContext for passing data between pipeline stages.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResultCache:
    """Caches tool results to avoid redundant calls across SubAgents.

    Cache key is computed from tool_name + sorted_args_json.
    Stores results and tracks hit count for optimization metrics.
    """
    results: Dict[str, Any] = field(default_factory=dict)
    hit_count: Dict[str, int] = field(default_factory=dict)
    miss_count: Dict[str, int] = field(default_factory=dict)

    def get_cache_key(self, tool_name: str, args: dict) -> str:
        """Generate a deterministic cache key from tool name and arguments.

        Args:
            tool_name: Name of the tool
            args: Tool arguments as dict

        Returns:
            Cache key string
        """
        # Sort args to ensure consistent key generation
        args_json = json.dumps(args, sort_keys=True, default=str)
        args_hash = hashlib.md5(args_json.encode()).hexdigest()[:8]
        return f"{tool_name}:{args_hash}"

    def get(self, tool_name: str, args: dict) -> Optional[Any]:
        """Get cached result if available.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached result or None if not found
        """
        key = self.get_cache_key(tool_name, args)
        if key in self.results:
            self.hit_count[key] = self.hit_count.get(key, 0) + 1
            logger.debug(f"Cache HIT for {tool_name} (key: {key})")
            return self.results[key]

        self.miss_count[tool_name] = self.miss_count.get(tool_name, 0) + 1
        logger.debug(f"Cache MISS for {tool_name}")
        return None

    def set(self, tool_name: str, args: dict, result: Any) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Result to cache
        """
        key = self.get_cache_key(tool_name, args)
        self.results[key] = result
        logger.debug(f"Cached result for {tool_name} (key: {key})")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with total_hits, total_misses, cache_size, hit_rate
        """
        total_hits = sum(self.hit_count.values())
        total_misses = sum(self.miss_count.values())
        total_requests = total_hits + total_misses

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "cache_size": len(self.results),
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "tools_cached": list(set(k.split(":")[0] for k in self.results.keys())),
        }

    def clear(self) -> None:
        """Clear all cached results."""
        self.results.clear()
        self.hit_count.clear()
        self.miss_count.clear()


@dataclass
class SharedContext:
    """Shared state across all SubAgents in a MetaAgent execution.

    Provides:
    - Abstract and initial keywords (input data)
    - Tool result cache (shared across all SubAgents)
    - Conversation memory (shared LLM message history)
    - Step results (outputs from each SubAgent)
    - Quality scores (per-step quality metrics)
    - Provider/model configuration
    """
    # Input data
    abstract: str = ""
    initial_keywords: List[str] = field(default_factory=list)
    input_type: str = "text"
    source_value: Optional[str] = None

    # Shared resources
    tool_result_cache: ToolResultCache = field(default_factory=ToolResultCache)
    conversation_memory: List[Dict] = field(default_factory=list)

    # Pipeline results
    step_results: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # LLM configuration
    provider: str = ""
    model: str = ""
    temperature: float = 0.5
    max_tokens: int = 4096

    # Working data (built during execution)
    working_title: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
    gnd_entries: List[Dict] = field(default_factory=list)
    selected_keywords: List[Dict] = field(default_factory=list)
    dk_classifications: List[Dict] = field(default_factory=list)

    def get_step_result(self, step_name: str) -> Optional[Dict]:
        """Get result from a specific pipeline step.

        Args:
            step_name: Name of the step (e.g., 'extraction', 'search')

        Returns:
            Step result dict or None if not found
        """
        return self.step_results.get(step_name)

    def set_step_result(self, step_name: str, result: Dict, quality: float = None) -> None:
        """Store result from a pipeline step.

        Args:
            step_name: Name of the step
            result: Result dictionary
            quality: Optional quality score for this step
        """
        self.step_results[step_name] = result
        if quality is not None:
            self.quality_scores[step_name] = quality
        logger.debug(f"Stored result for step '{step_name}' with quality {quality}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to shared conversation memory.

        Args:
            role: Message role ('system', 'user', 'assistant', 'tool')
            content: Message content
        """
        self.conversation_memory.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """Get recent messages from conversation memory.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        return self.conversation_memory[-limit:]

    def to_keyword_analysis_state(self):
        """Convert SharedContext to KeywordAnalysisState for output compatibility.

        Returns:
            KeywordAnalysisState with all gathered data
        """
        from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis, SearchResult

        # Build LLM analysis result with required fields
        final_analysis = LlmKeywordAnalysis(
            task_name="meta_agent",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="meta_agent_workflow",
            filled_prompt="",
            temperature=self.temperature,
            seed=None,
            response_full_text=json.dumps({
                "keywords": self.selected_keywords,
                "classifications": self.dk_classifications,
                "working_title": self.working_title,
            }),
            extracted_gnd_keywords=[kw.get("gnd_id", kw.get("title", ""))
                                   for kw in self.selected_keywords],
            extracted_gnd_classes=[cls.get("code", "")
                                  for cls in self.dk_classifications],
        )

        # Convert GND entries to SearchResult objects
        # SearchResult structure: search_term + results dict
        search_results = []
        for entry in self.gnd_entries:
            gnd_id = entry.get("gnd_id", "")
            title = entry.get("title", "")
            # Build results dict structure
            results = {
                gnd_id: {
                    "title": title,
                    "description": entry.get("description", ""),
                    "ddc_codes": entry.get("ddc_codes", []),
                    "gnd_parent": entry.get("gnd_parent"),
                    "synonyms": entry.get("synonyms", []),
                }
            }
            search_results.append(SearchResult(
                search_term=title,
                results=results,
            ))

        return KeywordAnalysisState(
            original_abstract=self.abstract,
            initial_keywords=self.initial_keywords,
            search_suggesters_used=["meta_agent"],
            working_title=self.working_title,
            final_llm_analysis=final_analysis,
            search_results=search_results,
            dk_classifications=[cls.get("code", "") for cls in self.dk_classifications],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SharedContext to a JSON-compatible dict for saving intermediate state.

        Tool result cache is NOT serialized (not needed for warm-start).

        Returns:
            Dict with all pipeline state fields
        """
        return {
            "abstract": self.abstract,
            "initial_keywords": self.initial_keywords,
            "input_type": self.input_type,
            "source_value": self.source_value,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "working_title": self.working_title,
            "extracted_keywords": self.extracted_keywords,
            "gnd_entries": self.gnd_entries,
            "selected_keywords": self.selected_keywords,
            "dk_classifications": self.dk_classifications,
            "step_results": self.step_results,
            "quality_scores": self.quality_scores,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedContext":
        """Deserialize SharedContext from a dict (e.g. loaded from JSON file).

        Creates a fresh ToolResultCache — the cache is not restored from disk.

        Args:
            data: Dict as produced by to_dict()

        Returns:
            SharedContext with pre-populated pipeline state
        """
        ctx = cls(
            abstract=data.get("abstract", ""),
            initial_keywords=data.get("initial_keywords", []),
            input_type=data.get("input_type", "text"),
            source_value=data.get("source_value"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.5),
            max_tokens=data.get("max_tokens", 4096),
        )
        ctx.working_title = data.get("working_title", "")
        ctx.extracted_keywords = data.get("extracted_keywords", [])
        ctx.gnd_entries = data.get("gnd_entries", [])
        ctx.selected_keywords = data.get("selected_keywords", [])
        ctx.dk_classifications = data.get("dk_classifications", [])
        ctx.step_results = data.get("step_results", {})
        ctx.quality_scores = data.get("quality_scores", {})
        return ctx

    def save_to_file(self, path: str) -> None:
        """Save context state to a JSON file.

        Args:
            path: File path to write to
        """
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved SharedContext to {path}")

    @classmethod
    def load_from_file(cls, path: str) -> "SharedContext":
        """Load context state from a JSON file.

        Args:
            path: File path to read from

        Returns:
            SharedContext with restored pipeline state
        """
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        ctx = cls.from_dict(data)
        logger.info(f"Loaded SharedContext from {path} "
                    f"(steps completed: {list(ctx.step_results.keys())})")
        return ctx

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the context state.

        Returns:
            Dict with key context information
        """
        return {
            "abstract_length": len(self.abstract),
            "initial_keywords_count": len(self.initial_keywords),
            "working_title": self.working_title,
            "extracted_keywords_count": len(self.extracted_keywords),
            "gnd_entries_count": len(self.gnd_entries),
            "selected_keywords_count": len(self.selected_keywords),
            "dk_classifications_count": len(self.dk_classifications),
            "cache_stats": self.tool_result_cache.get_stats(),
            "steps_completed": list(self.step_results.keys()),
            "provider": f"{self.provider}/{self.model}",
        }