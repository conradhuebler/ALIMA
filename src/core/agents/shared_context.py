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
    verbose: bool = False  # Log full prompts to stream + logger when True

    # Working data (built during execution)
    working_title: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
    gnd_entries: List[Dict] = field(default_factory=list)
    selected_keywords: List[Dict] = field(default_factory=list)
    keyword_chains: List[Dict] = field(default_factory=list)  # Schlagwortketten: [{chain: [...], reason: "..."}]
    missing_concepts: List[str] = field(default_factory=list)  # From selection, drives feedback loop
    dk_classifications: List[Dict] = field(default_factory=list)
    rvk_classifications: List[Dict] = field(default_factory=list)  # RVK classifications from classification step
    dk_search_results: List[Dict] = field(default_factory=list)  # DK catalog search results

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

        Produces a state that is fully compatible with the hardcoded pipeline's
        KeywordAnalysisState, including search_results, Schlagwortketten,
        DK classifications with statistics, and RVK provenance.

        Returns:
            KeywordAnalysisState with all gathered data
        """
        from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis, SearchResult

        # --- keyword titles (selected, GND-verified) ---
        kw_titles = [kw.get("title", kw.get("gnd_id", ""))
                     for kw in self.selected_keywords
                     if kw.get("title") or kw.get("gnd_id")]

        # --- DK codes as plain strings ---
        dk_codes = [cls.get("code", "") for cls in self.dk_classifications if cls.get("code")]

        # --- initial GND classes from entries with DDC codes ---
        initial_gnd_classes = []
        seen_ddc = set()
        for entry in self.gnd_entries:
            for ddc in entry.get("ddc_codes", []):
                if ddc and ddc not in seen_ddc:
                    seen_ddc.add(ddc)
                    initial_gnd_classes.append(ddc)

        # --- dk_search_results_flattened: [{dk, classification_type, titles, count, reasoning}] ---
        dk_search_results_flattened = []
        for cls in self.dk_classifications:
            code = cls.get("code", "")
            title = cls.get("title", "")
            reason = cls.get("reason", cls.get("reasoning", ""))
            if code:
                dk_search_results_flattened.append({
                    "dk": code,
                    "classification_type": "DK",
                    "titles": [title] if title else [],
                    "count": int(cls.get("confidence", 0) * 100),
                    "reasoning": reason,
                })

        # --- DK statistics ---
        dk_statistics = None
        if dk_search_results_flattened:
            from collections import Counter
            dk_counter = Counter(r["dk"] for r in dk_search_results_flattened)
            total = sum(dk_counter.values())
            unique = len(dk_counter)
            top_10 = [
                {"dk": dk, "count": cnt, "percentage": round(cnt / total * 100, 1) if total else 0}
                for dk, cnt in dk_counter.most_common(10)
            ]
            dk_statistics = {
                "total_classifications": total,
                "unique_dk_codes": unique,
                "deduplication_ratio": round(unique / total, 2) if total else 0,
                "top_10": top_10,
            }

        # --- search_results: GUI expects {title → {gndid: [gnd_id], ddc_codes: [...]}} format ---
        # Build from gnd_entries for full coverage (not just selected_keywords)
        # so the SearchTab can display all results
        search_results_dict: Dict[str, Dict] = {}
        for entry in self.gnd_entries:
            title = entry.get("title", "")
            if not title:
                continue
            gnd_ids = entry.get("gnd_ids", [])
            primary_gnd_id = entry.get("gnd_id", "")
            all_gnd_ids = list(gnd_ids) if gnd_ids else ([primary_gnd_id] if primary_gnd_id else [])
            search_results_dict[title] = {
                "gndid": all_gnd_ids,
                "ddc_codes": entry.get("ddc_codes", []),
            }
        # One SearchResult per extracted keyword (search_term = original search term)
        search_results = []
        for ekw in (self.extracted_keywords or ["meta_agent"]):
            search_results.append(SearchResult(
                search_term=ekw,
                results=search_results_dict,
            ))

        # --- Build Schlagwortketten text for final_llm_analysis ---
        chain_lines = []
        for chain_data in self.keyword_chains:
            chain_terms = chain_data.get("chain", [])
            reason = chain_data.get("reason", "")
            chain_str = " → ".join(chain_terms) if chain_terms else ""
            if chain_str:
                line = f"  {chain_str}"
                if reason:
                    line += f" ({reason})"
                chain_lines.append(line)

        # --- initial_llm_call_details: extraction step → abstract_tab ---
        extraction_result = self.step_results.get("extraction", {})
        initial_analysis = LlmKeywordAnalysis(
            task_name="extraction",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="initialisation",
            filled_prompt="",
            temperature=self.temperature,
            seed=None,
            response_full_text=extraction_result.get("raw_output", ""),
            extracted_gnd_keywords=self.extracted_keywords,
        )

        # --- final_llm_analysis: keyword selection → verifikation tab / analyse_keywords ---
        selection_result = self.step_results.get("selection", {})
        kw_lines = "\n".join(
            f"- {kw.get('title', kw.get('gnd_id', ''))} (GND-ID: {kw.get('gnd_id', '')})"
            for kw in self.selected_keywords
        )
        response_parts = [f"Ausgewählte GND-Schlagworte ({len(self.selected_keywords)}):\n{kw_lines}"]
        if chain_lines:
            response_parts.append(f"\nSchlagwortketten ({len(self.keyword_chains)}):\n" + "\n".join(chain_lines))
        final_analysis = LlmKeywordAnalysis(
            task_name="keywords",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="keywords_chunked",
            filled_prompt="",
            temperature=self.temperature,
            seed=None,
            response_full_text="\n".join(response_parts),
            extracted_gnd_keywords=kw_titles,
            missing_concepts=self.missing_concepts,
        )

        # --- dk_llm_analysis: classification step → dk_analysis_tab ---
        classification_result = self.step_results.get("classification", {})
        dk_reasoning = classification_result.get("reasoning", "")
        dk_raw = classification_result.get("raw_output", "")
        dk_text = dk_reasoning or dk_raw or ""
        dk_analysis = LlmKeywordAnalysis(
            task_name="dk_classification",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="dk_classification",
            filled_prompt="",
            temperature=self.temperature,
            seed=None,
            response_full_text=dk_text,
            extracted_gnd_classes=dk_codes,
        ) if dk_text else None

        # --- RVK provenance ---
        rvk_provenance = {}
        if self.rvk_classifications:
            for rvk in self.rvk_classifications:
                code = rvk.get("code", "")
                if code:
                    rvk_provenance[code] = {
                        "source": "agentic_classification",
                        "confidence": rvk.get("confidence", 0),
                        "title": rvk.get("title", ""),
                    }

        state = KeywordAnalysisState(
            original_abstract=self.abstract,
            initial_keywords=self.extracted_keywords or self.initial_keywords,
            search_suggesters_used=["swb", "lobid"],
            working_title=self.working_title,
            input_type=self.input_type,
            source_value=self.source_value,
            initial_gnd_classes=initial_gnd_classes,
            initial_llm_call_details=initial_analysis,
            final_llm_analysis=final_analysis,
            dk_llm_analysis=dk_analysis,
            search_results=search_results,
            dk_classifications=dk_codes,
        )
        state.dk_search_results_flattened = dk_search_results_flattened
        state.dk_search_results = self.dk_search_results
        state.dk_statistics = dk_statistics
        if rvk_provenance:
            state.rvk_provenance = rvk_provenance
        return state

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
            "keyword_chains": self.keyword_chains,
            "missing_concepts": self.missing_concepts,
            "dk_classifications": self.dk_classifications,
            "rvk_classifications": self.rvk_classifications,
            "dk_search_results": self.dk_search_results,
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
        ctx.keyword_chains = data.get("keyword_chains", [])
        ctx.missing_concepts = data.get("missing_concepts", [])
        ctx.dk_classifications = data.get("dk_classifications", [])
        ctx.rvk_classifications = data.get("rvk_classifications", [])
        ctx.dk_search_results = data.get("dk_search_results", [])
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