from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


@dataclass
class AbstractData:
    abstract: str
    keywords: str = ""


@dataclass
class KeywordResult:
    keyword: str
    gnd_id: str


@dataclass
class AnalysisResult:
    full_text: str
    matched_keywords: Dict[str, str] = field(default_factory=dict)
    gnd_systematic: Optional[str] = None
    keyword_chains: List[Dict] = field(default_factory=list)  # Schlagwortketten - Claude Generated


@dataclass
class PromptConfigData:
    prompt: str
    system: str
    temp: float
    p_value: float
    models: List[str]
    seed: Optional[int]
    output_format: Optional[str] = None  # None/"json" = JSON-Modus (default), "xml" = legacy - Claude Generated


@dataclass
class TaskState:
    abstract_data: AbstractData
    analysis_result: AnalysisResult
    prompt_config: Optional[PromptConfigData] = None
    status: str = "pending"  # e.g., pending, completed, failed
    task_name: Optional[str] = None
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    use_chunking_abstract: Optional[bool] = False
    abstract_chunk_size: Optional[int] = None
    use_chunking_keywords: Optional[bool] = False
    keyword_chunk_size: Optional[int] = None


@dataclass
class SearchResult:
    """Strukturierte Darstellung der Suchergebnisse für einen Suchbegriff."""

    search_term: str
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class LlmKeywordAnalysis:
    """Strukturierte Darstellung der LLM-Analyseergebnisse mit Details zum Aufruf."""

    task_name: str
    model_used: str
    provider_used: str
    prompt_template: str
    filled_prompt: str
    temperature: float
    seed: Optional[int]
    response_full_text: str
    extracted_gnd_keywords: List[str] = field(default_factory=list)
    extracted_gnd_classes: List[str] = field(default_factory=list)
    analyse_text: Optional[str] = None  # Analysis/thought section from LLM response - Claude Generated
    chunk_responses: List[str] = field(default_factory=list)  # Intermediate responses from chunked analysis - Claude Generated
    chunk_keywords: List[str] = field(default_factory=list)  # Deduplicated chunk-survivor keywords (pre-consolidation) for the GND-Recherche chunk tier - Claude Generated
    missing_concepts: List[str] = field(default_factory=list)  # Missing concepts identified for iterative refinement - Claude Generated
    keyword_chains: List[Dict] = field(default_factory=list)  # Schlagwortketten with reasons from LLM response - Claude Generated
    verification: Optional[Dict[str, Any]] = None  # GND pool verification results - Claude Generated


@dataclass
class KeywordAnalysisState:
    """Kapselt den gesamten Zustand des Keyword-Analyse-Workflows."""

    original_abstract: Optional[str]
    initial_keywords: List[str]
    search_suggesters_used: List[str]
    working_title: Optional[str] = None  # LLM-generated work title for identification - Claude Generated
    input_type: Optional[str] = None    # 'text', 'doi', 'pdf', 'img', 'url' - Claude Generated
    source_value: Optional[str] = None  # Original DOI, file path, URL, or None for plain text - Claude Generated
    initial_gnd_classes: List[str] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    initial_llm_call_details: Optional[LlmKeywordAnalysis] = None
    final_llm_analysis: Optional[LlmKeywordAnalysis] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_step_completed: Optional[str] = None  # For recovery tracking - Claude Generated
    dk_search_results: List[Dict[str, Any]] = field(default_factory=list)  # For DK catalog search results (keyword-centric) - Claude Generated
    dk_search_results_flattened: List[Dict[str, Any]] = field(default_factory=list)  # Deduplicated classifications for LLM prompt - Claude Generated Step 6
    dk_statistics: Optional[Dict[str, Any]] = None  # Deduplication metrics and frequency statistics - Claude Generated Step 6
    dk_llm_analysis: Optional[LlmKeywordAnalysis] = None  # For LLM classification details (AbstractTab view) - Claude Generated
    dk_classifications: List[str] = field(default_factory=list)  # Legacy alias for final DK/RVK classification strings - Claude Generated
    report_markdown: str = ""  # Generic workflow-report Markdown (e.g. title_list_search's duplicate table), rendered as an HTML block in the GUI - Claude Generated

    # Iterative refinement support - Claude Generated
    refinement_iterations: List[Dict[str, Any]] = field(default_factory=list)  # Iteration history with metadata
    convergence_achieved: bool = False  # True if converged before max iterations
    max_iterations_reached: bool = False  # True if stopped due to max iterations

    @property
    def classifications(self) -> List[str]:
        """Preferred neutral alias for final DK/RVK/DDC classification strings."""
        return self.dk_classifications

    @classifications.setter
    def classifications(self, value: List[str]) -> None:
        self.dk_classifications = list(value or [])

    # ------------------------------------------------------------------
    # Notation-agnostic aliases (general-notation generalization, WS2).
    # The ``dk_*`` fields stay the canonical/persisted names (saved-state JSON
    # back-compat); ``notation_*`` is the preferred name for new code that
    # treats DK/DDC/RVK uniformly. - Claude Generated
    # ------------------------------------------------------------------
    @property
    def notation_codes(self) -> List[str]:
        return self.dk_classifications

    @notation_codes.setter
    def notation_codes(self, value: List[str]) -> None:
        self.dk_classifications = list(value or [])

    @property
    def notation_search_results(self) -> List[Dict[str, Any]]:
        return self.dk_search_results

    @notation_search_results.setter
    def notation_search_results(self, value: List[Dict[str, Any]]) -> None:
        self.dk_search_results = value

    @property
    def notation_search_results_flattened(self) -> List[Dict[str, Any]]:
        return self.dk_search_results_flattened

    @notation_search_results_flattened.setter
    def notation_search_results_flattened(self, value: List[Dict[str, Any]]) -> None:
        self.dk_search_results_flattened = value

    @property
    def notation_statistics(self) -> Optional[Dict[str, Any]]:
        return self.dk_statistics

    @notation_statistics.setter
    def notation_statistics(self, value: Optional[Dict[str, Any]]) -> None:
        self.dk_statistics = value

    # ------------------------------------------------------------------
    # WP10 P-δ.1: Mutations-API (additive — direct field writes elsewhere
    # in PipelineManager remain in place). Each method emits a state event
    # via AlimaStateBus so subscribers (Review/Chat/Webapp) can refresh.
    # ------------------------------------------------------------------
    def _emit_state_event(self, op: str, **payload) -> None:
        """Best-effort emit on AlimaStateBus. Silent on import failure."""
        try:
            from src.core.state_bus import AlimaStateBus
            AlimaStateBus().emit_event("state.changed", {"op": op, **payload})
        except Exception:
            pass

    def apply_keyword_replacement(
        self, old: str, new: str, gnd_id: Optional[str] = None
    ) -> bool:
        """Replace ``old`` with ``new`` in ``initial_keywords``.

        Returns True if the replacement happened.
        """
        if not old or not new:
            return False
        replaced = False
        out: List[str] = []
        for kw in self.initial_keywords:
            if kw == old or kw.split(" (GND-ID:")[0].strip() == old:
                tag = f" (GND-ID: {gnd_id})" if gnd_id else ""
                out.append(f"{new}{tag}")
                replaced = True
            else:
                out.append(kw)
        self.initial_keywords = out
        if replaced:
            self._emit_state_event(
                "keyword_replacement", old=old, new=new, gnd_id=gnd_id
            )
        return replaced

    def apply_keyword_addition(
        self, keyword: str, gnd_id: Optional[str] = None
    ) -> bool:
        """Append a keyword to ``initial_keywords`` if not already present."""
        if not keyword:
            return False
        tag = f" (GND-ID: {gnd_id})" if gnd_id else ""
        entry = f"{keyword}{tag}"
        if entry in self.initial_keywords:
            return False
        self.initial_keywords = list(self.initial_keywords) + [entry]
        self._emit_state_event(
            "keyword_addition", keyword=keyword, gnd_id=gnd_id
        )
        return True

    def apply_keyword_removal(self, keyword: str) -> bool:
        """Remove the first keyword whose canonical form matches ``keyword``."""
        if not keyword:
            return False
        for i, kw in enumerate(self.initial_keywords):
            canon = kw.split(" (GND-ID:")[0].strip()
            if kw == keyword or canon == keyword:
                self.initial_keywords = (
                    list(self.initial_keywords[:i])
                    + list(self.initial_keywords[i + 1:])
                )
                self._emit_state_event("keyword_removal", keyword=keyword)
                return True
        return False

    def apply_classification_update(
        self, code: str, action: str = "add"
    ) -> bool:
        """Add or remove a classification code in ``dk_classifications``.

        ``action`` ∈ {``"add"``, ``"remove"``}.
        """
        if not code or action not in ("add", "remove"):
            return False
        codes = list(self.dk_classifications)
        if action == "add":
            if code in codes:
                return False
            codes.append(code)
        else:  # remove
            if code not in codes:
                return False
            codes.remove(code)
        self.dk_classifications = codes
        self._emit_state_event(
            "classification_update", code=code, action=action
        )
        return True

    def apply_step_result_override(
        self, step_id: str, key: str, value: object
    ) -> bool:
        """Override ``key`` on the ``LlmKeywordAnalysis`` for ``step_id``.

        Currently supports ``step_id="initial"`` (``initial_llm_call_details``)
        and ``step_id="final"`` (``final_llm_analysis``).
        """
        target = None
        if step_id == "initial":
            target = self.initial_llm_call_details
        elif step_id == "final":
            target = self.final_llm_analysis
        if target is None or not hasattr(target, key):
            return False
        setattr(target, key, value)
        self._emit_state_event(
            "step_result_override", step_id=step_id, key=key
        )
        return True


# ============================================================
# Agent / Tool-Calling Data Models - Claude Generated
# ============================================================

@dataclass
class ToolCall:
    """Represents a single tool call requested by an LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str
    content: str
    is_error: bool = False


class StopReason(Enum):
    """Why the LLM stopped generating."""
    END_TURN = "end_turn"        # Normal completion
    TOOL_USE = "tool_use"        # Wants to call tools
    MAX_TOKENS = "max_tokens"    # Hit token limit
    CANCELLED = "cancelled"      # User cancelled


@dataclass
class AgentResponse:
    """Response from an LLM that may contain tool calls."""
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    # Separate reasoning/thinking channel (e.g. OpenAI-compat reasoning_content).
    # Captured so a model that answers only in its reasoning channel — or runs out
    # of tokens mid-reasoning — doesn't surface as a silent empty response. - Claude Generated
    reasoning: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class AgentResult:
    """Final result from a complete agent run (potentially multi-turn)."""
    content: str
    tool_log: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tokens_used: int = 0
    agent_name: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)  # full conversation including tool calls
    # Stop reason of the final LLM turn (string form of StopReason) for logging
    # and empty-response diagnosis. - Claude Generated
    stop_reason: str = ""
    # Set when the run aborted on an LLM failure: content then holds an
    # error string, NOT a model answer. Callers must check this instead of
    # treating the run as successful - Claude Generated
    error: Optional[str] = None
