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
    chunk_responses: List[str] = field(default_factory=list)  # Intermediate responses from chunked analysis - Claude Generated
    missing_concepts: List[str] = field(default_factory=list)  # Missing concepts identified for iterative refinement - Claude Generated
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
    dk_classifications: List[str] = field(default_factory=list)  # For final DK classification codes - Claude Generated

    # Iterative refinement support - Claude Generated
    refinement_iterations: List[Dict[str, Any]] = field(default_factory=list)  # Iteration history with metadata
    convergence_achieved: bool = False  # True if converged before max iterations
    max_iterations_reached: bool = False  # True if stopped due to max iterations


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
