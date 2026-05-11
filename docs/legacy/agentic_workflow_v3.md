# Agentic Workflow Architecture

**Feature Status**: Planned (Experimental)
**Priority**: LOW (Long-term Enhancement)
**Effort**: 5-7 days
**Risk**: HIGH (Architectural change, experimental)

## Executive Summary

This feature introduces a self-reflective, multi-agent architecture that replaces ALIMA's linear pipeline with an intelligent, adaptive system. Specialized agents (SearchAgent, KeywordAgent, ClassificationAgent) work collaboratively under a Meta-Agent orchestrator, each validating their own outputs, selecting strategies dynamically, and iterating until quality thresholds are met.

**Key Capabilities**:
1. **Self-Reflection**: Agents evaluate their own outputs and retry if quality is insufficient
2. **Multi-Agent Collaboration**: Specialized agents with distinct responsibilities work together
3. **Adaptive Strategy Selection**: Dynamic prompt/model selection based on text characteristics
4. **Hierarchical Planning**: Meta-agent analyzes tasks and coordinates sub-agent execution

**Warning**: This is an experimental feature with significant token usage increase (3x). Enable only for quality-critical tasks.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Agent Specifications](#agent-specifications)
4. [Implementation Details](#implementation-details)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)
8. [Testing Strategy](#testing-strategy)
9. [Future Extensions](#future-extensions)

---

## Problem Statement

### Current Behavior (Linear Pipeline)

```
Input → Initialisation → Search → Keywords → Classification → Output
```

**Limitations**:
1. **No Quality Feedback**: Pipeline proceeds regardless of quality
2. **Fixed Strategy**: Same approach for all text types (scientific, fiction, reports)
3. **No Self-Correction**: Errors propagate through pipeline without detection
4. **Manual Intervention**: User must manually retry if quality is poor
5. **Brittle**: Single point of failure at each step

### Example Failure Scenario

**Input**: Scientific abstract with specialized terminology

**Linear Pipeline**:
1. Initialisation: Generates keywords (quality unknown)
2. Search: Returns sparse results (low coverage, but proceeds anyway)
3. Keywords: Selects from poor GND pool (quality degraded)
4. Classification: Works with suboptimal keywords (final output poor)
5. **Result**: Low-quality output, user must manually restart

**Desired Agentic Behavior**:
1. Meta-Agent: Analyzes abstract → Detects "scientific" type
2. SearchAgent: Executes search → Self-validates → Coverage 40% (below threshold 50%)
3. SearchAgent: Reflects → Adjusts strategy → Re-executes → Coverage 60% (acceptable)
4. KeywordAgent: Selects keywords → Self-validates → Relevance 85% (good)
5. ValidationAgent: Cross-checks → Overall quality 75% (acceptable)
6. **Result**: High-quality output, automatic quality assurance

---

## Solution Architecture

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    META-AGENT (Orchestrator)                        │
│                                                                    │
│  Responsibilities:                                                 │
│  • Analyze abstract → Determine text type (scientific/fiction/...) │
│  • Plan execution strategy → Select models, parameters             │
│  • Coordinate sub-agents → Execute in optimal order                │
│  • Aggregate results → Produce final output                        │
│  • Trigger retries → If validation fails                           │
└───────────────────┬────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        │           │           │              │
        ▼           ▼           ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│SearchAgent   │ │KeywordAgent │ │Classification│ │Validation    │
│              │ │             │ │Agent         │ │Agent         │
├──────────────┤ ├─────────────┤ ├──────────────┤ ├──────────────┤
│• GND search  │ │• Keyword    │ │• DK/RVK      │ │• Cross-check │
│• Strategy    │ │  selection  │ │  selection   │ │  outputs     │
│  selection   │ │• Quality    │ │• Diversity   │ │• Detect      │
│• Coverage    │ │  validation │ │  check       │ │  inconsis-   │
│  calculation │ │• Relevance  │ │• Self-       │ │  tencies     │
│• Self-       │ │  scoring    │ │  validation  │ │• Quality     │
│  validation  │ │• Iteration  │ │              │ │  metrics     │
└──────────────┘ └─────────────┘ └──────────────┘ └──────────────┘
        │                │                 │               │
        └────────────────┼─────────────────┴───────────────┘
                         │
                         ▼
                   ┌───────────┐
                   │ FINAL     │
                   │ RESULTS   │
                   └───────────┘
```

### Agent Interaction Flow

```
PHASE 1: Text Analysis
─────────────────────
Meta-Agent: Analyze abstract
    ├─> Detect text type: "scientific"
    ├─> Select strategy: "Wissenschaftliche Präzision"
    ├─> Choose model: cogito:32b (high accuracy)
    └─> Set parameters: min_keywords=10, temperature=0.25

PHASE 2: Search Execution
──────────────────────────
SearchAgent: Execute with self-reflection
    Iteration 1:
        ├─> Execute GND search for initial keywords
        ├─> Calculate coverage: 40%
        ├─> Self-validate: FAIL (below 50% threshold)
        └─> Reflect: "Low coverage - try broader terms"

    Iteration 2:
        ├─> Adjust strategy: Enable hierarchy search
        ├─> Execute search with broader terms
        ├─> Calculate coverage: 60%
        ├─> Self-validate: PASS (above threshold)
        └─> Return AgentResult(success=True, confidence=0.6, data={...})

PHASE 3: Keyword Selection
───────────────────────────
KeywordAgent: Execute with self-reflection
    Iteration 1:
        ├─> Call LLM for keyword selection
        ├─> Calculate relevance: 85%
        ├─> Self-validate: PASS (above 70% threshold)
        └─> Return AgentResult(success=True, confidence=0.85, data={keywords})

PHASE 4: Cross-Validation
──────────────────────────
ValidationAgent: Cross-check consistency
    ├─> Check search coverage: 60% ✓
    ├─> Check keyword quality: 85% ✓
    ├─> Check search-keyword consistency: 92% ✓
    └─> Overall quality: 79% ✓ (above 70% threshold)

PHASE 5: Classification (Optional)
───────────────────────────────────
ClassificationAgent: Select DK/RVK classifications
    ├─> Execute with self-reflection
    ├─> Calculate diversity: 0.75
    └─> Return classifications

PHASE 6: Aggregation
────────────────────
Meta-Agent: Combine all results
    ├─> Final keywords: 12 keywords
    ├─> Classifications: 8 DK codes
    ├─> Overall quality: 79%
    ├─> Agent iterations: Search=2, Keyword=1
    └─> Return to user
```

---

## Agent Specifications

### BaseAgent (Abstract Class)

**Purpose**: Provides common functionality for all agents

**Key Methods**:
- `execute()`: Perform agent's primary task (abstract)
- `self_validate()`: Check own output quality (abstract)
- `reflect_on_result()`: Identify potential issues
- `execute_with_self_reflection()`: Execute with auto-retry

**Self-Reflection Loop**:
```python
for iteration in range(1, max_iterations + 1):
    # Execute task
    result = self.execute(input_data, iteration=iteration)

    # Self-validate
    if self.self_validate(result):
        return result  # Quality acceptable

    # Reflect on issues
    reflection = self.reflect_on_result(result)

    # Last iteration? Return even if not perfect
    if iteration >= max_iterations:
        return result

    # Adjust strategy and retry
    kwargs = self._adjust_strategy_for_retry(result, kwargs)
```

**Data Structure**:
```python
@dataclass
class AgentResult:
    success: bool  # Did agent complete successfully?
    data: Any  # Agent-specific output
    confidence: float  # 0.0-1.0 (agent's confidence in result)
    quality_metrics: Dict[str, float]  # Detailed quality scores
    error_message: Optional[str]  # If failed
    iteration_count: int  # How many iterations needed
    reflection_notes: List[str]  # Self-reflection observations
```

### SearchAgent

**Responsibility**: Execute GND/SWB searches with adaptive strategy selection

**Input**:
```python
{
    "abstract": str,
    "initial_keywords": List[str],
    "text_type": str  # "scientific", "fiction", "report", "general"
}
```

**Output**:
```python
AgentResult(
    success=True,
    data={...search_results...},
    confidence=0.65,  # Coverage score
    quality_metrics={
        "coverage": 0.65,  # % keywords with GND matches
        "avg_results_per_keyword": 4.2
    }
)
```

**Quality Validation**:
- **Coverage >= 50%**: At least half of keywords found in GND
- **Avg results >= 2**: Each keyword has multiple GND options

**Adaptive Strategy**:

| Text Type | Search Depth | Hierarchy | Max Results |
|-----------|-------------|-----------|-------------|
| Scientific | Precise | Enabled | 10 |
| Fiction | Broad | Disabled | 20 |
| Report | Balanced | Enabled | 15 |
| General | Balanced | Enabled | 15 |

**Self-Reflection Logic**:
```python
def reflect_on_result(self, result: AgentResult) -> List[str]:
    notes = []

    coverage = result.quality_metrics.get("coverage", 0.0)
    if coverage < 0.5:
        notes.append(f"Low coverage: {coverage:.0%} - try broader search terms")

    avg_results = result.quality_metrics.get("avg_results_per_keyword", 0.0)
    if avg_results < 2.0:
        notes.append(f"Sparse results ({avg_results:.1f}/keyword) - try hierarchy search")

    return notes
```

### KeywordAgent

**Responsibility**: Select optimal GND keywords from search results

**Input**:
```python
{
    "abstract": str,
    "search_results": Dict[str, Any],
    "min_keywords": int,
    "max_keywords": int
}
```

**Output**:
```python
AgentResult(
    success=True,
    data={
        "keywords": List[str],  # Selected GND keywords
        "llm_analysis": LlmKeywordAnalysis
    },
    confidence=0.82,  # Relevance score
    quality_metrics={
        "relevance": 0.82,  # How relevant are keywords to abstract?
        "count": 12,  # Number of keywords selected
        "in_range": True  # Within min/max bounds?
    }
)
```

**Quality Validation**:
- **Relevance >= 70%**: Keywords highly relevant to abstract
- **Count in range**: Between min_keywords and max_keywords
- **All have GND-IDs**: Every keyword is GND-compliant

**Relevance Calculation**:
```python
def _calculate_relevance(self, abstract: str, keywords: List[str]) -> float:
    """
    Estimate keyword relevance using term overlap.

    Method:
        1. Count how many keywords appear (partially) in abstract
        2. Relevance = matched_keywords / total_keywords
    """
    abstract_lower = abstract.lower()

    overlap_count = sum(
        1 for kw in keywords
        if any(
            term.lower() in abstract_lower
            for term in kw.split()[:3]  # First 3 words of keyword
        )
    )

    return overlap_count / len(keywords)
```

### ClassificationAgent

**Responsibility**: Select relevant DK/RVK classifications

**Input**:
```python
{
    "abstract": str,
    "dk_search_results": List[Dict],
    "keywords": List[str]
}
```

**Output**:
```python
AgentResult(
    success=True,
    data=[...selected_classifications...],
    confidence=0.68,  # Diversity score
    quality_metrics={
        "diversity": 0.68,  # % unique top-level categories
        "count": 8
    }
)
```

**Quality Validation**:
- **Diversity >= 50%**: Classifications from multiple top-level categories
- **Count 3-15**: Reasonable number of classifications

**Diversity Calculation**:
```python
def _calculate_diversity(self, classifications: List[str]) -> float:
    """
    Measure classification diversity.

    Method:
        - For DK codes: Extract first 3 digits (e.g., "616" from "616.89")
        - For RVK codes: Extract first 2 letters (e.g., "QP" from "QP 340")
        - Diversity = unique_top_level / total_classifications
    """
    top_level_categories = set()

    for code in classifications:
        if "DK" in code.upper():
            numbers = ''.join(filter(str.isdigit, code))
            if len(numbers) >= 3:
                top_level_categories.add(numbers[:3])
        elif any(c.isalpha() for c in code):  # RVK
            letters = ''.join(filter(str.isalpha, code))
            if len(letters) >= 2:
                top_level_categories.add(letters[:2])

    return len(top_level_categories) / len(classifications) if classifications else 0.0
```

### ValidationAgent

**Responsibility**: Cross-validate outputs from multiple agents

**Input**:
```python
{
    "search_result": AgentResult,  # From SearchAgent
    "keyword_result": AgentResult,  # From KeywordAgent
    "classification_result": AgentResult  # From ClassificationAgent (optional)
}
```

**Output**:
```python
AgentResult(
    success=True,  # Overall validation passed
    data={
        "search_coverage": 0.65,
        "keyword_quality": 0.82,
        "keyword_search_consistency": 0.92,
        "classification_quality": 0.68
    },
    confidence=0.77,  # Average of all checks
    quality_metrics={...}
)
```

**Validation Checks**:

1. **Search Coverage**: >= 50% (from SearchAgent metrics)
2. **Keyword Quality**: >= 70% (from KeywordAgent metrics)
3. **Consistency Check**: Selected keywords match search results
4. **Classification Quality**: >= 50% diversity (if applicable)

**Consistency Logic**:
```python
def _validate_consistency(
    self,
    search_result: AgentResult,
    keyword_result: AgentResult
) -> float:
    """
    Verify selected keywords came from search results.

    Returns:
        Consistency score (0.0-1.0)
    """
    search_data = search_result.data
    keywords = keyword_result.data.get("keywords", [])

    if not keywords:
        return 0.0

    # Count keywords with GND-IDs (indicating they came from search)
    matched = sum(1 for kw in keywords if "GND-ID" in kw)

    return matched / len(keywords)
```

### MetaAgent (Orchestrator)

**Responsibility**: Analyze task, select strategy, coordinate sub-agents

**Text Type Detection**:
```python
def _analyze_text_type(self, abstract: str) -> str:
    """
    Analyze abstract to determine text type.

    Detection rules:
        - Scientific: Contains "methode", "studie", "analyse", "hypothese", etc.
        - Fiction: Contains "roman", "protagonist", "kapitel", "geschichte", etc.
        - Report: Contains "bericht", "empfehlung", "fazit", etc.
        - General: Default
    """
    abstract_lower = abstract.lower()

    scientific_keywords = [
        "methode", "ergebnis", "studie", "analyse", "untersuchung",
        "hypothese", "daten", "experiment", "conclusion"
    ]
    fiction_keywords = [
        "roman", "erzählung", "protagonist", "kapitel", "geschichte"
    ]

    scientific_count = sum(1 for kw in scientific_keywords if kw in abstract_lower)
    fiction_count = sum(1 for kw in fiction_keywords if kw in abstract_lower)

    if scientific_count >= 2:
        return "scientific"
    elif fiction_count >= 2:
        return "fiction"
    else:
        return "general"
```

**Strategy Planning**:
```python
def _plan_execution_strategy(
    self,
    abstract: str,
    text_type: str
) -> Dict[str, Any]:
    """
    Select optimal execution strategy based on text type.

    Strategies:
        - Scientific: High accuracy model, strict parameters, more keywords
        - Fiction: Fast model, broader parameters, fewer keywords
        - General: Balanced approach
    """
    strategies = {
        "scientific": {
            "name": "Wissenschaftliche Präzision",
            "min_keywords": 10,
            "max_keywords": 20,
            "preferred_model": "cogito:32b",  # More accurate
            "temperature": 0.25  # Low variance
        },
        "fiction": {
            "name": "Literarische Erschließung",
            "min_keywords": 5,
            "max_keywords": 15,
            "preferred_model": "cogito:14b",  # Faster
            "temperature": 0.4  # Higher creativity
        },
        "general": {
            "name": "Allgemeine Verschlagwortung",
            "min_keywords": 5,
            "max_keywords": 20,
            "preferred_model": "cogito:14b",
            "temperature": 0.3
        }
    }

    return strategies.get(text_type, strategies["general"])
```

**Execution Flow**:
```python
def execute_pipeline(
    self,
    abstract: str,
    enable_classification: bool = True
) -> Dict[str, Any]:
    """
    Execute complete agentic pipeline.

    Steps:
        1. Analyze abstract → text type
        2. Plan strategy → parameters
        3. Execute SearchAgent (with self-reflection)
        4. Execute KeywordAgent (with self-reflection)
        5. Cross-validate
        6. If validation fails → retry
        7. Execute ClassificationAgent (optional)
        8. Aggregate results
    """
    # Phase 1: Analysis
    text_type = self._analyze_text_type(abstract)
    strategy = self._plan_execution_strategy(abstract, text_type)

    # Phase 2: Execute agents with self-reflection
    search_result = self.search_agent.execute_with_self_reflection(
        input_data={
            "abstract": abstract,
            "initial_keywords": strategy["initial_keywords"],
            "text_type": text_type
        },
        max_iterations=3
    )

    keyword_result = self.keyword_agent.execute_with_self_reflection(
        input_data={
            "abstract": abstract,
            "search_results": search_result.data,
            "min_keywords": strategy["min_keywords"],
            "max_keywords": strategy["max_keywords"]
        },
        max_iterations=3
    )

    # Phase 3: Validate
    validation_result = self.validation_agent.execute(
        input_data={
            "search_result": search_result,
            "keyword_result": keyword_result
        }
    )

    # Phase 4: Aggregate
    return {
        "keywords": keyword_result.data["keywords"],
        "search_coverage": search_result.confidence,
        "keyword_confidence": keyword_result.confidence,
        "overall_quality": validation_result.confidence,
        "text_type": text_type,
        "strategy_used": strategy["name"],
        "agent_iterations": {
            "search": search_result.iteration_count,
            "keyword": keyword_result.iteration_count
        }
    }
```

---

## Implementation Details

### File Structure

```
src/core/agents/
├── __init__.py                 # Agent package
├── base_agent.py              # BaseAgent + AgentResult
├── search_agent.py            # SearchAgent
├── keyword_agent.py           # KeywordAgent
├── classification_agent.py    # ClassificationAgent
├── validation_agent.py        # ValidationAgent
└── meta_agent.py              # MetaAgent (orchestrator)
```

### Integration with PipelineManager

**File**: `src/core/pipeline_manager.py`

```python
class PipelineManager:
    def __init__(self, alima_manager, cache_manager, config, ui_callback=None):
        # ... existing initialization ...

        # NEW: Agentic workflow support - Claude Generated
        self.enable_agentic_mode = config.get("enable_agentic_workflow", False)

        if self.enable_agentic_mode:
            from src.core.agents.meta_agent import MetaAgent
            self.meta_agent = MetaAgent(alima_manager, cache_manager, config)

    def execute_pipeline_with_agents(
        self,
        abstract: str,
        enable_classification: bool = True
    ) -> KeywordAnalysisState:
        """
        Execute pipeline using agentic workflow.

        Replaces linear pipeline with agent-based orchestration.
        """
        if not self.enable_agentic_mode:
            raise ValueError("Agentic mode not enabled")

        # Execute meta-agent
        results = self.meta_agent.execute_pipeline(
            abstract=abstract,
            enable_classification=enable_classification,
            stream_callback=self._create_stream_callback("meta_agent")
        )

        # Convert agent results to KeywordAnalysisState
        state = KeywordAnalysisState(
            original_abstract=abstract,
            initial_keywords=results.get("initial_keywords", []),
            search_suggesters_used=["agentic_search"],
            # ... map agent results to state fields ...
        )

        return state
```

---

## Configuration

### Agentic Configuration Model

**File**: `src/utils/config_models.py`

```python
@dataclass
class AgenticConfig:
    """Configuration for agentic workflow - Claude Generated"""

    enable_agentic_mode: bool = False  # Default: OFF

    # Agent-specific settings
    search_agent_max_iterations: int = 3
    keyword_agent_max_iterations: int = 3
    classification_agent_max_iterations: int = 2

    # Quality thresholds
    min_search_coverage: float = 0.5  # 50%
    min_keyword_relevance: float = 0.7  # 70%
    min_overall_quality: float = 0.7  # 70%

    # Adaptive strategy
    enable_adaptive_strategy: bool = True
    enable_text_type_detection: bool = True

    # Model preferences by text type
    scientific_model: str = "cogito:32b"
    fiction_model: str = "cogito:14b"
    general_model: str = "cogito:14b"
```

### GUI Configuration

```
┌─────────────────────────────────────────────────┐
│ 🤖 Agentic Workflow (Experimental)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ☐ Agentic Modus aktivieren                    │
│                                                 │
│  ⚠️  WARNING: Experimentell                     │
│      • 3x Token-Verbrauch                      │
│      • 2-3x Ausführungszeit                    │
│      • Nur für qualitätskritische Aufgaben     │
│                                                 │
│  Search Agent Iterationen:     [3 ▼]           │
│  Keyword Agent Iterationen:    [3 ▼]           │
│  Classification Agent Iter.:   [2 ▼]           │
│                                                 │
│  Min. Qualitätsschwelle:       [0.7 ▼]         │
│                                                 │
│  ☑ Adaptive Strategie aktivieren               │
│  ☑ Texttyp-Erkennung aktivieren                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Scientific Abstract (Success)

**Input**:
```
Abstract: "Untersuchung zu katalytischen Prozessen in der organischen Synthese
unter Verwendung von Übergangsmetallkomplexen. Die Studie analysiert
Reaktionsmechanismen und Produktausbeuten bei verschiedenen Temperaturen."
```

**Execution Log**:
```
🤖 Agentic Workflow gestartet
============================================================

📝 Texttyp erkannt: scientific
📋 Strategie: Wissenschaftliche Präzision
   • Model: cogito:32b
   • Keywords: 10-20
   • Temperature: 0.25

🔍 SearchAgent wird ausgeführt...
   Iteration 1:
     - Coverage: 45% ✗ (below threshold)
     - Reflection: "Low coverage - try hierarchy search"

   Iteration 2:
     - Coverage: 62% ✓
     - Confidence: 0.62
     - ✓ Validation passed

🏷️  KeywordAgent wird ausgeführt...
   Iteration 1:
     - Keywords selected: 14
     - Relevance: 88% ✓
     - Confidence: 0.88
     - ✓ Validation passed

✅ ValidationAgent prüft Ergebnisse...
   • Search coverage: 62% ✓
   • Keyword quality: 88% ✓
   • Consistency: 95% ✓
   • Overall quality: 82% ✓

📊 ClassificationAgent wird ausgeführt...
   • Classifications selected: 6
   • Diversity: 0.75 ✓

============================================================
✓ Pipeline abgeschlossen
   • Keywords: 14
   • Classifications: 6
   • Quality: 82%
   • Iterations: Search=2, Keyword=1
```

**Result**:
- 14 high-quality GND keywords
- 6 diverse DK classifications
- Overall quality: 82% (excellent)
- Automatic quality assurance via self-reflection

### Example 2: Fiction Abstract (Adaptive Strategy)

**Input**:
```
Abstract: "Ein Roman über die Reise zweier Protagonisten durch das mittelalterliche
Europa. Die Geschichte behandelt Themen wie Freundschaft, Vertrauen und die Suche
nach Identität."
```

**Execution Log**:
```
📝 Texttyp erkannt: fiction
📋 Strategie: Literarische Erschließung
   • Model: cogito:14b (faster)
   • Keywords: 5-15
   • Temperature: 0.4 (more creative)

🔍 SearchAgent wird ausgeführt...
   Iteration 1:
     - Coverage: 55% ✓ (fiction has lower threshold)
     - ✓ Validation passed

🏷️  KeywordAgent wird ausgeführt...
   Iteration 1:
     - Keywords selected: 8
     - Relevance: 78% ✓
     - ✓ Validation passed

✓ Overall quality: 75%
```

**Result**:
- Faster execution (cogito:14b)
- Fewer keywords (8 vs. 14 for scientific)
- Higher temperature for creativity
- Adapted strategy worked perfectly

---

## Performance Considerations

### Token Usage

| Component | Single Request | Agentic (Avg) | Multiplier |
|-----------|---------------|---------------|------------|
| Search | 1x LLM | 2x LLM (avg) | 2x |
| Keywords | 1x LLM | 2x LLM (avg) | 2x |
| Classification | 1x LLM | 1.5x LLM (avg) | 1.5x |
| **Total** | **~5000-10000 tokens** | **~15000-30000 tokens** | **~3x** |

**Mitigation**:
- Opt-in feature (disabled by default)
- Clear warnings in UI about token usage
- Configurable max iterations (reduce for lower usage)
- Consider using cheaper models (haiku) for non-critical steps

### Time Overhead

| Scenario | Linear Pipeline | Agentic Pipeline | Overhead |
|----------|----------------|------------------|----------|
| Best case (1 iteration) | 30-60s | 60-120s | +30-60s |
| Average (2 iterations) | 30-60s | 90-180s | +60-120s |
| Worst case (3 iterations) | 30-60s | 120-240s | +90-180s |

**Acceptable for quality-critical tasks**: Users can disable for time-sensitive work.

### Quality Improvement (Expected)

Based on self-reflection and adaptive strategies:
- **Keyword coverage**: +10-20% (fewer missing concepts)
- **Keyword relevance**: +5-15% (better selection)
- **Classification diversity**: +10% (more balanced)
- **Overall satisfaction**: Significant improvement from automatic QA

---

## Testing Strategy

### Unit Tests

```python
class TestBaseAgent:
    """Test BaseAgent functionality"""

    def test_self_reflection_loop(self):
        """Test self-reflection with retry logic"""
        agent = MockAgent(...)

        # Mock to fail first iteration, pass second
        agent.mock_validate = [False, True]

        result = agent.execute_with_self_reflection(input_data={...}, max_iterations=3)

        assert result.iteration_count == 2
        assert result.success == True

    def test_max_iterations_limit(self):
        """Ensure iteration stops at max_iterations"""
        agent = MockAgent(...)

        # Mock to always fail validation
        agent.mock_validate = [False, False, False, False]

        result = agent.execute_with_self_reflection(input_data={...}, max_iterations=3)

        assert result.iteration_count == 3  # Stopped at max
        assert result.success == False  # Never validated

class TestSearchAgent:
    """Test SearchAgent specifically"""

    def test_coverage_calculation(self):
        """Test search coverage metric"""
        agent = SearchAgent(...)

        result = agent._calculate_coverage(
            initial_keywords=["A", "B", "C", "D"],
            search_results={"A": {...}, "B": {...}}  # 2/4 found
        )

        assert result == 0.5  # 50% coverage

    def test_adaptive_strategy_selection(self):
        """Test strategy selection based on text type"""
        agent = SearchAgent(...)

        strategy = agent._select_search_strategy(
            abstract="Scientific paper...",
            text_type="scientific"
        )

        assert strategy["search_depth"] == "precise"
        assert strategy["enable_hierarchy"] == True

class TestMetaAgent:
    """Test MetaAgent orchestration"""

    def test_text_type_detection(self):
        """Test text type detection logic"""
        meta = MetaAgent(...)

        scientific = meta._analyze_text_type(
            "Untersuchung zu Methoden der Datenanalyse..."
        )
        assert scientific == "scientific"

        fiction = meta._analyze_text_type(
            "Roman über einen Protagonisten..."
        )
        assert fiction == "fiction"

    def test_full_pipeline_execution(self):
        """Test complete agent orchestration"""
        meta = MetaAgent(...)

        results = meta.execute_pipeline(
            abstract="Test abstract...",
            enable_classification=True
        )

        assert "keywords" in results
        assert "overall_quality" in results
        assert results["overall_quality"] > 0.7  # Quality threshold
```

---

## Future Extensions

### Phase 2 Enhancements

1. **Ensemble Agents**: Multiple agents vote on best keywords
2. **Learning Agents**: Agents learn from user feedback to improve quality thresholds
3. **Specialized Agents**: Domain-specific agents (MedicalAgent, LegalAgent, etc.)
4. **Agent Communication**: Agents can request information from each other
5. **Confidence Voting**: Weighted voting based on agent confidence scores
6. **Dynamic Model Selection**: Agents switch models mid-execution based on task complexity

### Integration with Other Features

- **Iterative GND Search**: Agents can trigger iterative refinement automatically
- **DK Splitting**: ClassificationAgent uses splitting for large result sets
- **Batch Processing**: Meta-agent selects optimal strategy per document in batch

---

## Conclusion

The agentic workflow architecture represents a paradigm shift from deterministic pipelines to intelligent, adaptive systems. By introducing self-reflection, multi-agent collaboration, adaptive strategies, and hierarchical planning, ALIMA can achieve significantly higher quality outputs with minimal user intervention.

**Recommendation**: Implement iteratively:
1. **Phase 1**: Base agent infrastructure + SearchAgent (proof of concept)
2. **Phase 2**: KeywordAgent + ValidationAgent (core functionality)
3. **Phase 3**: ClassificationAgent + MetaAgent (complete system)
4. **Phase 4**: Adaptive strategies + text type detection (optimization)

**Trade-offs**:
- **Pro**: Automatic quality assurance, adaptive behavior, self-correction
- **Con**: 3x token usage, 2-3x execution time, increased complexity

**Best For**: Quality-critical tasks, complex abstracts, production environments where quality > speed

---

## References

- Main Plan: `/home/conrad/.claude/plans/stateless-soaring-squirrel.md`
- Pipeline Manager: `src/core/pipeline_manager.py`
- Config Models: `src/utils/config_models.py`
- Data Models: `src/core/data_models.py`
