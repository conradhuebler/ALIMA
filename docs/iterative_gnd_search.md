# Iterative GND Search with Missing Concepts

**Feature Status**: Planned
**Priority**: HIGH
**Effort**: 2-3 days
**Risk**: Low (infrastructure 80% ready)

## Executive Summary

This feature enables the ALIMA pipeline to iteratively enrich GND keyword pools by detecting missing concepts from LLM responses, searching for them in GND, and re-running analysis with the expanded context until convergence or maximum iterations are reached.

**Key Insight**: The `keywords` prompt in `prompts.json` ALREADY requests missing concepts in `<missing_list>` tags, but no code currently extracts this information. Infrastructure is 80% ready - we just need extraction logic and iteration control.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Data Flow](#data-flow)
4. [Implementation Details](#implementation-details)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Current Behavior

The keyword analysis step performs a single-pass selection of GND keywords:

```
Abstract → Initial Keywords → GND Search → LLM Analysis → Final Keywords
                                                ↓
                                          Missing concepts identified
                                          but IGNORED
```

**Issues**:
1. LLM identifies missing concepts (not covered by GND) but system ignores them
2. Users must manually search for missing concepts and re-run analysis
3. No automated refinement loop to improve coverage
4. Final keyword set may be incomplete for complex abstracts

### Example Scenario

**Abstract**: "Untersuchung zu Probenvorbereitung und Instrumentenspezifikationen in der analytischen Chemie"

**Initial GND Search** finds:
- Analytische Chemie (GND-ID: 4142176-0)
- Chemische Analyse (GND-ID: 4009840-0)

**LLM Response**:
```xml
<final_list>
Analytische Chemie (GND-ID: 4142176-0) | Chemische Analyse (GND-ID: 4009840-0)
</final_list>

<missing_list>
Probenvorbereitung, Instrumentenspezifikationen
</missing_list>
```

**Current System**: Ignores `<missing_list>`, returns only 2 keywords
**Desired Behavior**: Searches GND for "Probenvorbereitung" and "Instrumentenspezifikationen", enriches keyword pool, re-analyzes

---

## Solution Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                  Iterative Keyword Refinement                    │
│                                                                  │
│  ┌─────────────┐         ┌──────────────┐       ┌───────────┐  │
│  │  Iteration  │         │   Extract    │       │  Search   │  │
│  │   Counter   │────────▶│  <missing_   │──────▶│  GND for  │  │
│  │  (max: 2)   │         │   list>      │       │  Missing  │  │
│  └─────────────┘         └──────────────┘       │  Concepts │  │
│        │                         │               └─────┬─────┘  │
│        │                         │                     │        │
│        │                    ┌────▼──────┐              │        │
│        │                    │ Missing   │              │        │
│        └───────────────────▶│ Concepts  │◀─────────────┘        │
│                             │ Found?    │                       │
│                             └────┬──────┘                       │
│                                  │                              │
│                    ┌─────────────┼──────────────┐               │
│                    │             │              │               │
│                    ▼             ▼              ▼               │
│               ┌────────┐   ┌─────────┐   ┌──────────┐          │
│               │  No    │   │  Same   │   │   New    │          │
│               │Results │   │  as     │   │ Results  │          │
│               │        │   │ Before  │   │  Found   │          │
│               └────┬───┘   └────┬────┘   └─────┬────┘          │
│                    │            │              │               │
│                    └────────────┼──────────────┘               │
│                                 │                              │
│                                 ▼                              │
│                         ┌───────────────┐                      │
│                         │ STOP or       │                      │
│                         │ Continue?     │                      │
│                         └───────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

### Convergence Criteria

The iteration loop stops when ANY of these conditions is met:

1. **No Missing Concepts**: `<missing_list>` is empty
2. **Self-Consistency**: Identical missing concepts as previous iteration
3. **No New Results**: GND search finds no matches for missing concepts
4. **Max Iterations**: Configurable limit reached (default: 2)

### Components Modified

| Component | File | Changes |
|-----------|------|---------|
| Extraction Logic | `src/core/processing_utils.py` | Add `extract_missing_concepts_from_response()` |
| Data Models | `src/core/data_models.py` | Add iteration tracking to `LlmKeywordAnalysis` and `KeywordAnalysisState` |
| Search Logic | `src/utils/pipeline_utils.py` | Add `execute_fallback_gnd_search()` |
| Iteration Control | `src/utils/pipeline_utils.py` | Add `execute_iterative_keyword_refinement()` |
| Pipeline Integration | `src/core/pipeline_manager.py` | Update `_execute_keywords_step()` |
| Configuration | `src/utils/config_models.py` | Add refinement flags to `PipelineStepConfig` |
| GUI | `src/ui/pipeline_config_dialog.py` | Add refinement controls |
| GUI Display | `src/ui/analysis_review_tab.py` | Add iteration history widget |
| CLI | `src/alima_cli.py` | Add `--enable-iterative-search` flag |

---

## Data Flow

### Single Iteration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ ITERATION N                                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Format GND Pool → Prompt                                       │
│     ┌──────────────────────────────────────────┐                   │
│     │ Abstract: "..."                          │                   │
│     │                                          │                   │
│     │ GND Keywords Available:                  │                   │
│     │ - Analytische Chemie (GND-ID: 4142176-0)│                   │
│     │ - Chemische Analyse (GND-ID: 4009840-0) │                   │
│     │ - Probenvorbereitung (GND-ID: 4047326-1)│ ← NEW from iter N-1│
│     └──────────────────────────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  2. Call LLM                                                       │
│     [Model: cogito:14b, Temperature: 0.4]                          │
│                         │                                           │
│                         ▼                                           │
│  3. Parse Response                                                 │
│     ┌──────────────────────────────────────────┐                   │
│     │ <final_list>                             │                   │
│     │ Analytische Chemie | Probenvorbereitung  │                   │
│     │ </final_list>                            │                   │
│     │                                          │                   │
│     │ <missing_list>                           │                   │
│     │ Instrumentenspezifikationen              │                   │
│     │ </missing_list>                          │                   │
│     └──────────────────────────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  4. Check Convergence                                              │
│     ┌──────────────────────────────────────────┐                   │
│     │ Previous missing: [Probenvorbereitung]  │                   │
│     │ Current missing: [Instrumentenspez...]  │                   │
│     │ → Different → Continue                   │                   │
│     └──────────────────────────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  5. Search GND for Missing Concepts                                │
│     ┌──────────────────────────────────────────┐                   │
│     │ Query: "Instrumentenspezifikationen"     │                   │
│     │ Results: 3 GND entries found             │                   │
│     │   - Analysengerät (GND-ID: 4142987-X)   │                   │
│     │   - Messgerät (GND-ID: 4134029-1)       │                   │
│     └──────────────────────────────────────────┘                   │
│                         │                                           │
│                         ▼                                           │
│  6. Merge Results → Next Iteration                                 │
│     GND Pool now has 5 entries (2 initial + 1 from iter 1 + 2 new) │
└─────────────────────────────────────────────────────────────────────┘
```

### Iteration State Tracking

Each iteration records:

```python
{
    "iteration": 2,
    "missing_concepts": ["Instrumentenspezifikationen"],
    "keywords_selected": 4,
    "gnd_pool_size": 5,
    "new_gnd_results": 2,
    "convergence_reason": None  # or "no_missing_concepts", "self_consistency", etc.
}
```

---

## Implementation Details

### Phase 1: Extraction Function

**File**: `src/core/processing_utils.py`

```python
def extract_missing_concepts_from_response(text: str) -> List[str]:
    """
    Extract missing concepts from LLM response <missing_list> tag.

    Handles:
        - Multiple separators (comma, newline, semicolon)
        - Thinking blocks that need to be stripped
        - Case-insensitive tag matching
        - Malformed or missing tags

    Args:
        text: Full LLM response text

    Returns:
        List of missing concept strings (empty if none found)

    Examples:
        >>> text = "<missing_list>Konzept1, Konzept2</missing_list>"
        >>> extract_missing_concepts_from_response(text)
        ['Konzept1', 'Konzept2']

        >>> text = "<missing_list>\\nKonzept1\\nKonzept2\\n</missing_list>"
        >>> extract_missing_concepts_from_response(text)
        ['Konzept1', 'Konzept2']
    """
    # Remove thinking blocks first (they can contain unrelated text)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',
        '',
        cleaned,
        flags=re.DOTALL
    )

    # Extract missing_list content (case-insensitive)
    match = re.search(
        r'<missing_list>\s*([^<]+)\s*</missing_list>',
        cleaned,
        re.DOTALL | re.IGNORECASE
    )

    if not match:
        return []

    content = match.group(1).strip()

    # Split by comma, newline, or semicolon
    concepts = [
        c.strip()
        for c in re.split(r'[,\n;]+', content)
        if c.strip()
    ]

    return concepts
```

**Edge Cases**:
- Empty `<missing_list></missing_list>` → Returns `[]`
- Nested tags → Extracts innermost content
- Multiple `<missing_list>` tags → Uses first occurrence
- Missing closing tag → Returns `[]`

### Phase 2: Data Model Extensions

**File**: `src/core/data_models.py`

```python
@dataclass
class LlmKeywordAnalysis:
    """Strukturierte Darstellung der LLM-Analyseergebnisse mit Details zum Aufruf."""

    # ... existing fields ...

    missing_concepts: List[str] = field(default_factory=list)  # NEW - Claude Generated
    """
    Concepts identified by LLM as not covered by available GND keywords.
    Extracted from <missing_list> tag in response_full_text.
    """

@dataclass
class KeywordAnalysisState:
    """Kapselt den gesamten Zustand des Keyword-Analyse-Workflows."""

    # ... existing fields ...

    # NEW - Iterative search support - Claude Generated
    refinement_iterations: List[Dict[str, Any]] = field(default_factory=list)
    """
    Tracks each refinement iteration with metadata.

    Structure:
        [
            {
                "iteration": 1,
                "missing_concepts": ["Probenvorbereitung", "Instrumentenspez..."],
                "new_gnd_results": 5,  # New GND entries found
                "keywords_selected": 8,  # Keywords in final_list this iteration
                "gnd_pool_size": 25,  # Total GND entries available
                "convergence_reason": None  # or "no_missing_concepts", "self_consistency", etc.
            },
            ...
        ]
    """

    max_iterations_reached: bool = False
    """True if iteration stopped due to max_iterations limit."""

    convergence_achieved: bool = False
    """True if iteration stopped due to convergence (not max_iterations)."""
```

**Backward Compatibility**: All new fields use `field(default_factory=...)` so old JSON files load without errors.

### Phase 3: Fallback GND Search

**File**: `src/utils/pipeline_utils.py`

```python
def execute_fallback_gnd_search(
    self,
    missing_concepts: List[str],
    existing_results: Dict[str, Dict[str, Any]],
    stream_callback: Optional[callable] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Search GND for missing concepts identified by LLM.

    Search Strategy:
        1. Exact term search first
        2. If no results: try broader terms (from GND hierarchy) [FUTURE]
        3. If still no results: try related terms [FUTURE]
        4. Track which concepts found no matches

    Args:
        missing_concepts: List of concepts not covered by existing GND pool
        existing_results: Current search results to avoid duplicates
        stream_callback: Progress feedback callback

    Returns:
        Merged search results (existing + new)

    Example:
        >>> executor = PipelineStepExecutor(...)
        >>> missing = ["Probenvorbereitung", "Instrumentenspezifikationen"]
        >>> existing = {"Chemie": {"gndid": {"4009840-0"}}}
        >>> results = executor.execute_fallback_gnd_search(missing, existing)
        >>> len(results)
        3  # existing + 2 new
    """
    if stream_callback:
        stream_callback(
            f"\n🔍 Fallback-Suche für {len(missing_concepts)} fehlende Konzepte...\n",
            "keywords_refinement"
        )

    new_results = {}
    concepts_not_found = []

    for concept in missing_concepts:
        if stream_callback:
            stream_callback(f"  Suche: {concept}\n", "keywords_refinement")

        # Execute search via CacheManager
        search_result = self.cache_manager.search_gnd(
            query=concept,
            max_results=10
        )

        if search_result and len(search_result.get("gndid", set())) > 0:
            new_results[concept] = search_result
            if stream_callback:
                count = len(search_result.get("gndid", set()))
                stream_callback(
                    f"    ✓ {count} GND-Treffer gefunden\n",
                    "keywords_refinement"
                )
        else:
            concepts_not_found.append(concept)
            if stream_callback:
                stream_callback(
                    f"    ✗ Keine GND-Einträge gefunden\n",
                    "keywords_refinement"
                )

    # Merge with existing results
    merged_results = {**existing_results}

    for concept, data in new_results.items():
        if concept in merged_results:
            # Merge GND-IDs for existing keywords (avoid duplicates)
            merged_results[concept]["gndid"].update(data.get("gndid", set()))
        else:
            # Add completely new keyword
            merged_results[concept] = data

    # Progress summary
    if stream_callback:
        found_count = len(new_results)
        total_count = len(missing_concepts)
        stream_callback(
            f"\n📊 Fallback-Ergebnis: {found_count}/{total_count} Konzepte gefunden\n",
            "keywords_refinement"
        )
        if concepts_not_found:
            preview = ', '.join(concepts_not_found[:5])
            if len(concepts_not_found) > 5:
                preview += f" ... (+{len(concepts_not_found) - 5} weitere)"
            stream_callback(
                f"⚠️  Nicht gefunden: {preview}\n",
                "keywords_refinement"
            )

    return merged_results
```

### Phase 4: Iteration Control

**File**: `src/utils/pipeline_utils.py`

```python
def execute_iterative_keyword_refinement(
    self,
    original_abstract: str,
    initial_search_results: Dict[str, Dict[str, Any]],
    model: str,
    provider: str,
    max_iterations: int = 2,
    stream_callback: Optional[callable] = None,
    **kwargs
) -> Tuple[List[str], Dict[str, Any], LlmKeywordAnalysis]:
    """
    Iteratively refine keyword selection by searching for missing concepts.

    Process:
        1. Run keyword analysis with current GND pool
        2. Extract missing concepts from <missing_list>
        3. If missing concepts found AND iterations remaining:
           a. Search GND for missing concepts
           b. Merge results into GND pool
           c. Re-run keyword analysis
           d. Check for convergence
        4. Return final keywords + enriched state

    Args:
        original_abstract: The abstract text
        initial_search_results: Initial GND search results
        model: LLM model to use
        provider: LLM provider
        max_iterations: Maximum refinement iterations (default: 2)
        stream_callback: Progress callback

    Returns:
        Tuple of:
            - final_keywords: List[str] - Final selected keywords
            - iteration_metadata: Dict[str, Any] - Iteration statistics
            - llm_analysis: LlmKeywordAnalysis - Last LLM analysis details

    Example:
        >>> final_kw, metadata, analysis = executor.execute_iterative_keyword_refinement(
        ...     abstract="...",
        ...     initial_search_results={...},
        ...     model="cogito:14b",
        ...     provider="ollama",
        ...     max_iterations=2
        ... )
        >>> metadata['total_iterations']
        2
        >>> metadata['convergence_achieved']
        True
    """
    current_search_results = initial_search_results.copy()
    iteration_history = []
    previous_missing_concepts = []

    for iteration in range(1, max_iterations + 1):
        if stream_callback:
            stream_callback(
                f"\n{'='*60}\n"
                f"🔄 Iteration {iteration}/{max_iterations}\n"
                f"{'='*60}\n",
                "keywords_refinement"
            )

        # Execute keyword analysis with current GND pool
        final_keywords, _, llm_analysis = self.execute_final_keyword_analysis(
            original_abstract=original_abstract,
            search_results=current_search_results,
            model=model,
            provider=provider,
            stream_callback=stream_callback,
            **kwargs
        )

        # Extract missing concepts from LLM response
        missing_concepts = extract_missing_concepts_from_response(
            llm_analysis.response_full_text
        )

        # Store in llm_analysis for later reference
        llm_analysis.missing_concepts = missing_concepts

        # Record iteration data
        iteration_data = {
            "iteration": iteration,
            "missing_concepts": missing_concepts,
            "keywords_selected": len(final_keywords),
            "gnd_pool_size": len(current_search_results)
        }

        if stream_callback:
            stream_callback(
                f"\n📋 Iteration {iteration} Ergebnis:\n"
                f"  - Keywords: {len(final_keywords)}\n"
                f"  - Fehlende Konzepte: {len(missing_concepts)}\n",
                "keywords_refinement"
            )

        # === CONVERGENCE CHECKS ===

        # Check 1: No missing concepts
        if not missing_concepts:
            if stream_callback:
                stream_callback(
                    "✓ Konvergenz erreicht: Keine fehlenden Konzepte\n",
                    "keywords_refinement"
                )
            iteration_data["convergence_reason"] = "no_missing_concepts"
            iteration_history.append(iteration_data)
            break

        # Check 2: Self-consistency (identical missing concepts)
        if missing_concepts == previous_missing_concepts:
            if stream_callback:
                stream_callback(
                    "✓ Konvergenz erreicht: Identische fehlende Konzepte\n",
                    "keywords_refinement"
                )
            iteration_data["convergence_reason"] = "self_consistency"
            iteration_history.append(iteration_data)
            break

        # Not last iteration? Search for missing concepts
        if iteration < max_iterations:
            enriched_results = self.execute_fallback_gnd_search(
                missing_concepts=missing_concepts,
                existing_results=current_search_results,
                stream_callback=stream_callback,
                **kwargs
            )

            # Calculate new keywords found
            new_count = len(enriched_results) - len(current_search_results)
            iteration_data["new_gnd_results"] = new_count

            # Check 3: No new results
            if new_count == 0:
                if stream_callback:
                    stream_callback(
                        "⚠️  Keine neuen GND-Einträge gefunden - Iteration beendet\n",
                        "keywords_refinement"
                    )
                iteration_data["convergence_reason"] = "no_new_results"
                iteration_history.append(iteration_data)
                break

            # Update for next iteration
            current_search_results = enriched_results
            previous_missing_concepts = missing_concepts.copy()
            iteration_history.append(iteration_data)
        else:
            # Max iterations reached
            iteration_data["convergence_reason"] = "max_iterations"
            iteration_history.append(iteration_data)
            if stream_callback:
                stream_callback(
                    f"⚠️  Maximale Iterationen ({max_iterations}) erreicht\n",
                    "keywords_refinement"
                )

    # Build enriched state metadata
    state_metadata = {
        "total_iterations": len(iteration_history),
        "iteration_history": iteration_history,
        "final_gnd_pool_size": len(current_search_results),
        "convergence_achieved": any(
            "convergence_reason" in it and it["convergence_reason"] != "max_iterations"
            for it in iteration_history
        )
    }

    return final_keywords, state_metadata, llm_analysis
```

---

## Configuration

### Pipeline Configuration

**File**: `src/utils/config_models.py`

```python
@dataclass
class PipelineStepConfig:
    enabled: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

    # Iterative refinement - Claude Generated
    enable_iterative_refinement: bool = False  # Default: OFF
    max_refinement_iterations: int = 2  # Default: 2 iterations max

    # Existing chunking config
    enable_chunking: bool = False
    chunk_size: int = 500
```

### GUI Configuration

**Location**: Pipeline Config Dialog → Keywords Step Tab

```
┌─────────────────────────────────────────────────┐
│ 🔄 Iterative GND-Suche                         │
├─────────────────────────────────────────────────┤
│                                                 │
│  ☐ Iterative Suche aktivieren                  │
│                                                 │
│  Max. Iterationen:  [2 ▼]                      │
│                                                 │
│  ℹ️  Wenn aktiviert, sucht das System nach     │
│     fehlenden Konzepten und erweitert den      │
│     GND-Pool automatisch.                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### CLI Configuration

```bash
# Enable iterative search
python alima_cli.py pipeline \
    --input-text "..." \
    --enable-iterative-search \
    --max-iterations 3

# Disable (default)
python alima_cli.py pipeline --input-text "..."
```

---

## Usage Examples

### Example 1: Scientific Abstract

**Input**:
```
Abstract: "Untersuchung zu Probenvorbereitung und Instrumentenspezifikationen
in der analytischen Chemie mit Fokus auf Massenspektrometrie"
```

**Iteration 1**:
- GND Pool: Analytische Chemie, Chemische Analyse (2 entries)
- Keywords Selected: 2
- Missing Concepts: Probenvorbereitung, Instrumentenspezifikationen, Massenspektrometrie

**Fallback Search**:
- Found: Probenvorbereitung (GND-ID: 4047326-1)
- Found: Massenspektrometrie (GND-ID: 4037882-X)
- Not Found: Instrumentenspezifikationen

**Iteration 2**:
- GND Pool: 4 entries (2 initial + 2 new)
- Keywords Selected: 4
- Missing Concepts: Instrumentenspezifikationen

**Fallback Search**:
- Found: Analysengerät (GND-ID: 4142987-X)

**Iteration 3** (would continue if max_iterations > 2):
- GND Pool: 5 entries
- Keywords Selected: 5
- Missing Concepts: (empty)
- **Convergence**: No missing concepts

**Final Result**: 5 GND keywords (20% improvement over single-pass)

### Example 2: Convergence via Self-Consistency

**Iteration 1**:
- Missing Concepts: Konzept A, Konzept B

**Fallback Search**:
- Konzept A: Not found
- Konzept B: Not found

**Iteration 2**:
- Missing Concepts: Konzept A, Konzept B (identical to iteration 1)
- **Convergence**: Self-consistency

**Result**: Stops early, no infinite loop

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_iterative_search.py`

```python
import pytest
from src.core.processing_utils import extract_missing_concepts_from_response

class TestMissingConceptExtraction:
    """Test extract_missing_concepts_from_response()"""

    def test_comma_separated(self):
        text = "<missing_list>Konzept1, Konzept2, Konzept3</missing_list>"
        result = extract_missing_concepts_from_response(text)
        assert result == ["Konzept1", "Konzept2", "Konzept3"]

    def test_newline_separated(self):
        text = "<missing_list>\nKonzept1\nKonzept2\nKonzept3\n</missing_list>"
        result = extract_missing_concepts_from_response(text)
        assert result == ["Konzept1", "Konzept2", "Konzept3"]

    def test_empty_list(self):
        text = "<missing_list></missing_list>"
        result = extract_missing_concepts_from_response(text)
        assert result == []

    def test_no_list(self):
        text = "<final_list>Some keywords</final_list>"
        result = extract_missing_concepts_from_response(text)
        assert result == []

    def test_with_thinking_blocks(self):
        text = """
        <think>This is thinking...</think>
        <missing_list>Konzept1, Konzept2</missing_list>
        """
        result = extract_missing_concepts_from_response(text)
        assert result == ["Konzept1", "Konzept2"]

class TestIterativeRefinement:
    """Test execute_iterative_keyword_refinement()"""

    @pytest.fixture
    def mock_executor(self):
        # Mock PipelineStepExecutor
        pass

    def test_convergence_no_missing_concepts(self, mock_executor):
        """Test convergence when LLM returns empty missing_list"""
        # Mock LLM to return empty missing_list
        # Verify iteration stops at iteration 1
        pass

    def test_convergence_self_consistency(self, mock_executor):
        """Test convergence when missing concepts identical across iterations"""
        # Mock LLM to return same missing_list twice
        # Verify iteration stops at iteration 2
        pass

    def test_max_iterations_reached(self, mock_executor):
        """Test iteration stops at max_iterations"""
        # Mock LLM to always return different missing_list
        # Verify iteration stops at max_iterations
        pass
```

### Integration Tests

```python
class TestIterativeSearchIntegration:
    """End-to-end tests with real pipeline components"""

    def test_full_iteration_cycle(self):
        """Test complete iteration from abstract to final keywords"""
        abstract = "Test abstract about Probenvorbereitung..."
        # Execute pipeline with iterative refinement enabled
        # Verify:
        # - Iteration history recorded
        # - GND pool expanded
        # - Convergence achieved
        pass

    def test_cli_gui_parity(self):
        """Verify CLI and GUI produce identical results"""
        abstract = "..."

        # Execute via CLI
        cli_result = run_cli_with_iteration(abstract)

        # Execute via GUI
        gui_result = run_gui_with_iteration(abstract)

        assert cli_result.keywords == gui_result.keywords
        assert cli_result.iteration_count == gui_result.iteration_count
```

### Manual Testing Checklist

- [ ] Enable iterative search in GUI config
- [ ] Run with scientific abstract containing missing concepts
- [ ] Verify iteration history displayed in review tab
- [ ] Test convergence scenarios (no_missing, self_consistency, max_iterations)
- [ ] Test CLI with `--enable-iterative-search` flag
- [ ] Verify JSON export/import with iteration state
- [ ] Test with abstracts in different languages
- [ ] Verify streaming progress updates work correctly

---

## Performance Considerations

### Token Usage

| Scenario | LLM Calls | Estimated Tokens |
|----------|-----------|------------------|
| Single-pass (baseline) | 1 | 2000-5000 |
| Iterative (1 refinement) | 2 | 4000-10000 |
| Iterative (2 refinements) | 3 | 6000-15000 |

**Mitigation**: Feature is opt-in and disabled by default.

### Time Overhead

| Operation | Time per Iteration |
|-----------|-------------------|
| LLM Call | 10-30s |
| GND Search (per concept) | 0.5-2s |
| Merge Results | <0.1s |
| **Total per iteration** | ~12-35s |

**Expected with 2 iterations**: +24-70 seconds total

**Acceptable for quality improvement**: Users can disable if time-sensitive.

### Memory Impact

- Minimal: Only stores iteration history (small JSON objects)
- GND pool expansion: ~100-500 additional entries (negligible)

---

## Troubleshooting

### Issue: LLM not outputting `<missing_list>`

**Symptoms**: Iteration history shows empty missing_concepts for all iterations

**Cause**: Prompt may have been modified or LLM not following instructions

**Solution**:
1. Check prompts.json contains `<missing_list>` instructions
2. Verify LLM model supports structured output
3. Try increasing temperature for more creative responses
4. Check LLM response in `llm_analysis.response_full_text`

### Issue: Infinite iteration loop

**Symptoms**: Iteration reaches max_iterations without convergence

**Cause**: LLM continuously identifies new missing concepts

**Solution**:
1. Reduce `max_refinement_iterations` to 1-2
2. Check if abstract is too broad/complex
3. Review iteration history to see missing concepts pattern
4. Consider manual keyword selection for this abstract

### Issue: GND search finds no results

**Symptoms**: Fallback search returns 0 results for all missing concepts

**Cause**: Missing concepts may be too specific or not in GND

**Solution**:
1. Check GND database is populated
2. Verify search terms are in German (GND is primarily German)
3. Try broader search terms manually
4. Check if concepts should be broken down (e.g., "Instrumentenspezifikationen" → "Instrument" + "Spezifikation")

### Issue: Iteration history not displayed in GUI

**Symptoms**: Analysis review tab doesn't show iteration details

**Cause**: May be using old JSON export format

**Solution**:
1. Verify `refinement_iterations` field exists in `KeywordAnalysisState`
2. Check if `display_iteration_history()` is called in review tab
3. Re-run analysis with current version

---

## Future Enhancements

### Phase 2 Improvements (Post-MVP)

1. **Hierarchical GND Search**: Search for broader/narrower terms when exact match fails
2. **Semantic Similarity**: Use embeddings to find related GND terms
3. **Confidence Scoring**: Weight missing concepts by LLM confidence
4. **Adaptive Max Iterations**: Dynamically adjust based on abstract complexity
5. **Parallel Searches**: Execute fallback searches concurrently
6. **User Intervention**: GUI button to manually add missing concepts

### Integration with Other Features

- Combine with DK classification splitting for comprehensive optimization
- Use agentic workflow for adaptive iteration strategy selection
- Integrate with batch processing for bulk refinement

---

## References

- Main Plan: `/home/conrad/.claude/plans/stateless-soaring-squirrel.md`
- prompts.json: Line 17-18 (`<missing_list>` format)
- Data Models: `src/core/data_models.py:60-92`
- Pipeline Utils: `src/utils/pipeline_utils.py:600-700`
- CLAUDE.md Core: `src/core/CLAUDE.md`
