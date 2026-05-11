# ALIMA Pipeline Enhancement Plan: Iterative Search, Parallel Classification & Agentic Workflows

## Executive Summary

This plan outlines three major enhancements to the ALIMA pipeline architecture:

1. **Iterative GND Search**: Extract missing concepts from LLM responses, search GND for them, and re-run keyword analysis with enriched context
2. **DK Classification Splitting**: Split DK classification lists into two halves for parallel LLM processing to reduce token load
3. **Agentic Workflow Architecture**: Implement self-reflective, multi-agent, adaptive, and hierarchical planning capabilities

**Key Constraint**: All features must maintain CLI/GUI parity through shared `pipeline_utils.py` logic.

---

## Part 1: Iterative GND Search with Missing Concepts

### 1.1 Current State Analysis

**Critical Discovery**: The `keywords` prompt in `prompts.json` (line 17-18) ALREADY instructs LLMs to output missing concepts:

```
**FEHLENDE KONZEPTE**:
[Liste von Begriffen im Abstract, die *nicht* durch GND abgedeckt sind.]

Output format:
<missing_list>
Quantenmechanik
</missing_list>
```

**Problem**: No code currently extracts `<missing_list>` from LLM responses.

**Opportunity**: Infrastructure is 80% ready - just need extraction logic and iteration control.

### 1.2 Architecture Design

#### Data Flow
```
┌─────────────────────────────────────────────────────────┐
│ Keywords Step (execute_final_keyword_analysis)         │
│                                                         │
│ 1. Format GND pool → LLM prompt                       │
│ 2. Call LLM with abstract + GND keywords              │
│ 3. Extract <final_list> (EXISTING)                    │
│ 4. Extract <missing_list> (NEW)                       │
│                                                         │
│ If missing_list not empty AND iterations_left > 0:     │
│   ├─→ Search GND for missing concepts                  │
│   ├─→ Merge new results into existing GND pool        │
│   ├─→ Re-run LLM analysis with enriched pool          │
│   └─→ Repeat until convergence or max_iterations      │
└─────────────────────────────────────────────────────────┘
```

#### Convergence Detection
- **Self-consistency**: If `<missing_list>` empty or identical to previous iteration → STOP
- **Max iterations**: Configurable limit (default: 2)
- **No new results**: If GND search finds no matches → STOP

### 1.3 Implementation Plan

#### Phase 1: Extract Missing Concepts (Core Infrastructure)

**File**: `src/core/processing_utils.py`

Add new extraction function:

```python
def extract_missing_concepts_from_response(text: str) -> List[str]:
    """
    Extract missing concepts from LLM response <missing_list> tag.

    Args:
        text: Full LLM response text

    Returns:
        List of missing concept strings

    Example:
        Input: "<missing_list>Probenvorbereitung, Instrumentenspezifikationen</missing_list>"
        Output: ["Probenvorbereitung", "Instrumentenspezifikationen"]
    """
    # Remove thinking blocks first
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>', '', cleaned, flags=re.DOTALL)

    # Extract missing_list content
    match = re.search(r'<missing_list>\s*([^<]+)\s*</missing_list>', cleaned, re.DOTALL | re.IGNORECASE)
    if not match:
        return []

    content = match.group(1).strip()

    # Split by comma or newline
    concepts = [c.strip() for c in re.split(r'[,\n]+', content) if c.strip()]

    return concepts
```

**Testing Strategy**:
- Test with real LLM outputs containing `<missing_list>`
- Test with malformed tags (missing closing tag, nested tags)
- Test with empty missing_list
- Test with mixed separators (comma, newline, semicolon)

#### Phase 2: Update Data Models

**File**: `src/core/data_models.py`

Add missing concepts tracking to `LlmKeywordAnalysis`:

```python
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
    chunk_responses: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)  # NEW - Claude Generated
```

Add iteration tracking to `KeywordAnalysisState`:

```python
@dataclass
class KeywordAnalysisState:
    """Kapselt den gesamten Zustand des Keyword-Analyse-Workflows."""

    # ... existing fields ...

    # NEW - Iterative search support - Claude Generated
    refinement_iterations: List[Dict[str, Any]] = field(default_factory=list)
    """
    Tracks each refinement iteration:
    [
        {
            "iteration": 1,
            "missing_concepts": ["Probenvorbereitung", "..."],
            "new_gnd_results": {...},
            "new_keywords_found": 5,
            "final_keywords": [...]
        },
        ...
    ]
    """
    max_iterations_reached: bool = False
    convergence_achieved: bool = False
```

**Backward Compatibility**: All new fields use `field(default_factory=...)` so old JSON files load without errors.

#### Phase 3: Implement Fallback GND Search

**File**: `src/utils/pipeline_utils.py`

Add new method to `PipelineStepExecutor`:

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

    Args:
        missing_concepts: List of concepts not covered by existing GND pool
        existing_results: Current search results to avoid duplicates
        stream_callback: Progress feedback callback

    Returns:
        New search results for missing concepts (merged with existing)

    Search Strategy:
        1. Exact term search first
        2. If no results: try broader terms (from GND hierarchy)
        3. If still no results: try related terms
        4. Track which concepts found no matches
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

        # Execute search via SearchEngine
        search_result = self.cache_manager.search_gnd(
            query=concept,
            max_results=10
        )

        if search_result and len(search_result.get("gndid", set())) > 0:
            new_results[concept] = search_result
            if stream_callback:
                count = len(search_result.get("gndid", set()))
                stream_callback(f"    ✓ {count} GND-Treffer gefunden\n", "keywords_refinement")
        else:
            concepts_not_found.append(concept)
            if stream_callback:
                stream_callback(f"    ✗ Keine GND-Einträge gefunden\n", "keywords_refinement")

    # Merge with existing results
    merged_results = {**existing_results}
    for concept, data in new_results.items():
        if concept in merged_results:
            # Merge GND-IDs for existing keywords
            merged_results[concept]["gndid"].update(data.get("gndid", set()))
        else:
            merged_results[concept] = data

    if stream_callback:
        stream_callback(
            f"\n📊 Fallback-Ergebnis: {len(new_results)}/{len(missing_concepts)} Konzepte gefunden\n",
            "keywords_refinement"
        )
        if concepts_not_found:
            stream_callback(
                f"⚠️  Nicht gefunden: {', '.join(concepts_not_found[:5])}\n",
                "keywords_refinement"
            )

    return merged_results
```

#### Phase 4: Implement Iterative Refinement Loop

**File**: `src/utils/pipeline_utils.py`

Add new orchestration method:

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
) -> Tuple[List[str], Dict[str, Any], KeywordAnalysisState]:
    """
    Iteratively refine keyword selection by searching for missing concepts.

    Process:
        1. Run initial keyword analysis
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
        (final_keywords, iteration_metadata, enriched_state)
    """
    current_search_results = initial_search_results.copy()
    iteration_history = []
    previous_missing_concepts = []

    for iteration in range(1, max_iterations + 1):
        if stream_callback:
            stream_callback(
                f"\n{'='*60}\n🔄 Iteration {iteration}/{max_iterations}\n{'='*60}\n",
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

        # Store in llm_analysis
        llm_analysis.missing_concepts = missing_concepts

        # Record iteration
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

        # Check convergence conditions
        if not missing_concepts:
            if stream_callback:
                stream_callback(
                    "✓ Konvergenz erreicht: Keine fehlenden Konzepte\n",
                    "keywords_refinement"
                )
            iteration_data["convergence_reason"] = "no_missing_concepts"
            iteration_history.append(iteration_data)
            break

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

    # Build enriched state
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

#### Phase 5: UI Integration

**File**: `src/core/pipeline_manager.py`

Update `_execute_keywords_step()`:

```python
def _execute_keywords_step(self, auto_advance: bool = False) -> None:
    """Execute final keyword analysis step with optional iterative refinement."""

    # ... existing validation code ...

    step_config = self.config.pipeline_steps.get("keywords", PipelineStepConfig())

    # Check if iterative refinement enabled - Claude Generated
    enable_iteration = step_config.enable_iterative_refinement if hasattr(step_config, 'enable_iterative_refinement') else False
    max_iterations = step_config.max_refinement_iterations if hasattr(step_config, 'max_refinement_iterations') else 2

    if enable_iteration:
        # Iterative refinement path
        final_keywords, iteration_metadata, llm_analysis = (
            self.pipeline_executor.execute_iterative_keyword_refinement(
                original_abstract=self.state.original_abstract,
                initial_search_results=search_results,
                model=step_config.model,
                provider=step_config.provider,
                max_iterations=max_iterations,
                stream_callback=self._create_stream_callback("keywords"),
                **filtered_params
            )
        )

        # Store iteration metadata
        self.state.refinement_iterations = iteration_metadata["iteration_history"]
        self.state.convergence_achieved = iteration_metadata["convergence_achieved"]
    else:
        # Standard single-pass execution
        final_keywords, _, llm_analysis = (
            self.pipeline_executor.execute_final_keyword_analysis(
                # ... existing parameters ...
            )
        )

    # ... rest of existing code ...
```

**File**: `src/ui/pipeline_config_dialog.py`

Add refinement configuration controls:

```python
# In PipelineStepConfigWidget.__init__()

# Iterative Refinement Section - Claude Generated
self.refinement_group = QGroupBox("🔄 Iterative GND-Suche")
refinement_layout = QVBoxLayout()

self.enable_refinement = QCheckBox("Iterative Suche aktivieren")
self.enable_refinement.setToolTip(
    "Wenn aktiviert, sucht das System nach fehlenden Konzepten "
    "und erweitert den GND-Pool automatisch"
)

refinement_controls = QHBoxLayout()
refinement_controls.addWidget(QLabel("Max. Iterationen:"))
self.max_iterations_spin = QSpinBox()
self.max_iterations_spin.setRange(1, 5)
self.max_iterations_spin.setValue(2)
self.max_iterations_spin.setEnabled(False)

self.enable_refinement.toggled.connect(
    self.max_iterations_spin.setEnabled
)

refinement_controls.addWidget(self.max_iterations_spin)
refinement_controls.addStretch()

refinement_layout.addWidget(self.enable_refinement)
refinement_layout.addLayout(refinement_controls)
self.refinement_group.setLayout(refinement_layout)

self.layout.addWidget(self.refinement_group)
```

**File**: `src/ui/analysis_review_tab.py`

Add iteration history display:

```python
def display_iteration_history(self, state: KeywordAnalysisState):
    """Display iterative refinement history - Claude Generated"""

    if not state.refinement_iterations:
        return  # No iterations to display

    history_group = QGroupBox("🔄 Iterationsverlauf")
    layout = QVBoxLayout()

    for it_data in state.refinement_iterations:
        iteration_widget = QWidget()
        it_layout = QVBoxLayout()

        # Header
        header = QLabel(f"<b>Iteration {it_data['iteration']}</b>")
        it_layout.addWidget(header)

        # Stats
        stats_text = f"""
        Keywords: {it_data['keywords_selected']}
        GND-Pool: {it_data['gnd_pool_size']}
        Fehlende Konzepte: {len(it_data['missing_concepts'])}
        """
        stats_label = QLabel(stats_text)
        it_layout.addWidget(stats_label)

        # Missing concepts list
        if it_data['missing_concepts']:
            concepts_label = QLabel(f"<i>{', '.join(it_data['missing_concepts'][:5])}</i>")
            it_layout.addWidget(concepts_label)

        # Convergence reason
        if 'convergence_reason' in it_data:
            reason_map = {
                "no_missing_concepts": "✓ Keine fehlenden Konzepte",
                "self_consistency": "✓ Selbstkonsistenz erreicht",
                "no_new_results": "⚠️ Keine neuen GND-Ergebnisse",
                "max_iterations": "⚠️ Max. Iterationen erreicht"
            }
            reason_label = QLabel(reason_map.get(it_data['convergence_reason'], ""))
            it_layout.addWidget(reason_label)

        iteration_widget.setLayout(it_layout)
        layout.addWidget(iteration_widget)

    history_group.setLayout(layout)
    self.layout.addWidget(history_group)
```

#### Phase 6: CLI Integration

**File**: `src/alima_cli.py`

Add CLI flags:

```python
parser.add_argument(
    '--enable-iterative-search',
    action='store_true',
    help='Enable iterative GND search for missing concepts'
)

parser.add_argument(
    '--max-iterations',
    type=int,
    default=2,
    help='Maximum refinement iterations (default: 2)'
)
```

Update pipeline execution:

```python
if args.enable_iterative_search:
    final_keywords, iteration_metadata, llm_analysis = (
        pipeline_executor.execute_iterative_keyword_refinement(
            original_abstract=abstract,
            initial_search_results=search_results,
            model=args.final_model,
            provider=args.final_provider,
            max_iterations=args.max_iterations,
            stream_callback=lambda token, step_id=None: print(token, end='', flush=True)
        )
    )

    # Display iteration summary
    print("\n" + "="*60)
    print(f"Iterationen: {iteration_metadata['total_iterations']}")
    print(f"Konvergenz: {'Ja' if iteration_metadata['convergence_achieved'] else 'Nein'}")
    print("="*60)
```

#### Phase 7: Configuration Model Updates

**File**: `src/utils/config_models.py`

Add fields to `PipelineStepConfig`:

```python
@dataclass
class PipelineStepConfig:
    enabled: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None

    # NEW - Iterative refinement - Claude Generated
    enable_iterative_refinement: bool = False
    max_refinement_iterations: int = 2

    # Existing chunking config
    enable_chunking: bool = False
    chunk_size: int = 500
```

### 1.4 Testing Strategy

**Unit Tests** (`tests/test_iterative_search.py`):
1. Test `extract_missing_concepts_from_response()` with various formats
2. Test convergence detection logic
3. Test GND merge without duplicates
4. Test max iterations termination

**Integration Tests**:
1. Full iteration cycle with mock LLM responses
2. Test with real abstracts from library science
3. Verify CLI and GUI produce identical results
4. Test JSON save/resume with iteration state

**Edge Cases**:
- Missing concepts list is malformed
- GND search returns no results
- All concepts already in GND pool
- Infinite loop prevention (max iterations)

### 1.5 Performance Impact

**Token Usage**:
- Baseline: 1 LLM call (~2000-5000 tokens)
- With 2 iterations: 3 LLM calls (~6000-15000 tokens)
- Mitigation: Make opt-in, default disabled

**Time Overhead**:
- Per iteration: +10-30s (LLM call) + 2-5s (GND search)
- Total with 2 iterations: +24-70s
- Acceptable for quality improvement

### 1.6 Files to Create/Modify

**New Files**:
- None (all changes integrated into existing architecture)

**Modified Files**:
1. `src/core/processing_utils.py` - Add `extract_missing_concepts_from_response()`
2. `src/core/data_models.py` - Add iteration tracking fields
3. `src/utils/pipeline_utils.py` - Add `execute_fallback_gnd_search()` and `execute_iterative_keyword_refinement()`
4. `src/core/pipeline_manager.py` - Update `_execute_keywords_step()`
5. `src/ui/pipeline_config_dialog.py` - Add refinement UI controls
6. `src/ui/analysis_review_tab.py` - Add iteration history display
7. `src/alima_cli.py` - Add CLI flags
8. `src/utils/config_models.py` - Add config fields

---

## Part 2: DK Classification Splitting (50/50 Parallel Processing)

### 2.1 Current State Analysis

**Current Implementation** (`src/utils/pipeline_utils.py:1102-1330`):
- Single LLM request with ALL deduplicated DK classifications
- Can result in 50-200+ classifications in one prompt
- High token usage, potential context window issues

**Opportunity**: Split list in half, process in parallel, merge results

### 2.2 Architecture Design

#### Split Strategy
```
Input: 100 deduplicated DK classifications

Split 50/50:
├─ Chunk A: Classifications 1-50
│  ├─ Abstract (full)
│  ├─ DK codes with titles/keywords
│  └─ LLM Request 1 → Result A: [DK 616.89, DK 541.14, ...]
│
└─ Chunk B: Classifications 51-100
   ├─ Abstract (full)
   ├─ DK codes with titles/keywords
   └─ LLM Request 2 → Result B: [QP 340, DK 006.3, ...]

Merge Strategy:
├─ Combine: Result A + Result B
├─ Deduplicate: Remove duplicates (preserve first occurrence)
├─ Limit: Top 10-15 classifications
└─ Final: [DK 616.89, DK 541.14, QP 340, DK 006.3, ...]
```

#### Parallel vs Sequential Execution

**Option 1: True Parallel (Async)**
- Use `asyncio` or threading for simultaneous LLM calls
- Faster but more complex error handling
- Risk: One failure affects both

**Option 2: Sequential (Simpler)**
- Execute chunk 1, then chunk 2
- Easier error recovery
- Only ~10-15s slower than parallel

**Recommendation**: Start with sequential, add async later if needed

### 2.3 Implementation Plan

#### Phase 1: Splitting Logic

**File**: `src/utils/pipeline_utils.py`

Add helper method to `PipelineStepExecutor`:

```python
def _split_dk_classifications_50_50(
    self,
    dk_search_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split DK classification list into two equal halves.

    Args:
        dk_search_results: Deduplicated DK classification results

    Returns:
        (chunk_a, chunk_b) - Two roughly equal lists

    Algorithm:
        1. Calculate midpoint: len(results) // 2
        2. Handle odd-length lists by giving extra to first chunk
        3. Preserve original order
    """
    if not dk_search_results:
        return [], []

    total = len(dk_search_results)
    midpoint = total // 2

    # If odd length, first chunk gets the extra item
    chunk_a = dk_search_results[:midpoint + (total % 2)]
    chunk_b = dk_search_results[midpoint + (total % 2):]

    return chunk_a, chunk_b
```

**Edge Cases**:
- Empty list: Return ([], [])
- Single item: Return ([item], [])
- Two items: Return ([item1], [item2])

#### Phase 2: Chunk Execution

**File**: `src/utils/pipeline_utils.py`

Add chunk execution method:

```python
def _execute_dk_classification_chunk(
    self,
    chunk_id: int,
    original_abstract: str,
    dk_chunk: List[Dict[str, Any]],
    model: str,
    provider: str,
    stream_callback: Optional[callable] = None,
    **kwargs
) -> List[str]:
    """
    Execute DK classification for a single chunk.

    Args:
        chunk_id: Chunk identifier (1 or 2)
        original_abstract: The full abstract (included in both chunks!)
        dk_chunk: This chunk's DK classifications
        model: LLM model
        provider: LLM provider
        stream_callback: Progress callback

    Returns:
        List of selected DK classification codes from this chunk
    """
    if stream_callback:
        stream_callback(
            f"\n📦 Verarbeite Chunk {chunk_id} ({len(dk_chunk)} Klassifikationen)...\n",
            "dk_classification"
        )

    # Format chunk for prompt (same as existing logic)
    classification_text = ""

    for result in dk_chunk:
        classification_type = result.get("classification_type", "DK")
        dk_code = result.get("dk", "")
        count = result.get("count", 0)
        matched_keywords = result.get("matched_keywords", [])
        titles = result.get("titles", [])

        # Format entry
        keyword_text = ", ".join(matched_keywords[:5]) if matched_keywords else "—"
        title_text = " | ".join(titles[:3]) if titles else "—"

        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\n"
        entry += f"Keywords: {keyword_text}\n"
        entry += f"Beispieltitel: {title_text}\n\n"

        classification_text += entry

    # Build prompt (using existing template structure)
    prompt_variables = {
        "abstract": original_abstract,
        "classifications": classification_text
    }

    # Call LLM
    response_text = self.alima_manager.execute_task(
        task="dk_class",
        variables=prompt_variables,
        model=model,
        provider=provider,
        stream_callback=stream_callback,
        **kwargs
    )

    # Extract classifications from response
    classifications = extract_dk_classifications_from_response(response_text)

    if stream_callback:
        stream_callback(
            f"✓ Chunk {chunk_id}: {len(classifications)} Klassifikationen ausgewählt\n",
            "dk_classification"
        )

    return classifications
```

#### Phase 3: Merge Logic

**File**: `src/utils/pipeline_utils.py`

Add merge method:

```python
def _merge_dk_classification_chunks(
    self,
    chunk_a_results: List[str],
    chunk_b_results: List[str],
    max_results: int = 15
) -> List[str]:
    """
    Merge DK classification results from two chunks.

    Strategy:
        1. Combine both lists
        2. Remove duplicates (preserve first occurrence)
        3. Limit to max_results (default: 15)

    Args:
        chunk_a_results: Classifications from chunk A
        chunk_b_results: Classifications from chunk B
        max_results: Maximum classifications to return

    Returns:
        Merged and deduplicated classification list
    """
    # Combine results
    all_classifications = chunk_a_results + chunk_b_results

    # Deduplicate while preserving order
    seen = set()
    unique_classifications = []

    for code in all_classifications:
        # Normalize for comparison (case-insensitive, strip whitespace)
        normalized = code.strip().upper()

        if normalized not in seen:
            seen.add(normalized)
            unique_classifications.append(code)  # Keep original formatting

    # Limit to max results
    return unique_classifications[:max_results]
```

**Deduplication Logic**:
- Case-insensitive comparison: "DK 616.89" == "dk 616.89"
- Whitespace normalization: "DK  616.89" == "DK 616.89"
- Preserve original formatting from first occurrence

#### Phase 4: Main Execution Method

**File**: `src/utils/pipeline_utils.py`

Add new method to `PipelineStepExecutor`:

```python
def execute_dk_classification_split(
    self,
    original_abstract: str,
    dk_search_results: List[Dict[str, Any]],
    model: str,
    provider: str,
    stream_callback: Optional[callable] = None,
    **kwargs
) -> List[str]:
    """
    Execute DK classification with 50/50 splitting.

    Process:
        1. Split DK results into two equal halves
        2. Execute classification for chunk A
        3. Execute classification for chunk B
        4. Merge and deduplicate results

    Args:
        original_abstract: The abstract text
        dk_search_results: Deduplicated DK classification results
        model: LLM model
        provider: LLM provider
        stream_callback: Progress callback

    Returns:
        Final merged list of DK classifications
    """
    if stream_callback:
        stream_callback(
            f"\n{'='*60}\n"
            f"🔀 DK-Klassifikation mit 50/50-Splitting\n"
            f"{'='*60}\n"
            f"Gesamt: {len(dk_search_results)} Klassifikationen\n",
            "dk_classification"
        )

    # Split into two chunks
    chunk_a, chunk_b = self._split_dk_classifications_50_50(dk_search_results)

    if stream_callback:
        stream_callback(
            f"Chunk A: {len(chunk_a)} Klassifikationen\n"
            f"Chunk B: {len(chunk_b)} Klassifikationen\n\n",
            "dk_classification"
        )

    # Execute chunk A
    chunk_a_results = self._execute_dk_classification_chunk(
        chunk_id=1,
        original_abstract=original_abstract,
        dk_chunk=chunk_a,
        model=model,
        provider=provider,
        stream_callback=stream_callback,
        **kwargs
    )

    # Execute chunk B (only if not empty)
    chunk_b_results = []
    if chunk_b:
        chunk_b_results = self._execute_dk_classification_chunk(
            chunk_id=2,
            original_abstract=original_abstract,
            dk_chunk=chunk_b,
            model=model,
            provider=provider,
            stream_callback=stream_callback,
            **kwargs
        )

    # Merge results
    if stream_callback:
        stream_callback(
            f"\n🔗 Zusammenführung der Ergebnisse...\n",
            "dk_classification"
        )

    final_classifications = self._merge_dk_classification_chunks(
        chunk_a_results=chunk_a_results,
        chunk_b_results=chunk_b_results,
        max_results=15
    )

    if stream_callback:
        stream_callback(
            f"✓ Final: {len(final_classifications)} Klassifikationen\n"
            f"  (Chunk A: {len(chunk_a_results)}, Chunk B: {len(chunk_b_results)})\n",
            "dk_classification"
        )

    return final_classifications
```

#### Phase 5: Update Main Classification Method

**File**: `src/utils/pipeline_utils.py`

Modify `execute_dk_classification()` to support both modes:

```python
def execute_dk_classification(
    self,
    original_abstract: str,
    dk_search_results: List[Dict[str, Any]],
    model: str = None,
    provider: str = None,
    stream_callback: Optional[callable] = None,
    dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,
    enable_splitting: bool = False,  # NEW - Claude Generated
    mode=None,
    **kwargs,
) -> List[str]:
    """
    Execute LLM-based DK classification with optional 50/50 splitting.

    Args:
        ...existing args...
        enable_splitting: If True, split DK list and process in two chunks

    Returns:
        List of selected DK classification codes
    """
    # ... existing provider selection, filtering, validation ...

    # NEW: Check if splitting is enabled - Claude Generated
    if enable_splitting and len(results_with_titles) >= 10:
        # Use splitting for large result sets
        return self.execute_dk_classification_split(
            original_abstract=original_abstract,
            dk_search_results=results_with_titles,
            model=model,
            provider=provider,
            stream_callback=stream_callback,
            **kwargs
        )
    else:
        # Use standard single-request processing
        # ... existing single-request code ...
        pass
```

#### Phase 6: Configuration Updates

**File**: `src/utils/config_models.py`

Add field to `PipelineStepConfig`:

```python
@dataclass
class PipelineStepConfig:
    # ... existing fields ...

    # DK classification splitting - Claude Generated
    enable_dk_splitting: bool = False
    dk_split_threshold: int = 10  # Only split if >= N classifications
```

#### Phase 7: UI Integration

**File**: `src/ui/pipeline_config_dialog.py`

Add splitting controls to DK classification config:

```python
# In classification step config section

self.dk_splitting_group = QGroupBox("🔀 DK-Splitting")
splitting_layout = QVBoxLayout()

self.enable_dk_splitting = QCheckBox("50/50-Splitting aktivieren")
self.enable_dk_splitting.setToolTip(
    "Teilt DK-Liste in zwei Hälften für parallele Verarbeitung.\n"
    "Reduziert Token-Last pro LLM-Anfrage."
)

threshold_layout = QHBoxLayout()
threshold_layout.addWidget(QLabel("Min. Klassifikationen für Split:"))
self.dk_split_threshold = QSpinBox()
self.dk_split_threshold.setRange(5, 50)
self.dk_split_threshold.setValue(10)
self.dk_split_threshold.setEnabled(False)

self.enable_dk_splitting.toggled.connect(
    self.dk_split_threshold.setEnabled
)

threshold_layout.addWidget(self.dk_split_threshold)
threshold_layout.addStretch()

splitting_layout.addWidget(self.enable_dk_splitting)
splitting_layout.addLayout(threshold_layout)
self.dk_splitting_group.setLayout(splitting_layout)
```

**File**: `src/ui/analysis_review_tab.py`

Display chunk information:

```python
def display_dk_chunk_info(self, state: KeywordAnalysisState):
    """Display DK splitting information if available - Claude Generated"""

    if not hasattr(state, 'dk_chunk_metadata'):
        return  # No splitting used

    chunk_info = state.dk_chunk_metadata

    info_text = f"""
    <b>DK-Splitting Details:</b><br>
    Chunk A: {chunk_info['chunk_a_size']} Klassifikationen → {chunk_info['chunk_a_selected']} ausgewählt<br>
    Chunk B: {chunk_info['chunk_b_size']} Klassifikationen → {chunk_info['chunk_b_selected']} ausgewählt<br>
    Merge: {chunk_info['duplicates_removed']} Duplikate entfernt<br>
    Final: {chunk_info['final_count']} Klassifikationen
    """

    label = QLabel(info_text)
    self.layout.addWidget(label)
```

#### Phase 8: CLI Support

**File**: `src/alima_cli.py`

Add CLI flag:

```python
parser.add_argument(
    '--enable-dk-splitting',
    action='store_true',
    help='Enable 50/50 splitting for DK classification'
)

parser.add_argument(
    '--dk-split-threshold',
    type=int,
    default=10,
    help='Minimum classifications required to trigger splitting (default: 10)'
)
```

### 2.4 Testing Strategy

**Unit Tests**:
1. Test `_split_dk_classifications_50_50()` with various list sizes (0, 1, 2, 99, 100)
2. Test merge deduplication logic
3. Test chunk execution with mock LLM responses

**Integration Tests**:
1. Full split workflow with real DK data
2. Compare results: splitting vs single-request
3. Verify no classifications lost in merge
4. Test with duplicate classifications across chunks

**Edge Cases**:
- Empty DK list
- Single classification
- All duplicates between chunks
- One chunk returns empty results

### 2.5 Performance Analysis

**Token Savings**:
- Baseline (100 DKs, single request): ~8000 tokens
- Split (2x 50 DKs): 2x ~4500 tokens = 9000 tokens
- **Net change**: +1000 tokens (minimal overhead)

**Why splitting helps**:
- Reduces context window pressure
- Prevents truncation of large prompts
- Allows better focus per request
- More reliable parsing of results

**Time Impact**:
- Sequential: +10-20s (second LLM call)
- Parallel (future): Minimal (concurrent execution)

### 2.6 Files to Create/Modify

**New Files**:
- None (integrated into existing)

**Modified Files**:
1. `src/utils/pipeline_utils.py` - Add splitting/merge logic
2. `src/utils/config_models.py` - Add config fields
3. `src/ui/pipeline_config_dialog.py` - Add splitting UI
4. `src/ui/analysis_review_tab.py` - Add chunk info display
5. `src/alima_cli.py` - Add CLI flags
6. `src/core/data_models.py` - Add optional chunk metadata field

---

## Part 3: Agentic Workflow Architecture

### 3.1 Vision & Scope

The user requested ALL four agentic capabilities:
1. **Self-Reflection/Validation**: Agents validate their own outputs
2. **Multi-Agent Collaboration**: Specialized agents work together
3. **Adaptive Strategy Selection**: Dynamic prompt/model selection
4. **Hierarchical Planning**: Meta-agent coordinates sub-agents

**Challenge**: This is a significant architectural shift from linear pipeline to agent-based orchestration.

**Recommendation**: Implement in phases, starting with self-reflection and adaptive strategy (easier), then multi-agent and hierarchical planning (complex).

### 3.2 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Meta-Agent (Orchestrator)                  │
│                                                                 │
│  Responsibilities:                                              │
│  - Analyze abstract to determine strategy                       │
│  - Select appropriate specialized agents                        │
│  - Coordinate agent execution order                             │
│  - Aggregate and validate results                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Search Agent  │ │Keyword Agent  │ │ Class Agent   │
    │               │ │               │ │               │
    │ - GND search  │ │ - Keyword     │ │ - DK/RVK      │
    │ - Strategy    │ │   selection   │ │   selection   │
    │   selection   │ │ - Self-review │ │ - Validation  │
    │ - Quality     │ │ - Iteration   │ │               │
    └───────────────┘ └───────────────┘ └───────────────┘
           │                  │                  │
           │                  │                  │
           ▼                  ▼                  ▼
    ┌────────────────────────────────────────────────┐
    │         Validation Agent (Quality Gate)        │
    │                                                │
    │  - Cross-check agent outputs                   │
    │  - Detect inconsistencies                      │
    │  - Trigger re-execution if needed              │
    │  - Quality metrics calculation                 │
    └────────────────────────────────────────────────┘
```

### 3.3 Component Design

#### 3.3.1 Agent Base Class

**File**: `src/core/agents/base_agent.py` (NEW)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class AgentResult:
    """Result from an agent execution - Claude Generated"""
    success: bool
    data: Any
    confidence: float  # 0.0-1.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    iteration_count: int = 1
    reflection_notes: List[str] = field(default_factory=list)

class BaseAgent(ABC):
    """Base class for all ALIMA agents - Claude Generated"""

    def __init__(self, name: str, alima_manager, config: Dict[str, Any]):
        self.name = name
        self.alima_manager = alima_manager
        self.config = config
        self.execution_history = []

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Execute agent's primary task"""
        pass

    @abstractmethod
    def self_validate(self, result: AgentResult) -> bool:
        """Validate own output quality"""
        pass

    def reflect_on_result(self, result: AgentResult) -> List[str]:
        """
        Self-reflection: Identify potential issues with result.

        Returns:
            List of reflection notes (empty if result is good)
        """
        notes = []

        # Check confidence threshold
        if result.confidence < 0.6:
            notes.append(f"Low confidence: {result.confidence:.2f}")

        # Check quality metrics
        for metric, value in result.quality_metrics.items():
            if value < 0.5:
                notes.append(f"Quality concern - {metric}: {value:.2f}")

        return notes

    def execute_with_self_reflection(
        self,
        input_data: Any,
        max_iterations: int = 3,
        **kwargs
    ) -> AgentResult:
        """
        Execute task with self-reflection and auto-retry.

        Process:
            1. Execute task
            2. Self-validate result
            3. If validation fails AND iterations remaining:
               - Reflect on issues
               - Adjust strategy
               - Re-execute
            4. Return final result
        """
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Execute
            result = self.execute(input_data, iteration=iteration, **kwargs)
            result.iteration_count = iteration

            # Self-validate
            is_valid = self.self_validate(result)

            if is_valid:
                # Success!
                return result

            # Reflect on what went wrong
            reflection = self.reflect_on_result(result)
            result.reflection_notes = reflection

            # Record in history
            self.execution_history.append({
                "iteration": iteration,
                "confidence": result.confidence,
                "reflection": reflection
            })

            # Last iteration? Return even if not perfect
            if iteration >= max_iterations:
                return result

            # Prepare for retry with adjusted strategy
            kwargs = self._adjust_strategy_for_retry(result, kwargs)

        return result

    def _adjust_strategy_for_retry(
        self,
        previous_result: AgentResult,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adjust execution parameters for retry.
        Default: Increase temperature slightly for more creativity.
        Subclasses can override.
        """
        adjusted = kwargs.copy()

        if "temperature" in adjusted:
            # Increase temperature for more variation
            adjusted["temperature"] = min(adjusted["temperature"] + 0.1, 1.0)

        return adjusted
```

#### 3.3.2 Specialized Agents

**File**: `src/core/agents/search_agent.py` (NEW)

```python
from .base_agent import BaseAgent, AgentResult
from typing import Dict, Any

class SearchAgent(BaseAgent):
    """
    Specialized agent for GND/SWB search strategy selection.

    Responsibilities:
        - Analyze abstract to determine optimal search strategy
        - Execute searches with adaptive refinement
        - Validate search coverage (% of concepts found)
    """

    def __init__(self, alima_manager, cache_manager, config: Dict[str, Any]):
        super().__init__("SearchAgent", alima_manager, config)
        self.cache_manager = cache_manager

    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Execute GND/SWB search with strategy selection.

        Input:
            {
                "abstract": str,
                "initial_keywords": List[str],
                "text_type": str  # "scientific", "fiction", "report", etc.
            }

        Returns:
            AgentResult with search_results data
        """
        abstract = input_data["abstract"]
        initial_keywords = input_data["initial_keywords"]
        text_type = input_data.get("text_type", "general")

        # Adaptive strategy selection
        search_strategy = self._select_search_strategy(abstract, text_type)

        # Execute searches
        search_results = self._execute_searches(
            keywords=initial_keywords,
            strategy=search_strategy
        )

        # Calculate quality metrics
        coverage = self._calculate_coverage(initial_keywords, search_results)

        result = AgentResult(
            success=True,
            data=search_results,
            confidence=coverage,
            quality_metrics={
                "coverage": coverage,
                "avg_results_per_keyword": self._avg_results(search_results)
            }
        )

        return result

    def _select_search_strategy(self, abstract: str, text_type: str) -> Dict[str, Any]:
        """
        Adaptive strategy selection based on text characteristics.

        Strategies:
            - "scientific": Precise GND terms, use DDC for classification
            - "fiction": Broader terms, genre-based search
            - "report": Organizational keywords, geographic terms
        """
        strategies = {
            "scientific": {
                "search_depth": "precise",
                "enable_hierarchy": True,
                "max_results_per_keyword": 10
            },
            "fiction": {
                "search_depth": "broad",
                "enable_hierarchy": False,
                "max_results_per_keyword": 20
            },
            "general": {
                "search_depth": "balanced",
                "enable_hierarchy": True,
                "max_results_per_keyword": 15
            }
        }

        return strategies.get(text_type, strategies["general"])

    def _calculate_coverage(
        self,
        initial_keywords: List[str],
        search_results: Dict[str, Any]
    ) -> float:
        """
        Calculate search coverage: % of keywords with GND matches.
        """
        if not initial_keywords:
            return 0.0

        keywords_with_results = sum(
            1 for kw in initial_keywords
            if kw in search_results and search_results[kw].get("gndid")
        )

        return keywords_with_results / len(initial_keywords)

    def self_validate(self, result: AgentResult) -> bool:
        """
        Validate search quality.

        Criteria:
            - Coverage >= 50% (at least half keywords found)
            - Average results per keyword >= 2
        """
        coverage = result.quality_metrics.get("coverage", 0.0)
        avg_results = result.quality_metrics.get("avg_results_per_keyword", 0.0)

        return coverage >= 0.5 and avg_results >= 2.0
```

**File**: `src/core/agents/keyword_agent.py` (NEW)

```python
from .base_agent import BaseAgent, AgentResult
from typing import Dict, Any, List

class KeywordAgent(BaseAgent):
    """
    Specialized agent for GND keyword selection.

    Responsibilities:
        - Select optimal GND keywords from search results
        - Validate selection quality (relevance, coverage)
        - Iterate if quality insufficient
    """

    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Select GND keywords with quality validation.

        Input:
            {
                "abstract": str,
                "search_results": Dict[str, Any],
                "min_keywords": int,
                "max_keywords": int
            }
        """
        abstract = input_data["abstract"]
        search_results = input_data["search_results"]
        min_kw = input_data.get("min_keywords", 5)
        max_kw = input_data.get("max_keywords", 20)

        # Call LLM for keyword selection
        selected_keywords, llm_analysis = self._call_llm_for_selection(
            abstract=abstract,
            search_results=search_results,
            **kwargs
        )

        # Quality metrics
        relevance_score = self._calculate_relevance(
            abstract=abstract,
            keywords=selected_keywords
        )

        result = AgentResult(
            success=True,
            data={
                "keywords": selected_keywords,
                "llm_analysis": llm_analysis
            },
            confidence=relevance_score,
            quality_metrics={
                "relevance": relevance_score,
                "count": len(selected_keywords),
                "in_range": min_kw <= len(selected_keywords) <= max_kw
            }
        )

        return result

    def self_validate(self, result: AgentResult) -> bool:
        """
        Validate keyword selection quality.

        Criteria:
            - Relevance >= 0.7
            - Keyword count in expected range
            - All keywords have GND-IDs
        """
        relevance = result.quality_metrics.get("relevance", 0.0)
        in_range = result.quality_metrics.get("in_range", False)

        return relevance >= 0.7 and in_range

    def _calculate_relevance(self, abstract: str, keywords: List[str]) -> float:
        """
        Estimate keyword relevance using simple heuristics.

        Methods:
            1. Term overlap: How many keyword stems appear in abstract?
            2. Semantic density: Are keywords concentrated or scattered?
            3. Coverage: Do keywords cover main themes?
        """
        if not keywords:
            return 0.0

        # Simple implementation: Check term overlap
        abstract_lower = abstract.lower()

        overlap_count = sum(
            1 for kw in keywords
            if any(
                term.lower() in abstract_lower
                for term in kw.split()[:3]  # First 3 words
            )
        )

        return overlap_count / len(keywords)
```

**File**: `src/core/agents/classification_agent.py` (NEW)

```python
from .base_agent import BaseAgent, AgentResult
from typing import Dict, Any, List

class ClassificationAgent(BaseAgent):
    """
    Specialized agent for DK/RVK classification selection.

    Responsibilities:
        - Select relevant classifications from catalog results
        - Validate classification appropriateness
        - Ensure diversity (not all from same category)
    """

    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Select DK/RVK classifications with validation.

        Input:
            {
                "abstract": str,
                "dk_search_results": List[Dict],
                "keywords": List[str]
            }
        """
        abstract = input_data["abstract"]
        dk_results = input_data["dk_search_results"]
        keywords = input_data.get("keywords", [])

        # Call LLM for classification
        selected_classifications = self._call_llm_for_classification(
            abstract=abstract,
            dk_results=dk_results,
            **kwargs
        )

        # Quality metrics
        diversity = self._calculate_diversity(selected_classifications)

        result = AgentResult(
            success=True,
            data=selected_classifications,
            confidence=diversity,
            quality_metrics={
                "diversity": diversity,
                "count": len(selected_classifications)
            }
        )

        return result

    def self_validate(self, result: AgentResult) -> bool:
        """
        Validate classification quality.

        Criteria:
            - Diversity >= 0.5 (classifications from multiple categories)
            - Count between 3-15
        """
        diversity = result.quality_metrics.get("diversity", 0.0)
        count = result.quality_metrics.get("count", 0)

        return diversity >= 0.5 and 3 <= count <= 15

    def _calculate_diversity(self, classifications: List[str]) -> float:
        """
        Measure classification diversity.

        Method:
            Extract top-level categories (e.g., "616" from "616.89")
            Diversity = unique_categories / total_classifications
        """
        if not classifications:
            return 0.0

        top_level_categories = set()

        for code in classifications:
            # Extract first 3 digits for DK, first 2 letters for RVK
            if "DK" in code.upper():
                # Extract number part
                numbers = ''.join(filter(str.isdigit, code))
                if len(numbers) >= 3:
                    top_level_categories.add(numbers[:3])
            elif "RVK" in code.upper() or any(c.isalpha() for c in code):
                # RVK code
                letters = ''.join(filter(str.isalpha, code))
                if len(letters) >= 2:
                    top_level_categories.add(letters[:2])

        return len(top_level_categories) / len(classifications)
```

**File**: `src/core/agents/validation_agent.py` (NEW)

```python
from .base_agent import BaseAgent, AgentResult
from typing import Dict, Any, List

class ValidationAgent(BaseAgent):
    """
    Cross-validates outputs from multiple agents.

    Responsibilities:
        - Check consistency between agent outputs
        - Detect logical contradictions
        - Trigger re-execution if quality insufficient
        - Generate quality report
    """

    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Cross-validate agent results.

        Input:
            {
                "search_result": AgentResult,
                "keyword_result": AgentResult,
                "classification_result": AgentResult
            }
        """
        search_result = input_data["search_result"]
        keyword_result = input_data["keyword_result"]
        classification_result = input_data.get("classification_result")

        # Validation checks
        checks = {
            "search_coverage": self._validate_search_coverage(search_result),
            "keyword_quality": self._validate_keyword_quality(keyword_result),
            "keyword_search_consistency": self._validate_consistency(
                search_result, keyword_result
            )
        }

        if classification_result:
            checks["classification_quality"] = self._validate_classification_quality(
                classification_result
            )

        # Overall quality
        overall_quality = sum(checks.values()) / len(checks)

        result = AgentResult(
            success=overall_quality >= 0.7,
            data=checks,
            confidence=overall_quality,
            quality_metrics=checks
        )

        return result

    def _validate_consistency(
        self,
        search_result: AgentResult,
        keyword_result: AgentResult
    ) -> float:
        """
        Check if selected keywords actually came from search results.

        Returns:
            Consistency score (0.0-1.0)
        """
        search_data = search_result.data
        keywords = keyword_result.data.get("keywords", [])

        if not keywords:
            return 0.0

        # Count how many selected keywords have GND-IDs from search
        matched = 0
        for kw in keywords:
            # Extract GND-ID from keyword
            if "GND-ID" in kw:
                matched += 1

        return matched / len(keywords)

    def self_validate(self, result: AgentResult) -> bool:
        """Validation agent always returns True (meta-validator)"""
        return True
```

#### 3.3.3 Meta-Agent (Orchestrator)

**File**: `src/core/agents/meta_agent.py` (NEW)

```python
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResult
from .search_agent import SearchAgent
from .keyword_agent import KeywordAgent
from .classification_agent import ClassificationAgent
from .validation_agent import ValidationAgent

class MetaAgent:
    """
    Orchestrates specialized agents for complete pipeline execution.

    Responsibilities:
        - Analyze abstract to determine text type
        - Select and coordinate specialized agents
        - Implement hierarchical planning
        - Aggregate results from sub-agents
    """

    def __init__(
        self,
        alima_manager,
        cache_manager,
        config: Dict[str, Any]
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.config = config

        # Initialize specialized agents
        self.search_agent = SearchAgent(alima_manager, cache_manager, config)
        self.keyword_agent = KeywordAgent(alima_manager, cache_manager, config)
        self.classification_agent = ClassificationAgent(alima_manager, cache_manager, config)
        self.validation_agent = ValidationAgent(alima_manager, cache_manager, config)

    def execute_pipeline(
        self,
        abstract: str,
        enable_classification: bool = True,
        stream_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute complete agentic pipeline.

        Process:
            1. Analyze abstract → determine text type
            2. Plan execution strategy
            3. Execute search agent (with self-reflection)
            4. Execute keyword agent (with self-reflection)
            5. Cross-validate search + keyword results
            6. If validation fails: trigger retry
            7. Execute classification agent (optional)
            8. Final validation
            9. Return aggregated results
        """
        if stream_callback:
            stream_callback(
                "\n🤖 Agentic Workflow gestartet\n"
                "="*60 + "\n",
                "meta_agent"
            )

        # Phase 1: Text analysis
        text_type = self._analyze_text_type(abstract)
        if stream_callback:
            stream_callback(
                f"📝 Texttyp erkannt: {text_type}\n",
                "meta_agent"
            )

        # Phase 2: Plan strategy
        strategy = self._plan_execution_strategy(abstract, text_type)
        if stream_callback:
            stream_callback(
                f"📋 Strategie: {strategy['name']}\n\n",
                "meta_agent"
            )

        # Phase 3: Execute search agent
        if stream_callback:
            stream_callback("🔍 SearchAgent wird ausgeführt...\n", "search_agent")

        search_result = self.search_agent.execute_with_self_reflection(
            input_data={
                "abstract": abstract,
                "initial_keywords": strategy["initial_keywords"],
                "text_type": text_type
            },
            max_iterations=3
        )

        if stream_callback:
            stream_callback(
                f"  ✓ Confidence: {search_result.confidence:.2f}\n"
                f"  Iterationen: {search_result.iteration_count}\n\n",
                "search_agent"
            )

        # Phase 4: Execute keyword agent
        if stream_callback:
            stream_callback("🏷️  KeywordAgent wird ausgeführt...\n", "keyword_agent")

        keyword_result = self.keyword_agent.execute_with_self_reflection(
            input_data={
                "abstract": abstract,
                "search_results": search_result.data,
                "min_keywords": strategy["min_keywords"],
                "max_keywords": strategy["max_keywords"]
            },
            max_iterations=3
        )

        if stream_callback:
            stream_callback(
                f"  ✓ Confidence: {keyword_result.confidence:.2f}\n"
                f"  Iterationen: {keyword_result.iteration_count}\n\n",
                "keyword_agent"
            )

        # Phase 5: Cross-validation
        if stream_callback:
            stream_callback("✅ ValidationAgent prüft Ergebnisse...\n", "validation_agent")

        validation_result = self.validation_agent.execute(
            input_data={
                "search_result": search_result,
                "keyword_result": keyword_result
            }
        )

        if stream_callback:
            stream_callback(
                f"  ✓ Gesamtqualität: {validation_result.confidence:.2f}\n",
                "validation_agent"
            )

        # Phase 6: Retry if validation failed
        if not validation_result.success:
            if stream_callback:
                stream_callback(
                    "⚠️  Validierung fehlgeschlagen - Retry wird ausgelöst\n",
                    "meta_agent"
                )
            # Implement retry logic here...

        # Phase 7: Classification (optional)
        classification_result = None
        if enable_classification:
            if stream_callback:
                stream_callback("\n📊 ClassificationAgent wird ausgeführt...\n", "classification_agent")

            # Execute classification agent
            # ... implementation ...

        # Phase 8: Aggregate results
        final_results = {
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

        if classification_result:
            final_results["classifications"] = classification_result.data

        return final_results

    def _analyze_text_type(self, abstract: str) -> str:
        """
        Analyze abstract to determine text type.

        Types:
            - "scientific": Academic papers, research articles
            - "fiction": Novels, stories
            - "report": Reports, white papers
            - "general": General texts

        Method:
            1. Check for scientific indicators (citations, methodology, results)
            2. Check for narrative indicators (characters, plot)
            3. Check for report indicators (recommendations, findings)
            4. Default to general
        """
        abstract_lower = abstract.lower()

        # Scientific indicators
        scientific_keywords = [
            "methode", "ergebnis", "studie", "analyse", "untersuchung",
            "hypothese", "daten", "experiment", "conclusion"
        ]

        # Fiction indicators
        fiction_keywords = [
            "roman", "erzählung", "protagonist", "kapitel", "geschichte"
        ]

        # Count indicators
        scientific_count = sum(1 for kw in scientific_keywords if kw in abstract_lower)
        fiction_count = sum(1 for kw in fiction_keywords if kw in abstract_lower)

        if scientific_count >= 2:
            return "scientific"
        elif fiction_count >= 2:
            return "fiction"
        else:
            return "general"

    def _plan_execution_strategy(
        self,
        abstract: str,
        text_type: str
    ) -> Dict[str, Any]:
        """
        Plan execution strategy based on text characteristics.

        Returns:
            {
                "name": str,
                "initial_keywords": List[str],
                "min_keywords": int,
                "max_keywords": int,
                "preferred_model": str,
                "temperature": float
            }
        """
        # Generate initial keywords via LLM
        initial_keywords = self._generate_initial_keywords(abstract)

        strategies = {
            "scientific": {
                "name": "Wissenschaftliche Präzision",
                "initial_keywords": initial_keywords,
                "min_keywords": 10,
                "max_keywords": 20,
                "preferred_model": "cogito:32b",  # More accurate
                "temperature": 0.25
            },
            "fiction": {
                "name": "Literarische Erschließung",
                "initial_keywords": initial_keywords,
                "min_keywords": 5,
                "max_keywords": 15,
                "preferred_model": "cogito:14b",  # Faster
                "temperature": 0.4
            },
            "general": {
                "name": "Allgemeine Verschlagwortung",
                "initial_keywords": initial_keywords,
                "min_keywords": 5,
                "max_keywords": 20,
                "preferred_model": "cogito:14b",
                "temperature": 0.3
            }
        }

        return strategies.get(text_type, strategies["general"])
```

### 3.4 Integration with Existing Pipeline

**File**: `src/core/pipeline_manager.py`

Add agentic mode support:

```python
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
    Execute pipeline using agentic workflow instead of linear steps.
    """
    if not self.enable_agentic_mode:
        raise ValueError("Agentic mode not enabled in configuration")

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

### 3.5 Configuration

**File**: `src/utils/config_models.py`

Add agentic configuration:

```python
@dataclass
class AgenticConfig:
    """Configuration for agentic workflow - Claude Generated"""

    enable_agentic_mode: bool = False

    # Agent-specific settings
    search_agent_max_iterations: int = 3
    keyword_agent_max_iterations: int = 3
    classification_agent_max_iterations: int = 2

    # Quality thresholds
    min_search_coverage: float = 0.5
    min_keyword_relevance: float = 0.7
    min_overall_quality: float = 0.7

    # Adaptive strategy
    enable_adaptive_strategy: bool = True
    enable_text_type_detection: bool = True

    # Model preferences by text type
    scientific_model: str = "cogito:32b"
    fiction_model: str = "cogito:14b"
    general_model: str = "cogito:14b"

@dataclass
class Config:
    # ... existing fields ...

    agentic_config: AgenticConfig = field(default_factory=AgenticConfig)
```

### 3.6 UI Integration

**File**: `src/ui/pipeline_config_dialog.py`

Add agentic mode toggle:

```python
# Add to main config dialog

self.agentic_group = QGroupBox("🤖 Agentic Workflow (Experimental)")
agentic_layout = QVBoxLayout()

self.enable_agentic = QCheckBox("Agentic Modus aktivieren")
self.enable_agentic.setToolTip(
    "Aktiviert selbstreflektierende Agenten mit automatischer Qualitätskontrolle.\n"
    "WARNUNG: Experimentell, erhöht Token-Verbrauch signifikant."
)

# Agentic settings (disabled by default)
settings_layout = QFormLayout()

self.search_iterations_spin = QSpinBox()
self.search_iterations_spin.setRange(1, 5)
self.search_iterations_spin.setValue(3)
settings_layout.addRow("Search Agent Iterationen:", self.search_iterations_spin)

self.keyword_iterations_spin = QSpinBox()
self.keyword_iterations_spin.setRange(1, 5)
self.keyword_iterations_spin.setValue(3)
settings_layout.addRow("Keyword Agent Iterationen:", self.keyword_iterations_spin)

self.min_quality_spin = QDoubleSpinBox()
self.min_quality_spin.setRange(0.0, 1.0)
self.min_quality_spin.setSingleStep(0.1)
self.min_quality_spin.setValue(0.7)
settings_layout.addRow("Min. Qualitätsschwelle:", self.min_quality_spin)

agentic_layout.addWidget(self.enable_agentic)
agentic_layout.addLayout(settings_layout)
self.agentic_group.setLayout(agentic_layout)
```

**File**: `src/ui/analysis_review_tab.py`

Display agent execution details:

```python
def display_agent_execution_details(self, results: Dict[str, Any]):
    """Display agentic workflow execution details - Claude Generated"""

    details_group = QGroupBox("🤖 Agentic Workflow Details")
    layout = QVBoxLayout()

    # Text type and strategy
    info_text = f"""
    <b>Texttyp:</b> {results['text_type']}<br>
    <b>Strategie:</b> {results['strategy_used']}<br>
    <b>Gesamtqualität:</b> {results['overall_quality']:.2%}
    """
    layout.addWidget(QLabel(info_text))

    # Agent iterations
    iterations_table = QTableWidget(3, 2)
    iterations_table.setHorizontalHeaderLabels(["Agent", "Iterationen"])

    agents = [
        ("SearchAgent", results['agent_iterations']['search']),
        ("KeywordAgent", results['agent_iterations']['keyword']),
        ("ClassificationAgent", results['agent_iterations'].get('classification', 0))
    ]

    for row, (agent, iterations) in enumerate(agents):
        iterations_table.setItem(row, 0, QTableWidgetItem(agent))
        iterations_table.setItem(row, 1, QTableWidgetItem(str(iterations)))

    layout.addWidget(iterations_table)
    details_group.setLayout(layout)

    self.layout.addWidget(details_group)
```

### 3.7 Files to Create/Modify

**New Files**:
1. `src/core/agents/__init__.py` - Agent package
2. `src/core/agents/base_agent.py` - Base agent class
3. `src/core/agents/search_agent.py` - Search specialist
4. `src/core/agents/keyword_agent.py` - Keyword specialist
5. `src/core/agents/classification_agent.py` - Classification specialist
6. `src/core/agents/validation_agent.py` - Cross-validator
7. `src/core/agents/meta_agent.py` - Orchestrator

**Modified Files**:
1. `src/core/pipeline_manager.py` - Add agentic mode
2. `src/utils/config_models.py` - Add agentic config
3. `src/ui/pipeline_config_dialog.py` - Add agentic UI
4. `src/ui/analysis_review_tab.py` - Display agent details

### 3.8 Testing Strategy

**Unit Tests**:
1. Test each agent's `execute()` and `self_validate()` independently
2. Test `reflect_on_result()` with various quality scenarios
3. Test meta-agent's text type detection

**Integration Tests**:
1. Full agentic pipeline with mock LLM responses
2. Test retry logic when validation fails
3. Test adaptive strategy selection

**Performance Tests**:
1. Measure token usage: agentic vs. linear
2. Measure execution time with iterations
3. Quality comparison: agentic vs. linear results

### 3.9 Performance Impact

**Token Usage**:
- Linear pipeline: ~5000-10000 tokens per abstract
- Agentic (with 3 iterations avg): ~15000-30000 tokens
- **3x increase** - Make opt-in, warn users

**Time Overhead**:
- Linear: ~30-60 seconds
- Agentic: ~90-180 seconds
- Acceptable for quality improvement

**Quality Improvement** (Expected):
- Better keyword selection via self-reflection
- Higher search coverage via adaptive strategy
- Fewer missing concepts via validation loops

---

## Part 4: Documentation Strategy

### 4.1 Three Plan Files for docs/ Directory

After implementation, create these comprehensive documentation files:

**File**: `docs/iterative_gnd_search.md`
- Complete architecture documentation
- Data flow diagrams (ASCII art)
- Code examples and usage
- Configuration guide
- Testing instructions
- Troubleshooting section

**File**: `docs/dk_classification_splitting.md`
- Splitting algorithm documentation
- Merge strategy details
- Token optimization analysis
- Performance benchmarks
- Configuration guide
- CLI and GUI usage examples

**File**: `docs/agentic_workflow.md`
- Overall agent architecture
- Individual agent specifications
- Meta-agent orchestration
- Self-reflection mechanism
- Quality metrics explanation
- Integration guide
- Future extensions roadmap

### 4.2 CLAUDE.md Updates

**File**: `src/core/CLAUDE.md`

Add to Variable Section:

```markdown
### WIP: Agentic Workflow Architecture
- **Base Agent System**: Self-reflection, quality validation, iteration control
- **Specialized Agents**: SearchAgent, KeywordAgent, ClassificationAgent, ValidationAgent
- **Meta-Agent Orchestrator**: Text type detection, strategy selection, agent coordination
- **Integration**: Optional agentic mode in PipelineManager

### WIP: Iterative GND Search
- **Missing Concept Extraction**: Parse `<missing_list>` from LLM responses (prompt already supports)
- **Fallback Search**: GND search for missing concepts with hierarchy support
- **Iteration Control**: Max iterations + self-consistency convergence
- **UI Integration**: Manual trigger button, iteration history display
```

**File**: `src/utils/CLAUDE.md`

Add to Variable Section:

```markdown
### WIP: DK Classification Splitting
- **50/50 Split Logic**: Divide DK list into equal halves for parallel processing
- **Chunk Execution**: Each chunk gets full abstract + its half of classifications
- **Merge Strategy**: Combine results, deduplicate, limit to top 15
- **Configuration**: enable_dk_splitting, dk_split_threshold parameters
```

---

## Part 5: Implementation Phases

### Phase 1: Iterative GND Search (Priority: HIGH)
**Effort**: 2-3 days
**Risk**: Low (infrastructure 80% ready)
**Value**: Immediate quality improvement

**Steps**:
1. Day 1: Implement extraction + data models
2. Day 2: Implement fallback search + iteration loop
3. Day 3: UI integration + testing

### Phase 2: DK Classification Splitting (Priority: MEDIUM)
**Effort**: 1-2 days
**Risk**: Low (straightforward implementation)
**Value**: Token optimization, reliability

**Steps**:
1. Day 1: Implement split/merge logic + chunk execution
2. Day 2: UI integration + testing

### Phase 3: Agentic Workflow (Priority: LOW - Experimental)
**Effort**: 5-7 days
**Risk**: High (architectural change, experimental)
**Value**: Long-term quality improvement, future extensibility

**Steps**:
1. Day 1-2: Base agent class + SearchAgent
2. Day 3: KeywordAgent + ClassificationAgent
3. Day 4: ValidationAgent + MetaAgent
4. Day 5: Integration with PipelineManager
5. Day 6-7: UI integration + comprehensive testing

### Total Estimated Effort: 8-12 days

---

## Part 6: Backward Compatibility Checklist

✅ **Data Models**: All new fields use `field(default_factory=...)` for JSON compatibility
✅ **Configuration**: New config fields have sensible defaults, features are opt-in
✅ **CLI**: New flags are optional, existing commands work unchanged
✅ **GUI**: New UI elements are in separate sections, don't affect existing workflow
✅ **Pipeline**: Standard mode remains default, new modes are explicitly enabled
✅ **JSON Export**: Old exports load successfully, missing fields auto-populated

---

## Part 7: Success Metrics

### Iterative GND Search
- **Metric 1**: Average missing concepts found per iteration
- **Metric 2**: Coverage improvement (before vs. after iteration)
- **Metric 3**: Convergence rate (% reaching self-consistency vs. max iterations)
- **Target**: 20% increase in GND coverage

### DK Classification Splitting
- **Metric 1**: Token reduction per request (50% expected)
- **Metric 2**: Classification quality (split vs. single-request comparison)
- **Metric 3**: Execution time overhead
- **Target**: No quality degradation, <30s overhead

### Agentic Workflow
- **Metric 1**: Agent iteration rate (% needing retries)
- **Metric 2**: Overall quality score improvement
- **Metric 3**: Token usage increase
- **Target**: 10% quality improvement, <3x token increase

---

## Conclusion

This plan outlines three major enhancements to ALIMA's pipeline:

1. **Iterative GND Search** leverages existing prompt infrastructure to identify and search for missing concepts, significantly improving keyword coverage with minimal architectural changes.

2. **DK Classification Splitting** reduces token load and improves reliability by processing classifications in parallel chunks with intelligent merging.

3. **Agentic Workflow** introduces a self-reflective, multi-agent architecture that adaptively selects strategies, validates outputs, and iterates until quality thresholds are met.

All features maintain strict CLI/GUI parity through shared `pipeline_utils.py` logic, preserve backward compatibility, and are implemented as opt-in enhancements with comprehensive configuration control.

**Next Steps**: Proceed with Phase 1 (Iterative GND Search) as highest priority, then Phase 2 (DK Splitting), with Phase 3 (Agentic Workflow) as experimental long-term enhancement.
