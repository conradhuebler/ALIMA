# DK Classification Splitting (50/50 Parallel Processing)

**Feature Status**: Planned
**Priority**: MEDIUM
**Effort**: 1-2 days
**Risk**: Low (straightforward implementation)

## Executive Summary

This feature optimizes DK/RVK classification selection by splitting large classification lists into two equal halves, processing them in parallel (or sequentially), and merging results. This reduces token load per LLM request, prevents context window issues, and improves reliability while maintaining classification quality.

**Key Benefits**:
- Reduced token load per request (~50% reduction in prompt size)
- Better focus per request (LLM analyzes fewer classifications at once)
- More reliable parsing (smaller outputs)
- Backward compatible (opt-in feature)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Implementation Details](#implementation-details)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Testing Strategy](#testing-strategy)
7. [Performance Analysis](#performance-analysis)
8. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Current Behavior

The DK classification step processes ALL deduplicated classifications in a single LLM request:

```
Abstract + 100 DK Codes → Single LLM Request → Final 10-15 Classifications
```

**Issues**:
1. **Token Overload**: Large prompts (50-200+ classifications) consume 8000+ tokens
2. **Context Window Pressure**: Risk of truncation with very large lists
3. **Parsing Complexity**: LLM must evaluate hundreds of options simultaneously
4. **Reduced Focus**: More options → less careful consideration per classification

### Example Scenario

**Input**: 100 deduplicated DK classifications from catalog search

**Current Behavior**:
```
Prompt Size: ~8500 tokens
LLM Task: "Select up to 10 relevant classifications from these 100 options..."

Challenges:
- LLM must process all 100 classifications at once
- Risk of skipping less prominent but relevant classifications
- High token cost
- Potential for truncation errors
```

**Desired Behavior**:
```
Chunk A (50 classifications) → LLM Request 1 → Result A (5 classifications)
Chunk B (50 classifications) → LLM Request 2 → Result B (7 classifications)
Merge → Final 10 classifications (deduplicated)

Benefits:
- Prompt Size: ~4500 tokens each (47% reduction)
- Better focus per request
- More reliable parsing
- Minimal overhead
```

---

## Solution Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              DK Classification Splitting Workflow               │
│                                                                 │
│  Input: 100 Deduplicated DK Classifications                    │
│                                                                 │
│         ┌──────────────────────────────┐                       │
│         │  Check Split Threshold       │                       │
│         │  (min 10 classifications)    │                       │
│         └──────────┬───────────────────┘                       │
│                    │                                            │
│        ┌───────────┴──────────┐                                │
│        ▼                      ▼                                │
│   ┌─────────┐           ┌──────────┐                          │
│   │ < 10    │           │ >= 10    │                          │
│   │ Skip    │           │ Split    │                          │
│   │ Split   │           │ 50/50    │                          │
│   └─────────┘           └──────┬───┘                          │
│        │                       │                               │
│        │                  ┌────┴────┐                         │
│        │                  │         │                         │
│        │           ┌──────▼──┐  ┌──▼──────┐                  │
│        │           │Chunk A  │  │Chunk B  │                  │
│        │           │(50 DKs) │  │(50 DKs) │                  │
│        │           └────┬────┘  └────┬────┘                  │
│        │                │            │                        │
│        │         ┌──────▼──┐   ┌────▼──────┐                 │
│        │         │Abstract │   │ Abstract  │                 │
│        │         │+ Chunk  │   │ + Chunk   │                 │
│        │         │A Details│   │ B Details │                 │
│        │         └────┬────┘   └────┬──────┘                 │
│        │              │             │                         │
│        │         ┌────▼────┐   ┌────▼──────┐                 │
│        │         │ LLM     │   │  LLM      │                 │
│        │         │Request 1│   │ Request 2 │                 │
│        │         └────┬────┘   └────┬──────┘                 │
│        │              │             │                         │
│        │         ┌────▼────┐   ┌────▼──────┐                 │
│        │         │Result A │   │ Result B  │                 │
│        │         │(5 DKs)  │   │ (7 DKs)   │                 │
│        │         └────┬────┘   └────┬──────┘                 │
│        │              │             │                         │
│        │              └──────┬──────┘                         │
│        │                     │                                │
│        └─────────────────────┼────────────────────────────────┘
│                              │                                │
│                    ┌─────────▼──────────┐                     │
│                    │  Merge & Deduplicate│                     │
│                    │  - Combine results  │                     │
│                    │  - Remove duplicates│                     │
│                    │  - Limit to top 15  │                     │
│                    └─────────┬───────────┘                     │
│                              │                                │
│                    ┌─────────▼──────────┐                     │
│                    │ Final Classifications│                     │
│                    │     (10 DKs)       │                     │
│                    └────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Split Algorithm

**Input**: List of N deduplicated DK classifications

**Step 1**: Check threshold
```python
if len(dk_search_results) < dk_split_threshold:
    # Use standard single-request processing
    return execute_standard_dk_classification()
```

**Step 2**: Calculate midpoint
```python
total = len(dk_search_results)
midpoint = total // 2

# Handle odd-length lists
chunk_a = dk_search_results[:midpoint + (total % 2)]  # Extra item if odd
chunk_b = dk_search_results[midpoint + (total % 2):]
```

**Step 3**: Process chunks
- Both chunks receive FULL abstract (for context)
- Each chunk has its subset of DK classifications
- LLM selects relevant classifications from its chunk

**Step 4**: Merge results
```python
# Combine
all_classifications = chunk_a_results + chunk_b_results

# Deduplicate (case-insensitive, preserve first occurrence)
unique = []
seen = set()
for code in all_classifications:
    normalized = code.strip().upper()
    if normalized not in seen:
        seen.add(normalized)
        unique.append(code)  # Keep original formatting

# Limit to top N
return unique[:15]
```

---

## Implementation Details

### Phase 1: Splitting Helper

**File**: `src/utils/pipeline_utils.py`

```python
def _split_dk_classifications_50_50(
    self,
    dk_search_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split DK classification list into two equal halves.

    Algorithm:
        1. Calculate midpoint: len(results) // 2
        2. Handle odd-length lists by giving extra to first chunk
        3. Preserve original order

    Args:
        dk_search_results: Deduplicated DK classification results

    Returns:
        (chunk_a, chunk_b) - Two roughly equal lists

    Examples:
        >>> dk_results = [dk1, dk2, dk3, dk4, dk5]  # 5 items
        >>> chunk_a, chunk_b = _split_dk_classifications_50_50(dk_results)
        >>> len(chunk_a), len(chunk_b)
        (3, 2)  # First chunk gets extra item

        >>> dk_results = [dk1, dk2, dk3, dk4]  # 4 items (even)
        >>> chunk_a, chunk_b = _split_dk_classifications_50_50(dk_results)
        >>> len(chunk_a), len(chunk_b)
        (2, 2)  # Perfectly split

        >>> dk_results = []  # Empty
        >>> chunk_a, chunk_b = _split_dk_classifications_50_50(dk_results)
        >>> chunk_a, chunk_b
        ([], [])
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

### Phase 2: Chunk Execution

**File**: `src/utils/pipeline_utils.py`

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

    CRITICAL: Both chunks receive the FULL abstract for context.
    Only the DK classification list is split.

    Args:
        chunk_id: Chunk identifier (1 or 2) for logging
        original_abstract: The full abstract (included in both chunks!)
        dk_chunk: This chunk's DK classifications
        model: LLM model
        provider: LLM provider
        stream_callback: Progress callback

    Returns:
        List of selected DK classification codes from this chunk

    Example:
        >>> chunk_a_results = _execute_dk_classification_chunk(
        ...     chunk_id=1,
        ...     original_abstract="Full abstract...",
        ...     dk_chunk=chunk_a,  # 50 classifications
        ...     model="cogito:14b",
        ...     provider="ollama"
        ... )
        >>> len(chunk_a_results)
        5  # LLM selected 5 from this chunk
    """
    if stream_callback:
        stream_callback(
            f"\n📦 Verarbeite Chunk {chunk_id} ({len(dk_chunk)} Klassifikationen)...\n",
            "dk_classification"
        )

    # Format chunk for prompt (IDENTICAL to existing single-request logic)
    classification_text = ""

    for result in dk_chunk:
        classification_type = result.get("classification_type", "DK")
        dk_code = result.get("dk", "")
        count = result.get("count", 0)
        matched_keywords = result.get("matched_keywords", [])
        titles = result.get("titles", [])

        # Format entry (same as existing)
        keyword_text = ", ".join(matched_keywords[:5]) if matched_keywords else "—"
        title_text = " | ".join(titles[:3]) if titles else "—"

        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\n"
        entry += f"Keywords: {keyword_text}\n"
        entry += f"Beispieltitel: {title_text}\n\n"

        classification_text += entry

    # Build prompt (using existing template structure)
    # NOTE: abstract is FULL, classifications is CHUNK
    prompt_variables = {
        "abstract": original_abstract,  # FULL abstract
        "classifications": classification_text  # CHUNK classifications
    }

    # Call LLM (reuses existing logic)
    response_text = self.alima_manager.execute_task(
        task="dk_class",
        variables=prompt_variables,
        model=model,
        provider=provider,
        stream_callback=stream_callback,
        **kwargs
    )

    # Extract classifications from response (existing function)
    from src.core.processing_utils import extract_dk_classifications_from_response
    classifications = extract_dk_classifications_from_response(response_text)

    if stream_callback:
        stream_callback(
            f"✓ Chunk {chunk_id}: {len(classifications)} Klassifikationen ausgewählt\n",
            "dk_classification"
        )

    return classifications
```

### Phase 3: Merge Logic

**File**: `src/utils/pipeline_utils.py`

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
           - Case-insensitive comparison
           - Whitespace normalization
        3. Limit to max_results (default: 15)

    Args:
        chunk_a_results: Classifications from chunk A
        chunk_b_results: Classifications from chunk B
        max_results: Maximum classifications to return

    Returns:
        Merged and deduplicated classification list

    Examples:
        >>> chunk_a = ["DK 616.89", "DK 541.14", "QP 340"]
        >>> chunk_b = ["DK 616.89", "DK 006.3", "RVK ST 230"]  # DK 616.89 is duplicate
        >>> merged = _merge_dk_classification_chunks(chunk_a, chunk_b, max_results=15)
        >>> merged
        ['DK 616.89', 'DK 541.14', 'QP 340', 'DK 006.3', 'RVK ST 230']
        >>> # Note: Duplicate removed, first occurrence preserved

        >>> chunk_a = ["dk 616.89", "DK  541.14"]  # Different formatting
        >>> chunk_b = ["DK 616.89", "dk541.14"]
        >>> merged = _merge_dk_classification_chunks(chunk_a, chunk_b)
        >>> merged
        ['dk 616.89', 'DK  541.14']  # Originals preserved, duplicates removed
    """
    # Combine results
    all_classifications = chunk_a_results + chunk_b_results

    # Deduplicate while preserving order
    seen = set()
    unique_classifications = []

    for code in all_classifications:
        # Normalize for comparison (case-insensitive, strip whitespace)
        # Remove spaces for comparison but keep original formatting
        normalized = code.strip().upper().replace(" ", "")

        if normalized not in seen:
            seen.add(normalized)
            unique_classifications.append(code)  # Keep original formatting

    # Limit to max results
    return unique_classifications[:max_results]
```

**Deduplication Logic**:
- `"DK 616.89"` == `"dk 616.89"` (case-insensitive)
- `"DK  616.89"` == `"DK 616.89"` (whitespace normalized)
- `"DK616.89"` == `"DK 616.89"` (space-agnostic)
- **First occurrence preserved**: If chunk A has "DK 616.89" and chunk B has "dk 616.89", final result uses "DK 616.89"

### Phase 4: Main Split Execution

**File**: `src/utils/pipeline_utils.py`

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

    Example:
        >>> dk_results = get_dk_search_results()  # 100 classifications
        >>> final = executor.execute_dk_classification_split(
        ...     abstract="...",
        ...     dk_search_results=dk_results,
        ...     model="cogito:14b",
        ...     provider="ollama"
        ... )
        >>> len(final)
        10  # Final classifications after merge
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
        duplicates_removed = (
            len(chunk_a_results) + len(chunk_b_results)
        ) - len(final_classifications)

        stream_callback(
            f"✓ Final: {len(final_classifications)} Klassifikationen\n"
            f"  (Chunk A: {len(chunk_a_results)}, "
            f"Chunk B: {len(chunk_b_results)}, "
            f"Duplikate entfernt: {duplicates_removed})\n",
            "dk_classification"
        )

    return final_classifications
```

### Phase 5: Integration with Existing Method

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
    dk_split_threshold: int = 10,  # NEW - Claude Generated
    mode=None,
    **kwargs,
) -> List[str]:
    """
    Execute LLM-based DK classification with optional 50/50 splitting.

    Args:
        ...existing args...
        enable_splitting: If True, split DK list and process in two chunks
        dk_split_threshold: Minimum classifications required to trigger splitting

    Returns:
        List of selected DK classification codes
    """
    # ... existing provider selection, filtering, validation ...

    # NEW: Check if splitting is enabled - Claude Generated
    if enable_splitting and len(results_with_titles) >= dk_split_threshold:
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

---

## Configuration

### Pipeline Configuration

**File**: `src/utils/config_models.py`

```python
@dataclass
class PipelineStepConfig:
    # ... existing fields ...

    # DK classification splitting - Claude Generated
    enable_dk_splitting: bool = False  # Default: OFF
    dk_split_threshold: int = 10  # Only split if >= N classifications
```

### GUI Configuration

**Location**: Pipeline Config Dialog → Classification Step Tab

```
┌─────────────────────────────────────────────────┐
│ 🔀 DK-Splitting                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ☐ 50/50-Splitting aktivieren                  │
│                                                 │
│  Min. Klassifikationen für Split:  [10 ▼]      │
│                                                 │
│  ℹ️  Teilt DK-Liste in zwei Hälften für        │
│     parallele Verarbeitung. Reduziert Token-   │
│     Last pro LLM-Anfrage.                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### CLI Configuration

```bash
# Enable DK splitting
python alima_cli.py pipeline \
    --input-text "..." \
    --enable-dk-splitting \
    --dk-split-threshold 15

# Disable (default)
python alima_cli.py pipeline --input-text "..."
```

---

## Usage Examples

### Example 1: Large DK Result Set

**Input**:
```
Abstract: "Medizinische Studie zu Diagnoseverfahren..."
DK Search Results: 100 classifications
```

**Without Splitting** (Current):
```
Prompt Size: 8500 tokens
LLM Request: Single request with all 100 classifications
Result: 10 classifications
Time: 25 seconds
```

**With Splitting** (New):
```
Chunk A: 50 classifications → Prompt: 4500 tokens → Result: 6 classifications
Chunk B: 50 classifications → Prompt: 4500 tokens → Result: 5 classifications
Merge: 6 + 5 = 11 → Deduplicate → 10 unique → Final: 10 classifications

Total Time: 25s + 25s = 50 seconds (+25s overhead)
Token Reduction: 8500 → 9000 tokens (minimal overhead, but better reliability)
```

### Example 2: Small DK Result Set (Skips Splitting)

**Input**:
```
Abstract: "..."
DK Search Results: 8 classifications (below threshold)
```

**Behavior**:
```
Split threshold: 10
Result count: 8 < 10
Action: Skip splitting, use standard single-request processing
```

---

## Testing Strategy

### Unit Tests

```python
class TestDKSplitting:
    """Test DK classification splitting logic"""

    def test_split_even_count(self):
        """Test splitting with even number of classifications"""
        dk_results = [f"dk_{i}" for i in range(10)]
        chunk_a, chunk_b = executor._split_dk_classifications_50_50(dk_results)

        assert len(chunk_a) == 5
        assert len(chunk_b) == 5

    def test_split_odd_count(self):
        """Test splitting with odd number of classifications"""
        dk_results = [f"dk_{i}" for i in range(11)]
        chunk_a, chunk_b = executor._split_dk_classifications_50_50(dk_results)

        assert len(chunk_a) == 6  # First chunk gets extra
        assert len(chunk_b) == 5

    def test_split_empty(self):
        """Test splitting empty list"""
        chunk_a, chunk_b = executor._split_dk_classifications_50_50([])

        assert chunk_a == []
        assert chunk_b == []

    def test_merge_deduplication(self):
        """Test merge removes duplicates"""
        chunk_a = ["DK 616.89", "DK 541.14"]
        chunk_b = ["DK 616.89", "DK 006.3"]  # Duplicate

        merged = executor._merge_dk_classification_chunks(chunk_a, chunk_b)

        assert len(merged) == 3
        assert merged == ["DK 616.89", "DK 541.14", "DK 006.3"]

    def test_merge_case_insensitive(self):
        """Test merge handles different cases"""
        chunk_a = ["DK 616.89"]
        chunk_b = ["dk 616.89"]  # Same code, different case

        merged = executor._merge_dk_classification_chunks(chunk_a, chunk_b)

        assert len(merged) == 1  # Deduplicated
        assert merged[0] == "DK 616.89"  # First occurrence preserved
```

### Integration Tests

```python
class TestDKSplittingIntegration:
    """End-to-end tests with real pipeline"""

    def test_full_split_workflow(self):
        """Test complete split → execute → merge workflow"""
        abstract = "..."
        dk_results = get_mock_dk_results(count=100)

        final = executor.execute_dk_classification_split(
            abstract=abstract,
            dk_search_results=dk_results,
            model="cogito:14b",
            provider="ollama"
        )

        assert len(final) <= 15
        assert all(isinstance(code, str) for code in final)

    def test_quality_comparison(self):
        """Compare splitting vs single-request quality"""
        abstract = "..."
        dk_results = get_mock_dk_results(count=100)

        # Single request
        single_result = executor.execute_dk_classification(
            abstract=abstract,
            dk_search_results=dk_results,
            enable_splitting=False
        )

        # Split request
        split_result = executor.execute_dk_classification(
            abstract=abstract,
            dk_search_results=dk_results,
            enable_splitting=True
        )

        # Results should be similar (allow 20% variation)
        overlap = set(single_result) & set(split_result)
        assert len(overlap) / len(single_result) >= 0.8
```

---

## Performance Analysis

### Token Usage Comparison

| Scenario | Classifications | Single Request | Split Request | Change |
|----------|----------------|----------------|---------------|--------|
| Small | 10 | ~2000 tokens | N/A (skipped) | - |
| Medium | 50 | ~4500 tokens | 2x ~2500 = 5000 | +11% |
| Large | 100 | ~8500 tokens | 2x ~4500 = 9000 | +6% |
| Very Large | 200 | ~16000 tokens | 2x ~8500 = 17000 | +6% |

**Key Insight**: Minimal token overhead (~6-11%) but massive reliability improvement.

### Time Impact

| Operation | Single Request | Split Request | Overhead |
|-----------|---------------|---------------|----------|
| Prompt formatting | 0.1s | 0.2s | +0.1s |
| LLM call(s) | 25s | 50s | +25s |
| Merge | - | 0.05s | +0.05s |
| **Total** | **25.1s** | **50.25s** | **+25s** |

**Mitigation**: Feature is opt-in, disabled by default. Users can enable for quality-sensitive tasks.

### Quality Impact

Expected quality improvement:
- Better focus per request → more careful classification selection
- Reduced context window pressure → fewer truncation errors
- More reliable parsing → fewer malformed outputs

**Target**: No quality degradation, ideally slight improvement.

---

## Troubleshooting

### Issue: Both chunks return identical classifications

**Symptoms**: Chunk A and Chunk B select overlapping classifications

**Cause**: Both chunks see the same abstract, may select same popular classifications

**Solution**:
- This is expected behavior (LLM selects most relevant independently)
- Merge deduplication handles this automatically
- If overlap is >80%, consider increasing max_results to allow more diversity

### Issue: Split threshold not respected

**Symptoms**: Splitting occurs even with < 10 classifications

**Cause**: Configuration not properly propagated

**Solution**:
1. Verify `dk_split_threshold` in config
2. Check `execute_dk_classification()` receives `dk_split_threshold` parameter
3. Add logging to confirm threshold check

### Issue: Merge produces fewer classifications than expected

**Symptoms**: Chunk A returns 6, Chunk B returns 5, but final is only 8 (not 11)

**Cause**: Deduplication removed 3 overlapping classifications

**Solution**:
- This is expected behavior
- If you need more classifications, increase `max_results` parameter
- Review chunk results to see which classifications overlapped

---

## Future Enhancements

### Phase 2 Improvements

1. **True Parallel Execution**: Use `asyncio` for concurrent LLM calls
2. **Smart Splitting**: Split by frequency tiers instead of 50/50
3. **Adaptive Chunk Size**: Dynamically adjust split based on token limits
4. **N-way Splitting**: Split into 3+ chunks for very large result sets
5. **Frequency-Based Ranking**: Prioritize classifications with higher catalog frequency during merge

---

## References

- Main Plan: `/home/conrad/.claude/plans/stateless-soaring-squirrel.md`
- Pipeline Utils: `src/utils/pipeline_utils.py:1102-1330`
- Config Models: `src/utils/config_models.py`
- Processing Utils: `src/core/processing_utils.py`
