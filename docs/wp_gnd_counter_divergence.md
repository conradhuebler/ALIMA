# WP: GND-Häufigkeit — Pipeline vs. Agent divergieren beim Persistieren

> **Status:** Diagnosed + verified (July 10, 2026), **no code written**. Ready-to-execute
> ~2-edit fix. Split off from `docs/wp_records_as_first_class.md` (it is the sharpest
> single instance of the count fragmentation, F-5) because it is a self-contained bugfix,
> not the larger `BibRecord` normalization. **Recommended as a quick-win before that WP.**

## Symptom (operator, "nach wie vor")

For a *deterministic* GND search, the classic pipeline and the agentic pipeline show
**different** "Häufigkeit" for the same term. Concretely: the agentic run shows the real
frequency *live*, then it drops to **0** at completion and on any JSON reload; the classic
run keeps the real number.

## Root cause (verified against the code)

Both paths compute the **same** underlying numbers — pool `count=1` (ranking placeholder)
plus the real frequency in `display_count`, via the shared `aggregate_gnd_results` engine
(`src/core/search/aggregate.py`). Both default to `aggregate_from_raw=True`
(`pipeline_utils.py:487-489`, `deterministic_functions.py:77-81`). **The divergence is only
at the surfacing/persistence layer:**

- **Classic** preserves both fields into the persisted state: `search_from_raw` →
  `nested_from_aggregate` keeps `display_count` (`aggregate.py:188-195`), and
  `_convert_search_results_to_objects` (`src/core/pipeline_manager.py:2122-2125`) wraps the
  results dict **verbatim** into `List[SearchResult]`. GUI `flatten_gnd_hits` then does
  `cnt = max(count, display_count)` (`src/utils/pipeline_formatters.py:268-273`) → real number.
- **Agentic** drops both fields when it rebuilds the canonical state:
  `SharedContext.to_keyword_analysis_state()` writes only `{"gndid": …, "ddc_codes": …}`
  for each title — **no `count`, no `display_count`** —
  (`src/core/agents/shared_context.py:295-298`, and the identical fallback at `:308-311`).
  The end-of-run `_sync_classical_tabs_from_state` (`src/ui/_pipeline_tab_events.py:539-545`)
  re-populates the GND table from that count-less state → `max(0, 0) = 0`, overwriting the
  correct live snapshot.
- **Reload twin:** `state_bridge._flatten_search_results` (`src/core/agents/state_bridge.py:78-83`)
  drops the same two fields, so loading a saved agentic state also loses the counts.

### Concrete example (from the trace)
Term "Halbleiter" (GND `4129772-7`), lobid freq 87 / swb freq 42 →
`aggregate_gnd_results` merges via **max** (not sum) → pool `count=1, display_count=87,
source_count=2` (identical in both paths). Classic → `max(1,87)=87`. Agentic live → 87.
Agentic persisted/reloaded → `max(0,0)=0`. **Classic 87, agent 0** for the same search.

## The fix (landmine-safe)

In `SharedContext.to_keyword_analysis_state()` (`shared_context.py:295-298` **and** the
`:308-311` fallback), carry the two fields when building `kw_results[title]`:

```python
kw_results[title] = {
    "gndid": all_gnd_ids,
    "ddc_codes": entry.get("ddc_codes", []),
    "count": entry.get("count", 1),
    **({"display_count": entry["display_count"]} if entry.get("display_count") is not None else {}),
}
```

Mirror the same two-field addition in `state_bridge._flatten_search_results`
(`state_bridge.py:78-83`) so a reloaded state keeps the frequency too.

**Why this respects the count-landmine** (the risk to guard): it writes only into the
**display** structure `KeywordAnalysisState.search_results`. Ranking/chunking reads
`count`/`source_count` off `context.gnd_entries` inside `rank_pool`/`selection_chunks`,
which this does **not** touch — pool `count` stays `1` there. `flatten_gnd_hits` already
does `max(count, display_count)`, so no formatter change is needed.

## Secondary, distinct issue (optional, not this fix)

The LLM-facing raw `search_lobid`/`search_swb` tools serialize `res.to_gnd_keywords()`
verbatim (`src/mcp/tool_registry.py:1641,1666-1675`), i.e. they emit `count` (=1 on a cache
hit) plus a separate `display_count`. A model reading `count` reports "1"; one reading
`display_count` reports the real number → **model-dependent** within-agentic variance,
separate from the deterministic bug above (`aggregate_gnd_results`, the tool the workflow
actually uses, is unaffected). Optional follow-up: prefer/surface `display_count` in the
raw-tool output or note the convention in the tool description (`tool_schemas.py:74` already
hints at it). `source_count` is **not** conflated with frequency in any display surface.

## Verification (required — GUI not headless-checkable)

Static trace only so far. Before closing: run **one agentic + one classic** pipeline on the
same term (e.g. "Halbleiter") and compare the GND-Recherche "Häufigkeit" column — they must
match. Add a unit test asserting `to_keyword_analysis_state` preserves `count`/`display_count`
for a `gnd_entries` entry that has them.

## Key files
`src/core/agents/shared_context.py:289-316` · `src/core/agents/state_bridge.py:54-85` ·
`src/core/pipeline_manager.py:991, 2106-2126` · `src/core/search/aggregate.py:98-112,164-197` ·
`src/core/search/caching.py:126-159` · `src/utils/pipeline_formatters.py:248-328` ·
`src/ui/_pipeline_tab_events.py:438-442, 539-545`.
