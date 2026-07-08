# WP: Every API search = a cached plugin/tool (analysis + plan)

> **Status:** Analysis + plan (July 8, 2026). Operator direction: (1) the raw data
> of *any* search is written to the cache; (2) every search against an API/website
> is a standalone plugin exposed as a tool; (3) whether a plugin is cached is a
> per-plugin setting. Legacy breakage is acceptable ("alte Zöpfe abschneiden").
> Prerequisite done: GND search unified on the provider path, MetaSuggester retired
> (see `AIChangelog.md`, July 8).

## Current landscape

Two real plugin categories on the generic framework (`src/core/plugins/`), both
auto-generating MCP tools from declarative specs:

| Category | Members | Tool generation |
|---|---|---|
| `search_provider` | lobid, swb, catalog, finc, sru, gnd_local | `ProviderToolSpec` → `search_*` |
| `input_source` | text, file, pdf, image, url_fetch, doi_crossref/openalex/datacite | `InputToolSpec` → `resolve_doi_*` |

So "every API search = a plugin/tool" is already ~70% real — a new GND/DOI source
is just a plugin dir.

### The old braids (external-API access that is NOT a plugin)
- **`rvk_lookup`** → `ToolRegistry._handle_rvk_lookup` (`tool_registry.py:1086`) →
  `rvk_api_client.py` / `rvk_marc_index.py`. Live RVK-classification API, hand-wired,
  no plugin, no raw cache.
- **`scrape_url`** → hardcoded handler; overlaps the `url_fetch` input plugin.
- **`resolve_doi`** (merged) + **`k10plus_resolver.py`** → the 3 DOI *sources* are
  plugins, the merged resolver + k10plus are not.
- **`dnb_utils.py`** + the DNB/GND bulk import (`_main_window_data.py` direct
  `LobidSuggester`) → API access with no tool/plugin.

Local (non-API) tools — `search_gnd`, `aggregate_gnd_results`, pipeline/export —
are **core tools**, not plugins; they should stay that way. The braids to cut are
specifically the external-API handlers.

## Findings (reproduced)

### F1 — mapping cache returns empty without warmed facts *(live bug)*
`CachingProvider._items_from_cache` (`caching.py:107`) resolves each cached gnd_id's
title via `get_gnd_fact()` and **skips gnd_ids with no fact row**. GND-keyword
searches write only the *mapping* (`update_search_mapping`: term→gnd_ids+counts),
never `gnd_entries` facts — those are written only by the pipeline's enrichment
(`get_gnd_batch`). So a **standalone** search (chat agent, `find_keywords`
standalone) poisons its own cache:

```
1st search (live):      49 items
2nd search (cache hit):  0 items   ← same term, seconds later
search_gnd(term):        0         ← local DB never sees searched terms
```

Pre-existing (not from the MetaSuggester WP). Root of the "search_gnd liefert nichts
obwohl es den Grasfrosch gibt" symptom.

### F2 — `search_local_gnd` hides 1–2 hits
`unified_knowledge_manager.py:873`: `return entries[:min_results] if len(entries) >=
min_results else []` — with < 3 local matches it returns `[]`, discarding valid hits.

### F3 — raw-cache coverage is uneven
Search providers (`_store_raw_responses`/`_store_records_raw`) + input tools
(`cacheable` read-through, `tool_registry.py:1418-1450`) cache raw. `rvk_lookup`,
`scrape_url`, and every hardcoded API handler bypass it. The per-plugin toggle
half-exists: `SuggesterBackedProvider._cache_raw_enabled()` already reads a per-
instance `cache_responses` override, but there is **no `ConfigField`** for it (not
editable), and input tools use a static `InputToolSpec.cacheable`, not a setting.

## Phase A — foundation (non-breaking)

1. **Warm facts at the shared write seam.** New
   `UnifiedKnowledgeManager.warm_gnd_entries({gnd_id: title})` using
   `INSERT OR IGNORE` (never clobbers a richer enrichment fact — unlike
   `store_gnd_fact`'s `INSERT OR REPLACE`). Call it from `CachingProvider._live_search`
   where `gnd_counts` is already built → every GND provider + both pipelines warm the
   shared local knowledge DB. Fixes F1.
2. **Fix F2** — return the found local entries even when fewer than `min_results`.
3. **Generalize raw-caching** to the tool-execution layer so any API tool
   (`rvk_lookup`, `scrape_url`, …) caches its raw response, not just plugin-generated
   ones.
4. **Per-plugin `cache_responses` setting** — a standard BOOL `ConfigField` on every
   plugin, read at execution (search + input), surfaced in the plugin settings tab.

Verification: full suite green + a live cache-hit parity check (2nd search returns
the same items; `search_gnd` finds a just-searched term).

## Phase B — cut the braids (breaking; plan+confirm each)

5. `rvk_lookup` → a classification-lookup plugin; `scrape_url` → fold into
   `url_fetch`; `resolve_doi`/`k10plus` → plugin or explicit composition tool.
6. Formalize a **"plugin tool vs core tool" split** (external-API plugins vs local
   DB/pipeline/export) so "everything is a plugin" does not over-reach into the DB.
7. A **new `lookup`/`enrichment` category** (or the `CLASSIFICATION` capability) for
   RVK/DK — they return codes, not rankable records, so they don't fit
   `search_provider`.

### What breaks
- Plugin-izing `rvk_lookup`/`scrape_url`/`resolve_doi` changes tool names +
  registration → chat prompts, workflow YAMLs, tests referencing those names.
- The plugin/core split may move some tool schemas between registries.
