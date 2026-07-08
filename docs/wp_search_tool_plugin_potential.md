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

## Phase A — foundation (non-breaking) ✅ DONE (July 8, commit)

1. ✅ **Warm facts at the shared write seam.**
   `UnifiedKnowledgeManager.warm_gnd_entries({gnd_id: title})` (`INSERT OR IGNORE`,
   never clobbers a richer enrichment fact) called from `CachingProvider._live_search`
   where `gnd_counts` is built → every GND provider + both pipelines warm the shared
   local knowledge DB. **Fixes F1** (verified: cache hit 0→49; `search_gnd` finds a
   just-searched term; enriched fact preserved).
2. ✅ **Fixed F2** — `search_local_gnd` returns partial local hits instead of `[]`.
3. ⏭️ **Generalize raw-caching to the hardcoded API tools** (`rvk_lookup`,
   `scrape_url`) **folded into Phase B** — those tools become plugins there and inherit
   caching via the plugin path, so doing it now would be throwaway. Plugin tools
   (search + input) already cache, now governed per-plugin (item 4).
4. ✅ **Per-plugin `cache_responses` setting** — a standard tri-state `ConfigField`
   (`auto`/`on`/`off`, `schema.cache_field()` + `cache_pref_enabled()`) injected into
   both category forms (`SearchProviderCategory`/`InputSourceCategory.type_meta`), read
   at execution (search: `SuggesterBackedProvider._cache_raw_enabled`; input:
   `_make_input_handler`). `auto` follows the global `enable_response_cache`. GUI form
   is schema-driven → renders automatically. Tests: `test_cache_setting.py`.

Operator note: the global `enable_response_cache` is **off** in the current config —
set it on (or a plugin's `cache_responses` to `on`) to actually cache raw. Tests:
`test_gnd_cache_warming.py`, `test_cache_setting.py`. Suite 1159 passed.

## Phase B — cut the braids (operator-scoped)

Operator scope (July 8): k10plus→input plugin · unify URL-fetch core · RVK client→
lookup plugin · **add a new `lookup` category** + formalize external-API=plugin /
local=core. (Did *not* drop `scrape_url`.)

### B4+B3 ✅ DONE — new `lookup` category + RVK-API plugin
- `src/utils/lookups/` — third plugin category (`registry.py` `@register_lookup` +
  `LookupToolSpec`; `category.py` `LookupCategory` adapter, self-registers +
  injects the standard `cache_field`; `rvk.py` `RvkLookup` wrapping
  `RvkApiClient`). Tools **`rvk_search`** (keyword→ranked notations) + **`rvk_validate`**
  (notation→label+ancestors), generated via `ToolRegistry._generated_lookup_tools`
  (mirrors input-tool generation) and **raw-cached** through the same per-plugin
  `cache_responses` gate → realizes "cache any search" for lookups. The composed
  **`rvk_lookup` core tool stays unchanged** (pipeline anchor machinery). GUI
  auto-discovers the category (`list_categories`) + label. Verified live
  (Biologie→AN 94700, `rvk_validate` WI 1000). Tests: `test_lookup_plugins.py` (6).
- **plugin-tool vs core-tool boundary:** external-API interactions become plugins
  (search_provider / input_source / **lookup**), cached + per-plugin-toggleable;
  local DB/pipeline/export + composed tools (`rvk_lookup`, `resolve_doi`) stay core.

### B2 ✅ DONE — one guarded URL-fetch primitive
Extracted `url_fetch.fetch_guarded_response()` — the single SSRF-guarded fetch entry
point (net_guard + guard-settings resolution). Both `url_fetch.scrape_url` (main-
content) and the MCP `scrape_url` tool (full-page + PDF detection) now call it; the
two divergent content-shapings stay (intentional). Non-breaking; verified live
(Wikipedia: url_fetch 139k chars main-content; MCP tool title + truncation).

### B1 ✅ DONE — k10plus as a lookup plugin
`k10plus_resolver` is a *Paketsigel harvester* (Siegel → many records) — a one→many
API query that `input_source` (one→one text `extract`) cannot model, so it went into
the **`lookup`** category (same shape as RVK) instead of `input_source` (operator
decision July 8). `src/utils/lookups/k10plus.py` `K10PlusLookup` → tool
**`k10plus_package`** (siegel → records, capped by `max_records`, raw-cached). The
direct batch usages (`pipeline_cmd.fetch_dois_for_siegel`, batch dialog) stay.
Tested mocked (a live harvest fetches *all* records → large/slow; live verification
needs an operator Siegel). Tests: `test_lookup_plugins.py` (8 total).

## Phase B complete (July 8)
lookup category + RVK · URL-fetch core · k10plus lookup — all committed, additive
(no agent-facing tool renamed). Suite 1167 passed.

### What broke
- Nothing agent-facing: all new tools (`rvk_search`/`rvk_validate`/`k10plus_package`)
  are additive; `rvk_lookup`/`resolve_doi`/`scrape_url` kept their names + behavior.
