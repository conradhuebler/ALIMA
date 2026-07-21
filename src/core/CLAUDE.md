# Core - Central Business Logic and Data Management

## [Preserved Section - Permanent Documentation]

### Core Architecture
The `src/core/` directory contains the fundamental business logic and data management components of ALIMA:

**Primary Components:**
- `AlimaManager`: Central orchestration service coordinating LLM analysis workflows
- `PipelineManager`: Classic 5-step pipeline orchestration (composes `AlimaManager` + `PipelineStepExecutor`)
- `UnifiedKnowledgeManager`: Singleton GND/classification DB + mapping-first cache (thread-safe; per-thread connections — see root `MEMORY.md`)
- `DataModels`: Core data structures (AbstractData, AnalysisResult, TaskState, KeywordAnalysisState)
- `ProcessingUtils`: Text processing and keyword extraction utilities

**Plugin System (`src/core/plugins/`, Qt-free):**
- Category-agnostic framework: `ConfigField` schema, `PluginCategory` adapter registry, `plugin.toml` manifest + directory loader (multi-file package loading), AST security scanner + hash-pinning (all files, symlinks rejected). Concrete categories: search providers + input sources. Spec: [`docs/plugin_system.md`](../../docs/plugin_system.md), Authoring: [`docs/plugin_authoring.md`](../../docs/plugin_authoring.md).
- Search: `@register_provider` + `config_fields` per provider; `search/factory.py` `build_provider` is the single config→provider site (kills 3× hand-wiring). `sru` is a first-class provider type.
- ✅ **DK/RVK source is capability-driven (July 7)** — classic `execute_dk_search` picks its extractor via `factory.resolve_dk_extractor` from the enabled `CLASSIFICATION`-capable providers (finc opt-in → custom plugin → SRU/Libero), each exposing `dk_extractor()` (shared `extract_dk_classifications_for_keywords` contract). Any catalog plugin declaring `CLASSIFICATION` becomes a DK source with no core edit (completes the D-4 hand-wired site). Tests: `tests/test_dk_extractor_resolver.py`.
- ✅ **Self-contained blueprint dirs (July 6)** — each built-in provider is a copyable plugin dir `search/providers/<name>/` (plugin.toml + README + provider [+ suggester]); `SuggesterBackedProvider` is public API (`search/provider_base.py`); secret settings env-overridable (`ALIMA_PLUGIN_<ID>_<KEY>`); URL/SSRF guards in `src/utils/net_guard.py`. E2E: `tests/test_plugin_blueprint_e2e.py`.

**GND-keyword search:**
- Unified entry point `src/core/search/service.py` (`search_gnd_keywords` / `resolve_gnd_instances`): builds providers via `factory.build_provider` from `AlimaConfig.plugins` instances, merges, preserves the WP2 raw seam. Classic (`SearchCLI`), MCP (`ToolRegistry`), GUI (`find_keywords`) all route through it. **`MetaSuggester` retired July 8.**
- `BaseSuggester` (`src/core/search/base_suggester.py`, moved from `src/utils/suggesters/` July 19) is the per-source contract, wrapped by the capability-based providers; per-source suggesters live in their plugin dirs (`search/providers/<name>/suggester.py`).

**Key Design Patterns:**
- Signal/slot architecture for asynchronous communication
- Service layer pattern for business logic separation
- Cache-aside pattern for performance optimization
- Registry pattern for extensible steps/tools (`@register_step` / `@register_tool_fn`)

### Technical Specifications
- **Threading**: Extensive use of QThread for non-blocking operations
- **Database**: SQLite with prepared statements and per-thread connections
- **Logging**: Structured logging with configurable levels
- **Error Handling**: Comprehensive exception management with graceful degradation

### Integration Points
- **LLM Services**: Interfaces with `src/llm/` for AI-powered analysis
- **UI Components**: Provides data and services to `src/ui/` layer
- **Configuration**: Uses `src/utils/config_manager.py` for settings management
- **External APIs**: Lobid, SWB, Crossref/OpenAlex/DataCite (DOI input-source plugins + `doi_resolver.py`), local/finc catalogs

## [Variable Section - Short-term Information]

### Current Issues
- ✅ **Raw-First Response Cache (WP2, July 2)**: `search_response_cache` speichert die
  Quell-Antwort verbatim; `aggregate_gnd_results` (`src/core/search/aggregate.py`)
  leitet Pool + Counter (`display_count`) + Provenienz (`sources`/`source_count`) aus
  raw ab (**raw-first mit Mapping-Fallback**). Beide Pipelines nutzen es als Read-Pfad
  (`gnd_batch_search`, `SearchCLI.search_from_raw`), rollback via `aggregate_from_raw`.
  Count-Landmine bleibt (Pool-`count=1`). ⚠️ Klassisch default-on, aber GUI-Verifikation
  offen. Spec: [`docs/wp_raw_response_cache.md`](../../docs/wp_raw_response_cache.md).
- ✅ **F-4 GND „Häufigkeit zeigt 1" (gelöst June 29)**: Mapping-Cache speichert jetzt
  Per-GND-ID-Counts (`gnd_counts`); Cache-Treffer behalten Pool-`count=1`
  (Ranking/Chunking unverändert — Count-Landmine) und tragen ein separates
  `display_count` (echte Häufigkeit), das `flatten_gnd_hits`/GUI/Agentik anzeigen.
  Caching liegt im `CachingProvider` (`src/core/search/`), nicht mehr in
  `meta_suggester`. Offen: finaler agentischer Lauf zur Bestätigung (Chunk ≠ Final).

### WIP: Iterative GND Search
- Missing-concept feedback loop: `<missing_list>` → `extract_missing_concepts_from_response()` (processing_utils) + `execute_fallback_gnd_search()` / `execute_iterative_keyword_refinement()` (pipeline_utils). Details: `docs/iterative_gnd_search.md`.

### ✅ v4 Workflow System (Agentic)
Replaces the former MetaAgent + 4 SubAgents dispatch (removed April 2026).
- **Active path**: `PipelineManager._start_agentic_pipeline()` → `_start_v4_workflow_pipeline()` → `WorkflowExecutor`
- **Core classes** (`src/core/agents/`):
  - `WorkflowLoader` parses YAML into `WorkflowDef`/`StepConfig`
  - `WorkflowExecutor` runs steps sequentially against a `SharedContext`
  - `LLMAgentStep` + `DeterministicStep` (registered via `@register_step` in `registry.py`)
  - `deterministic_functions.py`: `gnd_batch_search`, `dk_classification_twophase`, `catalog_multi_search`, `catalog_title_search`, `gnd_entry_lookup`, `extract_gnd_related`, `gnd_batch_metadata`
- **Shared GND-search core** (`src/core/gnd_search_core.py`, classic↔agentic): `merge_code_entry` (also backs classic `SearchCLI.merge_results`; `classifications_field` merges per system), `merge_into_pool`/`parse_batch_response*`/`pool_entry_from_reduced`/`rank_pool`. ⚠️ pool `count` drives `selection_chunks`→`selection` — only `max`, never sum. Equal-chunk splitting shared via `src/utils/chunking.py`.
- ✅ **Canonical GND-pool vocabulary (WP-D1 P0, July 19; verified July 20; entries July 21)**: one shape end-to-end — `{count, gnd_ids, classifications: {system: [{code, count?, origin}]}, display_count?}` (suggester contract v2 → nested → pool → persisted KAS). System keys are **UPPERCASE** and equal-rank (`DK`/`DDC`/`RVK`/`BK`); `origin` separates an authority statement from statistical co-occurrence, entries are sorted authority-first then by descending evidence, and `count` merges by **max, never sum**. `src/utils/classification_systems.py` is the single owner (build/merge/normalise/read helpers). Classic↔agentic convergence is gated by `tests/test_e2e_smoke.py::TestClassicAgenticPoolConvergence`. Spec: [`docs/wp_records_as_first_class.md`](../../docs/wp_records_as_first_class.md) "Pinned decisions" + "P0 revision".
- ✅ **`BibRecord` (`src/core/bib_record.py`, WP-D1, July 20)** — one bibliographic record shape; `to_bibrecord(record, source)` normalizes finc/catalog/sru/k10plus. Not yet wired into the `ResultItem` seams: P1 (Record→analysis input) is the first consumer.
- **Workflows** (`workflows/`): `alima_classic`, `catalog_search`, `synonym_expansion`, `batch_metadata` (all v4). Legacy v3 YAMLs removed June 2026 (git history; see `docs/legacy/agentic_workflow_v3.md`).
- **Tool caching**: `CachingToolRegistry` (in `sub_agents/`) deduplicates tool calls
- **Agent Loop**: `src/core/agent_loop.py` — provider-agnostic tool-calling (used by LLMAgentStep)
- **Pipeline Integration**: `PipelineConfig.enable_agentic_mode` + `workflow_name`; CLI `alima workflow <name>` / `pipeline --agentic`; GUI workflow dropdown
- **Tests**: `tests/test_agents_v2.py` + `tests/test_agents.py` (SharedContext/ToolCache)
- **WARNING**: agentic mode still ~3x token usage vs rigid pipeline — opt-in only

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **Pipeline Enhancements**: Batch processing, templates, configuration UI
2. **Pipeline Webhooks**: External system notifications on step completion
3. **Performance**: Connection pooling, result pagination, memory optimization for large text

### Vision
- Establish core as the stable foundation for ALIMA's extensibility
- Maintain clean separation of concerns between data, business logic, and presentation
- Ensure scalability for large-scale library metadata processing
- Provide robust error handling and recovery
