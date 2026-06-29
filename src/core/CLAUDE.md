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

**Suggester System:**
- Located in `suggesters/` subdirectory
- Plugin-like architecture for different search providers (Lobid, SWB, local catalog, finc) + meta-suggester orchestrator
- Perspective: capability-based plugin standard — see `docs/search_provider_plugins.md`

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
- **External APIs**: Lobid, SWB, Crossref (`crossref_worker.py`, used by the DOI resolver), local/finc catalogs

## [Variable Section - Short-term Information]

### Current Issues
- **ADD — Agentic GND "Häufigkeit" zeigt 1 (display/sort entkoppeln)**: Mapping-Cache-Treffer bekommen in `meta_suggester._add_cached_results_to_combined` `count=1`; klassisch zeigt `flatten_gnd_hits` das Max über alle Suchbegriffe → agentisch wirkt inkonsistent (meist 1).
  - **Falle**: der Pool-`count` steuert auch die Reihenfolge (`gnd_batch_search` Sort `(source_count, count)` + Chunking `sort_by: count`). Ändert man die Pool-Counts (z. B. Max-Merge in `parse_batch_response*`), verschiebt sich `selection_chunks`→`selection` und Chunk/Final fallen zusammen (chunk = final). Naiver Max-Merge-Fix wurde aus genau diesem Grund zurückgenommen (Juni 2026).
  - **Richtung**: Count NUR anzeigeseitig korrigieren (`flatten_gnd_hits`/GUI), ohne die `gnd_entries` zu verändern, die Selektion/Sortierung speisen. Verifikation: agentischer Lauf/State, prüfen dass Chunk ≠ Final bleibt.

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
- **Shared GND-search core** (`src/core/gnd_search_core.py`, classic↔agentic): `merge_code_entry` (also backs classic `SearchCLI.merge_results`), `merge_into_pool`/`parse_batch_response*`/`rank_pool`. ⚠️ pool `count` drives `selection_chunks`→`selection` — only `max`, never sum. Equal-chunk splitting shared via `src/utils/chunking.py`.
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
