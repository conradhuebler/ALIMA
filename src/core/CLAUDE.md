# Core - Central Business Logic and Data Management

## [Preserved Section - Permanent Documentation]

### Core Architecture
The `src/core/` directory contains the fundamental business logic and data management components of ALIMA:

**Primary Components:**
- `AlimaManager`: Central orchestration service coordinating LLM analysis workflows
- `🆕 PipelineManager`: Complete pipeline orchestration extending AlimaManager functionality
- `SearchEngine`: Multi-source search coordination with signal-based communication
- `CacheManager`: SQLite-based caching system with real-time statistics
- `DataModels`: Core data structures (AbstractData, AnalysisResult, TaskState, KeywordAnalysisState)
- `ProcessingUtils`: Text processing and keyword extraction utilities

**Suggester System:**
- Located in `suggesters/` subdirectory
- Implements plugin-like architecture for different search providers
- Includes Lobid, SWB, local catalog, and meta-suggester implementations

**Key Design Patterns:**
- Signal/slot architecture for asynchronous communication
- Service layer pattern for business logic separation
- Cache-aside pattern for performance optimization
- Plugin pattern for extensible suggester system

### Technical Specifications
- **Threading**: Extensive use of QThread for non-blocking operations
- **Database**: SQLite with prepared statements and connection pooling
- **Network**: QNetworkAccessManager for HTTP/HTTPS requests
- **Logging**: Structured logging with configurable levels
- **Error Handling**: Comprehensive exception management with graceful degradation

### Integration Points
- **LLM Services**: Interfaces with `src/llm/` for AI-powered analysis
- **UI Components**: Provides data and services to `src/ui/` layer
- **Configuration**: Uses `src/utils/config.py` for settings management
- **External APIs**: Integrates with Lobid, SWB, Crossref, and local catalogs

## [Variable Section - Short-term Information]

### Recent Improvements (Claude Generated)
1. **Enhanced SearchEngine Signal System**: Fixed signal emission for proper GUI integration
2. **Thread Safety Improvements**: Enhanced SearchWorker implementation for stable operations
3. **Cache Connection Management**: Improved SQLite connection handling and error recovery
4. **Processing Utils Optimization**: Enhanced keyword extraction and matching algorithms
5. **🚀 MAJOR: Pipeline Manager Implementation**: Complete pipeline orchestration system
6. **Cache Statistics Enhancement**: Added `get_cache_stats()` method for real-time monitoring

### Current Issues
- **ADD — Agentic GND "Häufigkeit" zeigt 1 (display/sort entkoppeln)**: Mapping-Cache-Treffer bekommen in `meta_suggester._add_cached_results_to_combined` `count=1`; klassisch zeigt `flatten_gnd_hits` das Max über alle Suchbegriffe → agentisch wirkt inkonsistent (meist 1).
  - **Falle**: der Pool-`count` steuert auch die Reihenfolge (`gnd_batch_search` Sort `(source_count, count)` + Chunking `sort_by: count`). Ändert man die Pool-Counts (z. B. Max-Merge in `_parse_batch_response*`), verschiebt sich `selection_chunks`→`selection` und Chunk/Final fallen zusammen (chunk = final). Naiver Max-Merge-Fix wurde aus genau diesem Grund zurückgenommen (Juni 2026).
  - **Richtung**: Count NUR anzeigeseitig korrigieren (`flatten_gnd_hits`/GUI), ohne die `gnd_entries` zu verändern, die Selektion/Sortierung speisen. Verifikation: agentischer Lauf/State, prüfen dass Chunk ≠ Final bleibt.

### Development Notes
- All new functions marked as "Claude Generated" for traceability
- Comprehensive error handling implemented across all components
- Type hints maintained throughout the codebase

### WIP: Iterative GND Search
- **Missing Concept Extraction**: Parse `<missing_list>` from LLM responses (prompt already supports this!)
- **Fallback Search**: GND search for missing concepts with hierarchy support
- **Iteration Control**: Max iterations + self-consistency convergence detection
- **UI Integration**: Manual trigger button in pipeline config, iteration history display in review tab
- **Implementation**: `extract_missing_concepts_from_response()` in processing_utils.py, `execute_fallback_gnd_search()` + `execute_iterative_keyword_refinement()` in pipeline_utils.py
- **Documentation**: See `docs/iterative_gnd_search.md` for complete architecture and implementation plan

### ✅ v4 Workflow System (Agentic)
Replaces the former MetaAgent + 4 SubAgents dispatch (removed April 2026).
- **Active path**: `PipelineManager._start_agentic_pipeline()` → `_start_v4_workflow_pipeline()` → `WorkflowExecutor`
- **Core classes** (`src/core/agents/`):
  - `WorkflowLoader` parses YAML into `WorkflowDef`/`StepConfig`
  - `WorkflowExecutor` runs steps sequentially against a `SharedContext`
  - `LLMAgentStep` + `DeterministicStep` (registered via `@register_step` in `registry.py`)
  - `deterministic_functions.py`: `gnd_batch_search`, `dk_classification_twophase`, `catalog_multi_search`, `catalog_title_search`, `gnd_entry_lookup`, `extract_gnd_related`, `gnd_batch_metadata`
- **Workflows** (`workflows/`): `alima_classic`, `catalog_search`, `synonym_expansion`, `batch_metadata` (all v4)
- **Tool caching**: `CachingToolRegistry` (in `sub_agents/` dir, kept) deduplicates tool calls
- **MCP Tool Layer**: `src/mcp/` — 16 tools unchanged
- **Agent Loop**: `src/core/agent_loop.py` — provider-agnostic tool-calling (used by LLMAgentStep)
- **Pipeline Integration**: `PipelineConfig.enable_agentic_mode` + `workflow_name`; CLI `alima workflow <name>` / `pipeline --agentic`; GUI workflow dropdown in PipelineConfigDialog
- **Single-step**: `SharedContext.save_to_file/load_from_file` for warm-start + `WorkflowExecutor.run(..., only_step=<id>)`
- **Tests**: `tests/test_agents_v2.py` (59 tests) + `tests/test_agents.py` (SharedContext/ToolCache only)
- **Legacy**: `workflows/legacy/` holds archived v3 YAMLs (meta_agent_default, default_alima, extended, minimal), not discovered at runtime
- **WARNING**: agentic mode still ~3x token usage vs rigid pipeline — opt-in only

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **WIP - Pipeline Enhancements**: Batch processing, templates, configuration UI
2. **ADD - Pipeline Webhooks**: External system notifications on step completion
4. **Performance Optimization**: Implement connection pooling for database operations
5. **Result Pagination**: Add support for large dataset handling
6. **Memory Optimization**: Optimize memory usage for large text processing

### Recently ADDED Features
1. **✅ PipelineManager (`pipeline_manager.py`)**: 
   - Orchestrates complete ALIMA workflow using existing AlimaManager logic
   - 5-step pipeline: Input → Keywords → Search → Verification → Classification
   - Uses proven `KeywordAnalysisState` for data management
   - UI callback system for real-time progress updates
   - Auto-advance functionality for seamless workflow
   - **🔄 REFACTORED**: Now uses shared `PipelineStepExecutor` from utils

2. **✅ Enhanced CacheManager**:
   - `get_cache_stats()` method for real-time cache monitoring
   - Statistics include entry count, database size, file path
   - Integration with global status bar for live updates

3. **✅ Shared Pipeline Logic Integration**:
   - PipelineManager refactored to use `PipelineStepExecutor` from utils
   - Eliminates code duplication with CLI implementation
   - Added JSON save/resume functionality via `PipelineJsonManager`
   - `resume_pipeline_from_state()` method for continuing interrupted workflows

### ✅ PRODUCTION STATUS - Pipeline Architecture Complete

**Technical Implementation:**
- **PipelineManager**: Now uses shared `PipelineStepExecutor` eliminating ~150 lines of duplication
- **Parameter Handling**: Whitelist filtering ensures only valid AlimaManager parameters are passed
- **Stream Callback Adapter**: Converts GUI callbacks (token, step_id) to AlimaManager format (token)
- **JSON Persistence**: Complete save/resume functionality for interrupted workflows

**Verified Functionality:**
- ✅ All 5 pipeline steps executing correctly (input → initialisation → search → keywords → classification)
- ✅ Real-time streaming feedback working in GUI
- ✅ Parameter conflicts resolved (provider, model, temperature, step_id, enabled)
- ✅ Final keywords displaying correctly in pipeline tab
- ✅ CLI and GUI producing identical results using shared logic

### Vision
- Establish core as the stable foundation for ALIMA's extensibility
- Maintain clean separation of concerns between data, business logic, and presentation
- Ensure scalability for handling large-scale library metadata processing
- Provide robust error handling and recovery mechanisms for production use