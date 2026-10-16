# Core - Central Business Logic and Data Management

## [Preserved Section - Permanent Documentation]

### Core Architecture
The `src/core/` directory contains the fundamental business logic and data management components of ALIMA:

**Primary Components:**
- `AlimaManager`: Central orchestration service coordinating LLM analysis workflows
- `PipelineManager`: Classic 5-step pipeline orchestration (composes `AlimaManager` + `PipelineStepExecutor`). God-file split July 21 (F-15): die 10 klassischen Step-Executoren (`_execute_input/initialisation/search/keywords/dk_*_step` + Shape-Konverter) → `_pipeline_classic_steps.ClassicStepExecutorMixin`; der agentische Pfad → `_pipeline_agentic.AgenticPipelineMixin`; `execute_single_step` + `resume_pipeline_from_state` → `_pipeline_single_step.SingleStepExecutorMixin` (alle verbatim, via MRO). ⚠️ `PipelineStep`/`PipelineConfig`/`_AgenticStreamFilter` bleiben in `pipeline_manager.py`; die Mixins nutzen `from __future__ import annotations`, damit die `PipelineStep`-Typannotation keinen Import-Zyklus erzeugt. `execute_single_step` *instanziiert* `PipelineStep` zur Laufzeit (nicht nur Annotation) → dort method-lokaler `from .pipeline_manager import PipelineStep` (die eine nicht-verbatim Zeile des Splits; per realem Testlauf abgesichert).
- `UnifiedKnowledgeManager`: Singleton GND/classification DB + mapping-first cache (thread-safe; per-thread connections — see root `MEMORY.md`). God-file split (F-15): DDL/Migrationen → `_ukm_schema.SchemaMigrationMixin`, `catalog_dk_cache` → `_ukm_catalog_dk.CatalogDkCacheMixin`, mapping-first Smart-Search → `_ukm_smart_search.SmartSearchMixin` (verbatim Mixins, via MRO; 2077→1091 Z.). Singleton-Lebenszyklus + GND/Mapping-Query-Layer bleiben in `unified_knowledge_manager.py`.
- `DataModels`: Core data structures (AbstractData, AnalysisResult, TaskState, KeywordAnalysisState)
- `ProcessingUtils`: Text processing and keyword extraction utilities
- `user_rules`: personal indexing rules (`~/.config/alima/rules.yaml`) appended to agentic, planner and chat system prompts; Qt-free store + scoping + rendering. Spec: [`docs/user_rules.md`](../../docs/user_rules.md).

**Plugin System (`src/core/plugins/`, Qt-free):**
- Category-agnostic framework: `ConfigField` schema, `PluginCategory` adapter registry, `plugin.toml` manifest + directory loader (multi-file package loading), AST security scanner + hash-pinning (all files, symlinks rejected). Concrete categories: search providers + input sources. Spec: [`docs/plugin_system.md`](../../docs/plugin_system.md), Authoring: [`docs/plugin_authoring.md`](../../docs/plugin_authoring.md).
- Search: `@register_provider` + `config_fields` per provider; `search/factory.py` `build_provider` is the single config→provider site (kills 3× hand-wiring). `sru` is a first-class provider type.
- ✅ **DK/RVK source is capability-driven (July 7)** — classic `execute_notation_search` (renamed from `execute_dk_search`, WP-D2 sweep Aug 4) picks its extractor via `factory.resolve_dk_extractor` from the enabled `CLASSIFICATION`-capable providers (finc opt-in → custom plugin → SRU/Libero), each exposing `dk_extractor()` (shared `extract_dk_classifications_for_keywords` contract). Any catalog plugin declaring `CLASSIFICATION` becomes a DK source with no core edit (completes the D-4 hand-wired site). Tests: `tests/test_dk_extractor_resolver.py`.
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

### ✅ FIXED (Oct 2026): DatabaseConfig.charset/ssl_disabled hatten keinen Leser
`DatabaseManager.get_connection()`s MySQL/MariaDB-Zweige übergaben an
`setConnectOptions` nur `MYSQL_OPT_CONNECT_TIMEOUT` - beide Felder hatten ein
Bedienelement (GUI-Dialog + CLI `database set`), aber nichts darunter las sie;
eine Verbindung lief immer mit dem Server-Default für SSL und Zeichensatz.
Am tatsächlichen Treiber nachgeprüft (Strings in `libqsqlmysql.so`, nicht nur
Qt-Doku): `MYSQL_OPT_SSL_MODE=SSL_MODE_DISABLED` ist der vom Treiber erkannte
Connect-Options-Token für `ssl_disabled`, jetzt in `_mysql_connect_options()`.
Für `charset` gibt es **keinen** Connect-Options-Token — der Treiber setzt
intern hartkodiert `"utf8"` (3-Byte, nicht `"utf8mb4"`, der konfigurierte
Default). Fix: `SET NAMES '<charset>'` direkt nach dem Öffnen, hinter einer
Namensvalidierung (`_is_valid_charset_name`) — `SET NAMES` kennt keinen
Bind-Parameter. Tests: `tests/test_database_mysql_options.py` (reine
String-Bau-/Validierungslogik; der eigentliche `SET NAMES`-Aufruf folgt der
bestehenden Politik, den MySQL/MariaDB-Verbindungspfad nicht gegen einen
echten Server zu testen, wie der SQLite-Pragma-Block direkt daneben).

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

### ✅ FIXED (Oct 2026): Chat-Historie schnitt mitten in Tool-Austauschen
Beide Frontends (Qt6-Chat-Panel, Webapp) schnitten die letzten `CHAT_HISTORY_WINDOW` (5) Nachrichten per Index ab. Ab genug Tool-Aufrufen landet dieser Schnitt mitten in einem Austausch — auf einer verwaisten `tool`-Nachricht (ihr `assistant`-`tool_use` ist aus dem Fenster gefallen) oder auf einer `assistant`-Nachricht (Anthropic verlangt `user` als erste Nachricht). Beides führt zum selben 400er, den die Anthropic-Konvertierung in `llm_service.py` schon für falsch zusammengesetzte Nachrichten behoben hat — hier ist die Ursache eine andere: welche Nachrichten überhaupt ausgewählt werden. `chat_prompts.trim_history_to_turn_boundary` schiebt den Schnittpunkt rückwärts bis zur nächsten `user`-Nachricht statt ihn vorwärts zu kappen — mehr Kontext ist immer sicher, weniger kann wieder mitten im Austausch landen. Beide Aufrufstellen (`_chat_panel_chat_agent.py`, `webapp/routers/agent.py`) nutzen jetzt diese eine Funktion. Von einem Opus-Agenten gefunden, selbst am Code nachgerechnet vor der Behebung. Tests: `tests/test_chat_history_turn_boundary.py`.

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
- ✅ **`BibRecord` (`src/core/bib_record.py`, WP-D1, July 20)** — one bibliographic record shape; `to_bibrecord(record, source)` normalizes finc/catalog/sru/k10plus. **P1–P4 landed (Aug 3):** `to_analysis_text()` + `isbn`/`ppn` input sources (P1); record classifications as dk priors via KAS `input_record_classifications` + `record_sink` side channel (P2); `gnd_subjects` → `merge_record_gnd_subjects` pool injection (P3); `k10plus_resolve` identifier crosswalk, client-side verified (P4). Rest is surface adoption; agentic gets no record priors yet; the three `ResultItem` seams untouched.
- **Workflows** (`workflows/`): `alima_classic`, `catalog_search`, `synonym_expansion`, `batch_metadata` (all v4). Legacy v3 YAMLs removed June 2026 (git history; see `docs/legacy/agentic_workflow_v3.md`).
- **Tool caching**: `CachingToolRegistry` (in `sub_agents/`) deduplicates tool calls
- **Agent Loop**: `src/core/agent_loop.py` — provider-agnostic tool-calling (used by LLMAgentStep)
- **Step-Abbruch wirkt im Stream**: `PipelineManager.agentic_should_stop` → `WorkflowExecutor`/`MetaAgent` → `BaseStep.should_stop` → `AgentLoop` → Provider-Schleifen, die pro Chunk prüfen. Einmal-Semantik (`_step_abort_once`): nur der laufende Aufruf bricht ab, der nächste läuft normal. ⚠️ Ein neuer Step-Konstruktor-Aufruf ohne `should_stop=` nimmt den Abbruch wieder aus der Kette — er fällt dann still auf „wirkt erst nach der Generierung" zurück.
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
