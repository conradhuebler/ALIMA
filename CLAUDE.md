# CLAUDE: Your AI Assistant for the ALIMA Project

## Overview

**ALIMA** (Automatic Library Indexing and Metadata Analysis) — pipeline for library science workflows combining LLM-powered text analysis with GND/SWB keyword search and DK/RVK classification.

## Core
1. Don't assume. Don't hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

### Conservative Self-Assessment Rules for AI
When documenting implemented features, the AI must apply these rules:

1. **Automated tests pass ≠ correct** — tests only cover what was anticipated. Unknown failure modes exist.
2. **Agreement with reference ≠ general correctness** — the reference comparison is only as broad as the test set.
3. **No gaps visible ≠ no gaps exist** — absence of a known bug is not the same as correctness. Especially for AI-generated scientific code: the most dangerous bugs are those that produce plausible but wrong results.
4. **"Implemented" means the code compiles and runs** — it does not imply correctness, stability across all inputs, or completeness relative to the reference method.
5. **When in doubt, add a caveat** — a caveat that turns out to be unnecessary is harmless. A missing caveat on wrong code causes user errors.

## Very General Instructions for AI Coding
- Avoid flattery, compliments, or positive language. Be clear and concise. Do not use agreeable language to deceive.
- Do comprehensive verification before claiming completion.
- Show me proof of completion, don't just assert it.
- Prioritize thoroughness over speed.
- If I correct you, adapt your method for the rest of the task.
- No completion claims until you can demonstrate zero remaining instances.

## CLAUDE.md Hygiene
- Each source code dir has a CLAUDE.md with basic information and logic.
- **Keep CLAUDE.md files FOCUSED and CONCISE** — ONE clear idea per bullet, max 1-2 lines.
  - ❌ DON'T: Multi-paragraph explanations, code examples, historical details, completed features.
  - ✅ DO: Brief statements, links to detailed docs.
  - ✅ DO: `✅ **Feature name** — Brief description` for completed items.
- Remove completed/resolved items after 2-3 updates (move to git history / `AIChangelog.md`).
- Tasks corresponding to code go in the right CLAUDE.md.
- Each CLAUDE.md has a variable part (short-term info, bugs) and preserved part (permanent knowledge).
- **Instructions blocks** contain operator-defined future tasks and visions.
- Only include information important for ALL subdirectories in main CLAUDE.md.
- **Rule of thumb**: section >20 lines → place elsewhere.

## Implementation Standards
- Mark new functions as `Claude Generated` for traceability.
- Remove TODO hashtags after approved.
- Implement comprehensive error handling and logging.
- Maintain backward compatibility where possible.
- **Always check instructions blocks** in relevant CLAUDE.md files before implementing.
- Reformulate task/vision entries if not yet CLAUDE-formatted.
- Avoid hardcoded provider lists — read from `llmanager`.

## Workflow States
- **ADD**: to be added • **WIP**: in progress • **ADDED**: implemented • **TESTED**: works (operator confirmed) • **APPROVED**: → changelog, remove from CLAUDE.md.

## Documentation Update Rules
- Replace debugging details with architecture decisions when issues are resolved.
- Document the *why* behind decisions, not the *what*.
- Eliminate redundant info that doesn't add architectural value.
- Significant improvements → [`AIChangelog.md`](AIChangelog.md).

## Git Best Practices
- **Only commit source files**: `git add <file>`, never `git add -A` without review.
- **Review before committing**: `git diff` + `git status`.
- **Commit message**: action verb (Fix/Add/Improve/Refactor) + brief description.
- Include Claude Co-Author line.
- Test artifacts stay local (`.gitignore`).

## Quality Assurance — Test Maintenance Rules

**The test suite MUST stay green. Broken tests silently ignored are worse than no tests.**

### Refactoring or renaming APIs
- Update ALL affected tests in the same commit.
- If a class/method is renamed, grep for all test references and update them.
- If a table schema changes, update `test_search.py` to match.

### Test isolation
- Tests using `UnifiedKnowledgeManager` MUST call `UnifiedKnowledgeManager.reset()` in both `setUp` and `tearDown`.
- Tests MUST use `DatabaseConfig(db_type='sqlite', sqlite_path=<tempfile>)` — never the production config (may point to MariaDB).
- Tests using Qt classes (SearchEngine, LlmService) require a `QApplication` — use `Mock(spec=...)` to avoid it.

### Submitting/reviewing PRs
- Run `python -m pytest tests/ -v` locally before opening a PR.
- A PR introducing new test failures is not ready to merge.
- Pre-existing failures: fix in a separate commit and note explicitly.

### Incident: 9-month silent test debt
- `SearchEngine` rewritten async → Qt-signal-based in July 2025.
- `test_cache.py` / `test_search.py` not updated → silently broken for 9 months.
- Discovered during PR #9 review (March 2026). Apply the rules above to prevent recurrence.

## Critical Requirements

**All pipeline changes MUST be usable by both CLI and GUI.**
- Shared logic: `src/utils/pipeline_utils.py`.
- Configuration parity: identical params across interfaces.
- Use `PipelineConfigParser` + `PipelineConfigBuilder` (single source of truth).

## [Preserved Section — Permanent Documentation]
*Change only if explicitly wanted by operator.*

### Pipeline Modes
- **Classic (rigid)**: 5-step linear pipeline (input → initialisation → search → keywords → classification). Details: [`docs/pipeline_classic.md`](docs/pipeline_classic.md).
- **Agentic (v4)**: YAML-driven workflows via `WorkflowExecutor`. Default workflow: `alima_classic.yaml`. Details: [`docs/agentic_workflow.md`](docs/agentic_workflow.md), schema: [`docs/workflow_yaml_spec.md`](docs/workflow_yaml_spec.md).
- Both modes share `PipelineManager` state and `PipelineStepExecutor`.

### Database
- `alima_knowledge.db` — facts (`gnd_entries`, `classifications`) + mappings (`search_mappings`).
- `UnifiedKnowledgeManager` — singleton, mapping-first search. Thread-safety details in `MEMORY.md`.

## [Variable Section — Current Tasks]
- **WP Lookup-Plugin-Integration (Pipeline+Agent) — Phase D** ✅ CODE-COMPLETE
  (July 10): rvk_api/k10plus/dnb liefen bisher nur im Chat-Agent; jetzt *ein*
  Aufrufpfad je Quelle. Geteilter `build_lookup(config,id)`
  (`src/utils/lookups/resolve.py`, spiegelt `ToolRegistry._lookup_instances`);
  Workflow-Preset `lookup` + rvk in `classification` (`default_presets.yaml`);
  klassische Pipeline-RVK (`pipeline_utils.py`) + CLI/GUI-k10plus-Batch + DNB-GUI
  routen übers Plugin (k10plus: neuer uncapped `fetch_records`-Kern, `fetch_package`
  = Cap-Wrapper). Tests `test_lookup_plugins.py` +10, Suite 1241 grün.
  Verhaltensänderung: RVK-Validierungs-Timeout 4s→Plugin-Default, CLI-Siegel nutzt
  Plugin-`cache_dir`. Doc: `docs/wp_search_tool_plugin_potential.md` Phase D +
  `AIChangelog.md` (July 10). **Offen:** Operator-Commit + GUI-Sign-off (k10plus-
  Batch-Dialog, DNB-Sync-Click-Test).
- **WP Website-RAG-Chatbot (`webindex`-Lookup-Plugin)** ✅ CODE-COMPLETE
  (July 9): ALIMA als Chatbot für Webseiteninhalte. Eigenes Plugin hält eine DB
  über alle URLs einer konfigurierten Haupt-URL + eine **zentral synchronisierte
  Keyword-Tabelle**; Retrieval = Frage → Keyword-Match gegen `page_keywords`
  → gerankte Trefferseiten (Cache oder Live-Fetch) → Text → Antwort.
  Lookup-Plugin `src/utils/lookups/webindex/` (`store.py` eigene `webindex.db`
  nach `LocalGndStore`-Muster; `indexer.py` BeautifulSoup-Crawler mit injiziertem
  `keyword_extractor`; `provider.py` Tools `search_webindex`/`fetch_page`/
  `list_webindex_keywords`; `keywords.py` Model-Auflösung + Extractor-Glue).
  Indizieren: GUI-Button „Seite indizieren …" (`_TYPE_ACTIONS`-Registry +
  `WebIndexCrawlWorker` in `src/ui/webindex_crawl.py`) ODER CLI
  `alima webindex crawl/stats/list-keywords/search`. Keyword-Standprompt lebt als
  Workflow `workflows/webindex_keywords.yaml` (tool-less `llm_agent`, **nicht**
  prompts.json — veraltet); Antwort-Prompt `workflows/website_rag.yaml`.
  Model: CLI-Flags → Instanz `llm_provider`/`llm_model` → globaler agentic Default.
  Tests netzfrei (`test_webindex_{store,indexer,lookup,keywords,crawl_ui}.py`, 51),
  Suite 1231 grün. **Offen:** Operator-E2E gegen echte Biblio-URL + Sign-off.
  Sub-CLAUDE:
  [`src/utils/lookups/webindex/CLAUDE.md`](src/utils/lookups/webindex/CLAUDE.md).
- **WP GND-Suche vereinheitlicht / MetaSuggester retired** ✅ CODE-COMPLETE (July 8):
  ein Single-Entry `src/core/search/service.py` (`search_gnd_keywords` /
  `resolve_gnd_instances`) baut alle Provider über `factory.build_provider` aus
  `PluginInstanceConfig`; Klassik (`SearchCLI`), MCP (`ToolRegistry._provider_for` +
  `_source_transform`) und GUI (`find_keywords`) konvergiert; `meta_suggester.py`
  gelöscht (`grep "MetaSuggester("`→0). Live gegen lobid verifiziert (Klassik
  live/merge + raw-first, MCP-Tools, agentisches `aggregate_gnd_results`); Suite
  1152 grün. Defaults: lobid+swb zero-config, catalog/finc `is_available()`-gated
  Blueprints, gnd_local offline. **Residual:** finc-MCP-Handler bleibt
  `CatalogConfig`-basiert. **Offen:** Operator-Click-Test `find_keywords` (GUI nicht
  headless verifizierbar). Doc: [`AIChangelog.md`](AIChangelog.md) (July 8).
- **WP Plugin-Blueprints + Security-Härtung** ✅ CODE-COMPLETE (July 6): alle 6
  Built-in-Provider sind self-contained, kopierbare Plugin-Dirs
  (`src/core/search/providers/<name>/` mit plugin.toml + README); Loader lädt
  Multi-File-Plugins; Härtung: Symlink-Verbot, Hash über alle Dateien (⚠️ einmalige
  Re-Approval bestehender Code-Plugins), entry-Validierung, `net_guard`
  (SSRF/Timeouts), Secrets-Env-Override (`ALIMA_PLUGIN_<ID>_<KEY>`). E2E:
  `test_plugin_blueprint_e2e.py`. Offen: Operator-GUI-Click-Test (Plugin-Tab:
  Secret-Placeholder, URL-Warndialog, Approval-Dialog). Guide:
  [`docs/plugin_authoring.md`](docs/plugin_authoring.md).
- **WP Own-Plugins-Only POC** ✅ CODE-COMPLETE (July 6): alle 6 Built-ins als
  kopierbare `poc_*`-Plugins nachbaubar + Built-ins abschaltbar → App läuft nur
  auf eigenen Plugins. Deployer/Generator `examples/plugins_poc/deploy_poc.py`
  (nur `id` umbenannt, Tool-Namen/`source_label` bleiben), klassischer
  Leer-Schnittmengen-Fallback in `execute_gnd_search`, E2E
  `test_all_external_plugins_poc.py` (agentisch+klassisch grün). Grenzen:
  Built-in-*Klassen* bleiben registriert (nur Instanzen/Tools aus), WP2-Raw-Cache
  + GUI-`find_keywords` weiter auf `lobid`/`swb` verdrahtet. Offen:
  Operator-GUI-Sign-off (`examples/plugins_poc/README.md` §Verifikation).
  Guide: [`docs/plugin_authoring.md`](docs/plugin_authoring.md) §10.
- **WP Institutional Bundles** ✅ (July 6): `alima bundle
  {build,install,export,list,remove}` + GUI-Gruppe im Plugin-Tab — Einrichtungen
  rollen Plugins + beratendes Config-Profil gebündelt aus; `export` erfasst die
  laufende Einstellung (Secrets gestrippt+deklariert, synthetische declarative
  Instanzen mit eindeutiger id, `--plugin`-Auswahl, Code-Plugin setzt
  `enable_code_plugins`). Qt-frei in `src/utils/bundle.py` (CLI:
  `cli/commands/bundle_cmd.py`, GUI: `ui/plugin_settings_tab.py`), Provenienz-Ledger
  `AlimaConfig.installed_bundles` für präzises remove, `profile.json`-Whitelist
  schützt `unified_config`/Secrets (fail-closed), Per-User-Secrets nur deklariert
  (env-Override). Beispiel `examples/bundles/demo_institution/`, Tests
  `test_bundle.py` (14). Advisory, kein Lock/Signing/Auto-Update (bewusst). Doc:
  [`docs/institutional_bundles.md`](docs/institutional_bundles.md).
- **WP2 Raw-First Response Cache** ✅ (July 2, P1–P5): source responses cached verbatim
  (`search_response_cache`); pool + counter (`display_count`) + provenance derived from
  raw via `aggregate_gnd_results` (raw-first + mapping fallback); both pipelines converged
  (rollback `aggregate_from_raw`); `search_lobid` `agent_view` (member/totalItems); input
  tools `cacheable`. ⚠️ **Klassisch default-on, GUI/Webapp-Verifikation + Vergleichslauf
  offen** (Operator). Spec: [`docs/wp_raw_response_cache.md`](docs/wp_raw_response_cache.md).
- **WP Tool-Data-Passthrough** 🚧: agenten-facing Tools sollen die *vollständigen*
  Quelldaten durchreichen, nicht den alten Pipeline-Ausschnitt. DOI-Tools/`resolve_doi`/
  `scrape_url`/`read_pdf` ✅ done; **`search_finc` ✅ audited — clean** (reicht `raw`
  komplett durch; 8-Feld-Deckel ist der `fincsolrproxy` server-seitig, keine
  ALIMA-Änderung; live verifiziert July 1). **`search_lobid` ✅ audited**: Subjekt-
  Aggregation (Primärdaten) wird komplett durchgereicht (Buckets nur `{key,doc_count}`);
  gedroppt werden nur die `member`-Resource-Records — aber der Mapping-First-Cache
  umgeht Live-lobid, daher nicht zuverlässig surfacebar → Operator-Entscheid offen
  (Empfehlung: Per-Subjekt-Details via `get_gnd_entry`, kein Overload). Offen:
  **swb/catalog** (+ catalog_titles) — dort echte Per-Record-Reduktion; Pool braucht
  `{count,gndid,ddc,dk}`, daher volles `record` *zusätzlich*. Spec: [`docs/wp_tool_data_passthrough.md`](docs/wp_tool_data_passthrough.md).
- **Cleanup-Findings-Register** (prioritisiert, projektweit): offene Debt-Findings aus dem Juni-2026-Sweep + empfohlene Reihenfolge. Headline F-3 Search-Provider-Plugins + F-4 „Häufigkeit zeigt 1" ✅ DONE (June 29); F-6 webapp `app.py`-Split 2537→240 ✅ DONE (June 30, sandbox-verifiziert); F-5 GUI-God-Files ✅ CODE-COMPLETE (June 30, alle 5 gesplittet, statisch verifiziert — nur Operator-Click-Test-Sign-off offen). Offen: F-5/F-7 nur noch Operator-Click-Test (Refactor war nicht GUI-gated, nur das Sign-off), F-8 opportunistisch (decide-on-touch). Spec: [`docs/cleanup_findings.md`](docs/cleanup_findings.md).
- **TESTED — Kern-Konvergenz klassisch↔agentisch (WP-K1–K4)**: mapping-first GND-Suche in MCP-Tools, `verify_keywords`-Step, geteilte DK-Vorfilterung, Klassifikations-Gating, source_count-Ranking. Details: `AIChangelog.md` (June 10, 2026). Operator-Vergleichslauf June 29, 2026: beide Pfade laufen durch, reasonable results (klassisch≠agentisch by design). → bereit für APPROVED (Operator-Entscheid: in Changelog verschieben + hier entfernen).

## [Instructions Block — Operator-Defined Tasks]

### Vision
- Restructure code: consolidate distributed logic from `utils`, `core`, `suggestors`.
- ✅ **Search-Provider-Plugin-System** (done June 29) — capability-based provider standard + single `@register_provider` registry (`src/core/search/`); DB/API sources (lobid/swb/catalog/finc/gnd_local) are uniform, config-selectable tools; `SuggesterType` retired. Details: [`docs/search_provider_plugins.md`](docs/search_provider_plugins.md).
- Maintain unified pipeline architecture (CLI/GUI/Webapp parity).
- Extend agentic v4 to cover more workflow types beyond classical pipeline.
- **Chat-Agent as first-class frontend** — chat-first + headless dual-mode: same `AgentLoop`/toolset drives GUI-Chat, CLI (`alima agent --doi …`), and HTTP endpoint. GUI keeps role for visual inspection / high-risk operator mutations. Details: [`docs/chat_agent_roadmap.md`](docs/chat_agent_roadmap.md).

### Future Tasks
1. **Code Restructuring**: Consolidate distributed logic.
2. **Pipeline Enhancement**: Templates, advanced configuration UI.
3. **Batch Enhancement**: Extended image analysis, URL scraping.
4. **Performance**: Connection pooling, result pagination, memory optimization.
5. **Agentic Hauptagent**: `main_agent:` block in YAML — meta-orchestrator that calls sub-workflows as tools.
6. **Chat-Agent Phases P-δ.4 → P-ι**: Mini-Polish + UnifiedMessageWidget (shared renderer for chat + pipeline-logger) + Mutation Tools + Pipeline-Orchestration + Input-Beschaffung (DOI/URL/PDF/Image) + Export + Headless/CLI/API. Permission-Layer via `ChatConfig.autonomous_pipeline` (default explicit, confirmation dialog for destructive ops). Roadmap: [`docs/chat_agent_roadmap.md`](docs/chat_agent_roadmap.md).
7. **Streaming-with-Tools Backend** (P-δ.5): ✅ Ollama, OpenAI, Anthropic stream text deltas when tools are active (`_generate_*_with_tools` in `llm_service.py`). Remaining: Gemini (`_generate_gemini_with_tools` completes-then-delivers). Renderer is now QWebEngineView-based (`src/ui/web_log_view.py` `WebLogView`) — native `<details>` collapse, live token append; see `AIChangelog.md` (June 9, 2026).
8. **WP12 — Unified Render Layer (GUI ↔ Webapp)**: shared CSS+JS render layer + JSON render-event protocol; WP12.1–.4 committed in `9552d93`. Remaining work tracked as **WP12.5** §9.2–.5 (visual verification, error-event rendering, 2 operator decisions on webapp double-display & agentic tier). Spec: [`docs/wp12_unified_render_layer.md`](docs/wp12_unified_render_layer.md).
9. **WP13 — Cleanup**: ✅ DONE (commit `93ccc19`) — dead modules + dead test + tracked backups + `workflows/legacy/` removed, sub-CLAUDE.md hygiene. Remaining: one-time local SWB cache purge (`swb_gnd_cache.json`, operator, no code). Spec: [`docs/wp13_cleanup.md`](docs/wp13_cleanup.md).
10. **Search-Provider-Plugins**: ✅ DONE (June 29, P1+P2+P3) — capability-based `SearchProvider` + `@register_provider` registry (`src/core/search/`) replacing `SuggesterType`/MetaSuggester if-elif; `CachingProvider` wrapper (F-4 `display_count`); MCP tools generated from `ProviderToolSpec`; finc in the standard; `SearchProviderConfig` + GUI selector. Spec: [`docs/search_provider_plugins.md`](docs/search_provider_plugins.md). Remaining: GUI click-test + agentic verification lauf.

## Module Documentation
- [`src/core/CLAUDE.md`](src/core/CLAUDE.md) — Core business logic, pipeline orchestration, data management.
- `src/core/agents/` — Agentic v4: `WorkflowExecutor`, `LLMAgentStep`, `DeterministicStep`, optional `MetaAgent` loop. (v3 SubAgents removed April 2026 — see [`docs/legacy/agentic_workflow_v3.md`](docs/legacy/agentic_workflow_v3.md).)
- [`src/mcp/CLAUDE.md`](src/mcp/CLAUDE.md) — MCP tool layer: schemas, registry, handlers.
- [`src/ui/CLAUDE.md`](src/ui/CLAUDE.md) — PyQt6 GUI components.
- [`src/utils/CLAUDE.md`](src/utils/CLAUDE.md) — Configuration, batch processing, logging.
- [`docs/`](docs/) — Architecture docs (agentic, classic pipeline, subsystems, legacy).
- [`AIChangelog.md`](AIChangelog.md) — Detailed dated developer log.
- [`CHANGELOG.md`](CHANGELOG.md) — User-facing release notes.
