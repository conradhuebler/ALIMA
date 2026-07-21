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
- **Ein breiter `except` muss sagen, welche Art Fehler er gefangen hat.** Er ist
  richtig für *erwartbare* Fehlschläge (Netz, Parsing, fehlende Datei) und falsch
  für Programmierfehler (`AttributeError`, `NameError`, `ImportError`,
  `TypeError`) — die verschwinden sonst als `warning` zwischen den erwartbaren.
  Vier Defekte im Juli 2026 überlebten monatelang genau dadurch. Neuer Code nutzt
  `src/utils/error_visibility.log_caught(logger, e, kontext)`: fängt weiterhin
  alles, loggt aber Defekt-Formen auf ERROR. Ändert nie den Kontrollfluss.
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
- **➡️ Offene WPs: [`docs/open_workpackages.md`](docs/open_workpackages.md)** (July 19,
  ausführungsreifes Register): Daten-Achse **D1 Rest** (Vergleichslauf, dann P1–P4 +
  `to_bibrecord()`) + **D2 Logik-Generalisierung**; Konsolidierungen **K1**
  BusRenderBridge, **K2** Lobid-Label aus gnd_local, **K3** DOI-Mirror-Abbau (P8),
  **K4** Tool-Passthrough swb/catalog, **K5** SearchTab-Refresh; T-Reihe
  anlassbezogen (Gemini+LlmService-Split, Session-Persistenz, Keyring,
  i18n-Ausbau); V1 Agentic Hauptagent.
- **WP-D1 P0 ✅ DONE+VERIFIZIERT, `to_bibrecord()`+F-2 ✅ DONE (July 19+20, 11
  Commits ab `6991d57`):** EIN GND-Pool-Vokabular `{count, gnd_ids,
  classifications{system}, display_count?}` end-to-end; System-Keys **GROSS**
  (`classification_systems` = alleiniger Owner); `BibRecord` für
  finc/catalog/sru/k10plus; DOI-Keys klein. Verifikation = deterministischer
  Headless-Test beider Pfade. Suite 1357. **Offen: Konsumenten P1–P4** — P1 hat
  eine `input_type`-Parity-Landmine (Register). Details: `AIChangelog.md` +
  [`docs/wp_records_as_first_class.md`](docs/wp_records_as_first_class.md).
- **WP-D2 Ernte ✅ DONE (July 20–21, 4 Commits `2f34e8c`…`8670d6c`, Suite
  1405):** `classifications` trug in der Praxis **nichts** (0 von 5128 Einträgen
  in echten Läufen) — lobid liefert die Notationen die ganze Zeit mit, auf den
  `member`-Records. Jetzt geerntet als **Ko-Vorkommens-Heuristik**; dafür P0
  revidiert: ein Eintrag ist `{code, count?, origin}`, `origin ∈ {authority,
  cooccurrence}`, Merge per max (nie Summe). Abdeckung 6 % (strukturelle
  lobid-Grenze: Pool aus dem Aggregation-Facet, Klassifikationen nur aus den
  ausgelieferten Records). Details: `AIChangelog.md` (July 20–21).
- **Nebenbefund July 20 (behoben):** Batch-Speichern crashte an nicht
  konvertierten Sets (`TypeError`, pro Item verschluckt → Läufe meldeten
  Fehlschläge statt Ergebnisse); `rvk` fehlte in `SET_FIELDS`. Beides
  vorbestehend, keine P0-Regression.
- **Aufräumen A+C ✅ DONE (July 21, `a3bb317`/`11f77c8`/`06c4417`, Suite 1525):**
  drei weitere Aufrufe **nicht existierender** Methoden behoben + halbe
  CacheManager-Fassade gelöscht (−216 Z.); 84 Charakterisierungstests für die
  drei ungetesteten Logik-Module (`_pipeline_rvk_scoring`/`_pipeline_dk_steps`/
  `batch_processor` — der Batch-Crash lag genau dort); 5 reine RVK-Helfer
  verbatim auf Modulebene, `_source_rank`/`_status_rank`-Duplikat vereinigt.
  Register: F-12/F-13/F-14 in [`docs/cleanup_findings.md`](docs/cleanup_findings.md).
- **Aufräumen B ⏳ begonnen (July 21):** Politik entschieden und in den
  Implementation Standards festgeschrieben — `error_visibility.log_caught`
  trennt Defekt-Formen von erwartbaren Fehlern, ohne den Kontrollfluss zu
  ändern. **Übernommen bisher 9 von 189** stillen Blöcken um reinen Code
  (agentischer Tool-Pfad + `alima_manager`); Rest opportunistisch bei Berührung,
  kein Sweep. Verteilung: 61 core, 52 ui, 42 utils, 14 webapp.
- **Testpolitik (July 19):** GUI/Browser-Klick-Tests macht der Operator **on the fly
  beim Benutzen** — kein Gate, Brüche werden gemeldet. WPs gelten mit grüner Suite +
  statischer Verifikation als DONE.
- **WP Chat-UX-Aufräumen ✅ DONE (July 19):** 9 Commits (`07537d1`…`6985c8b`) +
  Link-Fix `e4101fd` — eine Webapp-Anzeigefläche (`#stream-text` weg), Live-Markdown
  beim Streamen, geteiltes Fehler-Chrome (`kind="error"`), `--alima-*`-Theming
  (Light-Mode erreicht das Log), GUI-Status-Strip, leichte i18n (`locales/`,
  `UIConfig.ui_language`), Chat-Doppelrender-Guard; wp12 §9.3–§9.5 geschlossen.
  Suite 1323. Details: `AIChangelog.md` (July 19).
- **WP Struktur-Aufräumen ✅ DONE (July 19):** keine toten Module; `suggesters/`-Rest +
  finc-Shims weg, 20 Docs → `docs/legacy/`, Lobid-Download lazy + persistenter Pfad,
  `pipeline_utils.py` 5166→2041 (`DkStepsMixin`/`RvkScoringMixin`). Register-Zeilen
  F-10/F-11 in [`docs/cleanup_findings.md`](docs/cleanup_findings.md).
- **WP Plugin-Konvergenz P1–P7 ✅ DONE (July 19, `27a2901`):** **eine**
  Konfigurationswahrheit — `CatalogConfig`/`SearchProviderConfig` gelöscht,
  `AlimaConfig.plugins` allein; Lesen `factory.primary_settings`, Schreiben
  `set_primary_settings`; Legacy-JSON = einmalige Migrations-Eingabe (absente Keys
  weggelassen). DOI-`SystemConfig`-Mirror bleibt (→ WP-K3). Doc:
  [`docs/wp_plugin_convergence.md`](docs/wp_plugin_convergence.md).
- **Ältere ✅-WPs** (Lookup-Plugins Phase D `16ca53b`, webindex-RAG, GND-Suche/
  MetaSuggester-Retire, Plugin-Blueprints+Härtung, Own-Plugins-POC, Institutional
  Bundles, WP2 Raw-Cache, Counter-Bug `038738e`): Details in `AIChangelog.md` +
  jeweiligen Docs/Sub-CLAUDEs; verbliebene Code-Reste sind ins WP-Register überführt
  (Tool-Passthrough swb/catalog → K4; POC-Grenze: WP2-Raw-Cache noch auf lobid/swb
  verdrahtet). BibRecord-Analyse → **WP-D1** ([`docs/wp_records_as_first_class.md`](docs/wp_records_as_first_class.md)).
- **Cleanup-Findings-Register**: F-1…F-9 ✅ bzw. decide-on-touch (F-7/F-8), F-10/F-11
  offen (→ K2 / Politik). Spec: [`docs/cleanup_findings.md`](docs/cleanup_findings.md).

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
5. **Agentic Hauptagent**: `main_agent:` block in YAML — meta-orchestrator that calls sub-workflows as tools. Spec-Skizze: [`docs/open_workpackages.md`](docs/open_workpackages.md) §WP-V1.
6. **Streaming-with-Tools Backend** (P-δ.5): Remaining: Gemini (`_generate_gemini_with_tools` completes-then-delivers) — gekoppelt mit dem `LlmService`-Split (Register §WP-T1); Ollama/OpenAI/Anthropic ✅ stream with tools.

(Erledigt + abgeräumt July 19: Chat-Agent-Phasen P-δ.4→P-ι ✅ Mai 2026, Roadmap `docs/chat_agent_roadmap.md`; WP13-Cleanup ✅ `93ccc19`; Search-Provider-Plugins ✅ June 29; **WP12 Unified Render Layer ✅** — §9.3–§9.5 im Chat-UX-WP geschlossen, §9.2-Sichtprüfung läuft on the fly ([`docs/wp12_unified_render_layer.md`](docs/wp12_unified_render_layer.md)). Details im `AIChangelog.md`.)

## Module Documentation
- [`src/core/CLAUDE.md`](src/core/CLAUDE.md) — Core business logic, pipeline orchestration, data management.
- `src/core/agents/` — Agentic v4: `WorkflowExecutor`, `LLMAgentStep`, `DeterministicStep`, optional `MetaAgent` loop. (v3 SubAgents removed April 2026 — see [`docs/legacy/agentic_workflow_v3.md`](docs/legacy/agentic_workflow_v3.md).)
- [`src/mcp/CLAUDE.md`](src/mcp/CLAUDE.md) — MCP tool layer: schemas, registry, handlers.
- [`src/ui/CLAUDE.md`](src/ui/CLAUDE.md) — PyQt6 GUI components.
- [`src/utils/CLAUDE.md`](src/utils/CLAUDE.md) — Configuration, batch processing, logging.
- [`docs/`](docs/) — Architecture docs (agentic, classic pipeline, subsystems, legacy).
- [`AIChangelog.md`](AIChangelog.md) — Detailed dated developer log.
- [`CHANGELOG.md`](CHANGELOG.md) — User-facing release notes.
