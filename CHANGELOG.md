# Changelog

This file summarizes notable changes in this branch since the last upstream release, [`v0.0.1`](https://github.com/conradhuebler/ALIMA/releases/tag/v0.0.1).

> **User-facing release notes.** Topic-grouped, deduplicated.
> For the detailed dated developer log (per-feature commits, file lists,
> internal refactors), see [`AIChangelog.md`](AIChangelog.md).

## [Unreleased]

### Agentic ↔ Classic Core Convergence

- Agentic GND search (`search_swb`/`search_lobid` MCP tools) now uses the same mapping-first cache as the classic pipeline (MetaSuggester with read + write-back), instead of always-live HTTP against raw suggesters.
- Source failures in the agentic GND search are now surfaced (stream warnings, `source_errors` in the result) and abort the step when all sources fail or the pool is empty after a partial outage — no more silently empty keyword context.
- New `verify_keywords` workflow step (alima_v51, alima_classic): LLM-selected keywords are verified against the GND search pool with DB fallback, correcting or attaching authoritative GND-IDs before the strict-validated DK catalog search.
- Agentic DK classification now uses the exact classic pre-filtering (frequency threshold, title filter, institution-library RVK filter, RVK guardrail) via a shared `prepare_dk_classification_context`, derives RVK anchor keywords, and is skipped when the DK search produced no usable catalog context.
- GND search results now carry `sources`/`source_count` and are ranked multi-source-first, matching what the selection prompts assume.

### Agentic Workflow System (v4)

- Replaced the hardcoded MetaAgent + 4 SubAgents architecture with a YAML-driven workflow system: every pipeline step is now defined declaratively and dispatched through a generic `WorkflowExecutor`.
- Added `LLMAgentStep` and `DeterministicStep` step types with a plugin registry so new step types and tool functions can be added without touching the dispatch code.
- Added context-path resolver (`${steps.X.Y}`, `${extra.Y}`) and free-form `SharedContext.extra` dict for cross-step data flow.
- Shipped reference workflows: `alima_classic`, `catalog_search`, `synonym_expansion`, `batch_metadata`, `title_list_search`.
- Added CLI commands `alima workflow <name>` and `alima workflows list`; GUI gained a workflow dropdown in the pipeline config dialog.
- Added live agentic context dock (`AgenticContextWidget`) with per-step state visualization.
- Optional MetaAgent planning loop (`PLAN → EXECUTE → REFLECT`) available behind `meta_agent.enabled: true` in any workflow.

### Pipeline And LLM Processing

- Added a dedicated pipeline manager and shared pipeline utilities so CLI, desktop, and web workflows can run the same multi-step analysis flow.
- Reworked LLM extraction around structured output, better prompt handling, improved fallback behavior, and richer per-step state.
- Added generated working titles, source-aware identifiers, repetition handling, and better support for OCR and image-based input.
- Expanded iterative keyword analysis with GND-aware verification and missing-concept refinement.

### DK And RVK Classification

- Expanded DK and RVK handling across the application, including richer candidate display, provenance tracking, and structured export data.
- Added RVK validation metadata so results distinguish standard, non-standard, and validation-error notations.
- Added RVK lookup support through the official API and a local MarcXML-backed GND index.
- Improved RVK selection by using thematic anchor terms, shortlist balancing, and DK-informed rescoring before final output is chosen.
- Tightened catalog-derived RVK handling by requiring explicit RVK source metadata in MARC `084` parsing, adding stronger rejection of single-library institutional branches without document context, and allowing promoted anchor terms to trigger supplemental RVK API fallback even when catalog RVK already exists.
- Standardized classification handling under `results.classifications` while keeping `results.dk_classifications` as a compatibility alias.

### Web Application

- Added a substantially expanded web interface with dedicated templates, custom styling, session isolation, and improved live progress reporting.
- Added autosave, recovery, reconnect handling, immediate export, browser notifications, and abort support for long-running sessions.
- Added markdown-aware in-place log rendering for streamed LLM output, including tables and structured DK/RVK profile blocks.
- Improved result serialization and rendering so classifications, RVK validation details, and flattened DK search data are shown more reliably.

### Desktop UI

- Reworked major PyQt views, including the pipeline tab, stream display, comparison and review areas, and main window layout.
- Added an Erschließungsvergleich view and improved data transfer between desktop views.
- Improved the settings dialog, provider configuration, first-start flow, and batch-processing behavior.

### Configuration, CLI, And Setup

- Replaced the older monolithic CLI with modular commands under `src/cli/commands`.
- Consolidated pipeline configuration into dedicated builders, parsers, defaults, and tests.
- Expanded setup and onboarding flows with stronger preset handling, `setup --force`, example configurations, and catalog/database setup support.
- Removed older legacy configuration and provider-dialog paths that duplicated newer unified settings logic.

### Providers, Database, And Infrastructure

- Added SQL dialect handling for SQLite and MariaDB and fixed related datetime and query behavior.
- Improved provider selection, preset handling, fuzzy model matching, think-flag behavior, and provider-status checks.
- Improved unified knowledge manager reset and shutdown handling.
- Added broader runtime cleanup, including Qt plugin setup helpers and dependency handling improvements.

### Metadata, Search, And Catalog Integrations

- Improved DOI-based metadata retrieval with OpenAlex and DataCite fallback handling and better provenance tracking.
- Added K10plus-related helpers, including PICA and MARC fixes, PaketSigel support, and resolver utilities.
- Made catalog integrations more configuration-driven by replacing hardcoded endpoints with presets and unified settings.
- Improved Libero and SOAP/SRU setup flows and strengthened RVK retrieval from catalog-backed metadata.

### Documentation And Examples

- Expanded the documentation with guides for configuration, the agentic workflow, iterative GND search, DK classification splitting, and webapp session behavior.
- Updated the README and examples to match the current setup and pipeline behavior.

### Dependencies

- Refreshed the Python dependency set for the current runtime.
- Added `pdf2image` support and documented the Poppler requirement for PDF OCR.

### Unified Render Layer (GUI ↔ Webapp)

- GUI and web app now render the same pipeline log and result cards from a single shared event stream, so a run started in one frontend shows the same collapsible blocks, badges, and summary layout in the other.
- Collapsible panels (DK/GND results, tool output, source notes) behave consistently across both frontends: native expand/collapse, state preserved while streaming, and consistent dark-theme styling.
- Browser sessions resume cleanly after a websocket reconnect (no duplicate events, no missed steps), and the log stays in sync when you switch tabs or come back to a running run.
- Pipeline status and step outcomes are reported the same way in both frontends, including failed steps surfacing as a clearly marked error block.
