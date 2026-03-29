# Changelog

This file summarizes notable changes in this branch since the last upstream release, [`v0.0.1`](https://github.com/conradhuebler/ALIMA/releases/tag/v0.0.1).

## [Unreleased]

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
- Standardized classification handling under `results.classifications` while keeping `results.dk_classifications` as a compatibility alias.

### Web Application

- Added a substantially expanded web interface with dedicated templates, custom styling, session isolation, and improved live progress reporting.
- Added autosave, recovery, reconnect handling, immediate export, browser notifications, and abort support for long-running sessions.
- Added a tabbed preview/log view so streaming output and finalized structured results are both available in the same session.
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
