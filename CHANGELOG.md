# Changelog

This file summarizes notable changes in this branch relative to [`conradhuebler/ALIMA`](https://github.com/conradhuebler/ALIMA), using `upstream/main` as the baseline.

## [Unreleased]

### Pipeline And LLM Processing

- Reworked the pipeline around structured JSON-mode LLM output, with updated parsing, schemas, prompt handling, and fallback behavior.
- Added GND-pool verification for model-extracted keywords and expanded iterative keyword refinement.
- Added generated working titles and richer per-step pipeline state across CLI, PyQt, and web runs.
- Added repetition detection and repetition-penalty controls for long or noisy model outputs.
- Improved OCR and image handling, including multi-image OCR, better OpenAI-compatible vision handling, and safer OCR error reporting.

### DK And RVK Classification

- Expanded DK/RVK transparency throughout the app, including full candidate display, deduplication statistics, and export of flattened DK search data.
- Added structured classification export under `results.classifications` while keeping `results.dk_classifications` as a backward-compatible alias.
- Added RVK validation metadata and UI surfacing for standard, non-standard, and validation-error RVK notations.
- Added RVK authority fallback infrastructure:
  - official RVK API lookup and validation
  - local RVK MarcXML GND index for `GND-ID -> RVK` lookup
  - periodic RVK dump release checks and index refresh
- Improved RVK candidate filtering to reject obvious artifacts, preserve plausible local variants, and record RVK provenance.
- Added thematic RVK anchor handling so RVK lookup can prefer a smaller, thematically central GND subset.
- Experimental deterministic RVK ranking was introduced and later disabled again as the active output path; final RVK output currently comes from the constrained LLM selection.
- Updated PyQt labels and data handling so DK/RVK classifications are treated more neutrally instead of implying DK-only output.

### Web Application

- Added a redesigned web UI with dedicated templates, stronger styling, and session isolation.
- Added auto-save, recovery, immediate export, reconnect handling, and step-level abort support.
- Improved live pipeline progress display, including more accurate step mapping and progress text.
- Fixed multiple web reliability issues:
  - final results rendering after completion
  - stale recovery banners and interrupted-connection UI
  - incorrect reuse of previous-run results while a new run is active
  - polling/rendering errors caused by unexpected result shapes
  - incorrect DK search summary rendering
- Added browser notifications and improved result presentation for classifications and RVK validation.

### PyQt GUI

- Reworked major parts of the desktop UI, especially the pipeline tab, stream widget, comparison/review views, and main window layout.
- Split and reorganized older DK/UB catalog functionality into clearer tabs and workers.
- Added an Erschließungsvergleich tab and improved result transfer between views.
- Improved the comprehensive settings dialog, provider settings, and first-start wizard.
- Added batch-processing improvements such as manual text input, DOI-aware filenames, and better non-modal behavior.
- Fixed settings-dialog sizing and scrolling issues on smaller screens.

### Configuration, Setup, And CLI

- Replaced the monolithic CLI with modular commands under `src/cli/commands`.
- Consolidated pipeline configuration handling with dedicated builders, parsers, defaults, and tests.
- Added richer setup flows, including `setup --force`, first-start wizard improvements, preset export, preset examples, and catalog/DB setup support.
- Added `config.example.lobid-gbv.json` as a sample configuration for Lobid plus GBV/GVK SRU usage.
- Removed older legacy config/editor code paths and redundant provider dialogs.

### Providers, Database, And Infrastructure

- Added SQL dialect abstraction for SQLite and MariaDB compatibility and fixed related datetime/query issues.
- Improved database reset and shutdown behavior in the unified knowledge manager.
- Added provider preset handling, fuzzy model matching, think-flag handling improvements, and better provider-status checks.
- Added Qt plugin setup helpers and broader dependency/runtime cleanup.

### Metadata, Search, And Catalog Integrations

- Added OpenAlex/DataCite fallback and improved DOI resolution, metadata formatting, and filename/source provenance.
- Added K10plus helpers including PICA/MARC fixes, PaketSigel support, and resolver utilities.
- Improved catalog configuration so hardcoded URLs are replaced by presets and configuration-driven endpoints.
- Added Libero token auto-creation/login dialog support and improved SOAP/SRU setup handling.

### Documentation And Examples

- Expanded documentation with new guides for configuration, the agentic workflow, iterative GND search, DK classification splitting, and webapp session behavior.
- Updated the README and examples to reflect the newer setup and pipeline behavior.
- Added `AIChangelog.md` and supporting internal documentation files.

### Dependencies

- Refreshed `requirements.txt` for the current environment.
- Added `pdf2image` to the Python dependencies and documented the Poppler requirement for PDF OCR.
