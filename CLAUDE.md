# CLAUDE: Your AI Assistant for the ALIMA Project

## Overview

**ALIMA** (Automatic Library Indexing and Metadata Analysis) is a comprehensive pipeline for library science workflows combining LLM-powered text analysis with GND/SWB keyword search and DK/RVK classification.

## Very General Instructions for AI Coding
- Avoid flattery, compliments, or positive language. Be clear and concise. Do not use agreeable language to deceive.
- Do comprehensive verification before claiming completion
- Show me proof of completion, don’t just assert it
- Prioritize thoroughness over speed
- If I correct you, adapt your method for the rest of the task
- No completion claims until you can demonstrate zero remaining instances

## General Instructions
- Each source code dir has a CLAUDE.md with basic information and logic
- **Keep CLAUDE.md files FOCUSED and CONCISE** - ONE clear idea per bullet, max 1-2 lines
  - ❌ DON'T: Multi-paragraph explanations, code examples, historical details
  - ✅ DO: Brief statements with links to detailed docs if needed
  - ✅ DO: `✅ **Feature name** - Brief description` for completed items
- Remove completed/resolved items after 2-3 updates (move to git history/AIChangelog.md)
- Task corresponding to code must be placed in the correct CLAUDE.md file
- Each CLAUDE.md has a variable part (short-term info, bugs) and preserved part (permanent knowledge)
- **Instructions blocks** contain operator-defined future tasks and visions for code development
- Only include information important for ALL subdirectories in main CLAUDE.md
- Preserve new knowledge from conversations but keep it brief
- Always suggest improvements to existing code
- **Keep entries concise and focused to save tokens**
- **Keep git commits concise and focused**
- **Rule of thumb**: If a CLAUDE.md section exceeds 20 lines, consider placing elsewhere

## Development Guidelines

### Code Organization
- Each `src/` subdirectory contains detailed CLAUDE.md documentation
- Variable sections updated regularly with short-term information
- Preserved sections contain permanent knowledge and patterns
- Instructions blocks contain operator-defined future tasks and visions

### Implementation Standards
- Mark new functions as "Claude Generated" for traceability
- Remove TODO Hashtags and text if done and approved
- Implement comprehensive error handling and logging
- Maintain backward compatibility where possible
- **Always check and consider instructions blocks** in relevant CLAUDE.md files before implementing
- Reformulate and clarify task and vision entries if not already marked as CLAUDE formatted
- Avoid hardcoded provider lists - read available providers and models from llmanager

### Workflow States
- **ADD**: Features to be added
- **WIP**: Currently being worked on
- **ADDED**: Basically implemented
- **TESTED**: Works (by operator feedback)
- **APPROVED**: Move to changelog, remove from CLAUDE.md

### Documentation Update Rules
- Replace debugging details with architecture decisions when issues are resolved
- Remove unnecessary pointer addresses and crash investigation specifics
- Focus on architectural clarity rather than technical debugging information
- Document the "why" behind design decisions for future reference
- Eliminate redundant information that doesn't add architectural value
- Prioritize clean, maintainable documentation over verbose troubleshooting history
- Keep track of significant improvements in [AIChangelog.md](AIChangelog.md)

### Git Best Practices
- **Only commit source files**: Use `git add <file>` for specific files, never `git add -A` without review
- **Review before committing**: Always check `git diff` and `git status` to avoid accidental commits
- **Commit message format**: Start with action verb (Fix, Add, Improve, Refactor), follow with brief description
- **Include Co-Author info**: All commits include Claude contribution notes with proper attribution
- **Test artifacts stay local**: Build outputs and temporary files are ignored by .gitignore

### Quality Assurance — Test Maintenance Rules

**The test suite MUST stay green. Broken tests that are silently ignored are worse than no tests.**

#### When refactoring or renaming APIs:
- Update ALL affected tests in the same commit — never leave tests failing
- If a class/method is renamed, grep for all test references and update them
- If a table schema changes, update `test_search.py` to reflect the new schema

#### Test isolation requirements:
- Tests using `UnifiedKnowledgeManager` MUST call `UnifiedKnowledgeManager.reset()` in both `setUp` and `tearDown`
- Tests MUST use `DatabaseConfig(db_type='sqlite', sqlite_path=<tempfile>)` — never rely on the production config (which may point to MariaDB)
- Tests using Qt classes (SearchEngine, LlmService) require a `QApplication` — use `Mock(spec=...)` to avoid it

#### When submitting or reviewing PRs:
- Run `python -m pytest tests/ -v` locally before opening a PR
- A PR that introduces new test failures is not ready to merge
- If existing tests were already broken (pre-existing failures), fix them in a separate commit and note it explicitly

#### Root cause of the 9-month debt (documented for future reference):
- `SearchEngine` was rewritten from async to Qt-signal-based in July 2025
- Tests in `test_cache.py` / `test_search.py` were not updated → silently broken for 9 months
- Discovered during PR #9 review (March 2026)

### Critical Requirements
**All pipeline changes MUST be usable by both CLI and GUI interfaces:**
- Shared Logic: All pipeline functionality uses `src/utils/pipeline_utils.py`
- Configuration Compatibility: Pipeline configurations work identically in both interfaces
- Parameter Consistency: All pipeline parameters supported uniformly across CLI and GUI

## [Preserved Section - Permanent Documentation]
**Change only if explicitly wanted by operator**

### ALIMA - Pipeline Architecture v2.0.0
**Step Definitions:**
- **1. `input`**: Any text, imported from documents, clipboard, or extracted from images via LLM visual analysis → TEXT
- **2. `initialisation`**: Free keyword generation using "initialisation" prompt → FREIE_SCHLAGWORTE
- **3. `search`**: GND/SWB/LOBID search based on free keywords to fill cache → GND_KEYWORD_POOL
- **4. `keywords`**: Verbale Erschließung using "keywords"/"rephrase"/"keywords_chunked" prompts with GND context → FINALE_GND_SCHLAGWORTE
- **5. `classification`**: Optional DDC/DK/RVK classification assignment via LLM → KLASSIFIKATIONEN

**Core Classes & Roles:**
- **`PipelineManager`**: High-level orchestrator managing workflow sequence and state
- **`PipelineStepExecutor`**: Helper knowing how to execute specific steps (LLM tasks, searches)
- **`AlimaManager`**: Low-level specialist for LLM calls via `LlmService`

**Technical Implementation:**
- ✅ CLI and GUI share identical `PipelineManager` code
- ✅ JSON export/resume capability for each step
- ✅ All prompts stored in prompts.json with provider/model adjustments
- ✅ Live streaming feedback system with step-by-step progress
- ✅ Consistent error handling and recovery across all steps
- ✅ **Unified Configuration System**: `PipelineConfigParser` + `PipelineConfigBuilder` consolidate CLI/GUI parameter handling
  - Single source of truth for parameter validation and parsing
  - Step-aware task validation with consistent rules across interfaces
  - Feature parity: CLI now supports all parameters (DK thresholds, chunking, etc.)

### Database Architecture - Facts/Mappings
- **`alima_knowledge.db`**: Single unified database
- **Facts Tables**: Immutable truths (`gnd_entries`, `classifications`)
- **Mappings Tables**: Dynamic search associations (`search_mappings`)
- **`UnifiedKnowledgeManager`**: Unified manager with mapping-first search strategy + singleton pattern

## [Variable Section - Current Tasks]

### ✅ DK Deduplication Statistics Display - Phase 2 COMPLETE
- ✅ **Critical Bug Fixes**: Fixed `dk_statistics` loading in CLI (3 functions), corrected GUI tab indices
- ✅ **CLI Statistics Display**: `format_dk_statistics()` shows deduplication metrics, Top 10 classifications, keyword coverage
- ✅ **GUI Statistics Tab**: Statistics panel in unified "📊 DK-Analyse" tab with deduplication summary, Top 10 table, keyword coverage mapping
- ✅ **Color-Coded Confidence**: Visual indicators based on catalog frequency (Green/Teal/Yellow/Red)
- ✅ **Backward Compatible**: Old JSON files gracefully handled with fallback messages

### ✅ Repetition Detection & Per-Model Chunking COMPLETE
- ✅ **Repetition Detector** (`src/utils/repetition_detector.py`): Detects LLM repetition loops during streaming
  - Three detection methods: char patterns, N-gram counting, window similarity (Jaccard)
  - Auto-abort with parameter variation suggestions (temperature, repetition_penalty, top_p)
  - Configurable via `RepetitionDetectionConfig` in config.json
  - **Grace Period** (2026-02-17): 2s wait before aborting to prevent false positives
    - Allows LLM self-correction during grace period
    - Shows countdown timer: "⏳ Auto-Abbruch in 1.5s"
    - Success message on resolution: "✅ Wiederholung behoben"
    - Configurable 0-10s (0 = immediate abort, 2s = default)
- ✅ **Per-Model Chunking Thresholds**: Model-specific keyword chunking configuration
  - `UnifiedProviderConfig.model_chunking_thresholds` stores per-provider/model settings
  - Priority: explicit override > per-model config > pattern match > default (500)
  - UI spinbox in Model Preferences table with auto-detect hint
- ✅ **Repetition Warning UI**: Orange warning panel in `PipelineStreamWidget`
  - Shows detection type and details
  - Retry buttons with suggested parameter variations
  - Emits `retry_with_variations` signal for pipeline retry
  - **Countdown Timer**: 100ms QTimer updates during grace period

### WIP: Token Control & Chunking
- Implement token size control with slider (1-50 keywords per chunk)
- Split keywords while keeping template+abstract constant
- Print currently processed keywords to console
- Bold highlighting for predefined keywords and GND numbers

### WIP: Results Processing
- Extract `<final_list>` sections from all results
- Match extracted terms against keyword database
- Final analysis of unprocessed keywords from chunks

## [Instructions Block - Operator-Defined Tasks]

### Vision
- Restructure code: consolidate distributed logic from utils, core, suggestors
- Keep CLAUDE.md compact, focus on actionable information not minor implementation details
- Maintain unified pipeline architecture ensuring CLI and GUI feature parity

### Future Tasks
1. Code Restructuring: Consolidate distributed logic from utils, core, suggestors
2. Pipeline Enhancement: Templates, advanced configuration UI
3. Batch Enhancement: Extended image analysis and URL scraping support
4. Performance Optimization: Connection pooling, result pagination, memory optimization

## Module Documentation
- **[src/core/CLAUDE.md](src/core/CLAUDE.md)** - Core business logic, pipeline orchestration, data management
- **[src/core/agents/](src/core/agents/)** - Agentic system: BaseAgent, SearchAgent, KeywordAgent, ClassificationAgent, ValidationAgent, MetaAgent
- **[src/mcp/CLAUDE.md](src/mcp/CLAUDE.md)** - MCP tool layer: tool schemas, registry, handlers for DB/web/pipeline access
- **[src/ui/CLAUDE.md](src/ui/CLAUDE.md)** - PyQt6 GUI components, dialogs, widgets
- **[src/utils/CLAUDE.md](src/utils/CLAUDE.md)** - Configuration management, batch processing, logging
- **[AIChangelog.md](AIChangelog.md)** - Complete history of implemented features (2025)

## Standards

### Unified Database Configuration (Single Source of Truth)
✅ **Problem Solved**: Eliminated duplicate path definitions → `DatabaseConfig.sqlite_path` is now the only database path source
- OS-specific default paths: Windows (`%APPDATA%\ALIMA\`), macOS (`~/Library/Application Support/ALIMA/`), Linux (`~/.config/alima/`)
- Singleton pattern: `UnifiedKnowledgeManager` with thread-safe `__new__()` override
- Auto-migration: Legacy configurations automatically migrate to unified JSON format

### Pipeline Integration (CLI and GUI)
✅ **Unified Logic**: Both interfaces use identical `PipelineManager` via shared `src/utils/pipeline_utils.py`
- `PipelineStepExecutor`: Handles initialisation, search, keywords, classification steps
- `PipelineJsonManager`: Save/resume functionality for interrupted workflows
- Real-time streaming: Token-by-token feedback via callback adapters

### Batch Processing
✅ **Complete Implementation**: Full pipeline execution for document batches
- Supported sources: DOI, PDF (with LLM-OCR fallback), TXT, IMG (vision model), URL (web scraping)
- Resume functionality: Continue interrupted batches via JSON state persistence
- Review UI: Toggle between batch table view and individual result details

### DK Classification Transparency
✅ **Transparency System**: Display which catalog titles led to each DK classification
- GUI: Color-coded confidence levels in PipelineStreamWidget and AnalysisReviewTab
- CLI: DK titles in detailed/compact/k10plus format via show-protocol

### K10+/WinIBW Export
✅ **Catalog Integration**: Direct export in K10+/WinIBW format for seamless integration
- GUI: K10+ Export Tab with Copy-Button
- CLI: `--format k10plus` mode

### First-Start Setup Wizard
✅ **Complete Implementation**: Interactive setup guides for new users on first launch
- **GUI Wizard** (`src/ui/first_start_wizard.py`): PyQt6 multi-page wizard with LLM provider setup and optional GND database download
  - Optional LLM connection testing (doesn't block proceeding)
  - GND skip confirmation dialog to prevent accidental skipping
- **CLI Wizard** (`src/utils/cli_setup_wizard.py`): Terminal-based interactive setup with provider selection, connection testing, and configuration saving
- **Shared Utilities** (`src/utils/setup_utils.py`): Reusable validation and setup functions (Ollama, OpenAI, Gemini, Anthropic, GND download)
  - Fixed: Correct UnifiedProvider initialization with proper parameters
  - Fixed: Proper TaskPreference creation with model_priority structure
- **Auto-Detection**: Both GUI and CLI check `first_run_completed` flag and prompt setup if needed
- **Provider Support**: Ollama (local/remote), OpenAI-compatible APIs, Google Gemini, Anthropic Claude

### Model Capabilities & Configuration
✅ **Auto-Detection System**: Intelligent model-specific configuration with pattern matching
- **Model Capabilities Registry** (`src/utils/model_capabilities.py`): 15+ patterns for optimal chunking thresholds
  - Large models (>30B): 1000 keywords before chunking
  - Medium models (13-14B): 500 keywords (default)
  - Small models (<7B): 200-300 keywords
- **Provider Selection Priority**: Explicit UI selection > Task preferences > Config defaults
  - Fixed: Manual UI selections now correctly override task preferences
  - Enhanced logging with visual indicators (🎯 explicit, 📋 preference, ⚙️ default)
- **GUI Integration**: Pipeline config dialog shows "Auto" for zero/None values (auto-detection)
