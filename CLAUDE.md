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

### Database Architecture - Facts/Mappings
- **`alima_knowledge.db`**: Single unified database
- **Facts Tables**: Immutable truths (`gnd_entries`, `classifications`)
- **Mappings Tables**: Dynamic search associations (`search_mappings`)
- **`UnifiedKnowledgeManager`**: Unified manager with mapping-first search strategy + singleton pattern

## [Variable Section - Current Tasks]

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
- **[src/ui/CLAUDE.md](src/ui/CLAUDE.md)** - PyQt6 GUI components, dialogs, widgets
- **[src/utils/CLAUDE.md](src/utils/CLAUDE.md)** - Configuration management, batch processing, logging
- **[AIChangelog.md](AIChangelog.md)** - Complete history of implemented features (2025)

## Standards

### Unified Database Configuration (Single Source of Truth)
✅ **Problem Solved**: Eliminated duplicate path definitions → `DatabaseConfig.sqlite_path` is now the only database path source
- OS-specific default paths: Windows (`%APPDATA%\ALIMA\`), macOS (`~/Library/Application Support/ALIMA/`), Linux (`~/.config/alima/`)
- Singleton pattern: `UnifiedKnowledgeManager` with thread-safe `__new__()` override
- Auto-migration: Old configs with `system_config.database_path` automatically migrate to `database_config.sqlite_path`

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
