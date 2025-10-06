# CLAUDE: Your AI Assistant for the ALIMA Project

## General Instructions

- Each source code dir has a CLAUDE.md with basic informations of the code, the corresponding knowledge and logic in the directory 
- If a file is not present or outdated, create or update it
- Task corresponding to code have to be placed in the correct CLAUDE.md file
- Each CLAUDE.md may contain a variable part, where short-term information, bugs etc things are stored. Obsolete information have to be removed
- Each CLAUDE.md has a preserved part, which should no be edited by CLAUDE, only created if not present
- Each CLAUDE.md may contain an **instructions block** filled by the operator/programmer and from CLAUDE if approved with future tasks and visions that must be considered during code development
- Each CLAUDE.md file content should be important for ALL subdirectories
- If new knowledge is obtained from Claude Code Conversation preserve it in the CLAUDE.md files
- Always give improvments to existing code
- **Remove bloated technical documentation and long-term outdated content** 
- **Keep only clear instructions, current tasks, and actionable information**
- **No usage documentation or tutorials belong here**

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
- reformulate and clarify task and vision entries if not alredy marked as CLAUDE formatted
- avoid hardcoded provider list like gemini, anthropic etc, read available provider and models from llmanager

### Workflow
- Features to be added have to be marked as ADD
- If the feature/function/task is being worked on, mark it as WIP
- If the feature/function/task is basically implemented, mark it as ADDED
- Summarize several functions to features/task
- If it works (by operator feedback), mark it as TESTED 
- Ask regulary if the TESTED feature is approved, if yes: it to the changelog (summarised) and remove from claude.md  

## [Preserved Section - Permanent Documentation]
**Change only if explicitly wanted by operator**

### ALIMA - Pipeline Architecture v2.0.0 ✅
**Step Definitions:**
- **1. `input`**: [any text, imported from documents, clipboard, or extracted from images via LLM visual analysis] → TEXT
- **2. `initialisation`**: Free keyword generation using "initialisation" prompt (formerly "abstract") → FREIE_SCHLAGWORTE  
- **3. `search`**: GND/SWB/LOBID search based on free keywords to fill cache with related GND keywords → GND_KEYWORD_POOL
- **4. `keywords`**: Verbale Erschließung using "keywords"/"rephrase"/"keywords_chunked" prompts with GND context → FINALE_GND_SCHLAGWORTE
- **5. `classification`**: Optional DDC/DK/RVK classification assignment via LLM → KLASSIFIKATIONEN

**Core Classes & Roles (Orchestrator vs. Engine):**
- **`PipelineManager` (The Conductor):** This is the high-level orchestrator for the entire workflow. It holds the `PipelineConfig`, manages the sequence of steps, and holds the overall state. It does *not* perform tasks itself but delegates them.
- **`PipelineStepExecutor` (The Foreman):** A helper used by the `PipelineManager`. It knows *how* to execute a specific type of step, calling the appropriate specialist for the job (e.g., `AlimaManager` for LLM tasks, `SearchCLI` for searches).
- **`AlimaManager` (The Engine):** This is the low-level specialist for LLM calls. It takes a precise, single task (a prompt, a model, parameters) and executes it via the `LlmService`, returning only the result of that single operation.

**Technical Requirements IMPLEMENTED:**
- ✅ **[UNIFIED]** CLI and GUI share identical `PipelineManager` code  
- ✅ JSON export/resume capability for each step
- ✅ All prompts stored in prompts.json with provider/model adjustments
- ✅ Live streaming feedback system with step-by-step progress
- ✅ Consistent error handling and recovery across all steps

### Database Architecture - Facts/Mappings ✅
- **`alima_knowledge.db`**: Single unified database 
- **Facts Tables**: Immutable truths (`gnd_entries`, `classifications`)  
- **Mappings Tables**: Dynamic search associations (`search_mappings`)
- **`UnifiedKnowledgeManager`**: Unified manager with mapping-first search strategy

## [Variable Section - Current Tasks]

### ✅ COMPLETED: Unified Logging System (Claude Generated)
- **Central Logging Infrastructure** (`src/utils/logging_utils.py`):
  - 4-level verbosity system (0=Quiet, 1=Normal, 2=Debug, 3=Verbose)
  - `setup_logging(level)` function with automatic third-party suppression
  - `print_result()` function for result output that respects quiet mode
- **CLI Integration**: `--log-level` argument (0-3, default=1)
- **GUI Integration**: Uses level 1 (Normal) by default
- **Migration Status**:
  - ✅ CLI: Critical pipeline callbacks and results migrated
  - ✅ Core: alima_manager.py debug output migrated
  - ✅ UI: main_window.py GND import logging migrated
  - ⏳ Future: CLI command status messages (~200+ print statements remain)
  - ⏳ Future: Suggester modules debug output

### WIP: Token Control & Chunking
- **AbstractTab Token Slider**: Implement token size control with slider (1-50 keywords per chunk)
- **Keyword Chunking**: Split keywords from `self.keywords` string, keep template+abstract constant
- **Console Logging**: Print currently processed keywords to console
- **Keyword Highlighting**: Bold highlighting for predefined keywords and GND numbers in results

### WIP: Results Processing
- **Final List Extraction**: Extract `<final_list>` sections from all results
- **Keyword Matching**: Match extracted terms against keyword database
- **Remaining Keywords Analysis**: Final analysis of unprocessed keywords from chunks

### TESTED: Three-Mode CLI System ✅
- **Smart Mode**: Uses task preferences from config.json automatically
- **Advanced Mode**: Manual provider|model override with `|` separator
- **Expert Mode**: Full parameter control (temperature, top-p, seed)

## [Instructions Block - Operator-Defined Tasks]

### Critical Requirements
**All pipeline changes MUST be usable by both CLI and GUI interfaces:**
- Shared Logic: All pipeline functionality uses `src/utils/pipeline_utils.py` 
- Configuration Compatibility: Pipeline configurations work identically in both interfaces
- Parameter Consistency: All pipeline parameters supported uniformly across CLI and GUI

### Vision
- Restructure Code: many functions and logics is distributed among utils, core, suggestors
- Keep Claude.MD compact, don't spam it with short-term informations about minor implementation bugs
- Maintain unified pipeline architecture ensuring CLI and GUI feature parity
- All future pipeline enhancements must implement both interfaces simultaneously

### Future Tasks
1. **Code Restructuring**: Consolidate distributed logic from utils, core, suggestors
2. **Pipeline Enhancement**: Batch processing, templates, advanced configuration UI
3. **Performance Optimization**: Connection pooling, result pagination, memory optimization