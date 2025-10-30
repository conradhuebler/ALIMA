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
- git commits have to be short and precise
- new features need a precise dokumentation in the readme, in special cases a comprehensive documention separatly and a note in claude.md that this feature exists, only for main and important features, not for updates and improvments

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

### ✅ PRODUCTION READY: Batch Processing System (Claude Generated)
**Automated stapelweise Verschlagwortung von Quellenstapeln mit vollständiger Pipeline-Ausführung**

**Core Components:**
- **`src/utils/batch_processor.py`**: Complete batch processing engine (689 lines)
  - ✅ `BatchProcessor`: Main processing class using `PipelineManager` for full pipeline execution
  - ✅ `BatchSourceParser`: Parses batch files with syntax validation
  - ✅ `BatchState`: Resume functionality with JSON persistence
  - ✅ **ALL Source Types Supported:**
    - **DOI**: Full support via `doi_resolver`
    - **PDF**: PyPDF2 with intelligent LLM-OCR fallback for poor quality
    - **TXT**: Direct file reading
    - **IMG**: Vision model analysis via `execute_input_extraction()`
    - **URL**: Web scraping with BeautifulSoup4

**CLI Integration:**
```bash
# Basic batch processing
python src/alima_cli.py batch --batch-file sources.txt --output-dir results/

# With pipeline configuration
python src/alima_cli.py batch --batch-file sources.txt --output-dir results/ \
  --step initialisation=ollama|cogito:14b --step keywords=gemini|gemini-1.5-flash

# Resume interrupted batch
python src/alima_cli.py batch --resume results/.batch_state.json
```

**GUI Integration:**
- **Menu**: Tools → Batch Processing... / Batch-Ergebnisse laden...
- **Dialog** (`src/ui/batch_processing_dialog.py`, 668 lines):
  - Tab 1: Batch File input
  - Tab 2: Directory Scan with filters (file types, recursive, name patterns)
  - Live preview list with checkboxes
  - QThread background processing with progress bar
  - Detailed progress log
- **✅ NEW: Batch Review UI** (`src/ui/analysis_review_tab.py`):
  - Toggle button "📋 Batch-Ansicht" / "📄 Einzelansicht"
  - Table view with columns: Status, Source, Keywords, Date, Actions
  - Double-click or "View" button to inspect individual results
  - Seamless switch between batch overview and detail view
  - Loads all JSONs from directory automatically

**Features:**
- ✅ **Complete Pipeline Execution**: Full ALIMA pipeline (init → search → keywords → classification)
- ✅ **Advanced PDF Support**: PyPDF2 with intelligent LLM-OCR fallback for scanned/poor quality PDFs
- ✅ **Image Analysis**: Vision model integration via `execute_input_extraction()`
- ✅ **URL Web Scraping**: BeautifulSoup4-based content extraction with heuristics
- ✅ **Batch Review Table**: GUI table view with toggle mode for reviewing all results
- ✅ **Quality Checks**: Automatic PDF quality detection and fallback strategies
- ✅ Continue-on-error vs. stop-on-error modes
- ✅ Resume functionality for interrupted batches
- ✅ Modular output (1 JSON per source)
- ✅ Pipeline configuration inheritance from global settings
- ✅ Real-time progress tracking and logging
- ✅ File type filters and preview for directory scanning

**Batch File Format:**
```
# Comments with #
DOI:10.1234/example
PDF:/path/to/document.pdf
TXT:/path/to/text.txt
IMG:/path/to/image.png
URL:https://example.com/abstract

# Extended format with custom name and overrides
DOI:10.1234/example ; MyPaper ; {"keywords": {"temperature": 0.3}}
```

### ✅ PRODUCTION READY: Protocol Display (`show-protocol` CLI Command) (Claude Generated)
CLI-Befehl zur Anzeige von Pipeline-Ergebnissen aus JSON-Dateien. Drei Modi: `--format detailed` (lesbar), `--format compact` (CSV), `--format k10plus` (Katalog-Export).
- **Implementation:** `src/alima_cli.py` (Zeilen 46-49: K10+ tags, 517-584: display_protocol_k10plus)
- **User Documentation:** README.md Sections 2.3, 2.4, 2.5

### ✅ PRODUCTION READY: DK Classification Transparency (Claude Generated)
Automatische Anzeige welche Katalog-Titel zu jeder DK-Klassifikation führten.
- **GUI**: PipelineStreamWidget zeigt Sample-Titel während DK-Suche, AnalysisReviewTab mit Farbcodierung (Konfidenz)
- **CLI**: show-protocol mit DK-Titeln in detailed/compact/k10plus format
- **Implementation:** `src/ui/pipeline_stream_widget.py` (256-310), `src/ui/analysis_review_tab.py` (389-399)
- **User Documentation:** README.md Section 2.4

### ✅ PRODUCTION READY: K10+/WinIBW Catalog Export (Claude Generated)
Direct export in K10+/WinIBW format für nahtlose Katalog-Integration.
- **GUI**: Neuer "K10+ Export" Tab mit Copy-Button
- **CLI**: --format k10plus für direktes Copy-Paste
- **Implementation:** `src/alima_cli.py` (48-49: K10+ tags, 517-584), `src/ui/analysis_review_tab.py` (29-32: K10+ tags, 215-235: K10+ tab, 342-375: _generate_k10plus_format)
- **Configuration**: K10PLUS_KEYWORD_TAG, K10PLUS_CLASSIFICATION_TAG (später in config.json)
- **User Documentation:** README.md Section 2.5

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
2. **✅ COMPLETED - Batch Processing**: Full batch processing with pipeline execution and review UI
3. **✅ COMPLETED - Batch Review Table**: AnalysisReviewTab with toggle mode and batch table
4. **Pipeline Enhancement**: Templates, advanced configuration UI
5. **Batch Enhancement**: Image analysis and URL scraping support
6. **Performance Optimization**: Connection pooling, result pagination, memory optimization
