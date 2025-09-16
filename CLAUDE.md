# CLAUDE: Your AI Assistant for the ALIMA Project

As Claude, your AI assistant, I'm here to provide comprehensive support for the ALIMA project and its ongoing development. My goal is to help you efficiently with programming, debugging, feature implementation, documentation, and architectural decisions.

## How I Support the ALIMA Project

## General instructions

- Each source code dir has a CLAUDE.md with basic informations of the code, the corresponding knowledge and logic in the directory 
- If a file is not present or outdated, create or update it
- Task corresponding to code have to be placed in the correct CLAUDE.md file
- Each CLAUDE.md may contain a variable part, where short-term information, bugs etc things are stored. Obsolete information have to be removed
- Each CLAUDE.md has a preserved part, which should no be edited by CLAUDE, only created if not present
- Each CLAUDE.md may contain an **instructions block** filled by the operator/programmer and from CLAUDE if approved with future tasks and visions that must be considered during code development
- Each CLAUDE.md file content should be important for ALL subdirectories
- If new knowledge is obtained from Claude Code Conversation preserve it in the CLAUDE.md files
- Always give improvments to existing code

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

### Add(ed)/Tested/Approved
- For Task/Features/Function use numeric identifieres (1,2,3,...) to organise the task/features/functions across the documents DURING development
### Workflow
- Features to be added have to be marked as ADD
- If the feature/function/task is being worked on, mark it as WIP
- If the feature/function/task is basically implemented, mark it as ADDED
- Summarize several functions to features/task
- If it works (by operator feedback), mark it as TESTED 
- Ask regulary if the TESTED feature is approved, if yes: it to the changelog (summarised) and remove from claude.md  


## [Preserved Section - Permanent Documentation]
**Change only if explicitly wanted by operator**
### ALIMA - Pipeline
#### v1.0.0 - FULLY CONSISTENT IMPLEMENTATION ✅
The pipeline has been made fully consistent across all components with unified naming and identifiers:

**Step Definitions:**
- **1. `input`**: [any text, imported from documents, clipboard, or extracted from images via LLM visual analysis] → TEXT
- **2. `initialisation`**: Free keyword generation using "initialisation" prompt (formerly "abstract") → FREIE_SCHLAGWORTE  
- **3. `search`**: GND/SWB/LOBID search based on free keywords to fill cache with related GND keywords → GND_KEYWORD_POOL
- **4. `keywords`**: Verbale Erschließung using "keywords"/"rephrase"/"keywords_chunked" prompts with GND context → FINALE_GND_SCHLAGWORTE
- **5. `classification`**: Optional DDC/DK/RVK classification assignment via LLM → KLASSIFIKATIONEN

**Consistent Implementation:**
- **Step-IDs**: `input` → `initialisation` → `search` → `keywords` → `classification`
- **UI-Names**: "Input" → "Initialisierung" → "Suche" → "Schlagworte" → "Klassifikation"
- **Prompt-Tasks**: `initialisation` → `keywords`/`rephrase`/`keywords_chunked`
- **Code-Methods**: All method names, variables, and identifiers use consistent step-IDs
- **Chunking-Support**: Available for both `initialisation` and `keywords` steps

**Technical Requirements IMPLEMENTED:**
- ✅ CLI and GUI share identical PipelineManager code  
- ✅ JSON export/resume capability for each step
- ✅ All prompts stored in prompts.json with provider/model adjustments
- ✅ Live streaming feedback system with step-by-step progress
- ✅ Consistent error handling and recovery across all steps

## [Variable Section - Short-term Information]

### ✅ MAJOR ENHANCEMENT COMPLETED - Two-Step DK Classification Architecture

**Status**: ✅ DK Classification split into separate search and analysis steps for optimal performance and recovery

**Implementation Details**:
- **Step 4a**: `execute_dk_search()` - Time-intensive catalog search (BiblioExtractor SOAP calls)  
- **Step 4b**: `execute_dk_classification()` - Fast LLM analysis using pre-fetched search results
- **Auto-save checkpoint**: Intermediate state saved after DK search step for recovery
- **Enhanced data model**: `KeywordAnalysisState` now stores both `dk_search_results` and `dk_classifications`
- **Pipeline step count**: Updated to 5 steps (Input → Keywords → Search → Final Keywords → DK Search → DK Classification)

**Performance Benefits**:
- **Caching**: Expensive catalog search results cached independently from LLM analysis
- **Recovery**: Pipeline can resume after catalog search without repeating time-intensive operations  
- **Modularity**: Search and analysis steps can be configured and run independently
- **Error isolation**: Search failures don't affect LLM analysis step and vice versa

### 🚀 MAJOR REFACTORING COMPLETED - Pipeline Logic Unification

**Status**: ✅ CLI and GUI now share maximum logic through abstracted utilities

**Abstracted Components** (`src/utils/pipeline_utils.py`):
- **PipelineStepExecutor**: Unified step execution (initialisation, search, keywords)
- **PipelineJsonManager**: Complete JSON serialization/deserialization
- **PipelineResultFormatter**: Consistent result formatting across implementations
- **execute_complete_pipeline()**: End-to-end pipeline function used by both CLI and GUI

**CLI Enhancements**:
- **New `pipeline` command**: Unified pipeline execution using shared logic
- **JSON save/resume**: `--output-json` and `--resume-from` parameters
- **Refactored utilities**: Uses `PipelineJsonManager` for all JSON operations

**GUI Refactoring**:
- **PipelineManager**: Now uses `PipelineStepExecutor` eliminating duplication
- **JSON functionality**: Added save/resume capabilities to GUI pipeline
- **Fixed display issue**: Final GND keywords now display correctly in pipeline tab

**Benefits Achieved**:
- **Code reduction**: ~200 lines of duplicate code eliminated
- **Consistency**: Both CLI and GUI use identical pipeline logic
- **Maintainability**: Single source of truth for pipeline operations
- **JSON compatibility**: Unified serialization across both interfaces

### ✅ TESTING COMPLETED - Pipeline Verified Working

**CLI Command (New Pipeline):**
```bash
python alima_cli.py pipeline \
  --input-text "Your analysis text here" \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b" \
  --provider "ollama" \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434 \
  --suggesters "lobid" "swb" \
  --output-json "results.json"
```

**GUI Pipeline:**
- ✅ Auto-Pipeline button functional
- ✅ Real-time streaming display
- ✅ Final GND keywords correctly displayed
- ✅ All 5 pipeline steps working (Input → Initialisation → Search → Keywords → Classification)

**Status**: Both CLI and GUI pipelines tested and working with shared logic from `pipeline_utils.py`

### ✅ COMPREHENSIVE SETTINGS DIALOG COMPLETED - Unified Configuration Management

**Status**: ✅ Complete settings dialog integrated replacing old configuration system

**Features Implemented**:
- **Tabbed Interface**: 6 comprehensive tabs (Database, LLM Providers, Catalog, Prompts, System, About)
- **Database Configuration**: Full SQLite and MySQL/MariaDB support with connection testing
- **LLM API Management**: Secure API key storage for all providers (Gemini, Anthropic, OpenAI, Comet, ChatAI)
- **Catalog Settings**: Library catalog token and URL configuration
- **Integrated Prompt Editor**: JSON-based prompt editing with validation and task management
- **System Configuration**: Debug settings, log levels, directory paths, and OS-specific config scopes

**Technical Integration**:
- **Main Window Integration**: `ComprehensiveSettingsDialog` replaces old `SettingsDialog` 
- **Configuration Backend**: Uses new `ConfigManager` with OS-specific paths and fallback system
- **Real-time Validation**: Database connection testing with threaded workers
- **Component Refresh**: Automatic refresh of all application components after configuration changes
- **Cross-platform Support**: Native configuration paths for Windows, macOS, and Linux

**GUI Implementation**:
```python
# Main window integration
from .comprehensive_settings_dialog import ComprehensiveSettingsDialog

def show_settings(self):
    dialog = ComprehensiveSettingsDialog(parent=self)
    dialog.config_changed.connect(self._on_config_changed)
    if dialog.exec():
        self.load_settings()
        self._refresh_components()
```

**Verification Results**:
- ✅ All 6 tabs functional with proper configuration loading/saving
- ✅ Database connection testing working for both SQLite and MySQL
- ✅ Prompt editor with JSON validation operational  
- ✅ OS-specific configuration paths correctly implemented
- ✅ Component refresh system integrated with main application
- ✅ Replaces fragmented configuration dialogs with unified interface

### ✅ DATABASE UNIFICATION COMPLETED - Facts/Mappings Architecture  

**Status**: ✅ WEEK 1 IMPLEMENTATION COMPLETED - Unified knowledge database with Facts/Mappings separation

**Database Architecture Implemented**:
- **`alima_knowledge.db`**: Single unified database replacing separate `search_cache.db` + `dk_classifications.db`
- **Facts Tables**: Immutable truths (`gnd_entries`, `classifications`)  
- **Mappings Tables**: Dynamic search associations (`search_mappings`)
- **Clean separation**: Facts never change; Mappings track search term → results relationships

**Technical Implementation**:
- **`UnifiedKnowledgeManager`**: New unified manager replacing both `CacheManager` + `DKCacheManager`
- **Compatibility adapters**: Existing interfaces maintained via adapter methods
- **Full system integration**: All 15+ files updated to use unified system
- **Schema migration**: Clean database design with proper relationships and indexing

**Files Updated**:
- **Core services**: `pipeline_utils.py`, `pipeline_manager.py`, `search_cli.py`, `biblioextractor.py`
- **UI components**: `main_window.py`, `pipeline_tab.py`, `global_status_bar.py` 
- **CLI tools**: `alima_cli.py`, test files
- **New unified manager**: `unified_knowledge_manager.py` with Facts/Mappings architecture

**Verification Results**:
- ✅ Database schema created and operational
- ✅ GND fact storage working 
- ✅ Search mapping storage/retrieval working
- ✅ Compatibility adapters functional for existing code
- ✅ All components using unified system successfully

**Performance Benefits Achieved**:
- **Single database**: Eliminates management complexity of separate databases
- **Facts/Mappings separation**: Enables progressive learning and advanced caching strategies  
- **Unified interface**: Simplified development and maintenance
- **Foundation for Week 2**: Smart search integration with mapping-based optimization ready

## [Instructions Block - Operator-Defined Tasks]
### Future Tasks
- Task 1: Description
- Task 2: Description

### ✅ WEEK 2 COMPLETED - Smart Search Integration with Mappings

**Status**: ✅ Mapping-first search strategy fully implemented and tested

**Week 2 Implementation Details**:
- **UnifiedKnowledgeManager Enhanced**: Added `search_with_mappings_first()` method for intelligent caching
- **MetaSuggester Refactored**: Now uses mapping-first strategy with automatic fallback to live search
- **Progressive Learning System**: Search results automatically cached as mappings for future use
- **Cache Performance Metrics**: Real-time statistics show mapping hits vs live searches

**Performance Improvements Achieved**:
- **Mapping Hit Example**: "artificial intelligence" → 3 results from cache (instant)
- **Live Search + Caching**: "machine learning" → 100 results from Lobid + auto-stored mapping
- **Statistics Tracking**: 21 search mappings with 5.9 average results per mapping
- **Database Growth**: Unified knowledge database now contains 206,870+ entries (34MB)

**Technical Architecture**:
```python
# Week 2: Mapping-first search flow
cached_gnd_ids, was_cached = ukm.search_with_mappings_first(
    search_term=term,
    suggester_type="lobid", 
    max_age_hours=24,
    live_search_fallback=lambda t: suggester.search([t])
)
# Result: 📊 lobid: 1 mapping hits, 1 live searches
```

**CLI Enhancements Added**:
- **Cache Management**: `python alima_cli.py clear-cache --type [all|gnd|search|classifications]`
- **DNB Import**: `python alima_cli.py dnb-import --force --debug` with progress information
- **Interactive Confirmation**: Safety prompts with current cache statistics before clearing

**Verified Functionality**:
- ✅ Mapping-first search working in MetaSuggester
- ✅ Automatic cache updates after live searches
- ✅ Cache statistics tracking and reporting
- ✅ CLI cache management commands functional
- ✅ DNB import with progress information working

### ✅ CRITICAL: Pipeline Development Requirements (IMPLEMENTED)
**All pipeline changes MUST be usable by both CLI and GUI interfaces:**
- ✅ **Shared Logic**: All pipeline functionality uses `src/utils/pipeline_utils.py` 
- ✅ **Task Selection**: Both CLI (--initial-task, --final-task) and GUI (dropdown selections) support prompt task selection
- ✅ **Configuration Compatibility**: Pipeline configurations work identically in both interfaces
- ✅ **Parameter Consistency**: All pipeline parameters supported uniformly across CLI and GUI
- ✅ **Result Format**: Identical output formats and data structures between CLI and GUI
- ✅ **Stream Callback**: Unified streaming feedback system adapted for both interfaces

**Implemented Prompt Selection Features:**
- **CLI**: `--initial-task` and `--final-task` parameters for pipeline command
  ```bash
  python alima_cli.py pipeline --initial-task "initialisation" --final-task "rephrase"
  ```
- **GUI**: Task selection dropdowns in pipeline configuration dialog  
  - Keywords step: "initialisation", "keywords", "rephrase"
  - Verification step: "keywords", "rephrase", "keywords_chunked"
- **Shared**: `execute_complete_pipeline()` function with `initial_task` and `final_task` parameters
- **Both interfaces** can now use all available prompt tasks for flexible analysis workflows

### Vision
- Restructure Code, many functions and logics is distributed among utlis, core, suggestors
- Keep Claude.MD compact, don't spam it with short-term informations about minoar implementation bugs, that occur during development
- Long-term goals and architectural directions
- Maintain unified pipeline architecture ensuring CLI and GUI feature parity
- All future pipeline enhancements must implement both interfaces simultaneously
```

### 1. Code Analysis and Development

**Python Code Generation**: I can generate clean, efficient Python code for ALIMA's core components:
- PyQt6 UI components and event handling
- LLM integration and API management
- Database operations and caching mechanisms
- Search algorithms and result processing
- Multi-threaded operations for background tasks

**Code Review and Optimization**: I analyze existing code to suggest improvements for:
- Performance optimization
- Code readability and maintainability
- Error handling and robustness
- Threading and asynchronous operations
- Memory management and resource usage

**Architecture Guidance**: I help design scalable and maintainable solutions:
- MVC pattern implementation
- Service layer architecture
- Plugin system for different LLM providers
- Configuration management
- Testing strategies

### 2. ALIMA-Specific Expertise

**Library Science Integration**: I understand the specialized requirements of:
- GND (Gemeinsame Normdatei) keyword management
- Bibliographic metadata processing
- Library catalog integration (Lobid, SWB, local catalogs)
- DDC classification systems
- SOAP-based library services

**LLM Integration**: I provide guidance on:
- Multi-provider LLM support (Ollama, Gemini, OpenAI, Anthropic)
- Prompt engineering and optimization
- Stream processing and real-time output
- Error handling for different providers
- Model selection and configuration

**Search and Indexing**: I help with:
- Keyword extraction and matching algorithms
- Search result ranking and relevance
- Database caching strategies
- Multi-source search aggregation
- Result deduplication and merging

### 3. GUI Development and User Experience

**PyQt6 Interface Design**: I assist with:
- Modern UI layout and styling
- Signal/slot architecture
- Custom widgets and components
- Threading for non-blocking operations
- Responsive design patterns

**User Workflow Optimization**: I help improve:
- Multi-step analysis workflows
- Progress tracking and feedback
- Error reporting and recovery
- Data import/export functionality
- Configuration management

### 4. CLI and Automation

**Command Line Interface**: I support:
- Argument parsing and validation
- JSON-based state management
- Batch processing capabilities
- Pipeline integration
- Error handling and logging

**Workflow Automation**: I help implement:
- Multi-stage analysis pipelines
- Resume/restart functionality
- Progress tracking and reporting
- Output formatting and export
- Integration with external tools

## ALIMA System Architecture Understanding

### Core Components

**1. Core Services**
- `AlimaManager`: Central coordination service
- `LlmService`: Multi-provider LLM integration
- `PromptService`: Template and prompt management
- `CacheManager`: SQLite-based result caching
- `SearchEngine`: Multi-source search coordination

**2. UI Components**
- `MainWindow`: Central application window
- `AbstractTab`: Text analysis interface
- `SearchTab`: GND keyword search interface
- `AnalysisReviewTab`: Analysis workflow review
- `UBSearchTab`: University library search
- `CrossrefTab`: DOI metadata lookup

**3. Search Integration**
- `MetaSuggester`: Unified search interface
- `LobidSuggester`: German National Library API
- `SWBSuggester`: Southwest German Library Network
- `CatalogSuggester`: Local library catalog integration

**4. Data Models**
- `AbstractData`: Text and keyword containers
- `AnalysisResult`: Analysis output structure
- `TaskState`: Workflow state management
- `KeywordAnalysisState`: Multi-step analysis tracking

### Key Workflows

**1. Keyword Analysis Pipeline**
```
Text Input → Initial Keyword Extraction → GND Search → Result Filtering → Final Analysis
```

**2. GUI Search Workflow**
```
User Input → SearchWorker Thread → MetaSuggester → Cache Check → API Calls → Result Display
```

**3. CLI Analysis Workflow**
```
Arguments → AlimaManager → LLM Analysis → Result Processing → JSON Export
```

## Best Practices I Follow

### 1. Code Quality
- **Type Hints**: Full type annotation support
- **Error Handling**: Comprehensive exception management
- **Logging**: Structured logging throughout the application
- **Documentation**: Clear docstrings and comments
- **Testing**: Unit tests for critical components

### 2. Performance
- **Caching**: SQLite-based result caching
- **Threading**: Non-blocking UI operations
- **Memory Management**: Efficient data structures
- **Database Optimization**: Indexed queries and connection pooling

### 3. User Experience
- **Responsive UI**: Non-blocking operations
- **Progress Feedback**: Real-time status updates
- **Error Recovery**: Graceful error handling
- **Data Persistence**: State saving and restoration

## How to Best Utilize My Support

### For Development Tasks
- **Be Specific**: Include exact error messages, code snippets, and desired behavior
- **Provide Context**: Explain the purpose and surrounding logic
- **Show Examples**: Include sample input/output data
- **Iterative Approach**: Work through complex problems step by step

### For Architecture Decisions
- **Describe Requirements**: Explain functional and non-functional requirements
- **Consider Constraints**: Mention performance, scalability, or compatibility constraints
- **Discuss Trade-offs**: I'll help evaluate different approaches

### For Debugging
- **Include Logs**: Provide relevant log output and error traces
- **Describe Reproduction**: Explain how to reproduce the issue
- **Share Environment**: Include Python version, library versions, and OS details

## Recent Enhancements I've Helped Implement

### 1. Enhanced Search Functionality
- Fixed `SearchEngine` signal emission for proper result handling
- Implemented `SearchWorker` thread safety for GUI operations
- Added comprehensive result processing and display

### 2. UI Improvements
- Enhanced `AbstractTab` with splitter layout and result history
- Implemented larger fonts and improved readability
- Added dynamic input area resizing during LLM streaming

### 3. Analysis Review System
- Created `AnalysisReviewTab` for comprehensive analysis review
- Implemented JSON-based analysis import/export
- Added workflow step navigation and data transfer between tabs

### 4. Cache Management
- Fixed `CacheManager` connection handling
- Improved SQLite connection management
- Enhanced error handling and logging

### 5. CLI Enhancements
- Fixed stream callback functionality
- Improved error handling and user feedback
- Enhanced JSON export capabilities

## Recent Major Enhancement: Pipeline Architecture (IMPLEMENTED)

### 🚀 **Vertical Pipeline UI with Live Streaming - COMPLETED**
**Status**: ✅ Fully implemented and integrated with real-time feedback

**New Pipeline Tab Features:**
- **Chat-like vertical workflow**: Pipeline steps displayed as conversation flow
- **Auto-Pipeline functionality**: One-click processing through entire workflow
- **Visual step indicators**: Real-time status (Pending ▷, Running ▶, Completed ✓, Error ✗)
- **Integrated input methods**: DOI, Image, PDF, Text in unified interface
- **Live result display**: Each step shows results immediately
- **🆕 Live Token Streaming**: Real-time LLM token display during generation
- **🆕 Continuous Feedback Window**: Gemeinsames stetig sich füllendes Textfenster

**Live Streaming System:**
- **PipelineStreamWidget**: Console-style dark theme with live updates
- **Token-Level Streaming**: Individual LLM tokens displayed as generated
- **Color-Coded Messages**: Time-stamped output with level-based coloring
- **Auto-Scrolling**: Automatic scroll to latest messages
- **Pipeline Progress**: Visual progress bars and step timing
- **Stream Controls**: Pause, cancel, clear, save log functionality

**Architecture Based on Existing Logic:**
- **PipelineManager**: Extends `AlimaManager` functionality with streaming callbacks
- **Utilizes `KeywordAnalysisState`**: Complete workflow state management
- **Integrates CLI pipeline logic**: Proven analysis workflow in GUI
- **Signal/Slot communication**: Thread-safe UI updates with streaming support
- **Stream Callback System**: Real-time token forwarding from LLM to UI

## 📋 **PIPELINE ARCHITECTURE & TASK SYSTEM**

### Pipeline Flow & Step Configuration

**Complete Pipeline**: Input → Keywords → Search → Verification → Classification (optional)

**Prompt System Integration** (`prompts.json`):
- **Available Tasks**: `"keywords"`, `"keywords_chunked"`
- **Task Structure**: Each task contains multiple prompt variants with model preferences
- **Template Variables**: `{abstract}` (input text), `{keywords}` (search results)
- **German Library Focus**: Specialized prompts for GND (Gemeinsame Normdatei) workflow

### Detailed Step Documentation

**1. Input Step** (`step_id: "input"`):
- **Purpose**: Multi-source text acquisition and validation
- **LLM Required**: ❌ No - Pure text processing
- **Supported Sources**: Drag-n-drop files, clipboard paste, DOI resolution, URL extraction
- **Output**: Validated text ready for analysis
- **UI Component**: `UnifiedInputWidget` with drag-drop zone

**2. Keywords Step** (`step_id: "keywords"`):
- **Purpose**: Extract GND-compatible keywords from abstract text
- **LLM Required**: ✅ Yes
- **Task Reference**: `"keywords"` (from prompts.json)
- **Prompt Role**: German librarian extracting OGND keywords
- **Default Providers**: ollama (cogito:14b, cogito:32b, magistral:latest)
- **Template**: German text → keyword extraction → format: ANALYSE/Schlagworte/OGND Einträge
- **Output**: List of extracted keywords + GND classifications

**3. Search Step** (`step_id: "search"`):
- **Purpose**: Query GND database for keyword matches
- **LLM Required**: ❌ No - Uses search suggesters
- **Suggesters**: configurable `["lobid", "swb"]` (Lobid API, SWB catalog)
- **Input**: Keywords from step 2
- **Process**: Multi-threaded search across configured suggester services
- **Output**: GND search results with metadata (titles, IDs, descriptions)

**4. Verification Step** (`step_id: "verification"`):
- **Purpose**: Verify and refine keyword selection using search results
- **LLM Required**: ✅ Yes
- **Task Reference**: `"keywords"` (reused with search context)
- **Input**: Original abstract + search results as "zur Auswahl stehende GND-Schlagworte"
- **Process**: LLM analyzes original text against found GND entries
- **Output**: Verified, refined keyword list with confidence analysis

**5. Classification Step** (`step_id: "classification"`, OPTIONAL):
- **Purpose**: Assign DDC (Dewey Decimal Classification) categories
- **LLM Required**: ✅ Yes (if enabled)
- **Default State**: Disabled in standard pipeline config
- **Use Case**: Library science specific classification workflow
- **Output**: Classification codes and hierarchical categories

### Pipeline Configuration System

```python
# Default Configuration Structure with Streaming Support
PipelineConfig(
    auto_advance=True,              # Automatic progression through steps
    stop_on_error=True,             # Halt on first error
    save_intermediate_results=True, # Cache step outputs
    
    step_configs={
        "keywords": {
            "step_id": "keywords",
            "enabled": True,
            "provider": "ollama",       # gemini | ollama | openai | anthropic
            "model": "cogito:14b", # Provider-specific model name
            "temperature": 0.7,         # LLM creativity parameter
            "top_p": 0.1,              # LLM sampling parameter  
            "task": "abstract"          # Analyse text to free keywords
        },
        "verification": {
            "step_id": "verification", 
            "enabled": True,
            "provider": "ollama",
            "model": "cogito:32b", # different models for different tasks
            "temperature": 0.7,
            "top_p": 0.1,
            "task": "keywords"          # Different task/different prompt with gnd context
        },
        "classification": {
            "step_id": "classification",
            "enabled": False,           # Typically disabled
            "provider": "gemini",
            "model": "gemini-1.5-flash"
        }
    },
    
    search_suggesters=["lobid", "swb"]  # GND search service configuration
)

# Streaming Integration in Pipeline Manager
def stream_callback(token, step_id):
    """Forward LLM tokens to UI in real-time"""
    if self.stream_callback:
        self.stream_callback(token, step_id)

# UI Integration for Live Streaming
@pyqtSlot(str, str)
def on_llm_stream_token(self, token: str, step_id: str):
    """Handle streaming LLM tokens - Claude Generated"""
    if not self.stream_widget.is_streaming:
        self.stream_widget.start_llm_streaming(step_id)
    self.stream_widget.add_streaming_token(token, step_id)
```

### Prompt System Deep Dive

**File Location**: `prompts.json` in project root
**Structure**: Task-based organization with multiple prompt variants per task

**Example Task Configuration**:
```json
{
    "keywords": {
        "fields": ["prompt", "system", "temp", "p-value", "model", "seed"],
        "required": ["abstract", "keywords"],
        "prompts": [
            [
                "Du bist ein korrekter Bibliothekar...",  // German prompt template
                "Your role as an assistant involves...",   // System prompt for reasoning
                "0.25",                                    // Temperature override
                "0.1",                                     // Top-P override
                ["cogito:32b", "qwq:latest"],             // Preferred models
                "0"                                        // Seed for reproducibility
            ]
        ]
    }
}
```

**Task Resolution Process**:
1. Pipeline step references task by name (`"keywords"`)
2. `PromptService` loads task configuration from prompts.json
3. Model-specific prompts preferred, falls back to "default" variant
4. Template variables `{abstract}` and `{keywords}` substituted with actual content
5. LLM called with resolved prompt, system message, and parameters

### 🌐 **Global Status Bar - COMPLETED** 
**Status**: ✅ Fully implemented and active

**Features:**
- **Provider Information**: Current LLM provider and model display
- **Cache Statistics**: Live cache entries and size monitoring
- **Pipeline Progress**: Real-time pipeline step tracking
- **Color-coded indicators**: Status visualization (Green/Orange/Red/Blue)

### 🔄 **Automated Data Flow - COMPLETED**
**Status**: ✅ Implemented between AbstractTab and AnalysisReviewTab

**Features:**
- **Automatic transfer**: Analysis results flow to verification tab
- **Signal-based communication**: `analysis_completed` signal implementation
- **Immediate display**: Results appear in review tab upon completion

### 🗄️ **COMPLETED: Database Unification & Facts/Mappings Separation**

**Status**: ✅ Week 1 COMPLETED - Implementation successfully finished

**Vision**: Self-Learning Knowledge Database mit Facts/Mappings-Trennung
- **Von**: Separate Caches (`search_cache.db` + `dk_classifications.db`) + Web-APIs
- **Zu**: Unified Knowledge Database (`alima_knowledge.db`) + Progressive Learning System

#### Unified Database Schema

```sql
-- === FAKTEN-TABELLEN (Unveränderliche Wahrheiten) ===

-- 1. GND-Einträge (Facts only, keine Suchbegriffe)
CREATE TABLE gnd_entries (
    gnd_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    synonyms TEXT,
    ddcs TEXT,
    ppn TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. DK/RVK Klassifikationen (Facts only, keine Keywords)
CREATE TABLE classifications (
    code TEXT PRIMARY KEY,           -- "530.12" oder "Q175"
    type TEXT NOT NULL,              -- "DK" oder "RVK"
    title TEXT,                      -- Offizieller Titel der Klassifikation
    description TEXT,                
    parent_code TEXT,                -- Hierarchie (optional)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- === SUCH-MAPPING TABELLEN (Dynamische Zuordnungen) ===

-- 3. Search Mappings (Suchbegriff → gefundene Ergebnisse)
CREATE TABLE search_mappings (
    search_term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,   -- Für Fuzzy-Matching
    suggester_type TEXT NOT NULL,    -- "lobid", "swb", "catalog", "fuzzy"
    
    -- Gefundene GND-IDs (JSON Array)
    found_gnd_ids TEXT,              -- ["4010435-7", "4156984-3"]
    
    -- Gefundene Klassifikationen (JSON Array) 
    found_classifications TEXT,       -- [{"code":"530.12","type":"DK"}, {"code":"Q175","type":"RVK"}]
    
    result_count INTEGER DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (search_term, suggester_type)
);
```

#### Implementation Roadmap

**Phase 1: Database Unification (Week 1) ⚡ Critical**
1. **Unified Schema Migration**
   - Merge `search_cache.db` + `dk_classifications.db` → `alima_knowledge.db`
   - Migrate existing GND entries and search cache
   - Create new classification and search mapping tables

2. **UnifiedKnowledgeManager Class**
   - Replace `CacheManager` + `DKCacheManager` → `UnifiedKnowledgeManager`
   - Implement all CRUD operations for unified schema
   - Separate methods for Facts vs. Mappings

**Phase 2: Smart Search Integration (Week 2) 🔍 High**
3. **Mapping-basierte Suche**
   - GND-Suche prüft Mappings first
   - Live-Suche nur bei Mapping-Miss oder veralteten Mappings
   - Automatische Mapping-Updates nach jeder Live-Suche

4. **Local-First GND Search**
   - Priority: Local DB → Web APIs only for unknowns
   - Progressive cache filling through usage
   - Fuzzy search with mapping integration

**Phase 3: Testing & Optimization (Week 3) 🔧 Medium**
5. **Mapping-Qualität Tests**
   - Mapping-Freshness Logic
   - Fuzzy-Search Qualität mit verschiedenen Schwellenwerten
   - Performance-Tests: Mapping vs. Live-Search

6. **Pipeline Integration Tests**
   - Vollständige Pipeline-Runs mit Mapping-System
   - JSON-Output Deduplizierung 
   - Cache-Hit-Rate Monitoring

#### Expected Performance Impact

**Nach 50 Pipeline-Runs:**
- **GND-Mapping-Hits**: 60-70% Begriffe aus Mappings  
- **DK-Mapping-Hits**: 40-50% Klassifikationen aus Mappings
- **Web-Request-Reduktion**: 70-80% weniger API-Calls
- **Mapping-Qualität**: Umfassende Abdeckung häufiger Begriffe

**Architektur-Prinzip**: Saubere Trennung zwischen unveränderlichen Fakten und dynamischen Such-Zuordnungen für maximale Flexibilität und Performance.

## Future Development Areas

### 1. Pipeline Enhancements (Next Phase)
- **Batch Processing Pipeline**: Process multiple inputs simultaneously  
- **Pipeline Templates**: Save/load different workflow configurations
- **Advanced Configuration UI**: Graphical pipeline step configuration
- **Export Pipeline Results**: Complete pipeline output in various formats
- **Pipeline Resume/Restart**: Resume from failed steps with cached intermediate results

### 2. Streaming & Feedback Improvements  
- **🔄 Token Count Display**: Show processed token statistics in real-time
- **🔄 Streaming Performance Metrics**: Display tokens/second and ETA
- **🔄 Enhanced Error Streaming**: Stream error details with recovery suggestions
- **🔄 Multi-Step Streaming**: Parallel streaming from multiple LLM steps
- **🔄 Stream Export**: Export streaming logs with timestamps and metadata

### 3. Performance Optimization
- Implement connection pooling for database operations
- Add result pagination for large datasets
- Optimize memory usage for large text processing
- Pipeline step caching and resume functionality
- Streaming buffer optimization for high-frequency token updates

### 4. Enhanced User Experience
- **Drag & Drop Integration**: File drag-and-drop to pipeline steps
- **Keyboard Shortcuts**: Navigation and pipeline control
- **Step Editing**: Inline editing of intermediate results
- **Pipeline Visualization**: Animated progress and connections
- **Dark/Light Theme**: Theme support for streaming widget
- **Stream Search/Filter**: Search through streaming output history

### 5. Integration Improvements
- Add REST API for external integration
- Implement plugin system for custom suggesters
- Add support for additional metadata formats
- Pipeline webhook notifications
- Streaming API endpoints for external monitoring

## I'm Ready to Help!

Whether you're implementing new features, fixing bugs, optimizing performance, or planning architectural changes, I'm here to provide detailed, contextual assistance. I understand ALIMA's codebase, architecture, and requirements, and I'm ready to help you continue building this powerful tool for library science and information management.

Let's build something great together!

---

*This document serves as a comprehensive guide to my capabilities and understanding of the ALIMA project. Feel free to reference it when seeking assistance, and don't hesitate to ask for clarification or additional details on any aspect of the system.*
