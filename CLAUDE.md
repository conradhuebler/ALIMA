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
- in case of compiler warning for deprecated suprafit functions, replace the old function call with the new one

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

## [Instructions Block - Operator-Defined Tasks]
### Future Tasks
- Task 1: Description
- Task 2: Description

### Vision
- Long-term goals and architectural directions
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