# CLAUDE: Your AI Assistant for the ALIMA Project

As Claude, your AI assistant, I'm here to provide comprehensive support for the ALIMA project and its ongoing development. My goal is to help you efficiently with programming, debugging, feature implementation, documentation, and architectural decisions.

## How I Support the ALIMA Project

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

## Future Development Areas

### 1. Performance Optimization
- Implement connection pooling for database operations
- Add result pagination for large datasets
- Optimize memory usage for large text processing

### 2. Feature Enhancements
- Add batch processing capabilities
- Implement advanced search filters
- Add support for additional LLM providers

### 3. User Experience
- Implement keyboard shortcuts
- Add drag-and-drop file support
- Enhance progress tracking and cancellation

### 4. Integration Improvements
- Add REST API for external integration
- Implement plugin system for custom suggesters
- Add support for additional metadata formats

## I'm Ready to Help!

Whether you're implementing new features, fixing bugs, optimizing performance, or planning architectural changes, I'm here to provide detailed, contextual assistance. I understand ALIMA's codebase, architecture, and requirements, and I'm ready to help you continue building this powerful tool for library science and information management.

Let's build something great together!

---

*This document serves as a comprehensive guide to my capabilities and understanding of the ALIMA project. Feel free to reference it when seeking assistance, and don't hesitate to ask for clarification or additional details on any aspect of the system.*