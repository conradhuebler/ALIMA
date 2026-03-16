# Core - Central Business Logic and Data Management

## [Preserved Section - Permanent Documentation]

### Core Architecture
The `src/core/` directory contains the fundamental business logic and data management components of ALIMA:

**Primary Components:**
- `AlimaManager`: Central orchestration service coordinating LLM analysis workflows
- `🆕 PipelineManager`: Complete pipeline orchestration extending AlimaManager functionality
- `SearchEngine`: Multi-source search coordination with signal-based communication
- `CacheManager`: SQLite-based caching system with real-time statistics
- `DataModels`: Core data structures (AbstractData, AnalysisResult, TaskState, KeywordAnalysisState)
- `ProcessingUtils`: Text processing and keyword extraction utilities

**Suggester System:**
- Located in `suggesters/` subdirectory
- Implements plugin-like architecture for different search providers
- Includes Lobid, SWB, local catalog, and meta-suggester implementations

**Key Design Patterns:**
- Signal/slot architecture for asynchronous communication
- Service layer pattern for business logic separation
- Cache-aside pattern for performance optimization
- Plugin pattern for extensible suggester system

### Technical Specifications
- **Threading**: Extensive use of QThread for non-blocking operations
- **Database**: SQLite with prepared statements and connection pooling
- **Network**: QNetworkAccessManager for HTTP/HTTPS requests
- **Logging**: Structured logging with configurable levels
- **Error Handling**: Comprehensive exception management with graceful degradation

### Integration Points
- **LLM Services**: Interfaces with `src/llm/` for AI-powered analysis
- **UI Components**: Provides data and services to `src/ui/` layer
- **Configuration**: Uses `src/utils/config.py` for settings management
- **External APIs**: Integrates with Lobid, SWB, Crossref, and local catalogs

## [Variable Section - Short-term Information]

### Recent Improvements (Claude Generated)
1. **Enhanced SearchEngine Signal System**: Fixed signal emission for proper GUI integration
2. **Thread Safety Improvements**: Enhanced SearchWorker implementation for stable operations
3. **Cache Connection Management**: Improved SQLite connection handling and error recovery
4. **Processing Utils Optimization**: Enhanced keyword extraction and matching algorithms
5. **🚀 MAJOR: Pipeline Manager Implementation**: Complete pipeline orchestration system
6. **Cache Statistics Enhancement**: Added `get_cache_stats()` method for real-time monitoring

### Current Issues
- No critical issues reported

### Development Notes
- All new functions marked as "Claude Generated" for traceability
- Comprehensive error handling implemented across all components
- Type hints maintained throughout the codebase

### WIP: Iterative GND Search
- **Missing Concept Extraction**: Parse `<missing_list>` from LLM responses (prompt already supports this!)
- **Fallback Search**: GND search for missing concepts with hierarchy support
- **Iteration Control**: Max iterations + self-consistency convergence detection
- **UI Integration**: Manual trigger button in pipeline config, iteration history display in review tab
- **Implementation**: `extract_missing_concepts_from_response()` in processing_utils.py, `execute_fallback_gnd_search()` + `execute_iterative_keyword_refinement()` in pipeline_utils.py
- **Documentation**: See `docs/iterative_gnd_search.md` for complete architecture and implementation plan

### ✅ ADDED: Agentic Workflow with MCP Tools
- **Agent System**: `src/core/agents/` - BaseAgent with self-reflection, SearchAgent, KeywordAgent, ClassificationAgent, ValidationAgent, MetaAgent
- **MCP Tool Layer**: `src/mcp/` - 16 tools wrapping DB (UnifiedKnowledgeManager), web services (suggesters), and pipeline results
- **Agent Loop**: `src/core/agent_loop.py` - Provider-agnostic tool-calling loop with safety features (max iterations, diminishing-returns detection)
- **Tool-Calling in LlmService**: `generate_with_tools()` supporting Ollama, OpenAI, Anthropic, Gemini native tool-calling + text fallback
- **Pipeline Integration**: `PipelineConfig.enable_agentic_mode`, CLI `--agentic`, GUI checkbox
- **Pipeline Results as Tools**: `list_pipeline_results`, `get_pipeline_result` etc. allow agents to access/discuss saved results
- **WARNING**: Experimental feature with 3x token usage increase - opt-in only

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **WIP - Pipeline Enhancements**: Batch processing, templates, configuration UI
2. **ADDED - Pipeline Step Caching**: Cache intermediate results for resume functionality  
3. **ADD - Pipeline Webhooks**: External system notifications on step completion
4. **Performance Optimization**: Implement connection pooling for database operations
5. **Result Pagination**: Add support for large dataset handling
6. **Memory Optimization**: Optimize memory usage for large text processing

### Recently ADDED Features
1. **✅ PipelineManager (`pipeline_manager.py`)**: 
   - Orchestrates complete ALIMA workflow using existing AlimaManager logic
   - 5-step pipeline: Input → Keywords → Search → Verification → Classification
   - Uses proven `KeywordAnalysisState` for data management
   - UI callback system for real-time progress updates
   - Auto-advance functionality for seamless workflow
   - **🔄 REFACTORED**: Now uses shared `PipelineStepExecutor` from utils

2. **✅ Enhanced CacheManager**:
   - `get_cache_stats()` method for real-time cache monitoring
   - Statistics include entry count, database size, file path
   - Integration with global status bar for live updates

3. **✅ Shared Pipeline Logic Integration**:
   - PipelineManager refactored to use `PipelineStepExecutor` from utils
   - Eliminates code duplication with CLI implementation
   - Added JSON save/resume functionality via `PipelineJsonManager`
   - `resume_pipeline_from_state()` method for continuing interrupted workflows

### ✅ PRODUCTION STATUS - Pipeline Architecture Complete

**Technical Implementation:**
- **PipelineManager**: Now uses shared `PipelineStepExecutor` eliminating ~150 lines of duplication
- **Parameter Handling**: Whitelist filtering ensures only valid AlimaManager parameters are passed
- **Stream Callback Adapter**: Converts GUI callbacks (token, step_id) to AlimaManager format (token)
- **JSON Persistence**: Complete save/resume functionality for interrupted workflows

**Verified Functionality:**
- ✅ All 5 pipeline steps executing correctly (input → initialisation → search → keywords → classification)
- ✅ Real-time streaming feedback working in GUI
- ✅ Parameter conflicts resolved (provider, model, temperature, step_id, enabled)
- ✅ Final keywords displaying correctly in pipeline tab
- ✅ CLI and GUI producing identical results using shared logic

### Vision
- Establish core as the stable foundation for ALIMA's extensibility
- Maintain clean separation of concerns between data, business logic, and presentation
- Ensure scalability for handling large-scale library metadata processing
- Provide robust error handling and recovery mechanisms for production use