# Utils - Configuration and Utility Services

## [Preserved Section - Permanent Documentation]

### Utils Architecture
The `src/utils/` directory provides essential configuration management and utility services for ALIMA:

**Core Components:**
- `Config`: Centralized configuration management with YAML persistence
- `TextProcessor`: Advanced text analysis and processing utilities

### Configuration Management System
**Config Class Features:**
- **Singleton Pattern**: Single instance ensuring consistent configuration access
- **Section-Based Organization**: Modular configuration with enum-defined sections
- **YAML Persistence**: Human-readable configuration files with automatic saving/loading
- **Type-Safe Configuration**: Dataclass-based configuration objects with validation
- **Environment Override Support**: Environment variable configuration overrides

**Configuration Sections:**
- `GeneralConfig`: Language, debug settings, log levels
- `AIConfig`: Multi-provider LLM configuration with API keys and models
- `SearchConfig`: Search parameters, thresholds, and provider settings
- `CacheConfig`: Database caching configuration and cleanup policies
- `UIConfig`: Interface preferences, themes, and window settings
- `ExportConfig`: Output format settings and file handling
- `PromptConfig`: LLM prompt templates with variable substitution

### Advanced Configuration Features
**AI Provider Management:**
- Multi-provider support (Gemini, OpenAI, Anthropic, Local)
- Provider-specific model lists and API endpoints
- Dynamic provider switching and configuration
- Secure API key storage and management

**Prompt Template System:**
- JSON-based template storage with variable substitution
- Task-specific prompt optimization
- Model-specific prompt variations
- Required variable validation and documentation

**Configuration Validation:**
- Comprehensive validation with error reporting
- Missing dependency detection
- Directory creation and permission checks
- Environment-specific configuration adjustments

### TextProcessor Capabilities
**Text Analysis Features:**
- **Language Detection**: Automatic language identification
- **Keyword Extraction**: Advanced keyword identification algorithms
- **Text Cleaning**: Unicode normalization and special character handling
- **Statistical Analysis**: Text metrics and content analysis
- **Stopword Filtering**: Multi-language stopword removal

**Processing Pipeline:**
- Multi-stage text processing with configurable steps
- Result caching for performance optimization
- Error handling and graceful degradation
- Support for various text formats and encodings

## [Variable Section - Short-term Information]

### Recent Enhancements (Claude Generated)
1. **Enhanced Config Loading**: Better YAML parsing and error handling
2. **Provider Configuration**: Improved AI provider management and validation
3. **Template System**: Enhanced prompt template loading and variable substitution
4. **Text Processing**: Optimized keyword extraction and language detection
5. **🚀 MAJOR: Pipeline Utils (`pipeline_utils.py`)**: Shared logic abstraction for CLI and GUI
   - **PipelineStepExecutor**: Unified pipeline step execution logic
   - **PipelineJsonManager**: JSON serialization/deserialization utilities
   - **PipelineResultFormatter**: Result formatting for display/prompts
   - **execute_complete_pipeline()**: End-to-end pipeline execution function

### Configuration Status
- **Config Location**: `~/.config/gnd-fetcher/config.yaml`
- **Default Values**: Comprehensive defaults for all configuration sections
- **Validation**: Active validation with detailed error reporting
- **Migration**: Automatic configuration migration and updates

### Development Notes
- All new utility functions marked as "Claude Generated"
- Comprehensive logging throughout configuration system
- Type hints maintained for better IDE integration
- Environment variable support for containerized deployments

### Current Prompt Templates
- `abstract_analysis`: Schlagwort extraction from abstracts
- `results_verification`: GND keyword quality verification
- `concept_extraction`: General concept identification
- `ub_search`: University library search optimization
- `classification`: DDC classification assignment

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **ADD - Configuration UI**: Graphical configuration editor for complex settings
2. **ADD - Template Editor**: Advanced template editor with syntax highlighting
3. **ADD - Profile Management**: Multiple configuration profiles for different use cases
4. **ADD - Cloud Sync**: Configuration synchronization across multiple installations
5. **ADDED - Pipeline Abstraction**: Shared CLI/GUI pipeline logic in `pipeline_utils.py`

### Recently ADDED Features
1. **✅ PipelineStepExecutor**: 
   - Unified execution logic for initialisation, search, and final keyword analysis
   - Consistent error handling and logging across CLI and GUI
   - Stream callback support for real-time feedback
   - Configurable parameters and model selection

2. **✅ PipelineJsonManager**:
   - Complete JSON serialization/deserialization for `TaskState` and `KeywordAnalysisState`
   - Set-to-list conversion for JSON compatibility
   - Save/load functionality with comprehensive error handling

3. **✅ Complete Pipeline Function**:
   - `execute_complete_pipeline()` runs full workflow from input to final keywords
   - Used by both CLI (`pipeline` command) and GUI (PipelineManager)
   - Eliminates code duplication between implementations

### ✅ PRODUCTION READY - Tested and Verified

**Pipeline Utils Functions:**
- **PipelineStepExecutor**: Successfully handles all 3 LLM steps (initialisation, keywords, classification)
- **PipelineJsonManager**: Save/resume functionality working for both CLI and GUI
- **Stream Callback Compatibility**: Adapts between GUI (token, step_id) and CLI (token) formats
- **Parameter Filtering**: Correctly filters config parameters for AlimaManager compatibility

**CLI Integration:**
```bash
# New unified pipeline command
python alima_cli.py pipeline --input-text "..." --initial-model "cogito:14b" --final-model "cogito:32b"

# Resume from saved state
python alima_cli.py pipeline --resume-from "results.json"

# List available models
python alima_cli.py list-models --ollama-host "http://server" --ollama-port 11434
```

**GUI Integration:**
- PipelineManager refactored to use PipelineStepExecutor
- All pipeline steps working with real-time streaming
- JSON save/resume capabilities added to GUI

### Vision
- Establish comprehensive configuration ecosystem supporting all ALIMA features
- Provide intuitive configuration management for non-technical users
- Support for advanced deployment scenarios (Docker, cloud, enterprise)
- Integration with external configuration management systems (Ansible, Terraform)
- Real-time configuration updates without application restart