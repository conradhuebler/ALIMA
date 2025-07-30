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

### Vision
- Establish comprehensive configuration ecosystem supporting all ALIMA features
- Provide intuitive configuration management for non-technical users
- Support for advanced deployment scenarios (Docker, cloud, enterprise)
- Integration with external configuration management systems (Ansible, Terraform)
- Real-time configuration updates without application restart