# LLM - Large Language Model Integration Layer

## [Preserved Section - Permanent Documentation]

### LLM Architecture
The `src/llm/` directory provides a unified interface for integrating multiple Large Language Model providers:

**Core Components:**
- `LlmService`: Multi-provider LLM client with unified API
- `PromptService`: Template management and prompt configuration system

### Multi-Provider Support
**Supported Providers:**
- **Ollama**: Local LLM hosting with configurable URL/port
- **OpenAI**: GPT models via API
- **Anthropic**: Claude models via API  
- **Google Gemini**: Gemini models via API
- **Extensible Architecture**: Easy addition of new providers

**Key Features:**
- **Unified Interface**: Consistent API across all providers
- **Provider Abstraction**: Transparent switching between models
- **Configuration Management**: JSON-based provider and model configuration
- **Streaming Support**: Real-time text generation with PyQt signals
- **Error Handling**: Graceful fallback and comprehensive error reporting

### LlmService Technical Details
**Signal-Based Communication:**
- `text_received`: Real-time streaming text chunks
- `generation_finished`: Completion notifications
- `generation_error`: Error handling and reporting
- `generation_cancelled`: User cancellation support

**Thread Safety:**
- Thread-safe operations for GUI integration
- Non-blocking generation requests
- Proper resource cleanup and cancellation

**Configuration Management:**
- JSON-based configuration storage
- API key management with secure storage
- Per-provider settings (URLs, models, parameters)
- Dynamic provider initialization

### PromptService Features
**Template System:**
- JSON-based prompt templates
- Task-specific prompt configurations
- Model-specific prompt optimization
- Variable substitution and formatting

**Model Management:**
- Task-to-model mapping
- Provider-specific model selection
- Model capability tracking
- Performance optimization hints

## [Variable Section - Short-term Information]

### Recent Enhancements (Claude Generated)
1. **Enhanced Signal System**: Improved PyQt signal handling for real-time streaming
2. **Provider Stability**: Better error handling and connection management
3. **Configuration Management**: Enhanced JSON config loading and validation
4. **Threading Improvements**: Better thread safety and resource management
5. **✅ Lazy Initialization System**: Implemented comprehensive lazy loading to prevent GUI startup blocking
6. **✅ Ping Test Implementation**: Added simple server reachability test before full connection attempts

### Current Provider Status
- **Ollama**: Primary local provider, stable operation
- **External APIs**: Configured based on available API keys
- **Fallback Strategy**: Automatic provider selection based on availability

### Development Notes
- All new LLM functions marked as "Claude Generated"
- Comprehensive logging for debugging provider issues
- Type hints maintained for better IDE integration
- **Lazy Loading Performance**: GUI startup reduced from 30+ seconds to 0.13 seconds with lazy provider initialization
- **Ping Test Strategy**: Socket connection test (0.7ms) with ping fallback for better connection reliability

### Configuration Structure
```json
{
  "task_name": {
    "models": ["model1", "model2"],
    "prompt_template": "...",
    "parameters": {...}
  }
}
```

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **ADD - Model Caching**: Implement intelligent model loading and caching
2. **ADD - Performance Monitoring**: Add response time and quality metrics
3. **ADD - Advanced Streaming**: Support for structured streaming (JSON, markdown)
4. **ADD - Provider Health Checks**: Automatic provider availability monitoring

### Vision
- Establish ALIMA as provider-agnostic platform supporting all major LLMs
- Implement intelligent model selection based on task requirements
- Provide seamless fallback and load balancing across providers
- Support for specialized models (embeddings, classification, generation)
- Integration with emerging LLM standards and protocols