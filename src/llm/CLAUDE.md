# LLM - Large Language Model Integration Layer

## [Preserved Section - Permanent Documentation]

### LLM Architecture
The `src/llm/` directory provides a unified interface for integrating multiple Large Language Model providers:

**Core Components:**
- `LlmService`: Multi-provider LLM client with unified API (Qt-signal-based streaming)
- `PromptService`: Template management and prompt configuration system

### Multi-Provider Support
**Supported Providers:** Ollama (local, configurable URL/port), OpenAI-compatible (incl. GitHub/Azure), Anthropic, Google Gemini. Extensible via the provider registry.

**Key Features:**
- Unified interface + transparent model switching
- JSON-based provider/model configuration
- Streaming via PyQt signals
- Graceful fallback + comprehensive error reporting

### LlmService Technical Details
- **Signals:** `text_received` (streaming chunks), `generation_finished`, `generation_error`, `generation_cancelled`.
- **Thread safety:** non-blocking generation, proper cleanup/cancellation; lazy provider init (avoids GUI-startup blocking); ping/socket reachability test before full connect.
- **Configuration:** JSON storage, per-provider settings (URLs, models, params), API-key management. ⚠️ keys stored in plaintext in `~/.config/alima/config.json` (see `src/utils/CLAUDE.md`).

### PromptService Features
- JSON prompt templates: task-specific + model-specific variants, variable substitution.
- Task-to-model mapping, provider-specific model selection, capability tracking.

## [Variable Section - Short-term Information]

### Provider Dispatch (live set)
- **Text path** (`generate_response`): dispatches via `supported_providers[p]['generator']` — 4 live generators: `_generate_gemini`, `_generate_anthropic`, `_generate_ollama_native`, `_generate_openai_compatible` (GitHub/Azure run as `openai_compatible`).
- **Tool path** (`generate_with_tools`): dispatches by `provider_type`, wrapped in `_retry_on_rate_limit`; 5 `_generate_*_with_tools` generators (+ text fallback).
- Shared scaffolding: `_convert_messages_for_{ollama,openai}`, `_retry_on_rate_limit`, `_apply_openai_think`, `_extract_reasoning`. Per-provider generators are genuinely provider-specific (no further safe dedup; superseded HTTP-Ollama/GitHub/Azure generators removed June 2026).

### Reasoning-Modelle über OpenAI-kompatible Endpunkte
- **Zwei Feld-Dialekte für denselben Kanal**: vLLM/SGLang/DeepSeek senden `reasoning_content`, Ollamas `/v1` und OpenRouter senden `reasoning` → immer `_extract_reasoning()` benutzen, nie ein Feld direkt.
- **Zwei Think-Dialekte**: `extra_body.chat_template_kwargs.enable_thinking` erreicht vLLM, `reasoning_effort` erreicht Ollama; `_apply_openai_think` sendet beide. Nur `reasoning_effort="none"` schaltet den Kanal wirklich ab.
- **Reasoning-Tokens zählen gegen `max_tokens`** — ein Reasoning-Modell kann das Budget aufbrauchen, bevor die Antwort beginnt (`finish_reason="length"`, leerer Inhalt).
- `_generate_ollama_native_with_tools` reicht `max_tokens` nicht als `num_predict` durch und liefert nie `StopReason.MAX_TOKENS`.

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **Model Caching**: intelligent model loading and caching
2. **Performance Monitoring**: response time and quality metrics
3. **Advanced Streaming**: structured streaming (JSON, markdown)
4. **Provider Health Checks**: automatic availability monitoring

### Vision
- Provider-agnostic platform supporting all major LLMs.
- Intelligent model selection by task; seamless fallback / load balancing.
- Support for specialized models (embeddings, classification) + emerging standards.
