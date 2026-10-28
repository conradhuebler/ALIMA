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

### Gemini ✅ PORTED (Oct 2026): war komplett tot
`google.generativeai` existiert nicht — die gepinnte, installierte SDK ist `google-genai` (Modul `google.genai`), eine andere API: `Client()` + `client.models.*` statt Modul-weitem `configure()`/`GenerativeModel()`. Jeder Init-Versuch war `ModuleNotFoundError`. Auf `google.genai` portiert (`_init_gemini`, `_generate_gemini`, `_generate_gemini_with_tools`, die Modell-Liste), dabei vier echte Funde mitbehoben: `self.clients["gemini"]` war hart kodiert statt des konfigurierten Instanznamens (derselbe Fehler wie bei Anthropic); ein Modellname ohne `gemini-`-Präfix wurde stillschweigend durch `gemini-1.5-flash` ersetzt — lief einfach auf dem falschen Modell, ohne Fehler; `hasattr(part, "function_call")` ist auf jedem Part-Objekt True (ein deklariertes, nur meist leeres Feld) und hätte jeden Text-Teil fälschlich als Tool-Aufruf behandelt — jetzt `is not None`; zwei parallele Aufrufe desselben Tools bekamen dieselbe `id` (`f"gemini_{name}"`), jetzt mit Index eindeutig. Bild-MIME-Erkennung mit Anthropic geteilt (`_detect_image_media_type`), Gemini akzeptiert kein GIF inline und konvertiert zu PNG. Tool-Schemas gehen über `parameters_json_schema` (nimmt das rohe JSON-Schema direkt, `parameters` verlangt Googles eigenen `Schema`-Typ). Der Einrichtungsassistent validierte einen Gemini-Schlüssel ebenfalls über den toten Import (`setup_utils.APIKeyValidator.validate_gemini`) — jeder gültige Schlüssel wurde als ungültig gemeldet; mitbehoben. Von einem Opus-Agenten gefunden, von mir am Code nachgerechnet vor dem Porting. Tests: `tests/test_gemini_generation.py`, `tests/test_gemini_key_validation.py`.

### Start: Anbieter werden beim ersten Gebrauch verbunden ✅ FIXED (Oct 2026)
`LlmService(lazy_initialization=True)` speicherte das Flag und ignorierte es: der Konstruktor verband jeden konfigurierten Anbieter, und allein die SDK-Importe (`google.genai` 1,5 s, `anthropic` 1,6 s, `openai` 0,75 s, `ollama` 0,5 s) kosteten rund 5 s GUI-Start (GUI und Webapp übergeben das Flag). Jetzt `_setup_clients`: lazy registriert nur Platzhalter, `clients` ist eine `_LazyClients`-Map, die beim ersten Indexzugriff verbindet (`in`/`keys()`/`peek` verbinden nicht; ein gescheiterter Anbieter wird wieder entfernt, wie im eifrigen Modus; Sperre gegen doppeltes Verbinden aus zwei Threads). Eifriger Modus (CLI, Standard) unverändert. ⚠️ Der erste Zugriff auf einen Anbieter zahlt seinen SDK-Import (≈1 bis 1,5 s) im aufrufenden Thread; beim GUI-Start passiert das im Hintergrund-Statuslauf. Tests: `tests/test_llm_lazy_clients.py`.

### Gemini-Tool-Pfad: Verlauf ✅ FIXED (Oct 2026)
`_generate_gemini_with_tools` schickte Tool-Ergebnisse mit `role="function"` (400 INVALID_ARGUMENT, „Role 'function' is not supported"), und der Aufruf des Modells ging nie zurück in den Verlauf, nur sein Text. Neu `_convert_messages_for_gemini` (Familie der anderen drei): Funktionsantworten in einem **user**-Turn, parallele Antworten als ein Content, `function_call`-Part des Assistenten wird zurückgespielt, `thought_signature` (Gemini 3) wird je Call-ID gemerkt und mitgegeben. Gemini und Anthropic laufen laut Operator (Oct 2026); die Tests laufen gegen das installierte `google-genai`, nicht gegen die API. Tests: `tests/test_gemini_generation.py`.

### Zwei kleinere Lücken im Provider-Dispatch ✅ FIXED (Oct 2026)
`_convert_messages_for_ollama` sendete Tool-Ergebnisse nur als `{"role": "tool",
"content": ...}`. Ollama hat kein `tool_call_id`-Konzept — `tool_name`
(`ollama._types.Message.tool_name`) ist das einzige Korrelationssignal für
einen parallelen Multi-Tool-Turn; `AgentLoop` setzt `"name"` bereits auf jedes
generische Tool-Result-Dict, der Konverter las es nur nie. Jetzt
`"tool_name": msg.get("name", "")`.

`cancel_generation`s serverseitiger Abbruch-Dispatch verglich `current_provider`
gegen literale Strings (`"openai"`/`"anthropic"`/...) statt gegen `self.clients`'
Instanznamen — derselbe Fehlertyp wie die Anthropic/Gemini-Bugs. Ein korrekter
Treffer hätte trotzdem nichts bewirkt: die installierten `anthropic`/`openai`/
`ollama`-Clients haben keine `.cancel()`-Methode (empirisch per `hasattr`
bestätigt), jeder Zweig warf `AttributeError`, leise gefangen. Entfernt statt
geflickt — der echte Abbruch läuft über `cancel_requested`/`stream_running`,
in jeder Provider-Streaming-Schleife geprüft, unabhängig vom toten Dispatch.
Tests: `tests/test_ollama_message_conversion.py`, `tests/test_cancel_generation.py`.

### OpenAI-Reasoning-Modelle im Tool-Pfad ✅ FIXED (Oct 2026)
`_generate_openai_compatible` (Text-Pfad) kennt seit einer früheren Reparatur die
OpenAI-/Reasoning-Modelle, die `temperature`/`top_p` auf Chat-Completions rundweg
ablehnen (`o1`/`o3`/`o4`/`gpt-5`*) — aber nur hinter einem `if image`-Gate, für
Vision-OCR-Kompatibilität gedacht. `_generate_openai_with_tools` (der
agentische/Tool-Pfad) hatte überhaupt keinen Guard: ein Reasoning-Modell über
AgentLoop/MCP-Tools scheiterte an jedem einzelnen Aufruf mit einem 400er. Beide
Methoden teilen sich jetzt `LlmService.STRICT_DEFAULT_SAMPLING_MODELS`; der
Tool-Pfad wendet ihn unconditional an (keine Bilder in diesem Pfad, das alte
`if image`-Gate hätte dort ohnehin nie gegriffen) — in beiden Params-Dicts
(Streaming + Non-Streaming), die unabhängig voneinander aufgebaut werden. Tests:
`tests/test_openai_tools_reasoning_models.py`.

### Anthropic ✅ FIXED (Oct 2026)
`anthropic` was never in `requirements.txt` and never installed — every init attempt hit `ImportError`, logged at `warning`, indistinguishable from "not configured". Now `anthropic==1.15.0`. Both generators also sent `temperature`/`top_p` unconditionally (Opus 4.7+ rejects both outright; Sonnet 5.5/Haiku 5.5 reject any non-default value) and the text path sent `seed` (no such field) and read `content[0].text`/`chunk.delta.text` unconditionally — a `ThinkingBlock`/`ThinkingDelta` has `.thinking`, not `.text`, and every current model runs adaptive thinking by default. All fixed in `_generate_anthropic`/`_generate_anthropic_with_tools`; vision also silently dropped the image (imported SDK classes that never existed under those names). Two more, found while actually running it: `self.clients` is keyed by the configured instance **name** (e.g. "claude"), not the provider_type `"anthropic"` — both generators hardcoded the literal and raised "client not initialized" for any instance not named exactly that; and every tool-calling turn 400'd ("unexpected tool_use_id ... Each tool_result block must have a corresponding tool_use block in the previous message") because the generic assistant-with-tool-calls shape (`{"content": str, "tool_calls": [...]}`, OpenAI's own wire format - see `_convert_messages_for_{ollama,openai}`) had no Anthropic converter at all: the `tool_use` block was silently dropped, so the id a tool_result referenced existed nowhere. Now `_convert_messages_for_anthropic` (same family as the other two) reconstructs it, and also merges a parallel turn's several `{"role": "tool", ...}` entries into one Anthropic user message (required; `AgentLoop` emits one per call, valid for OpenAI/Ollama but not Anthropic). Two more: chat streaming never showed Claude's reasoning (`thinking_callback` reached Ollama/OpenAI but not Anthropic - now wired, requests `thinking: {type: adaptive, display: summarized}` only when a caller actually wants it shown, iterates the raw stream instead of the text-only `.text_stream` convenience so thinking deltas aren't filtered out); and vision sent every image as `media_type: "image/jpeg"` unconditionally regardless of the real format - PNG is first in the file picker's own filter and the webcam path writes PNG, so this routinely mislabeled the bytes. `_anthropic_image_block()` detects the real format via Pillow and converts BMP/TIFF (offered by the same picker, not accepted by Anthropic at all) to PNG. Tests: `tests/test_anthropic_generation.py`.

### Provider Dispatch (live set)
- **Text path** (`generate_response`): dispatches via `supported_providers[p]['generator']` — 4 live generators: `_generate_gemini`, `_generate_anthropic`, `_generate_ollama_native`, `_generate_openai_compatible` (GitHub/Azure run as `openai_compatible`).
- **Tool path** (`generate_with_tools`): dispatches by `provider_type`, wrapped in `_retry_on_rate_limit`; 5 `_generate_*_with_tools` generators (+ text fallback).
- Shared scaffolding: `_convert_messages_for_{ollama,openai}`, `_retry_on_rate_limit`, `_apply_openai_think`, `_extract_reasoning`. Per-provider generators are genuinely provider-specific (no further safe dedup; superseded HTTP-Ollama/GitHub/Azure generators removed June 2026).

### Reasoning-Modelle über OpenAI-kompatible Endpunkte
- **Drei Feld-Dialekte für denselben Kanal**: vLLM/SGLang/DeepSeek senden `reasoning_content`, Ollamas `/v1` und OpenRouter senden `reasoning`, der native Ollama-Client `message.thinking` → nie ein Feld direkt lesen (`_extract_reasoning()` für die OpenAI-Pfade, `_field(msg, "thinking")` nativ).
- **`_field(obj, name)`** für alle Ollama-Payloads: der Client liefert je nach Aufruf und Release pydantic-Modelle oder Dicts.
- **Budget nativ**: `options["num_predict"]` (nicht `max_tokens`), Truncation meldet Ollama als `done_reason == "length"` — non-streaming am Response, streaming am letzten Chunk.
- **Zwei Think-Dialekte**: `extra_body.chat_template_kwargs.enable_thinking` erreicht vLLM, `reasoning_effort` erreicht Ollama; `_apply_openai_think` sendet beide. Nur `reasoning_effort="none"` schaltet den Kanal wirklich ab.
- **Reasoning-Tokens zählen gegen `max_tokens`** — ein Reasoning-Modell kann das Budget aufbrauchen, bevor die Antwort beginnt (`finish_reason="length"`, leerer Inhalt).
- **Budget anheben verschiebt die Grenze, garantiert aber nichts** — der Denkkanal wächst mit. Gemessen am echten `alima_v51:extraction` (deepseek-v4-flash:cloud): 4096 → 1 von 2 Läufen ohne Antwort, 8192 → 3 von 3 mit, 16384 → 1 von 3 ohne (56886 Zeichen Reasoning, 122 s), 32768 → 5 von 5 mit (7,5k–32k Zeichen Reasoning, 15–70 s pro Aufruf). `think=false` antwortet in 1,6–4,5 s ohne Reasoning.
- Nachmessen pro Modell: `scripts/probe_thinking.py --model P/M --workflow-step alima_v51:extraction` (Matrix think × max_tokens, meldet Antwort-/Reasoning-Länge und Stop-Grund).

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
