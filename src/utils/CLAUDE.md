# Utils - Configuration and Utility Services

## [Preserved Section - Permanent Documentation]

### Core Components
- `ConfigManager`: thread-safe singleton; JSON-persisted unified config + provider system. `reset()` for test isolation.
- `TextProcessor`: language detection, keyword extraction, text cleaning, stopword filtering.
- `model_capabilities`: auto-detection of model-specific chunking thresholds (15+ patterns; explicit > pattern > default).
- `repetition_detector`: LLM repetition-loop detection (char/n-gram/window) + parameter-variation suggestions.
- `smart_provider_selector`: provider/model resolution (explicit UI > task prefs > config defaults).
- `pdf_extractor` (P-η): PyPDF2 extraction + quality heuristic + optional Vision-LLM OCR fallback.
- `image_analyzer` (P-η): sync wrapper over `LlmService.generate_response(image=...)`.
- `exporters` / `report_renderer` (P-θ): JSON/CSV/TeX/MARC writers + `load_state`; Jinja2 LaTeX reports (delimiters `(((  )))` / `((* *))`).
- `batch_processor`: batch engine over `PipelineStepExecutor` (DOI/PDF/TXT/IMG/URL; resume via `BatchState`).
- `pipeline_config_parser` / `pipeline_config_builder`: single source of truth for CLI/GUI pipeline param parsing + validation.

### Pipeline modules
Split out of the former `pipeline_utils.py` god-module; all re-exported from `pipeline_utils` via a facade, so `from …pipeline_utils import X` keeps working:
- `pipeline_utils.py`: `PipelineStepExecutor` (shared CLI/GUI/Webapp step logic) + classic-step helpers (`_emit_classic_*`, `_run_classic_step`).
- `pipeline_input.py`: `execute_input_extraction` (PDF/image/text/OCR).
- `gnd_keyword_utils.py`: GND-pool verification + keyword/RVK canonicalisation (leaf).
- `pipeline_text_utils.py`: pure text/display/title helpers (leaf, shared by executor + formatter).
- `pipeline_formatters.py`: `PipelineResultFormatter`.
- `pipeline_persistence.py`: `PipelineJsonManager`, `export_analysis_state_to_file`, `AnalysisPersistence`.
- `chunking.py`: `split_into_equal_chunks` (shared with agentic `llm_agent_step`).

### Configuration
- Location: `~/.config/alima/config.json` (unified JSON; legacy migration complete).
- Sections (dataclasses): `AlimaConfig`, `DatabaseConfig`, `CatalogConfig`, `PromptConfig`, `SystemConfig`, `UIConfig`, `UnifiedProviderConfig`.
- Providers: Ollama, OpenAI-compatible, Gemini, Anthropic; multi-host, priority ordering, task-specific preferences; environment-variable overrides.

## [Variable Section - Short-term Information]

### Known state — API keys in plaintext
`~/.config/alima/config.json` stores provider API keys unencrypted. Encryption/keyring is a separate (security, platform-dependent) package — recorded here so it is not mistaken for an oversight.

### WIP: DK Classification Splitting
Split the DK list into equal halves for parallel LLM classification, merge with dedup (top 15). `execute_dk_classification_split()` in `pipeline_utils.py`, gated by `enable_dk_splitting` + `dk_split_threshold`. Details: `docs/dk_classification_splitting.md`.

### Known Issues
- **Keyword parser robustness**: LLM sometimes emits comma- instead of pipe-separated keywords; fallback handles both. Improvement: tighten prompt format instructions across models.
- **Non-matched keywords for DK search**: keywords without GND-cache match are passed to DK catalog search as plain keywords. Investigate cache staleness / incomplete GND coverage.

### Current Prompt Templates
`abstract_analysis`, `results_verification`, `concept_extraction`, `ub_search`, `classification`.

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **Configuration UI**: graphical editor for complex settings
2. **Template Editor**: advanced prompt editor with syntax highlighting
3. **Profile Management**: multiple configuration profiles per use case
4. **Cloud Sync**: configuration synchronization across installations

### Vision
- Comprehensive configuration ecosystem for all ALIMA features, usable by non-technical users.
- Advanced deployment scenarios (Docker, cloud, enterprise); real-time config updates without restart.
