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
- `_pipeline_dk_steps.py` / `_pipeline_rvk_scoring.py`: `DkStepsMixin` + `RvkScoringMixin` — the DK/RVK classification methods of `PipelineStepExecutor` (verbatim mixin extraction July 19; methods stay on the class via MRO, cross-calls über `self`). Seam for the geplante `(system, notation)`-Generalisierung.
  - ⚠️ **`_pipeline_rvk_scoring` ist zur Hälfte untestbar konstruiert**: 14 Funktionen liegen weiterhin in Methodenkörpern verschachtelt (Closure-gebunden, u.a. `_validate_code` 160 Z.). Die 5 **reinen** Helfer sind seit July 21 auf Modulebene und getestet — `_compact_rvk`/`_branch_key`/`_is_parent_like` (Hierarchie: entscheidet, ob eine Notation ihre Eltern verdrängt) + `_source_rank`/`_status_rank` (existierten byte-identisch doppelt). Neue reine Helfer **nicht** wieder verschachteln; ein Test prüft, dass jeder genau einmal definiert ist.
  - Getestet sind die Primitive, **nicht** die großen Entscheidungsmethoden (`_select_final_rvk_candidates` 209 Z., `_validate_catalog_rvk_candidates` 394 Z.).
- `pipeline_input.py`: `execute_input_extraction` (PDF/image/text/OCR).
- `gnd_keyword_utils.py`: GND-pool verification + keyword/RVK canonicalisation (leaf).
- `pipeline_text_utils.py`: pure text/display/title helpers (leaf, shared by executor + formatter).
- `pipeline_formatters.py`: `PipelineResultFormatter`.
- `pipeline_persistence.py`: `PipelineJsonManager`, `export_analysis_state_to_file`, `AnalysisPersistence`.
- `chunking.py`: `split_into_equal_chunks` (shared with agentic `llm_agent_step`).

### Configuration
- Location: `~/.config/alima/config.json` (unified JSON; legacy migration complete).
- Sections (dataclasses): `AlimaConfig`, `DatabaseConfig`, `PromptConfig`, `SystemConfig`, `UIConfig`, `UnifiedProviderConfig`.
- **Plugin instances** (`AlimaConfig.plugins`, `PluginInstanceConfig`) are the *only* per-provider/source config — `CatalogConfig`/`SearchProviderConfig` deleted (WP P7). Read via `factory.primary_settings`, write via `set_primary_settings`. Spec: [`docs/plugin_system.md`](../../docs/plugin_system.md).
- Legacy `catalog_config`/`search_provider_config` JSON sections are one-way migration input (`plugin_migration.synthesize_search_instances`), dropped on next save. Absent keys are **omitted** so the plugin's `ConfigField` default applies — never written as `None`.
- The DOI `SystemConfig` fields remain a **derived mirror** (`derive_input_mirrors` on save).

### Input sources (`input_sources/`)
- `INPUT_SOURCE_REGISTRY` + `@register_input_source`; `execute_input_extraction` is a registry dispatcher (text/file/pdf/image byte-parity). `url_fetch` (extracted from `batch_processor`) + three separately-configurable DOI plugins (`doi_crossref`/`openalex`/`datacite`) wrapping `UnifiedResolver`.
- Providers: Ollama, OpenAI-compatible, Gemini, Anthropic; multi-host, priority ordering, task-specific preferences; environment-variable overrides.

## [Variable Section - Short-term Information]

### Known state — API keys in plaintext
`~/.config/alima/config.json` stores provider API keys unencrypted. Encryption/keyring is a separate (security, platform-dependent) package — recorded here so it is not mistaken for an oversight. Mitigation (July 6): plugin SECRET fields can be supplied per env var `ALIMA_PLUGIN_<INSTANCE_ID>_<KEY>` (runtime-only, never persisted); legacy mirror readers do not see env overrides.

### net_guard (July 6)
`net_guard.py` — two-posture URL validation: `check_operator_url`/`require_http_url` for operator-configured endpoints (scheme gate, intranet allowed) and `assert_public_http_url`/`fetch_guarded` (strict SSRF guard, redirect-per-hop, size cap) for runtime/LLM-supplied URLs (`url_fetch`, MCP `scrape_url`). Exceptions: `SystemConfig.url_fetch_allowlist`.

### Suggester relocation (July 6, completed July 19)
`lobid/swb/biblio/finc`-Suggester live in their plugin dirs (`src/core/search/providers/<name>/suggester.py`); the shared contract `base_suggester.py` moved to `src/core/search/` and **`src/utils/suggesters/` is gone** (July 19). `meta_suggester.py` retired July 8 (folded into `src/core/search/service.py`). Shared transport clients stay in `src/utils/clients/`; the finc clients are vendored in `providers/finc/` and their old `clients/finc_*` shim paths were **deleted** (July 19) — import the plugin paths directly. `biblio_client.py` god-file split July 21 (F-15): MARC/MAB-Parser → `_biblio_parsing.BiblioParsingMixin`, Transport-Reliability (Rate-Limit/Circuit-Breaker/Session) → `_biblio_transport.BiblioTransportMixin`, SOAP/Web-Request-Achse (`search`/`_search_attempt`/`get_title_details` + Web-Fallbacks) → `_biblio_soap.BiblioRequestMixin` (verbatim Mixins, via MRO; 2106→994 Z.). ⚠️ `logger` heißt hier `"biblio_extractor"` (nicht `__name__`) — in allen drei Mixins mit exakt diesem Namen neu angelegt, sonst wandern Log-Zeilen still in einen Fehl-Logger.

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
