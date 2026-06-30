# UI - PyQt6 User Interface Layer

## [Preserved Section - Permanent Documentation]

### UI Architecture
`src/ui/` implements the PyQt6 GUI for ALIMA.

**Main Components:**
- `MainWindow`: central window, tab + menu management
- `PipelineTab`: vertical pipeline UI (primary interface) orchestrating the full workflow via `PipelineManager`; integrated input tabs (DOI/Image/PDF/Text), per-step result + live streaming, Auto-Pipeline button
- `GlobalStatusBar`: provider info + cache stats + pipeline progress (auto-refresh)
- `AbstractTab`: text analysis with chunking workflow + real-time streaming; auto-sends results to `AnalysisReviewTab` (`analysis_completed` signal)
- `SearchTab` (`find_keywords.py`): GND keyword search/browse
- `AnalysisReviewTab`: analysis review + result management (JSON import/export, auto-receive)
- `DkAnalysisUnifiedTab`: unified DK-Zuordnung + DK-Statistik + UB-Suche (inherits `AbstractTab`)
- `ImageAnalysisTab`: Vision-LLM image analysis
- Crossref: DOI lookup via `src/core/crossref_worker.py` (the standalone Crossref tab was removed)

**Supporting Components:**
- `PipelineConfigDialog` / `workflow_editor_dialog.py`: pipeline config + form editor for agentic v4 workflow YAML (ruamel round-trip, validates `load_workflow(strict=True)`, saves to `~/.config/alima/workflows/`; engine read-only)
- `batch_processing_dialog.py`: batch file/directory processing (QThread, progress, continue-on-error)
- `SettingsDialog` / `PromptEditorDialog`: configuration + prompt template editing
- `TableWidget`, `Styles`: reusable display/theming (the unused `widgets.py` was removed June 2026)

### Design Patterns
- Signal/slot, thread-safe UI updates; QThread for non-blocking LLM/search/IO with streaming + cancellation.
- Responsive `QSplitter` layouts with dynamic resizing during streaming.

## [Variable Section - Short-term Information]

### F-5 god-file split (June 30)
- ✅ **`pipeline_chat_panel` 1923→698** — leaf widgets → `chat_input_widgets.py`; repetition bar → `repetition_warning_bar.py`; behavior → `PipelineLogMixin` / `ChatAgentMixin` / `BusEventMixin` (`_chat_panel_*.py`). Verbatim moves, reachable via MRO; `bus.subscribe` wiring stays in `__init__`.
- **Mixin pattern** (new here): for a single god-*class* with no embeddable sub-widgets, split methods into behavioral mixins — zero call-site changes, existing stub tests pass unchanged.
- Remaining F-5: `pipeline_tab` 2882, `main_window` 2650.

### Findings — `ProviderModelSelector` adoption (decide when working through cleanup)
Shared `provider_model_selector.ProviderModelSelector` is used by `unified_provider_tab`, `image_analysis_tab`, `pipeline_chat_panel`, `pipeline_tab`, and now `single_step_dialog` (migrated June 2026). The remaining builders of their own provider/model combos are **not** clean dedup — each has an *intentional* behavior difference, so adopting the selector would change UX. Decide deliberately (not as "cleanup"):
- **`comprehensive_settings_dialog`**: `load_providers` intentionally lists the 4 common providers (ollama/gemini/openai/anthropic) **plus** enabled ones; the selector shows **enabled-only**. Also has a separate `custom_model_input`. → adoption shrinks the provider list.
- **`abstract_tab`**: model combo is filled from a **pushed `self.available_models` cache** (not async pull); tracks `explicit_provider/model_selection` + `user_interaction_mode`; recommended-model and history-restore use synchronous `findText`+`setCurrentIndex` that breaks under the selector's **async** loading. → adoption is a re-architecture, not a dedup.
- **`pipeline_config_dialog`**: per-step provider/model grid (~26/50 combo refs) — large, out of scope.

Detection note: the selector's default detection service wraps `LlmService.get_available_models` (TTL-cached), so the **model list source is the same** — the differences above are about provider lists / load mechanics, not the underlying models.

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **Pipeline Configuration UI**: graphical configuration for pipeline steps/models
2. **Pipeline Templates**: save/load workflow configurations
3. **Batch Review Table**: enhanced table view for batch results
4. **Drag & Drop to Pipeline**: files dropped directly onto pipeline steps
5. **Pipeline Step Editing**: inline editing of intermediate results
6. **Keyboard Shortcuts**: comprehensive keyboard navigation
7. **Advanced Theming**: dark mode + customizable themes
8. **Enhanced Export**: pipeline results to PDF/Excel/etc.

### Vision
- Intuitive, efficient interface for library-science workflows.
- Accessibility, multi-monitor/varied-resolution support.
- Seamless integration across all analysis and search functions.
