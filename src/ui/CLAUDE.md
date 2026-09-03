# UI - PyQt6 User Interface Layer

## [Preserved Section - Permanent Documentation]

### UI Architecture
`src/ui/` implements the PyQt6 GUI for ALIMA.

**Main Components:**
- `MainWindow`: central window, tab + menu management
- `PipelineTab`: vertical pipeline UI (primary interface) orchestrating the full workflow via `PipelineManager`; integrated input tabs (DOI/Image/PDF/Text), per-step result + live streaming, Auto-Pipeline button
- `GlobalStatusBar`: provider info + cache stats + pipeline progress (auto-refresh)
- `AbstractTab`: text analysis with chunking workflow + real-time streaming; auto-sends results to `AnalysisReviewTab` (`analysis_completed` signal)
- `SearchTab` (`find_keywords.py`): **the** unified search tab — GND keyword search + embedded UB catalog (DK/RVK) + pipeline post-processing, ONE shared search input. Reworked Aug 2026 (WP-K5): async + cancellable (`GndSearchWorker`), source checkboxes live from the plugin system (`refresh_sources`), Häufigkeit = `display_count` (count landmine), classification column. Results are tabs: GND-Schlagwörter / UB-Katalog (injected `UBCatalogTab`, `set_embedded` shares the input) / Pipeline-Mapping (hidden until pipeline data). `SearchTabUnified` (combo switcher) was deleted. External contract: ctor signature (+`ub_catalog_tab`), `update_data`/`update_search_field`/`display_search_results`/`refresh_styles`/`refresh_sources`, signal `selection_changed`.
- `AnalysisReviewTab`: analysis review + result management (JSON import/export, auto-receive)
- `DkAnalysisUnifiedTab`: unified DK-Zuordnung + DK-Statistik + UB-Suche (inherits `AbstractTab`)
- `ImageAnalysisTab`: Vision-LLM image analysis
- DOI lookup: via the DOI input-source plugins / `src/utils/doi_resolver.py` (the standalone Crossref tab + `crossref_worker.py` were removed)

**Supporting Components:**
- `PipelineConfigDialog` / `workflow_editor_dialog.py`: pipeline config + form editor for agentic v4 workflow YAML (ruamel round-trip, validates `load_workflow(strict=True)`, saves to `~/.config/alima/workflows/`; engine read-only)
- `batch_processing_dialog.py`: batch file/directory processing (QThread, progress, continue-on-error)
- `SettingsDialog` / `PromptEditorDialog`: configuration + prompt template editing
- `plugin_settings_tab.py` (`PluginSettingsTab`): category-grouped per-plugin config (search + input), auto-built from each plugin's `config_fields`; enable/primary/usage_hint + add/duplicate/remove. Single editor for provider/source config — **the Catalog tab + DOI System entries were removed** (folded into the `catalog` / DOI plugins; only the DOI `SystemConfig` fields are still derived on save — the `CatalogConfig` mirror is gone, WP P7). Replaced the checkbox-only `provider_selector.py` (deleted). Spec: [`docs/plugin_system.md`](../../docs/plugin_system.md).
- `dialogs/rules_dialog.py` (`RulesDialog`): manage the personal indexing rules — list/edit/enable/delete plus export+import (provenance preserved). Reached from Settings → Chat-Agent and from 📌 in the chat panel header. Store: `src/core/user_rules.py`.
- `chat_tools/rules.py`: `list_rules` / `propose_rule` / `set_rule_enabled` / `delete_rule`; the writing ones ask through the existing `ProposalGateway` and ignore `autonomous_pipeline` (a rule changes every *future* run, not the current one).
- `proposal_bar.py` (`ProposalBar`): the confirmation surface for every gateway proposal — a one-shot strip between chat log and input, hidden again once answered. The log block is a record only; ⚠️ never put accept/reject anchors back there — a custom scheme is dropped by QWebEngine, and an https one stays clickable forever.
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
0. ✅ **`SearchTab` überarbeitet** (Aug 4, WP-K5) — async + abbrechbar, Plugin-live,
   `display_count`, Klassifikations-Spalte; Klick-Test-Feedback eingearbeitet
   (Suchmodus-Label, Mapping nur im Pipeline-Modus, Stop-Button). Offener
   Polish-Punkt: Zeilenfarben sind Light-Mode-Hexes (Dark-Theme).
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
