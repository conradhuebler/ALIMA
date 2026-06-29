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
- `TableWidget`, `Widgets`, `Styles`: reusable display/components/theming

### Design Patterns
- Signal/slot, thread-safe UI updates; QThread for non-blocking LLM/search/IO with streaming + cancellation.
- Responsive `QSplitter` layouts with dynamic resizing during streaming.

## [Variable Section - Short-term Information]

_(no open UI-specific items; operator tab-restructuring intents are tracked separately)_

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
