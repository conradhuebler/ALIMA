# ALIMA AI Changelog

## 2025

### Unified Database Configuration (November 2025)
- Eliminated duplicate `SystemConfig.database_path` + `DatabaseConfig.sqlite_path` → single source of truth
- Implemented OS-specific default paths (Windows, macOS, Linux) via `get_default_db_path()`
- Singleton pattern for UnifiedKnowledgeManager with thread-safe `__new__()` override
- Automatic backward compatibility migration for old configs
- All 12 UnifiedKnowledgeManager instantiations now use singleton automatically

### K10+/WinIBW Catalog Export (October 2025)
- Direct export in K10+/WinIBW format for seamless catalog integration
- GUI: New "K10+ Export" Tab with Copy-Button
- CLI: `--format k10plus` for direct Copy-Paste
- Configuration: K10PLUS_KEYWORD_TAG, K10PLUS_CLASSIFICATION_TAG

### DK Classification Transparency (October 2025)
- Automatic display of which catalog titles led to each DK classification
- GUI: PipelineStreamWidget shows sample titles during DK search, AnalysisReviewTab with color coding
- CLI: show-protocol with DK titles in detailed/compact/k10plus format

### Protocol Display CLI Command (October 2025)
- `show-protocol` command for displaying pipeline results from JSON files
- Three modes: `--format detailed` (readable), `--format compact` (CSV), `--format k10plus` (catalog export)

### Batch Processing System (August 2025)
- Complete batch processing engine using PipelineManager for full pipeline execution
- ALL Source Types Supported: DOI (via doi_resolver), PDF (PyPDF2 + LLM-OCR fallback), TXT, IMG (vision model), URL (BeautifulSoup4)
- Batch Review UI: Toggle mode for batch overview vs. detail view, table with Status/Source/Keywords/Date/Actions
- Continue-on-error vs. stop-on-error modes with detailed error reporting
- Resume functionality for interrupted batches via JSON persistence
- Pipeline configuration inheritance from global settings

### Unified Logging System (August 2025)
- Central logging infrastructure with 4-level verbosity system (0=Quiet, 1=Normal, 2=Debug, 3=Verbose)
- CLI: `--log-level` argument (0-3, default=1)
- GUI: Uses level 1 (Normal) by default
- Setup function: `setup_logging(level)` with automatic third-party suppression
- Result output respecting quiet mode: `print_result()` function

### Three-Mode CLI System (July 2025)
- Smart Mode: Uses task preferences from config.json automatically
- Advanced Mode: Manual provider|model override with `|` separator
- Expert Mode: Full parameter control (temperature, top-p, seed)

### Vertical Pipeline UI (June 2025)
- Chat-like vertical workflow with 5 pipeline steps
- Visual status indicators: ▷ (Pending), ▶ (Running), ✓ (Completed), ✗ (Error)
- Auto-Pipeline button for one-click complete analysis
- Integrated input tabs (DOI, Image, PDF, Text) in first step
- Real-time result display in each step
- Direct integration with PipelineManager for workflow orchestration

### Global Status Bar (June 2025)
- Unified provider information display across all tabs
- Real-time cache statistics (entries count, database size)
- Pipeline progress tracking with color-coded status
- Auto-updating every 5 seconds for live monitoring
- Integration with LlmService and CacheManager

### Pipeline Manager (May 2025)
- Orchestrates complete ALIMA workflow using existing AlimaManager logic
- 5-step pipeline: Input → Keywords → Search → Verification → Classification
- Uses proven `KeywordAnalysisState` for data management
- UI callback system for real-time progress updates
- Auto-advance functionality for seamless workflow
- Refactored to use shared `PipelineStepExecutor` from utils

### Automated Data Flow (May 2025)
- AbstractTab automatically sends results to AnalysisReviewTab
- New `analysis_completed` signal in AbstractTab
- `receive_analysis_data()` method in AnalysisReviewTab
- Seamless workflow progression without manual data transfer
