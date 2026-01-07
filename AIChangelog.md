# ALIMA AI Changelog

## 2026

### WebApp Auto-Save & Recovery System (January 6, 2026)
- **Complete reliability upgrade** for long-running pipeline analyses in web interface
- **Auto-Save Infrastructure**: Incremental JSON saving after each pipeline step
  - Auto-save directory: `/tmp/alima_webapp_autosave/` with session-specific files
  - Metadata tracking: session_id, timestamp, last_step, status
  - Uses existing `PipelineJsonManager` for consistent serialization
- **Extended WebSocket Timeout**: Increased from 5 minutes to 30 minutes
  - Heartbeat mechanism: Sends heartbeat every 5 seconds to maintain connection
  - Prevents timeout during long DK searches (100+ keywords)
  - Frontend filters heartbeat messages (no console spam)
- **Recovery Mechanism**: Complete result restoration after connection loss
  - New API endpoint: `GET /api/session/{id}/recover`
  - Auto-detection of WebSocket errors (code 1006, 1011)
  - Recovery UI: Orange "🔄 Ergebnisse wiederherstellen" button with status messages
  - Full result reconstruction using shared `_extract_results_from_analysis_state()` helper
- **Auto-Cleanup**: Automatic deletion of old auto-save files (>24h) on webapp startup
- **Progress Enhancement**: DK search now shows percentage progress `[idx/total] (pct%)`
- **Code Quality**: DRY principle - shared result extraction logic between callback and recovery
- **Backward Compatibility**: Old sessions without auto-save continue to work
- **Files Modified**:
  - `src/webapp/app.py`: +4 functions, +1 endpoint, auto-save infrastructure
  - `src/webapp/static/index.html`: Recovery button + message span
  - `src/webapp/static/app.js`: +2 recovery functions, WebSocket handler enhancements
  - `src/utils/pipeline_utils.py`: Percentage display in DK search
  - `src/webapp/CLAUDE.md`: Documentation update

### DK Deduplication Statistics Display (January 2026)
- **Phase 2 Complete**: Comprehensive statistics visualization for DK classification deduplication
- **CLI Statistics Display**: New `format_dk_statistics()` in `show-protocol` detailed mode
  - Shows deduplication metrics: original→deduplicated count, duplicates removed, rate, token savings
  - Top 10 most frequent classifications with keyword provenance and title counts
  - Keyword coverage summary showing keywords→DK codes mapping
- **GUI Statistics Tab**: New "📊 DK-Statistik" tab in AnalysisReviewTab (index 9)
  - Deduplication Summary box with 5 key metrics
  - Top 10 table with rank, DK code, type, count, keywords, and color-coded confidence
  - Keyword Coverage table showing keyword→DK codes relationships
  - Color-coded confidence indicators: Green (>50 titles), Teal (>20), Yellow (>5), Red (<5)
- **Critical Bug Fixes**:
  - Fixed `dk_statistics` not being loaded from JSON in CLI display functions (3 locations)
  - Fixed incorrect tab navigation indices in GUI `on_step_selected()` method
  - Added missing navigation cases: chunk_details, k10plus, dk_statistics
- **Backward Compatibility**: Old JSON files without statistics handled gracefully with fallback messages
- **Files Modified**: `src/alima_cli.py`, `src/ui/analysis_review_tab.py`, `CLAUDE.md`

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
