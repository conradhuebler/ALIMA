# UI - PyQt6 User Interface Layer

## [Preserved Section - Permanent Documentation]

### UI Architecture
The `src/ui/` directory implements the complete PyQt6-based graphical user interface for ALIMA:

**Main Components:**
- `MainWindow`: Central application window with tab management and menu system
- `🆕 PipelineTab`: Vertical pipeline UI orchestrating complete ALIMA workflow
- `🆕 GlobalStatusBar`: Unified status display for providers, cache, and pipeline progress
- `AbstractTab`: Primary text analysis interface with chunking workflow
- `SearchTab` (find_keywords.py): GND keyword search and browse interface
- `AnalysisReviewTab`: Analysis workflow review and result management (enhanced with auto-receive)
- `CrossrefTab`: DOI metadata lookup and Crossref integration
- `UBSearchTab`: University library search interface
- `ImageAnalysisTab`: AI-powered image analysis functionality

**Supporting Components:**
- `SettingsDialog`: Application configuration interface
- `PromptEditorDialog`: LLM prompt template editor
- `TableWidget`: Enhanced table displays for search results
- `Widgets`: Reusable UI components and utilities
- `Styles`: Centralized styling and theme management

### PyQt6 Design Patterns
**Signal/Slot Architecture:**
- Comprehensive event-driven communication
- Thread-safe UI updates via signals
- Proper separation of UI and business logic
- Real-time streaming text display

**Threading Integration:**
- QThread-based background processing
- Non-blocking LLM generation
- Progress tracking and cancellation support
- Thread-safe GUI updates

**Layout Management:**
- Responsive design with QSplitter components
- Dynamic UI resizing during streaming
- Consistent spacing and alignment
- Adaptive tab management

### Key UI Features

**AbstractTab - Text Analysis:**
- Multi-step chunking workflow
- Real-time LLM streaming display
- Dynamic text area resizing
- Result history and navigation
- Integrated keyword extraction

**Search Interfaces:**
- Unified search across multiple providers
- Real-time result display
- Advanced filtering and sorting
- Export functionality
- Cache management integration

**Analysis Review System:**
- JSON-based analysis import/export
- Step-by-step workflow navigation
- Result comparison and validation
- Data transfer between analysis stages

### Threading and Responsiveness
**Background Operations:**
- SearchWorker threads for non-blocking searches
- LLM generation threads with streaming
- File I/O operations in separate threads
- Progress tracking and user feedback

**UI Responsiveness:**
- Dynamic content updates during operations
- Proper cursor management (busy/normal states)
- Real-time progress indicators
- Graceful error handling and user notification

## [Variable Section - Short-term Information]

### Recent Enhancements (Claude Generated)
1. **Enhanced AbstractTab Layout**: Improved splitter-based layout with better text streaming
2. **Search Result Display**: Better formatting and navigation of search results
3. **Analysis Review Integration**: Seamless workflow between analysis steps
4. **Threading Improvements**: More stable background operations and UI updates
5. **Enhanced Analysis Feedback**: Real-time status updates during LLM analysis with visual indicators
6. **Dynamic UI Adjustment**: Automatic layout optimization for streaming text display
7. **Manual UI Controls**: Toggle buttons for hiding/showing input areas during analysis
8. **Improved Error Handling**: Better error feedback with status messages and auto-recovery

### 🚀 MAJOR NEW FEATURES (Recently ADDED)
9. **✅ Vertical Pipeline UI (`pipeline_tab.py`)**:
   - Chat-like vertical workflow with 5 pipeline steps
   - Visual status indicators: ▷ (Pending), ▶ (Running), ✓ (Completed), ✗ (Error)
   - Auto-Pipeline button for one-click complete analysis
   - Integrated input tabs (DOI, Image, PDF, Text) in first step
   - Real-time result display in each step
   - Direct integration with PipelineManager for workflow orchestration

10. **✅ Global Status Bar (`global_status_bar.py`)**:
    - Unified provider information display across all tabs
    - Real-time cache statistics (entries count, database size)
    - Pipeline progress tracking with color-coded status
    - Auto-updating every 5 seconds for live monitoring
    - Integration with LlmService and CacheManager

11. **✅ Automated Data Flow Enhancement**:
    - AbstractTab automatically sends results to AnalysisReviewTab
    - New `analysis_completed` signal in AbstractTab
    - `receive_analysis_data()` method in AnalysisReviewTab
    - Seamless workflow progression without manual data transfer

### Current UI State
- **Font Sizes**: Larger fonts implemented for better readability
- **Layout System**: QSplitter-based responsive design with auto-adjustment during streaming
- **Theme Support**: Consistent styling across all components
- **Error Handling**: Comprehensive user feedback for all operations
- **Analysis Status**: Real-time feedback with color-coded status indicators
- **UI Controls**: Cancel analysis, toggle input visibility, auto-scroll for streaming text
- **🆕 Pipeline-First Design**: Pipeline tab as primary interface (first tab)
- **🆕 Global Monitoring**: Unified status bar showing system-wide information
- **🆕 Auto-Workflow**: Seamless data flow between analysis components

### Development Notes
- All new UI functions marked as "Claude Generated"
- Comprehensive signal/slot documentation maintained
- Type hints for better IDE integration and debugging

### Performance Considerations
- Efficient text streaming with minimal UI blocking
- Optimized result display for large datasets
- Memory-conscious handling of analysis results
- Responsive design patterns for various screen sizes

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **WIP - Pipeline Configuration UI**: Graphical configuration for pipeline steps and models
2. **ADD - Pipeline Templates**: Save/load different pipeline workflow configurations
3. **ADD - Batch Pipeline Processing**: Process multiple inputs through pipeline simultaneously
4. **ADD - Drag & Drop to Pipeline**: File drag-and-drop directly to pipeline steps
5. **ADD - Pipeline Step Editing**: Inline editing of intermediate results in pipeline
6. **ADD - Keyboard Shortcuts**: Implement comprehensive keyboard navigation
7. **ADD - Advanced Theming**: Dark mode and customizable UI themes
8. **ADD - Enhanced Export**: Pipeline result export in multiple formats (PDF, Excel, etc.)

### Recently COMPLETED Tasks
1. **✅ TESTED - Vertical Pipeline UI**: Complete implementation with visual step indicators
2. **✅ TESTED - Global Status Integration**: Real-time provider and cache monitoring  
3. **✅ TESTED - Automated Data Transfer**: AbstractTab → AnalysisReviewTab workflow
4. **✅ APPROVED - Pipeline Manager Integration**: Full workflow orchestration system

### Vision
- Create intuitive and efficient interface for library science workflows
- Implement accessibility features for diverse user needs
- Provide seamless integration between all analysis and search functions
- Support for multi-monitor setups and varying screen resolutions
- Establish ALIMA as the standard for library metadata analysis tools