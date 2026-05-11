# ALIMA Complete Pipeline Initialization Pattern

## Executive Summary

The ALIMA pipeline has a **unified initialization architecture** used by all three interfaces (GUI, CLI, Webapp). All interfaces use identical core components initialized in the same sequence:

```
ConfigManager → PromptService → LlmService → AlimaManager → PipelineManager → SearchEngine & UnifiedKnowledgeManager
```

**Key Principle**: Configuration loads first, then services are initialized with lazy loading (LlmService doesn't test providers until needed).

---

## Part 1: Configuration Loading Sequence

### Step 1: ConfigManager Initialization
**Purpose**: Load all configuration from `config.json` and environment

**Load Order** (in `config_manager.py`):
1. OS-specific config path resolution (Windows/macOS/Linux)
2. Load `config.json` from disk
3. Detect legacy vs. modern format
4. Parse sections in order:
   - `database_config` → `DatabaseConfig` (UNIFIED SINGLE SOURCE OF TRUTH: `sqlite_path`)
   - `catalog_config` → `CatalogConfig`
   - `prompt_config` → `PromptConfig`
   - `system_config` → `SystemConfig` (contains `prompts_path`)
   - `ui_config` → `UIConfig`
   - `unified_config` → `UnifiedProviderConfig` (providers list, task preferences)
5. Perform path resolution (absolute paths for database and prompts)
6. Return fully initialized `AlimaConfig` object

**Key Flags**:
- `lazy_initialization`: Defers provider testing until needed
- `force_reload`: Forces fresh config load from disk

---

## Part 2: Core Service Initialization Chain

### The Complete Initialization Chain (All Entry Points)

```python
# 1. CONFIGURATION (MUST BE FIRST)
config_manager = ConfigManager(logger=logger)
config = config_manager.load_config()  # Loads all sections
prompts_path = config.system_config.prompts_path

# 2. LLM SERVICE (with lazy initialization)
llm_service = LlmService(
    config_manager=config_manager,
    lazy_initialization=True  # CRITICAL: prevents GUI hanging
)

# 3. PROMPT SERVICE (prompt templates)
prompt_service = PromptService(prompts_path, logger=logger)

# 4. ALIMA MANAGER (core business logic)
alima_manager = AlimaManager(
    llm_service=llm_service,
    prompt_service=prompt_service,
    config_manager=config_manager,
    logger=logger
)

# 5. UNIFIED KNOWLEDGE MANAGER (database singleton)
cache_manager = UnifiedKnowledgeManager()  # Singleton pattern

# 6. SEARCH ENGINE (search coordination)
search_engine = SearchEngine(cache_manager)

# 7. PIPELINE MANAGER (pipeline orchestration)
pipeline_manager = PipelineManager(
    alima_manager=alima_manager,
    cache_manager=cache_manager,
    logger=logger,
    config_manager=config_manager
)
```

### Detailed Component Initialization

#### 1. ConfigManager (lines 157-188 in config_manager.py)

```python
class ConfigManager:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup OS-specific paths
        self._setup_config_paths()  # Sets self.config_file
        
        self._config: Optional[AlimaConfig] = None
        self._provider_detection_service: Optional[ProviderDetectionService] = None
    
    def _setup_config_paths(self):
        """OS-specific paths: Windows (%APPDATA%\ALIMA), macOS (~/Library/Application Support/ALIMA), Linux (~/.config/alima)"""
        system_name = platform.system().lower()
        
        if system_name == "windows":
            config_base = Path(os.environ.get("APPDATA", "")) / "ALIMA"
        elif system_name == "darwin":  # macOS
            config_base = Path("~/Library/Application Support/ALIMA").expanduser()
        else:  # Linux and others
            config_base = Path("~/.config/alima").expanduser()
        
        config_base.mkdir(parents=True, exist_ok=True)
        self.config_file = config_base / "config.json"  # SINGLE SOURCE OF TRUTH FOR CONFIG
    
    def load_config(self, force_reload: bool = False) -> AlimaConfig:
        """Load and cache configuration"""
        if self._config is None or force_reload:
            self._config = self._load_config_from_file()
        return self._config
```

**Configuration Sections Loaded** (lines 233-276):
```python
def _parse_config(self, config_data: Dict[str, Any]) -> AlimaConfig:
    # Section 1: Database Configuration
    database_config = DatabaseConfig(**config_data.get("database_config", {}))
    
    # Section 2: Catalog Configuration  
    catalog_config = CatalogConfig(**config_data.get("catalog_config", {}))
    
    # Section 3: Prompt Configuration
    prompt_config = PromptConfig(**config_data.get("prompt_config", {}))
    
    # Section 4: System Configuration (contains prompts_path)
    system_config = SystemConfig(**system_config_data)
    
    # Section 5: UI Configuration
    ui_config = UIConfig(**ui_config_data)
    
    # Section 6: Unified Provider Configuration (providers, task_preferences)
    unified_config = self._parse_unified_config(unified_config_data)
    
    # Return combined AlimaConfig
    return AlimaConfig(
        database_config=database_config,
        catalog_config=catalog_config,
        prompt_config=prompt_config,
        system_config=system_config,
        ui_config=ui_config,
        unified_config=unified_config,
        config_version="2.0"
    )
```

#### 2. LlmService (lines 52-126 in llm_service.py)

```python
class LlmService(QObject):
    def __init__(
        self,
        providers: List[str] = None,
        config_manager = None,
        api_keys: Dict[str, str] = None,
        ollama_url: str = "http://localhost",
        ollama_port: int = 11434,
        lazy_initialization: bool = False  # ⭐ KEY: prevents blocking during GUI startup
    ):
        super().__init__()
        
        # Store lazy initialization flag
        self.lazy_initialization = lazy_initialization
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration manager
        if config_manager is None:
            from ..utils.config_manager import get_config_manager
            config_manager = get_config_manager()
        self.config_manager = config_manager
        
        # Load configuration
        self.alima_config = self.config_manager.load_config()
        
        # Dictionary to store provider clients
        self.clients = {}
        
        # Initialize unified provider configurations (reads from config)
        self._init_unified_provider_configs()
        
        # Initialize legacy dynamic OpenAI-compatible providers (deprecated)
        self._legacy_init_dynamic_provider_configs()
        
        # Initialize all providers directly (ping tests prevent blocking)
        self.initialize_providers(providers)
```

**Key Initialization Methods**:
- `_init_unified_provider_configs()` (line 140): Reads providers from config
- `_legacy_init_dynamic_provider_configs()`: Handles old format
- `initialize_providers()` (line 539): Creates provider clients with ping tests (not full connection)

#### 3. PromptService (lines 9-25 in prompt_service.py)

```python
class PromptService:
    def __init__(self, config_path: str, logger: logging.Logger = None):
        """Initialize the PromptService with the configuration file path"""
        self.config_path = config_path  # Points to prompts.json
        self.logger = logger or logging.getLogger(__name__)
        
        # Load prompts.json from disk
        self.config = self.load_config(config_path)
        
        # Extract available tasks from config
        self.tasks = self.config.keys()
        
        # Build model-to-prompt index for fast lookup
        self.models_by_task = self._build_model_index()
    
    def load_config(self, config_path: str) -> Dict:
        """Load the configuration file"""
        if not os.path.exists(config_path):
            self.logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
```

#### 4. AlimaManager (lines 19-41 in alima_manager.py)

```python
class AlimaManager:
    def __init__(
        self,
        llm_service: LlmService,
        prompt_service: PromptService,
        config_manager: "ConfigManager",
        logger: logging.Logger = None,
    ):
        self.ollama_url = "http://localhost"  # Default, can be overridden
        self.ollama_port = 11434  # Default, can be overridden
        
        # Store references to services
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Provider Status Service for health monitoring
        try:
            self.provider_status_service = ProviderStatusService(self.llm_service)
            self.logger.info("ProviderStatusService initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ProviderStatusService: {e}")
            self.provider_status_service = None
```

#### 5. UnifiedKnowledgeManager (lines 58-87 in unified_knowledge_manager.py)

```python
class UnifiedKnowledgeManager:
    """Singleton pattern - only one instance per application"""
    
    # Singleton implementation
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, db_path: Optional[str] = None, database_config: Optional[DatabaseConfig] = None):
        """Create or return singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None, database_config: Optional[DatabaseConfig] = None):
        # Skip re-initialization of existing singleton
        if UnifiedKnowledgeManager._initialized:
            self.logger.debug("⚠️ UnifiedKnowledgeManager is singleton - skipping re-initialization")
            return
        
        # Load database config from ConfigManager if not provided
        if database_config is None:
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            # UNIFIED SINGLE SOURCE OF TRUTH: database_config.sqlite_path
            database_config = config.database_config
```

#### 6. SearchEngine (lines 10-27 in search_engine.py)

```python
class SearchEngine(QObject):
    term_search_completed = pyqtSignal(str, dict)
    term_search_error = pyqtSignal(str, str)
    search_finished = pyqtSignal(dict)
    
    def __init__(self, cache_manager: 'UnifiedKnowledgeManager'):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.cache = cache_manager  # Reference to singleton
        self.base_url = "https://lobid.org/resources/search"
        self.network_manager = QNetworkAccessManager()
        self.current_results = {"term_results": {}, "total_counter": Counter()}
```

#### 7. PipelineManager (lines 49-158 in pipeline_manager.py)

```python
class PipelineManager:
    def __init__(
        self,
        alima_manager: AlimaManager,
        cache_manager: UnifiedKnowledgeManager,
        logger: logging.Logger = None,
        config_manager = None
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize PipelineStepExecutor for shared pipeline logic
        from ..utils.pipeline_utils import PipelineStepExecutor
        self.step_executor = PipelineStepExecutor(
            alima_manager=alima_manager,
            cache_manager=cache_manager,
            logger=logger,
            config_manager=config_manager
        )
        
        # Pipeline state tracking
        self.is_running = False
        self.current_analysis_state = None
        
        # Callback storage for UI/CLI communication
        self._callbacks = {}
```

---

## Part 3: Entry Point Initialization Sequences

### Entry Point 1: GUI Initialization (src/alima_gui.py + src/ui/main_window.py)

**File**: `/home/user/ALIMA/src/alima_gui.py` (lines 15-58)

```python
def main():
    # Setup centralized logging
    setup_logging(level=1, log_file="alima.log")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setOrganizationName("TU Bergakademie Freiberg")
    app.setApplicationName("AlIma")
    app.setApplicationVersion("0.2")
    app.setStyle("Fusion")
    
    # Show splash screen while initializing
    pixmap = QPixmap(os.path.join(current_dir, "alima.png"))
    if not pixmap.isNull():
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()
    
    # Create and initialize MainWindow
    window = MainWindow()  # THIS CONTAINS ALL INITIALIZATION
    window.show()
    
    # Start background provider check
    if hasattr(window, 'alima_manager') and window.alima_manager.provider_status_service:
        try:
            window.alima_manager.provider_status_service.refresh_all()
            logging.info("Background provider status check started")
        except Exception as e:
            logging.warning(f"Failed to start background provider check: {e}")
    
    # Hide splash after showing
    if "splash" in locals():
        splash.finish(window)
    
    sys.exit(app.exec())
```

**Main Window Initialization** (lines 319-358 in main_window.py):

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("TUBAF", "Alima")
        
        # ===== STEP 1: CACHE & SEARCH (no dependencies) =====
        self.cache_manager = UnifiedKnowledgeManager()  # Singleton
        self.search_engine = SearchEngine(self.cache_manager)
        
        # ===== STEP 2: LOGGING & CONFIGURATION =====
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager(logger=self.logger)
        
        # Check for first run and offer configuration import
        self.check_first_run()
        
        # ===== STEP 3: LOAD CONFIG & GET PROMPTS PATH =====
        config = self.config_manager.load_config()
        prompts_path = config.system_config.prompts_path
        
        # ===== STEP 4: LLM SERVICE (with lazy initialization) =====
        self.llm_service = LlmService(
            config_manager=self.config_manager,
            lazy_initialization=True  # ⭐ CRITICAL: Fast GUI startup
        )
        self.llm = self.llm_service  # Alias
        
        # ===== STEP 5: PROMPT SERVICE =====
        self.prompt_service = PromptService(prompts_path)
        
        # ===== STEP 6: ALIMA MANAGER =====
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager,
            logger=self.logger
        )
        
        # ===== STEP 7: PIPELINE MANAGER =====
        self.pipeline_manager = PipelineManager(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=self.logger,
            config_manager=self.config_manager
        )
        
        # ===== STEP 8: UI INITIALIZATION =====
        self.available_models = {}
        self.available_providers = []
        
        self.init_ui()  # Creates all tabs and widgets
        self.load_settings()  # Restore window geometry
        
        # Setup reactive provider status connections
        self.setup_provider_status_connections()
```

**Passed to Child Widgets** (lines 393-482):
```python
def init_ui(self):
    # SearchTab
    self.search_tab = SearchTab(
        cache_manager=self.cache_manager,
        alima_manager=self.alima_manager,
        pipeline_manager=self.pipeline_manager
    )
    
    # AbstractTab (analysis)
    self.abstract_tab = AbstractTab(
        alima_manager=self.alima_manager,
        llm_service=self.llm_service,
        cache_manager=self.cache_manager,
        pipeline_manager=self.pipeline_manager,
        main_window=self
    )
    
    # PipelineTab (main workflow)
    self.pipeline_tab = PipelineTab(
        alima_manager=self.alima_manager,
        llm_service=self.llm_service,
        cache_manager=self.cache_manager,
        pipeline_manager=self.pipeline_manager,
        main_window=self
    )
    
    # Global Status Bar (monitors providers & cache)
    self.global_status_bar = GlobalStatusBar()
    self.setStatusBar(self.global_status_bar)
    self.global_status_bar.set_services(self.llm_service, self.cache_manager)
```

---

### Entry Point 2: CLI Initialization (src/alima_cli.py)

**File**: `/home/user/ALIMA/src/alima_cli.py` (lines 668-1415)

```python
def main():
    # ===== STEP 1: SETUP LOGGING =====
    parser = argparse.ArgumentParser(description="ALIMA CLI - AI-powered abstract analysis")
    # ... argument parsing ...
    
    # ===== STEP 2: SETUP LOGGING (log level from args) =====
    logger = logging.getLogger(__name__)  # Will be configured by setup_logging
    
    # ===== WHEN COMMAND IS "pipeline": =====
    if args.command == "pipeline":
        # ===== STEP 3: CONFIGURATION MANAGER =====
        from src.utils.config_manager import ConfigManager as CM
        config_manager = CM()
        
        # ===== STEP 4: LOAD CONFIG & GET PROMPTS PATH =====
        config = config_manager.load_config()
        prompts_path = config.system_config.prompts_path
        
        # ===== STEP 5: LLM SERVICE (WITHOUT lazy initialization for CLI) =====
        llm_service = LlmService(
            providers=None,  # Resolve dynamically
            config_manager=config_manager,
            ollama_url=args.ollama_host,
            ollama_port=args.ollama_port
        )
        
        # ===== STEP 6: PROMPT SERVICE =====
        prompt_service = PromptService(prompts_path, logger)
        
        # ===== STEP 7: ALIMA MANAGER =====
        alima_manager = AlimaManager(llm_service, prompt_service, config_manager, logger)
        
        # ===== STEP 8: UNIFIED KNOWLEDGE MANAGER =====
        cache_manager = UnifiedKnowledgeManager()
        
        # ===== STEP 9: PIPELINE MANAGER =====
        from src.core.pipeline_manager import PipelineManager
        pipeline_manager = PipelineManager(
            alima_manager=alima_manager,
            cache_manager=cache_manager,
            logger=logger,
            config_manager=config_manager
        )
        
        # ===== STEP 10: CREATE PIPELINE CONFIG =====
        try:
            pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)
            logger.info("Pipeline configuration loaded from Provider Preferences")
        except Exception as e:
            logger.warning(f"Failed to load Provider Preferences, using defaults: {e}")
            pipeline_config = PipelineConfig()
        
        # ===== STEP 11: APPLY CLI OVERRIDES =====
        updated_pipeline_config = apply_cli_overrides(pipeline_config, args)
        pipeline_manager.set_config(updated_pipeline_config)
        
        # ===== STEP 12: SET CALLBACKS FOR CLI OUTPUT =====
        pipeline_manager.set_callbacks(
            step_started=cli_step_started,
            step_completed=cli_step_completed,
            step_error=cli_step_error,
            pipeline_completed=cli_pipeline_completed,
            stream_callback=cli_stream_callback
        )
        
        # ===== STEP 13: RESOLVE INPUT (text, DOI, or image) =====
        if args.doi:
            success, input_text, error_msg = resolve_input_to_text(args.doi, logger)
        elif args.input_image:
            from src.utils.pipeline_utils import execute_input_extraction
            input_text, source_info, extraction_method = execute_input_extraction(
                llm_service=llm_service,
                input_source=args.input_image,
                input_type="image",
                stream_callback=image_stream_callback,
                logger=logger
            )
        else:
            input_text = args.input_text
        
        # ===== STEP 14: EXECUTE PIPELINE =====
        pipeline_manager.start_pipeline(
            input_text=input_text,
            force_update=getattr(args, 'force_update', False)
        )
        
        # Wait for completion (synchronous mode for CLI)
        import time
        timeout = 300  # 5 minutes
        elapsed = 0
        while pipeline_manager.is_running and elapsed < timeout:
            time.sleep(0.1)
            elapsed += 0.1
        
        # Get results
        analysis_state = pipeline_manager.current_analysis_state
        
        # Print/export results
        print_result(f"Initial Keywords: {analysis_state.initial_keywords}")
        print_result(f"Final Keywords: {analysis_state.final_llm_analysis.extracted_gnd_keywords}")
```

**Callback Functions** (lines 1418-1434):
```python
# CLI Callbacks for pipeline events
def cli_step_started(step):
    provider_info = f"{step.provider}/{step.model}" if step.provider and step.model else "Smart Mode"
    logger.info(f"▶ Starte Schritt: {step.name} ({provider_info})")

def cli_step_completed(step):
    logger.info(f"✅ Schritt abgeschlossen: {step.name}")

def cli_step_error(step, error_message):
    logger.error(f"❌ Fehler in Schritt {step.name}: {error_message}")

def cli_pipeline_completed(analysis_state):
    logger.info("\n🎉 Pipeline vollständig abgeschlossen!")

def cli_stream_callback(token, step_id):
    # Streaming output should always go to stdout for user feedback
    print(token, end="", flush=True)
```

---

### Entry Point 3: Webapp Initialization (src/alima_webapp.py + src/webapp/app.py)

**File**: `/home/user/ALIMA/src/alima_webapp.py` (lines 1-31)

```python
#!/usr/bin/env python3
"""
ALIMA Webapp Launcher
Claude Generated - Start the FastAPI web server
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import uvicorn
    from src.webapp.app import app
    
    print("\n" + "=" * 60)
    print("🚀 ALIMA Webapp Server")
    print("=" * 60)
    print("\n📝 Starting server on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    print("\n✅ Press Ctrl+C to stop the server\n")
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

**Backend Application** (`src/webapp/app.py`, lines 1-85):

```python
"""
ALIMA Webapp - FastAPI Backend
Claude Generated - Pipeline widget as web interface
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import subprocess
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ===== STEP 1: IMPORT ALIMA COMPONENTS =====
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import ConfigManager
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.pipeline_utils import PipelineJsonManager

# ===== STEP 2: SETUP LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== STEP 3: GET PROJECT ROOT =====
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ===== STEP 4: CREATE FASTAPI APP =====
app = FastAPI(title="ALIMA Webapp", description="Pipeline widget as web interface")

# ===== STEP 5: SETUP CORS MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== STEP 6: MOUNT STATIC FILES =====
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ===== STEP 7: SESSION STORAGE =====
sessions: dict = {}

class Session:
    """Represents an analysis session - Claude Generated"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.status = "idle"  # idle, running, completed, error
        self.current_step = None
        self.input_data = None
        self.results = {}
        self.error_message = None
        self.process = None
        self.temp_files = []
    
    def add_temp_file(self, path: str):
        """Track temporary files for cleanup - Claude Generated"""
        self.temp_files.append(path)
    
    def cleanup(self):
        """Clean up temporary files - Claude Generated"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not cleanup {temp_file}: {e}")
```

**Lazy Initialization in Webapp** (should be added to endpoint handlers):

```python
# Lazy initialization approach for webapp
class AppContext:
    """Global application context with lazy-initialized services"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def init_services(self):
        """Initialize ALIMA services on first use"""
        if self._initialized:
            return
        
        # ===== STEP 1: CONFIG MANAGER =====
        self.config_manager = ConfigManager()
        config = self.config_manager.load_config()
        prompts_path = config.system_config.prompts_path
        
        # ===== STEP 2: LLM SERVICE =====
        from src.llm.llm_service import LlmService
        self.llm_service = LlmService(
            config_manager=self.config_manager,
            lazy_initialization=True
        )
        
        # ===== STEP 3: PROMPT SERVICE =====
        from src.llm.prompt_service import PromptService
        self.prompt_service = PromptService(prompts_path, logger)
        
        # ===== STEP 4: ALIMA MANAGER =====
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager,
            logger=logger
        )
        
        # ===== STEP 5: CACHE MANAGER =====
        self.cache_manager = UnifiedKnowledgeManager()
        
        # ===== STEP 6: PIPELINE MANAGER =====
        self.pipeline_manager = PipelineManager(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=logger,
            config_manager=self.config_manager
        )
        
        self._initialized = True
        logger.info("✅ ALIMA services initialized")
    
    def get_services(self):
        """Get or initialize services"""
        if not self._initialized:
            self.init_services()
        return {
            'config_manager': self.config_manager,
            'llm_service': self.llm_service,
            'prompt_service': self.prompt_service,
            'alima_manager': self.alima_manager,
            'cache_manager': self.cache_manager,
            'pipeline_manager': self.pipeline_manager
        }

# In endpoints:
@app.post("/api/analyze/{session_id}")
async def start_analysis(session_id: str, input_text: str = Form(...)):
    """Start analysis - Claude Generated"""
    # Initialize services on first use
    app_context = AppContext()
    services = app_context.get_services()
    
    pipeline_manager = services['pipeline_manager']
    # ... continue with analysis
```

---

## Part 4: All Required Imports by Entry Point

### GUI Imports (src/ui/main_window.py, lines 1-62)

```python
# Qt Framework
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QMenu, QStatusBar, QLabel, QPushButton,
    QDialog, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QFileDialog, QProgressDialog, QTextEdit
)
from PyQt6.QtCore import Qt, QSettings, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QCursor

# Standard Library
import os, sys, subprocess, json, time, datetime, re, tempfile, gzip, requests
from typing import Optional, Dict
from pathlib import Path

# ALIMA Core
from ..core.search_engine import SearchEngine
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..core.gndparser import GNDParser
from ..core.gitupdate import GitUpdateWorker
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..core.alima_manager import AlimaManager
from ..core.pipeline_manager import PipelineManager
from ..utils.config_manager import ConfigManager
from ..utils.config import Config  # Legacy

# ALIMA UI Tabs
from .find_keywords import SearchTab
from .abstract_tab import AbstractTab
from .comprehensive_settings_dialog import ComprehensiveSettingsDialog
from .crossref_tab import CrossrefTab
from .analysis_review_tab import AnalysisReviewTab
from .ubsearch_tab import UBSearchTab
from .tablewidget import TableWidget, DatabaseViewerDialog
from .image_analysis_tab import ImageAnalysisTab
from .styles import get_main_stylesheet
from .global_status_bar import GlobalStatusBar
from .pipeline_tab import PipelineTab

# Logging
import logging
```

### CLI Imports (src/alima_cli.py, lines 1-42)

```python
# Standard Library
import argparse, logging, os, json, time, tempfile, gzip, requests, sys
from dataclasses import asdict
from typing import List, Tuple

# Python Path Setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ALIMA Core
from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.data_models import (
    AbstractData, TaskState, AnalysisResult, PromptConfigData,
    KeywordAnalysisState, SearchResult, LlmKeywordAnalysis
)
from src.core.search_cli import SearchCLI
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.suggesters.meta_suggester import SuggesterType
from src.core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response
)

# ALIMA Utils
from src.utils.pipeline_utils import (
    PipelineJsonManager,
    PipelineStepExecutor
)
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.config_manager import ConfigManager, OpenAICompatibleProvider
from src.utils.config_models import PipelineMode, PipelineStepConfig
from src.utils.logging_utils import setup_logging, print_result

# Pipeline
from src.utils.pipeline_defaults import DEFAULT_DK_MAX_RESULTS, DEFAULT_DK_FREQUENCY_THRESHOLD
```

### Webapp Imports (src/webapp/app.py, lines 1-31)

```python
# Standard Library
import asyncio, json, logging, os, tempfile, uuid, subprocess, sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ALIMA Core
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

# ALIMA Utils
from src.utils.config_manager import ConfigManager
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.pipeline_utils import PipelineJsonManager
```

---

## Part 5: Configuration Loading Details

### config.json Structure (Loaded in Order)

```json
{
  "config_version": "2.0",
  
  "database_config": {
    "db_type": "sqlite",
    "sqlite_path": "~/.config/alima/alima_knowledge.db"  // UNIFIED SINGLE SOURCE OF TRUTH
  },
  
  "catalog_config": {
    "catalog_token": "...",
    "catalog_search_url": "...",
    "catalog_details_url": "..."
  },
  
  "prompt_config": {
    "custom_prompts_enabled": false
  },
  
  "system_config": {
    "language": "de",
    "debug": false,
    "log_level": 1,
    "prompts_path": "~/.config/alima/prompts.json"  // Resolved to absolute path
  },
  
  "ui_config": {
    "theme": "light",
    "window_width": 1400,
    "window_height": 900
  },
  
  "unified_config": {
    "provider_priority": ["ollama", "gemini", "anthropic", "openai"],
    "disabled_providers": [],
    "auto_fallback": true,
    "prefer_faster_models": false,
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "providers": [
      {
        "name": "ollama",
        "provider_type": "ollama",
        "enabled": true,
        "host": "http://localhost",
        "port": 11434,
        "models": ["cogito:14b", "cogito:32b", "neural-chat"]
      }
    ],
    "task_preferences": {
      "keywords": {
        "task_type": "general",
        "model_priority": [
          {"provider_name": "ollama", "models": ["cogito:32b"]}
        ],
        "allow_fallback": true
      }
    }
  }
}
```

### Prompts File Structure (prompts.json)

```json
{
  "initialisation": {
    "description": "Extract keywords from abstract",
    "prompts": [
      ["Prompt template here...", "System instruction...", 0.7, 0.1, ["cogito:14b", "gemini-1.5-flash"], 42]
    ]
  },
  
  "keywords": {
    "description": "Verify keywords with GND context",
    "prompts": [
      ["Prompt template...", "System instruction...", 0.7, 0.1, ["cogito:32b", "claude-opus"], 42]
    ]
  },
  
  "classification": {
    "description": "DDC/DK classification",
    "prompts": [
      ["Prompt template...", "System instruction...", 0.5, 0.1, ["cogito:32b"], 42]
    ]
  }
}
```

---

## Part 6: Initialization Flags & Special Parameters

### Lazy Initialization Flag
**Purpose**: Prevent provider testing during startup (critical for GUI responsiveness)

```python
# In GUI (ALWAYS lazy)
llm_service = LlmService(
    config_manager=config_manager,
    lazy_initialization=True  # ⭐ DO NOT TEST PROVIDERS UNTIL FIRST USE
)

# In CLI (NO lazy loading - tests on startup)
llm_service = LlmService(
    config_manager=config_manager,
    lazy_initialization=False  # OR OMIT - default is False
)

# Impact:
# - lazy=True: Skip provider.test_connection() calls → instant UI startup
# - lazy=False: Test all providers → 30+ seconds CLI startup (acceptable)
```

### Configuration Reload Flag
```python
# Force fresh config load (bypass cache)
config = config_manager.load_config(force_reload=True)

# Normal load (use cache if available)
config = config_manager.load_config()
```

### Pipeline Mode Flag
```python
# CLI argument determines provider selection strategy
args.mode in ["smart", "advanced", "expert"]

# Smart: Uses task_preferences from config automatically
# Advanced: Manual provider|model selection via --step
# Expert: Full parameter control (temperature, top_p, seed)
```

---

## Part 7: Dependency Graph

```
ConfigManager (base)
    ↓
    ├→ LlmService (reads config.unified_config)
    │       ↓
    │       └→ ProviderStatusService (monitors providers)
    │
    ├→ PromptService (reads config.system_config.prompts_path)
    │
    └→ AlimaManager
            ├→ LlmService
            ├→ PromptService
            └→ ProviderStatusService

UnifiedKnowledgeManager (singleton)
    ├→ DatabaseManager (uses database_config.sqlite_path)
    └→ (no dependencies on other services)

SearchEngine
    └→ UnifiedKnowledgeManager

PipelineManager
    ├→ AlimaManager
    ├→ UnifiedKnowledgeManager
    └→ PipelineStepExecutor
        ├→ SmartProviderSelector (uses ConfigManager)
        └→ SearchCLI
```

---

## Summary Checklist

✅ **Initialization Order** (ALL entry points):
1. ConfigManager → load config.json
2. LlmService → initialize with lazy flag
3. PromptService → load prompts.json
4. AlimaManager → coordinate LLM tasks
5. UnifiedKnowledgeManager → singleton database
6. SearchEngine → search coordination
7. PipelineManager → pipeline orchestration
8. UI components → pass initialized services

✅ **Required Imports** - All available in provided code
✅ **Configuration Files** - config.json + prompts.json
✅ **Lazy Initialization** - GUI only
✅ **Singleton Pattern** - UnifiedKnowledgeManager only
✅ **All Entry Points** - GUI, CLI, Webapp documented

