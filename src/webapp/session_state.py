"""Webapp shared state — session registry, Session model, lazy service context.

Claude Generated — extracted verbatim from ``app.py`` during the F-6 god-file
split. Imports nothing from ``app.py`` so routers can depend on it without a
cycle. ``app.py`` re-exports these names for backward compatibility.
"""

import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import ALIMA Pipeline components - Claude Generated
from src.core.pipeline_manager import PipelineManager
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Auto-Save Configuration - Claude Generated (2026-01-06)
# These settings control the auto-save and recovery system
AUTOSAVE_ENABLED = True  # Enable/disable auto-save system
AUTOSAVE_MAX_AGE_HOURS = 24  # Auto-cleanup files older than this (hours)
WEBSOCKET_TIMEOUT_SECONDS = 1800  # WebSocket idle timeout (30 minutes = 1800s)
WEBSOCKET_HEARTBEAT_INTERVAL = 5  # Heartbeat interval in seconds (5s)

# Auto-save directory for session recovery - Claude Generated
AUTOSAVE_DIR = Path(tempfile.gettempdir()) / "alima_webapp_autosave"
AUTOSAVE_DIR.mkdir(exist_ok=True)
logger.info(f"Auto-save directory: {AUTOSAVE_DIR}")

# Store active sessions and their results
sessions: dict = {}


class AppContext:
    """Global application context with lazy-initialized services - Claude Generated"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init_services(self):
        """Initialize ALIMA services on first use"""
        if self._initialized:
            return

        logger.info("Initializing ALIMA services...")

        # Step 1: ConfigManager (load config.json)
        self.config_manager = ConfigManager(logger=logger)
        config = self.config_manager.load_config()
        prompts_path = config.system_config.prompts_path

        # Step 2: LlmService (with lazy initialization for webapp responsiveness)
        self.llm_service = LlmService(
            config_manager=self.config_manager,
            lazy_initialization=True
        )

        # Step 3: PromptService (load prompts.json)
        self.prompt_service = PromptService(prompts_path, logger=logger)

        # Step 4: AlimaManager (core business logic)
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager,
            logger=logger
        )

        # Step 5: UnifiedKnowledgeManager (singleton database)
        self.cache_manager = UnifiedKnowledgeManager()

        # Step 6: PipelineManager (pipeline orchestration)
        self.pipeline_manager = PipelineManager(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=logger,
            config_manager=self.config_manager
        )

        AppContext._initialized = True
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


class Session:
    """Represents an analysis session - Claude Generated"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.status = "idle"  # idle, running, completed, error
        self.current_step = None
        self.current_step_status = None  # 'running' or 'completed' - Claude Generated
        self.input_data = None
        self.results = {}
        self.error_message = None
        self.process = None
        self.temp_files = []
        self.streaming_buffer = {}  # Buffer for streaming tokens by step_id - Claude Generated
        self.streaming_buffer_sent_count = {}  # Track how many tokens sent per step - Claude Generated
        self._streaming_lock = threading.Lock()  # Thread-safe access to streaming buffers - Claude Generated
        # WP12: append-only render-event log for the shared chrome (DK/GND
        # cards). Broadcast over the WS (per-connection replay on reconnect) and
        # surfaced to polling clients via render_buffer_sent_count. - Claude Generated
        self.render_buffer = []
        self.render_buffer_sent_count = 0  # cursor for polling clients
        self.abort_requested = False  # Flag to signal pipeline abort - Claude Generated
        # Auto-save support - Claude Generated
        self.autosave_path = AUTOSAVE_DIR / f"session_{session_id}.json"
        self.autosave_enabled = AUTOSAVE_ENABLED  # Use global config
        self.autosave_failed = False
        self.autosave_timestamp = None  # Last auto-save timestamp for status indicator
        self.current_analysis_state = None  # Reference to PipelineManager state
        self.working_title = None  # Working title from initialisation step - Claude Generated
        self.dk_search_progress = None  # DK search progress info (current/total/percent) - Claude Generated
        self.pipeline_manager_ref = None  # Reference for step-abort - Claude Generated
        self.chat_thread = None  # Running chat-agent StoppableAgentThread (for cancel) - Claude Generated
        self.workflow_name = None  # Selected workflow for this session - Claude Generated
        self.last_provider: Optional[str] = None  # Effective provider from last pipeline run - Claude Generated
        self.last_model: Optional[str] = None  # Effective model from last pipeline run - Claude Generated
        self.chat_history: list = []  # Chat-agent conversation history - Claude Generated

    def add_temp_file(self, path: str):
        """Track temporary files for cleanup - Claude Generated"""
        self.temp_files.append(path)

    def add_streaming_token(self, token: str, step_id: str):
        """Add token to streaming buffer - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            if step_id not in self.streaming_buffer:
                self.streaming_buffer[step_id] = []
            self.streaming_buffer[step_id].append(token)

    def get_and_clear_streaming_buffer(self) -> dict:
        """Get all buffered tokens and clear - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            result = dict(self.streaming_buffer)
            self.streaming_buffer.clear()
            self.streaming_buffer_sent_count.clear()
            return result

    def get_new_streaming_tokens(self) -> dict:
        """Get only newly added tokens since last retrieval - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            result = {}
            for step_id, tokens in self.streaming_buffer.items():
                sent_count = self.streaming_buffer_sent_count.get(step_id, 0)
                new_tokens = tokens[sent_count:]
                if new_tokens:
                    result[step_id] = new_tokens
                    self.streaming_buffer_sent_count[step_id] = len(tokens)
            return result

    def append_render_event(self, event: dict):
        """Append a WP12 render event to the per-session log - Thread-safe.

        Stamps each event with a monotonic ``seq`` (its buffer index) so a
        client can dedup across WS-reconnect replay / polling re-delivery.
        """
        with self._streaming_lock:
            event = {**event, "seq": len(self.render_buffer)}
            self.render_buffer.append(event)

    def get_render_events_since(self, index: int):
        """Return (events_since_index, new_length) - for WS replay/incremental."""
        with self._streaming_lock:
            return list(self.render_buffer[index:]), len(self.render_buffer)

    def get_new_render_events(self) -> list:
        """Return render events not yet sent to a polling client - Thread-safe."""
        with self._streaming_lock:
            new = list(self.render_buffer[self.render_buffer_sent_count:])
            self.render_buffer_sent_count = len(self.render_buffer)
            return new

    def clear(self):
        """Complete session reset - clear all data - Claude Generated"""
        self.status = "idle"
        self.current_step = None
        self.current_step_status = None
        self.input_data = None
        self.results = {}
        self.error_message = None
        with self._streaming_lock:  # Thread-safe buffer clearing - Claude Generated
            self.streaming_buffer.clear()
            self.streaming_buffer_sent_count.clear()  # Reset token tracking - Claude Generated
            self.render_buffer.clear()  # WP12: reset render-event log - Claude Generated
            self.render_buffer_sent_count = 0
        self.abort_requested = False
        self.workflow_name = None
        self.chat_history = []
        self.cleanup()
        logger.info(f"Session {self.session_id} cleared")

    def cleanup(self):
        """Clean up temporary files - Claude Generated"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not cleanup {temp_file}: {e}")
        self.temp_files.clear()
