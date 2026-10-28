"""
Global Status Bar - Unified status display for ALIMA Pipeline
Claude Generated - Shows provider info, cache status, and pipeline progress
"""

from PyQt6.QtWidgets import QStatusBar, QLabel, QProgressBar, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from typing import Dict, Optional, List
import logging

from ..llm.llm_service import LlmService
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from .styles import set_style_prop  # chip colours: QLabel[chip=...] in the main stylesheet


class GlobalStatusBar(QStatusBar):
    """Global status bar showing provider info, cache status, and pipeline progress"""

    # Signals for status updates
    provider_changed = pyqtSignal(str, str)  # provider, model
    pipeline_step_changed = pyqtSignal(str, str)  # step_name, status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Status components
        self.provider_label = QLabel("Provider: Nicht verfügbar")
        self.cache_label = QLabel("Cache: --")
        self.pipeline_label = QLabel("Pipeline: Bereit")
        self.progress_bar = QProgressBar()

        # Setup UI
        self.setup_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_cache_status)
        self.update_timer.start(5000)  # Update every 5 seconds

        # Current state
        self.current_provider = "Nicht verfügbar"
        self.current_model = ""
        self.cache_manager: Optional[UnifiedKnowledgeManager] = None
        self.llm_service: Optional[LlmService] = None

    def setup_ui(self):
        """Setup the status bar UI - Claude Generated"""
        # Create main widget for status components
        status_widget = QWidget()
        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(5, 0, 5, 0)

        # Provider info
        set_style_prop(self.provider_label, "chip", "info")
        layout.addWidget(self.provider_label)

        # Cache status
        set_style_prop(self.cache_label, "chip", "success")
        layout.addWidget(self.cache_label)

        # Pipeline status
        set_style_prop(self.pipeline_label, "chip", "warning")
        layout.addWidget(self.pipeline_label)

        # Progress bar (hidden by default)
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        # Border, ground and the gradient chunk: QProgressBar rules of the main stylesheet
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # Add to status bar
        self.addPermanentWidget(status_widget)

    def set_services(self, llm_service: LlmService, cache_manager: UnifiedKnowledgeManager):
        """Set the services for status monitoring - Claude Generated"""
        self.llm_service = llm_service
        self.cache_manager = cache_manager

        # Connect to LLM service signals if available
        if hasattr(llm_service, "provider_changed"):
            llm_service.provider_changed.connect(self.update_provider_info)

        # Initial update - don't load models during startup - Claude Generated
        self.update_provider_info_lazy()
        self.update_cache_status()

    def update_provider_info_lazy(self):
        """Update provider display without initializing providers - Claude Generated"""
        if self.llm_service:
            try:
                # Just show that providers are available without initializing them
                providers = self.llm_service.get_available_providers()
                if providers:
                    provider_count = len(providers)
                    self.provider_label.setText(f"Provider: {provider_count} verfügbar")
                    set_style_prop(self.provider_label, "chip", "success")
                else:
                    self.provider_label.setText("Provider: Nicht verfügbar")
                    set_style_prop(self.provider_label, "chip", "error")
            except Exception as e:
                self.logger.error(f"Error updating provider info lazily: {e}")
                self.provider_label.setText("Provider: Error")

    def update_provider_info(self, provider: str = None, model: str = None,
                             allow_network: bool = True):
        """Update provider information display - Claude Generated

        ``allow_network=False`` fragt keine Modellliste ab (das ist ein
        synchroner Netzaufruf im UI-Thread, der bei einem nicht antwortenden
        Server beliebig lange blockiert); ein bekanntes Modell desselben
        Providers bleibt stehen.
        """
        if self.llm_service:
            try:
                if not provider:
                    providers = self.llm_service.get_available_providers()
                    provider = providers[0] if providers else "Nicht verfügbar"

                if not model and provider != "Nicht verfügbar":
                    if allow_network:
                        models = self.llm_service.get_available_models(provider)
                        model = models[0] if models else "Kein Modell"
                    elif provider == getattr(self, "current_provider", None):
                        model = getattr(self, "current_model", "")

                self.current_provider = provider
                self.current_model = model or ""

                if provider == "Nicht verfügbar":
                    self.provider_label.setText("Provider: Nicht verfügbar")
                    set_style_prop(self.provider_label, "chip", "error")
                else:
                    display_text = f"Provider: {provider}"
                    if model:
                        # Truncate long model names
                        short_model = model[:20] + "..." if len(model) > 20 else model
                        display_text += f" ({short_model})"

                    self.provider_label.setText(display_text)
                    set_style_prop(self.provider_label, "chip", "info")

            except Exception as e:
                self.logger.error(f"Error updating provider info: {e}")
                self.provider_label.setText("Provider: Fehler")
        else:
            self.provider_label.setText("Provider: Nicht initialisiert")

    def update_cache_status(self):
        """Update cache status display - Claude Generated"""
        if self.cache_manager:
            try:
                # Get cache statistics
                cache_stats = self.cache_manager.get_cache_stats()
                if cache_stats:
                    entries = cache_stats.get("total_entries", 0)
                    size_mb = cache_stats.get("size_mb", 0)

                    if entries > 0:
                        self.cache_label.setText(
                            f"Cache: {entries} Einträge ({size_mb:.1f}MB)"
                        )
                        set_style_prop(self.cache_label, "chip", "success")
                    else:
                        self.cache_label.setText("Cache: Leer")
                        set_style_prop(self.cache_label, "chip", "warning")
                else:
                    self.cache_label.setText("Cache: Nicht verfügbar")

            except Exception as e:
                self.logger.error(f"Error updating cache status: {e}")
                self.cache_label.setText("Cache: Fehler")
        else:
            self.cache_label.setText("Cache: Nicht initialisiert")

    def update_pipeline_status(self, step_name: str, status: str, progress: int = -1):
        """Update pipeline status display - Claude Generated"""
        if status == "running":
            self.pipeline_label.setText(f"Pipeline: {step_name} läuft...")
            set_style_prop(self.pipeline_label, "chip", "info")

            if progress >= 0:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(progress)
            else:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate

        elif status == "completed":
            self.pipeline_label.setText(f"Pipeline: {step_name} abgeschlossen ✓")
            set_style_prop(self.pipeline_label, "chip", "success")
            self.progress_bar.setVisible(False)

        elif status == "error":
            self.pipeline_label.setText(f"Pipeline: {step_name} Fehler ✗")
            set_style_prop(self.pipeline_label, "chip", "error")
            self.progress_bar.setVisible(False)

        elif status == "ready":
            self.pipeline_label.setText("Pipeline: Bereit")
            set_style_prop(self.pipeline_label, "chip", "warning")
            self.progress_bar.setVisible(False)

        # Emit signal for other components
        self.pipeline_step_changed.emit(step_name, status)

    def show_temporary_message(self, message: str, duration: int = 3000):
        """Show a temporary message in the status bar - Claude Generated"""
        self.showMessage(message, duration)

    def get_current_provider_info(self) -> Dict[str, str]:
        """Get current provider information - Claude Generated"""
        return {"provider": self.current_provider, "model": self.current_model}
