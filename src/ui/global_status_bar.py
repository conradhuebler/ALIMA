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
from ..core.cache_manager import CacheManager


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
        self.cache_manager: Optional[CacheManager] = None
        self.llm_service: Optional[LlmService] = None

    def setup_ui(self):
        """Setup the status bar UI - Claude Generated"""
        # Create main widget for status components
        status_widget = QWidget()
        layout = QHBoxLayout(status_widget)
        layout.setContentsMargins(5, 0, 5, 0)

        # Provider info
        self.provider_label.setStyleSheet(
            """
            QLabel {
                color: #2196f3;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 3px;
                background-color: rgba(33, 150, 243, 0.1);
            }
        """
        )
        layout.addWidget(self.provider_label)

        # Cache status
        self.cache_label.setStyleSheet(
            """
            QLabel {
                color: #4caf50;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 3px;
                background-color: rgba(76, 175, 80, 0.1);
            }
        """
        )
        layout.addWidget(self.cache_label)

        # Pipeline status
        self.pipeline_label.setStyleSheet(
            """
            QLabel {
                color: #ff9800;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 3px;
                background-color: rgba(255, 152, 0, 0.1);
            }
        """
        )
        layout.addWidget(self.pipeline_label)

        # Progress bar (hidden by default)
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 2px;
            }
        """
        )
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # Add to status bar
        self.addPermanentWidget(status_widget)

    def set_services(self, llm_service: LlmService, cache_manager: CacheManager):
        """Set the services for status monitoring - Claude Generated"""
        self.llm_service = llm_service
        self.cache_manager = cache_manager

        # Connect to LLM service signals if available
        if hasattr(llm_service, "provider_changed"):
            llm_service.provider_changed.connect(self.update_provider_info)

        # Initial update
        self.update_provider_info()
        self.update_cache_status()

    def update_provider_info(self, provider: str = None, model: str = None):
        """Update provider information display - Claude Generated"""
        if self.llm_service:
            try:
                if not provider:
                    providers = self.llm_service.get_available_providers()
                    provider = providers[0] if providers else "Nicht verfügbar"

                if not model and provider != "Nicht verfügbar":
                    models = self.llm_service.get_available_models(provider)
                    model = models[0] if models else "Kein Modell"

                self.current_provider = provider
                self.current_model = model or ""

                if provider == "Nicht verfügbar":
                    self.provider_label.setText("Provider: Nicht verfügbar")
                    self.provider_label.setStyleSheet(
                        """
                        QLabel {
                            color: #d32f2f;
                            font-weight: bold;
                            padding: 2px 8px;
                            border-radius: 3px;
                            background-color: rgba(211, 47, 47, 0.1);
                        }
                    """
                    )
                else:
                    display_text = f"Provider: {provider}"
                    if model:
                        # Truncate long model names
                        short_model = model[:20] + "..." if len(model) > 20 else model
                        display_text += f" ({short_model})"

                    self.provider_label.setText(display_text)
                    self.provider_label.setStyleSheet(
                        """
                        QLabel {
                            color: #2196f3;
                            font-weight: bold;
                            padding: 2px 8px;
                            border-radius: 3px;
                            background-color: rgba(33, 150, 243, 0.1);
                        }
                    """
                    )

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
                        self.cache_label.setStyleSheet(
                            """
                            QLabel {
                                color: #4caf50;
                                font-weight: bold;
                                padding: 2px 8px;
                                border-radius: 3px;
                                background-color: rgba(76, 175, 80, 0.1);
                            }
                        """
                        )
                    else:
                        self.cache_label.setText("Cache: Leer")
                        self.cache_label.setStyleSheet(
                            """
                            QLabel {
                                color: #ff9800;
                                font-weight: bold;
                                padding: 2px 8px;
                                border-radius: 3px;
                                background-color: rgba(255, 152, 0, 0.1);
                            }
                        """
                        )
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
            self.pipeline_label.setStyleSheet(
                """
                QLabel {
                    color: #2196f3;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                    background-color: rgba(33, 150, 243, 0.1);
                }
            """
            )

            if progress >= 0:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(progress)
            else:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate

        elif status == "completed":
            self.pipeline_label.setText(f"Pipeline: {step_name} abgeschlossen ✓")
            self.pipeline_label.setStyleSheet(
                """
                QLabel {
                    color: #4caf50;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                    background-color: rgba(76, 175, 80, 0.1);
                }
            """
            )
            self.progress_bar.setVisible(False)

        elif status == "error":
            self.pipeline_label.setText(f"Pipeline: {step_name} Fehler ✗")
            self.pipeline_label.setStyleSheet(
                """
                QLabel {
                    color: #d32f2f;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                    background-color: rgba(211, 47, 47, 0.1);
                }
            """
            )
            self.progress_bar.setVisible(False)

        elif status == "ready":
            self.pipeline_label.setText("Pipeline: Bereit")
            self.pipeline_label.setStyleSheet(
                """
                QLabel {
                    color: #ff9800;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                    background-color: rgba(255, 152, 0, 0.1);
                }
            """
            )
            self.progress_bar.setVisible(False)

        # Emit signal for other components
        self.pipeline_step_changed.emit(step_name, status)

    def show_temporary_message(self, message: str, duration: int = 3000):
        """Show a temporary message in the status bar - Claude Generated"""
        self.showMessage(message, duration)

    def get_current_provider_info(self) -> Dict[str, str]:
        """Get current provider information - Claude Generated"""
        return {"provider": self.current_provider, "model": self.current_model}
