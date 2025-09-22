"""
Centralized Worker Threads for ALIMA UI Components
Claude Generated - Provides shared worker threads to avoid code duplication
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional
import logging

from ..core.pipeline_manager import PipelineManager


class PipelineWorker(QThread):
    """Worker thread for pipeline execution - Claude Generated"""

    # Signals for pipeline events
    step_started = pyqtSignal(object)  # PipelineStep
    step_completed = pyqtSignal(object)  # PipelineStep
    step_error = pyqtSignal(object, str)  # PipelineStep, error_message
    pipeline_completed = pyqtSignal(object)  # analysis_state
    stream_token = pyqtSignal(str, str)  # token, step_id

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        input_text: str,
        input_type: str = "text",
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.input_text = input_text
        self.input_type = input_type
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute pipeline in background thread - Claude Generated"""
        try:
            # Set up callbacks to emit signals
            self.pipeline_manager.set_callbacks(
                step_started=self.step_started.emit,
                step_completed=self.step_completed.emit,
                step_error=self.step_error.emit,
                pipeline_completed=self.pipeline_completed.emit,
                stream_callback=self.stream_token.emit,
            )

            # Start pipeline
            pipeline_id = self.pipeline_manager.start_pipeline(
                self.input_text, self.input_type
            )
            self.logger.info(f"Pipeline {pipeline_id} completed in worker thread")

        except Exception as e:
            self.logger.error(f"Pipeline worker error: {e}")
            # Emit error signal if needed