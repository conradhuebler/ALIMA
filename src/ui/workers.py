"""
Centralized Worker Threads for ALIMA UI Components
Claude Generated - Provides shared worker threads to avoid code duplication
"""

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from typing import Optional
import logging
import threading

from ..core.pipeline_manager import PipelineManager, PipelineConfig


class StoppableWorker(QThread):
    """Generic base class for stoppable worker threads - Claude Generated

    Provides interrupt mechanism for graceful worker termination.
    Child classes should check is_interrupted() at appropriate points.
    """

    # Signal emitted when worker is aborted by user request
    aborted = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_interrupted = False
        self._interrupt_lock = threading.Lock()  # Use threading.Lock instead of QMutex for simplicity
        self.logger = logging.getLogger(__name__)

    def request_stop(self):
        """Request worker to stop gracefully - Claude Generated

        Sets interrupt flag. Worker should check this flag periodically
        and exit cleanly when interrupted.
        """
        with self._interrupt_lock:
            self._is_interrupted = True
            self.logger.debug(f"{self.__class__.__name__}: Stop requested")

    def is_interrupted(self) -> bool:
        """Check if interruption was requested - Claude Generated

        Thread-safe check of interrupt flag.

        Returns:
            bool: True if stop has been requested, False otherwise
        """
        with self._interrupt_lock:
            return self._is_interrupted

    def check_interruption(self):
        """Check and raise exception if interrupted - Claude Generated

        Raises:
            InterruptedError: If interruption was requested
        """
        if self.is_interrupted():
            self.logger.debug(f"{self.__class__.__name__}: Interruption detected, raising InterruptedError")
            raise InterruptedError("Operation cancelled by user")


class SingleStepWorker(QThread):
    """Worker thread for single pipeline step execution - Claude Generated"""

    # Signals
    step_completed = pyqtSignal(object)  # PipelineStep
    step_error = pyqtSignal(str)  # error_message
    stream_token = pyqtSignal(str, str)  # token, step_id

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        step_config: PipelineConfig,
        input_data: Optional[str] = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.step_config = step_config
        self.input_data = input_data
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute single step in background thread - Claude Generated"""
        try:
            # Set streaming callback
            self.pipeline_manager.stream_callback = self.stream_token.emit

            # Execute single step
            # For now, we use the config's first enabled step
            for step_id, config in self.step_config.step_configs.items():
                if getattr(config, 'enabled', True):
                    result_step = self.pipeline_manager.execute_single_step(
                        step_id, self.step_config, self.input_data
                    )
                    self.step_completed.emit(result_step)
                    break

        except Exception as e:
            self.logger.error(f"Single step worker error: {e}")
            self.step_error.emit(str(e))


class PipelineWorker(StoppableWorker):
    """Worker thread for pipeline execution - Claude Generated

    Extends StoppableWorker to support graceful pipeline interruption.
    """

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
        force_update: bool = False,  # Claude Generated
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.input_text = input_text
        self.input_type = input_type
        self.force_update = force_update  # Claude Generated
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute pipeline in background thread - Claude Generated"""
        try:
            # Check for interruption before starting
            self.check_interruption()

            # Set up callbacks to emit signals
            self.pipeline_manager.set_callbacks(
                step_started=self.step_started.emit,
                step_completed=self.step_completed.emit,
                step_error=self.step_error.emit,
                pipeline_completed=self.pipeline_completed.emit,
                stream_callback=self.stream_token.emit,
            )

            # Set interrupt flag in pipeline manager for step-level checks
            if hasattr(self.pipeline_manager, 'set_interrupt_flag'):
                self.pipeline_manager.set_interrupt_flag(self._interrupt_lock, self.is_interrupted)

            # Start pipeline - Claude Generated (added force_update parameter)
            pipeline_id = self.pipeline_manager.start_pipeline(
                self.input_text, self.input_type, force_update=self.force_update
            )
            self.logger.info(f"Pipeline {pipeline_id} completed in worker thread")

        except InterruptedError:
            self.logger.info("Pipeline interrupted by user")
            self.aborted.emit()
        except Exception as e:
            self.logger.error(f"Pipeline worker error: {e}")
            # Emit error signal if needed