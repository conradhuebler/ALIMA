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
    repetition_detected = pyqtSignal(object, list, bool, bool, float)  # result, suggestions, grace_period, resolved, grace_seconds - Claude Generated (2026-02-17)

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

    def abort_current_step(self) -> None:
        """Abort only the current LLM generation; pipeline continues - Claude Generated"""
        self.pipeline_manager.abort_current_step()

    def run(self):
        """Execute pipeline in background thread - Claude Generated"""
        try:
            # Check for interruption before starting
            self.check_interruption()

            # Set up callbacks to emit signals - Claude Generated (updated 2026-02-17)
            self.pipeline_manager.set_callbacks(
                step_started=self.step_started.emit,
                step_completed=self.step_completed.emit,
                step_error=self.step_error.emit,
                pipeline_completed=self.pipeline_completed.emit,
                stream_callback=self.stream_token.emit,
                repetition_detected=self.repetition_detected.emit,
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


class DNBSyncWorker(QThread):
    """Worker thread for batch DNB sync operations - Claude Generated

    Fetches updated DNB classifications for multiple GND entries in parallel.
    Emits progress signals for UI feedback.
    """

    # Signals
    progress = pyqtSignal(int)  # Progress percentage (0-100)
    entry_synced = pyqtSignal(str, bool)  # gnd_id, success
    finished = pyqtSignal(int, int)  # success_count, error_count

    def __init__(self, gnd_ids, cache_manager):
        """Initialize DNB sync worker

        Args:
            gnd_ids: List of GND-IDs to sync
            cache_manager: UnifiedKnowledgeManager instance for DB updates
        """
        super().__init__()
        self.gnd_ids = gnd_ids
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute batch DNB sync in background thread - Claude Generated"""
        from ..core.dnb_utils import get_dnb_classification

        success = 0
        errors = 0
        total = len(self.gnd_ids)

        self.logger.info(f"Starting DNB sync for {total} entries")

        for i, gnd_id in enumerate(self.gnd_ids):
            try:
                # Fetch DNB classification
                dnb_class = get_dnb_classification(gnd_id)

                if dnb_class and dnb_class.get("status") == "success":
                    # Extract data
                    term = dnb_class.get("preferred_name", "")

                    # DDCs extrahieren und formatieren
                    ddc_list = dnb_class.get("ddc", [])
                    ddc = ";".join(f"{d['code']}({d['determinancy']})" for d in ddc_list)

                    # GND-Kategorien extrahieren
                    gnd_category = dnb_class.get("gnd_subject_categories", [])
                    gnd_category = ";".join(gnd_category)

                    # Allgemeine Kategorie
                    category = dnb_class.get("category", "")

                    # Update database
                    self.cache_manager.update_gnd_entry(
                        gnd_id,
                        title=term,
                        ddcs=ddc,
                        gnd_systems=gnd_category,
                        classification=category,
                    )

                    success += 1
                    self.entry_synced.emit(gnd_id, True)
                    self.logger.debug(f"Successfully synced {gnd_id}")
                else:
                    errors += 1
                    self.entry_synced.emit(gnd_id, False)
                    self.logger.warning(f"Failed to sync {gnd_id}: {dnb_class.get('error_message', 'Unknown error')}")

            except Exception as e:
                errors += 1
                self.entry_synced.emit(gnd_id, False)
                self.logger.error(f"Error syncing {gnd_id}: {e}")

            # Update progress
            progress_percent = int((i + 1) / total * 100)
            self.progress.emit(progress_percent)

        self.finished.emit(success, errors)
        self.logger.info(f"DNB sync completed: {success} success, {errors} errors")