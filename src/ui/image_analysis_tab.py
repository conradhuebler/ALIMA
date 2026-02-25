import sys
import os
import logging
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QComboBox,
    QGroupBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QProgressBar,
    QFrame,
    QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from .workers import StoppableWorker
from PyQt6.QtGui import QPixmap, QFont
import PIL.Image
import json
from pathlib import Path
import uuid
from .styles import (
    get_main_stylesheet,
    get_button_styles,
    get_status_label_styles,
    get_image_preview_style,
    LAYOUT,
    COLORS,
)

# Simple text recognition prompt
DEFAULT_PROMPT = """Bitte extrahiere den gesamten lesbaren Text aus diesem Bild. 
Gib nur den Text zurück, ohne zusätzliche Formatierung oder Kommentare.
Achte darauf, dass der Text genau so ausgegeben wird, wie er im Bild steht."""


class ImageAnalysisWorker(StoppableWorker):
    """Worker thread for image analysis - Claude Generated

    Extends StoppableWorker to support graceful analysis interruption.
    """

    analysis_finished = pyqtSignal(str)
    analysis_error = pyqtSignal(str)

    def __init__(
        self,
        llm_service,
        provider,
        model,
        prompt,
        image_path,
        temperature=0.7,
        seed=None,
    ):
        super().__init__()
        self.llm_service = llm_service
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.image_path = image_path
        self.temperature = temperature
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            # Check for interruption before starting analysis
            self.check_interruption()

            request_id = str(uuid.uuid4())

            # Check for interruption before making LLM request
            self.check_interruption()

            response = self.llm_service.generate_response(
                provider=self.provider,
                model=self.model,
                prompt=self.prompt,
                request_id=request_id,
                temperature=self.temperature,
                seed=self.seed,
                image=self.image_path,
                stream=False,
            )

            # Check for interruption after receiving response
            self.check_interruption()

            # Handle generator objects (e.g., from Ollama)
            if hasattr(response, "__iter__") and not isinstance(response, str):
                # If response is a generator, collect all chunks
                full_response = ""
                for chunk in response:
                    # Check for interruption during iteration
                    self.check_interruption()

                    if isinstance(chunk, str):
                        full_response += chunk
                    elif hasattr(chunk, "text"):
                        full_response += chunk.text
                    elif hasattr(chunk, "content"):
                        full_response += chunk.content
                    else:
                        full_response += str(chunk)
                response = full_response

            self.analysis_finished.emit(response)

        except InterruptedError:
            self.logger.info("Image analysis interrupted by user")
            self.aborted.emit()
        except Exception as e:
            self.logger.error(f"Image analysis error: {e}")
            self.analysis_error.emit(str(e))


class ImageAnalysisTab(QWidget):
    """Tab for image analysis with text extraction"""

    # Signal to send extracted text to abstract tabs
    text_extracted = pyqtSignal(str)

    def __init__(self, llm_service, main_window=None, parent=None):
        super().__init__(parent)
        self.llm_service = llm_service
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

        self.image_path = None
        self.current_worker = None
        self._analysis_was_aborted = False  # Claude Generated - Track if analysis was aborted by user

        self.setup_ui()
        self.load_providers_and_models()

    def setup_ui(self):
        """Setup the user interface"""
        # Apply main stylesheet
        self.setStyleSheet(get_main_stylesheet())
        btn_styles = get_button_styles()

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(LAYOUT["spacing"])
        main_layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"]
        )

        # Control bar at the top
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)

        # Status label
        self.status_label = QLabel("Status: Bereit")
        self.status_label.setStyleSheet(get_status_label_styles()["info"])
        control_bar.addWidget(self.status_label)

        control_bar.addStretch()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFormat("Analysiere... %p%")
        control_bar.addWidget(self.progress_bar)

        #main_layout.addLayout(control_bar)

        # Create splitter for main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel for image and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(LAYOUT["spacing"])

        # File selection group
        file_group = QGroupBox("Bildauswahl")
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(10, 20, 10, 10)
        file_layout.setSpacing(LAYOUT["inner_spacing"])

        self.file_button = QPushButton("📁 Bild auswählen")
        self.file_button.setStyleSheet(btn_styles["secondary"])
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_button)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(350, 250)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet(get_image_preview_style())
        self.image_preview.setText(
            "📷\n\nKein Bild ausgewählt\n\nKlicken Sie auf 'Bild auswählen'"
        )
        self.image_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        file_layout.addWidget(self.image_preview)

        left_layout.addWidget(file_group)

        # Provider settings
        provider_group = QGroupBox("LLM Provider Einstellungen")
        provider_layout = QVBoxLayout(provider_group)
        provider_layout.setContentsMargins(10, 20, 10, 10)
        provider_layout.setSpacing(LAYOUT["inner_spacing"])

        # Provider selection
        provider_grid = QGridLayout()
        provider_grid.setSpacing(LAYOUT["inner_spacing"])

        provider_grid.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_grid.addWidget(self.provider_combo, 0, 1)

        provider_grid.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        provider_grid.addWidget(self.model_combo, 1, 1)

        provider_layout.addLayout(provider_grid)

        # Temperature slider
        temp_layout = QHBoxLayout()
        self.temperature_label = QLabel("Temperatur: 0.70")
        temp_layout.addWidget(self.temperature_label)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        temp_layout.addWidget(self.temperature_slider)
        provider_layout.addLayout(temp_layout)

        # Seed input
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Seed:"))
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1000000000)
        self.seed_input.setValue(0)
        seed_layout.addWidget(self.seed_input)
        provider_layout.addLayout(seed_layout)

        left_layout.addWidget(provider_group)
        left_layout.addStretch()
        main_splitter.addWidget(left_panel)

        # Right panel for prompt and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(LAYOUT["spacing"])

        # Prompt group
        prompt_group = QGroupBox("Prompt Konfiguration")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setContentsMargins(10, 20, 10, 10)
        prompt_layout.setSpacing(LAYOUT["inner_spacing"])

        prompt_label = QLabel("Prompt für Textextraktion:")
        prompt_label.setStyleSheet("font-weight: bold;")
        prompt_layout.addWidget(prompt_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Prompt für die Bildanalyse...")
        self.prompt_input.setText(DEFAULT_PROMPT)
        self.prompt_input.setMaximumHeight(120)
        self.prompt_input.setFont(QFont("Segoe UI", LAYOUT["input_font_size"]))
        prompt_layout.addWidget(self.prompt_input)

        right_layout.addWidget(prompt_group)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(LAYOUT["spacing"])

        self.analyze_button = QPushButton("🔍 Text extrahieren")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setStyleSheet(btn_styles["primary"])
        button_layout.addWidget(self.analyze_button)

        # Stop button
        self.stop_analysis_button = QPushButton("⏹️ Stop")
        self.stop_analysis_button.setStyleSheet(btn_styles["error"])
        self.stop_analysis_button.setVisible(False)
        self.stop_analysis_button.clicked.connect(self.on_stop_analysis_requested)
        button_layout.addWidget(self.stop_analysis_button)

        self.send_to_abstract_button = QPushButton("📤 An Abstract-Tab senden")
        self.send_to_abstract_button.clicked.connect(self.send_to_abstract_tab)
        self.send_to_abstract_button.setEnabled(False)
        self.send_to_abstract_button.setStyleSheet(btn_styles["success"])
        button_layout.addWidget(self.send_to_abstract_button)

        button_layout.addStretch()
        right_layout.addLayout(button_layout)

        # Results group
        results_group = QGroupBox("Extrahierter Text")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(10, 20, 10, 10)

        self.output_field = QTextEdit()
        self.output_field.setReadOnly(True)
        self.output_field.setPlaceholderText(
            "Der extrahierte Text wird hier angezeigt..."
        )
        self.output_field.setFont(QFont("Segoe UI", LAYOUT["input_font_size"]))
        results_layout.addWidget(self.output_field)

        right_layout.addWidget(results_group)

        main_splitter.addWidget(right_panel)

        # Set splitter sizes
        main_splitter.setSizes([400, 600])

        main_layout.addWidget(main_splitter)

    def refresh_styles(self):
        """Re-apply styles after theme change — Claude Generated"""
        from .styles import get_main_stylesheet, get_status_label_styles
        self.setStyleSheet(get_main_stylesheet())
        if hasattr(self, 'status_label'):
            self.status_label.setStyleSheet(get_status_label_styles()["info"])

    def load_providers_and_models(self):
        """Load available providers and models"""
        try:
            providers = self.llm_service.get_available_providers()
            self.provider_combo.addItems(providers)

            if providers:
                self.update_models(providers[0])
                self.status_label.setText(
                    f"Status: Bereit - {len(providers)} Provider verfügbar"
                )
                self.status_label.setStyleSheet(get_status_label_styles()["success"])
            else:
                self.status_label.setText("Status: Keine Provider verfügbar")
                self.status_label.setStyleSheet(get_status_label_styles()["error"])

        except Exception as e:
            self.logger.error(f"Error loading providers: {e}")
            self.status_label.setText(f"Status: Fehler beim Laden der Provider")
            self.status_label.setStyleSheet(get_status_label_styles()["error"])

    def update_models(self, provider):
        """Update available models when provider changes"""
        self.model_combo.clear()

        try:
            models = self.llm_service.get_available_models(provider)
            self.model_combo.addItems(models)

            if models:
                self.model_combo.setCurrentIndex(0)
                self.status_label.setText(
                    f"Status: {provider} bereit - {len(models)} Modelle verfügbar"
                )
                self.status_label.setStyleSheet(get_status_label_styles()["success"])
            else:
                self.status_label.setText(
                    f"Status: Keine Modelle für {provider} verfügbar"
                )
                self.status_label.setStyleSheet(get_status_label_styles()["warning"])

        except Exception as e:
            self.logger.error(f"Error loading models for {provider}: {e}")
            self.status_label.setText(f"Status: Fehler beim Laden der Modelle")
            self.status_label.setStyleSheet(get_status_label_styles()["error"])

    def update_temperature_label(self, value):
        """Update temperature label"""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")

    def select_file(self):
        """File selection dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Bild auswählen", "", "Bilder (*.png *.jpg *.jpeg *.gif *.bmp *.tiff)"
        )

        if file_path:
            self.image_path = file_path
            self.file_button.setText(f"📁 {Path(file_path).name}")
            self.update_image_preview()
            self.status_label.setText(
                f"Status: Bild ausgewählt - {Path(file_path).name}"
            )
            self.status_label.setStyleSheet(get_status_label_styles()["info"])

    def update_image_preview(self):
        """Update image preview"""
        if self.image_path:
            try:
                pixmap = QPixmap(self.image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.image_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.image_preview.setPixmap(scaled_pixmap)
                else:
                    self.image_preview.setText("Bild konnte nicht geladen werden")
            except Exception as e:
                self.logger.error(f"Error loading image preview: {e}")
                self.image_preview.setText("Fehler beim Laden des Bildes")

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.image_path:
            self.update_image_preview()

    def analyze_image(self):
        """Analyze the selected image"""
        # Reset abort flag for new analysis - Claude Generated
        self._analysis_was_aborted = False

        if not self.image_path:
            QMessageBox.warning(
                self, "Warnung", "Bitte wählen Sie zuerst ein Bild aus!"
            )
            return

        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText().strip()
        prompt = self.prompt_input.toPlainText().strip()

        if not provider or not model:
            QMessageBox.warning(
                self, "Warnung", "Bitte wählen Sie einen Provider und ein Modell aus!"
            )
            return

        if not prompt:
            QMessageBox.warning(self, "Warnung", "Bitte geben Sie einen Prompt ein!")
            return

        # Disable UI during analysis
        self.analyze_button.setEnabled(False)
        self.stop_analysis_button.setVisible(True)  # Show stop button - Claude Generated
        self.stop_analysis_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText(
            f"Status: Analysiere Bild mit {provider} ({model})..."
        )
        self.status_label.setStyleSheet(get_status_label_styles()["info"])

        # Start analysis in worker thread
        self.current_worker = ImageAnalysisWorker(
            self.llm_service,
            provider,
            model,
            prompt,
            self.image_path,
            self.temperature_slider.value() / 100,
            self.seed_input.value() if self.seed_input.value() > 0 else None,
        )

        self.current_worker.analysis_finished.connect(self.on_analysis_finished)
        self.current_worker.analysis_error.connect(self.on_analysis_error)
        self.current_worker.aborted.connect(self.on_analysis_aborted)  # Claude Generated
        self.current_worker.start()

    @pyqtSlot(str)
    def on_analysis_finished(self, result):
        """Handle successful analysis completion"""
        self.output_field.setText(result)
        self.send_to_abstract_button.setEnabled(True)

        # Re-enable UI
        self.analyze_button.setEnabled(True)
        self.stop_analysis_button.setVisible(False)  # Hide stop button - Claude Generated
        self.progress_bar.setVisible(False)
        self.status_label.setText("Status: Analyse erfolgreich abgeschlossen")
        self.status_label.setStyleSheet(get_status_label_styles()["success"])

        self.logger.info(f"Image analysis completed successfully")

    @pyqtSlot(str)
    def on_analysis_error(self, error):
        """Handle analysis error"""
        self.output_field.setText(f"❌ Fehler bei der Analyse:\n\n{error}")

        # Re-enable UI
        self.analyze_button.setEnabled(True)
        self.stop_analysis_button.setVisible(False)  # Hide stop button - Claude Generated
        self.progress_bar.setVisible(False)
        self.status_label.setText("Status: Fehler bei der Analyse")
        self.status_label.setStyleSheet(get_status_label_styles()["error"])

        # Don't show error dialog if analysis was aborted - Claude Generated
        if self._analysis_was_aborted:
            self._analysis_was_aborted = False  # Reset flag
            self.logger.info("Suppressing error dialog - analysis was aborted by user")
            return

        self.logger.error(f"Image analysis error: {error}")
        QMessageBox.critical(self, "Fehler", f"Fehler bei der Bildanalyse:\n{error}")

    def on_stop_analysis_requested(self):
        """Handle stop button click - Claude Generated"""
        if self.current_worker and self.current_worker.isRunning():
            self.logger.info("User requested image analysis stop")
            self.stop_analysis_button.setEnabled(False)
            self.stop_analysis_button.setText("⏹️ Stopping...")
            self.status_label.setText("Status: Beende Analyse...")
            self.current_worker.request_stop()

    @pyqtSlot()
    def on_analysis_aborted(self):
        """Handle analysis abort signal - Claude Generated"""
        self.logger.info("Image analysis aborted by user")

        # Set abort flag to suppress error dialog - Claude Generated
        self._analysis_was_aborted = True

        # Reset UI
        self.analyze_button.setEnabled(True)
        self.stop_analysis_button.setVisible(False)
        self.stop_analysis_button.setText("⏹️ Stop")
        self.stop_analysis_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Update status
        self.status_label.setText("Status: Analyse abgebrochen")
        self.status_label.setStyleSheet(get_status_label_styles()["warning"])

        # Show abort message
        self.output_field.setText("⏹️ Analyse wurde vom Benutzer abgebrochen")

    def send_to_abstract_tab(self):
        """Send extracted text to abstract tab"""
        text = self.output_field.toPlainText().strip()

        if not text:
            QMessageBox.warning(self, "Warnung", "Kein Text zum Senden vorhanden!")
            return

        # Emit signal to send text to abstract tab
        self.text_extracted.emit(text)

        # Show confirmation
        QMessageBox.information(
            self, "Erfolg", "Text wurde an die Abstract-Tab gesendet!"
        )

        # Switch to abstract tab if main window is available
        if self.main_window and hasattr(self.main_window, "tabs"):
            # Find abstract tab (usually index 2 after Crossref and Bilderkennung)
            for i in range(self.main_window.tabs.count()):
                if "Abstract" in self.main_window.tabs.tabText(i):
                    self.main_window.tabs.setCurrentIndex(i)
                    break

        self.status_label.setText("Status: Text an Abstract-Tab gesendet")
        self.status_label.setStyleSheet(get_status_label_styles()["success"])
