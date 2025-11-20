#!/usr/bin/env python3
"""
ALIMA First-Start Setup Wizard
Guide new users through initial configuration (LLM provider and GND database)
Claude Generated
"""

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QRadioButton, QButtonGroup, QComboBox, QPushButton, QProgressDialog,
    QMessageBox, QGroupBox, QSpinBox, QCheckBox, QFileDialog, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QPixmap
from pathlib import Path
import logging
from typing import Optional, List

from ..utils.config_manager import ConfigManager
from ..utils.setup_utils import (
    OllamaConnectionValidator, APIKeyValidator, GNDDatabaseDownloader,
    ConfigurationBuilder, PromptValidator, SetupResult
)


logger = logging.getLogger(__name__)


class FirstStartWizard(QWizard):
    """Main wizard for first-start setup - Claude Generated"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ALIMA Setup Wizard")
        self.setWindowIcon(QIcon("assets/alima.png"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Initialize pages
        self.welcome_page = WelcomePage()
        self.addPage(self.welcome_page)

        self.llm_page = LLMSetupPage()
        self.addPage(self.llm_page)

        self.model_page = ModelSelectionPage()
        self.addPage(self.model_page)

        self.gnd_page = GNDDatabasePage()
        self.addPage(self.gnd_page)

        self.summary_page = SummaryPage()
        self.addPage(self.summary_page)

        # Store configuration results
        self.config = None

    def accept(self):
        """Handle wizard completion - Claude Generated"""
        try:
            # Get LLM provider configuration from the LLM page
            provider_type = getattr(self.llm_page, 'provider_type', 'ollama')
            provider_name = getattr(self.llm_page, 'provider_name', 'ALIMA Provider')
            base_url = getattr(self.llm_page, 'base_url', 'http://localhost:11434')
            api_key = getattr(self.llm_page, 'api_key', '')
            models = getattr(self.llm_page, 'available_models', [])

            # Get task model selections from model page - Claude Generated
            task_model_selections = getattr(
                self.model_page, 'task_model_selections', {}
            ) if hasattr(self, 'model_page') else {}

            # If no models were tested, use default model list
            if not models:
                if provider_type == 'ollama':
                    models = ['mistral']  # Default model
                elif provider_type == 'openai_compatible':
                    models = ['unknown']
                elif provider_type == 'gemini':
                    models = ['gemini-1.5-flash']
                elif provider_type == 'anthropic':
                    models = ['claude-3-5-haiku-20241022']

            # Build configuration with task selections - Claude Generated
            from ..utils.setup_utils import ConfigurationBuilder
            self.config = ConfigurationBuilder.create_initial_config(
                provider_type=provider_type,
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                models=models,
                task_model_selections=task_model_selections
            )

            # Mark first run as completed
            self.config.system_config.first_run_completed = True

            # Save configuration
            config_manager = ConfigManager()
            config_manager.save_config(self.config)

            # Handle GND download for post-wizard background import - Claude Generated
            if hasattr(self.gnd_page, 'downloaded_xml_path') and self.gnd_page.downloaded_xml_path:
                import json
                import tempfile
                from pathlib import Path

                marker_file = Path(tempfile.gettempdir()) / "alima_gnd_pending.json"
                marker_data = {
                    'xml_path': self.gnd_page.downloaded_xml_path,
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                }
                marker_file.write_text(json.dumps(marker_data))
                logger.info(f"GND import marker saved: {marker_file}")

                QMessageBox.information(
                    self,
                    "GND-Import läuft im Hintergrund",
                    "✅ Setup abgeschlossen!\n\n"
                    "Die GND-Datenbank wird im Hintergrund importiert (~5-10 Min).\n\n"
                    "Sie können ALIMA sofort nutzen. Lobid-API wird als Fallback "
                    "verwendet, bis der Import fertig ist."
                )

            logger.info(f"First-start wizard completed: provider={provider_type}, models={len(models)}")
            super().accept()

        except Exception as e:
            logger.error(f"Error completing wizard: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Setup Error", f"Error saving configuration:\n{str(e)}")


class WelcomePage(QWizardPage):
    """Welcome page introducing ALIMA - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to ALIMA")
        self.setSubTitle("Automatic Library Indexing and Metadata Analysis")

        layout = QVBoxLayout()

        # Title
        title = QLabel("Welcome to ALIMA!")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Description
        description = QLabel(
            "ALIMA helps you analyze library materials and extract metadata with AI.\n\n"
            "This wizard will guide you through:\n"
            "1. Setting up an LLM provider (local or cloud)\n"
            "2. Downloading optional GND authority data\n"
            "3. Reviewing your configuration\n\n"
            "Let's get started!"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Info box
        info = QLabel(
            "💡 Tip: You can always change these settings later in the application menu."
        )
        info.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        self.setLayout(layout)

    def validatePage(self) -> bool:
        """Page validation - Claude Generated"""
        return True


class LLMSetupPage(QWizardPage):
    """LLM Provider configuration page - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("LLM Provider Setup")
        self.setSubTitle("Choose where to run your AI models")

        # Store values from current configuration
        self.provider_type = "ollama"
        self.provider_name = "My Ollama Server"
        self.base_url = "http://localhost:11434"
        self.api_key = ""
        self.available_models = []

        layout = QVBoxLayout()

        # Provider selection
        self.provider_group = QButtonGroup()
        provider_box = QGroupBox("Select LLM Provider")
        provider_layout = QVBoxLayout()

        self.ollama_radio = QRadioButton("🔹 Ollama (Local Server) - Recommended")
        self.ollama_radio.setChecked(True)
        self.ollama_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.ollama_radio, 0)
        provider_layout.addWidget(self.ollama_radio)

        self.openai_radio = QRadioButton("🔷 OpenAI-Compatible API")
        self.openai_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.openai_radio, 1)
        provider_layout.addWidget(self.openai_radio)

        self.gemini_radio = QRadioButton("🟡 Google Gemini API")
        self.gemini_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.gemini_radio, 2)
        provider_layout.addWidget(self.gemini_radio)

        self.anthropic_radio = QRadioButton("🔴 Anthropic Claude API")
        self.anthropic_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.anthropic_radio, 3)
        provider_layout.addWidget(self.anthropic_radio)

        provider_box.setLayout(provider_layout)
        layout.addWidget(provider_box)

        # Configuration fields (dynamic based on provider)
        config_box = QGroupBox("Configuration")
        self.config_layout = QVBoxLayout()

        # Ollama - Host and Port
        self.ollama_host_label = QLabel("Host:")
        self.ollama_host_input = QLineEdit()
        self.ollama_host_input.setText("localhost")
        self.config_layout.addWidget(self.ollama_host_label)
        self.config_layout.addWidget(self.ollama_host_input)

        self.ollama_port_label = QLabel("Port:")
        self.ollama_port_input = QSpinBox()
        self.ollama_port_input.setValue(11434)
        self.ollama_port_input.setRange(1, 65535)
        self.config_layout.addWidget(self.ollama_port_label)
        self.config_layout.addWidget(self.ollama_port_input)

        # API Key (for cloud providers)
        self.api_key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setVisible(False)
        self.api_key_label.setVisible(False)
        self.config_layout.addWidget(self.api_key_label)
        self.config_layout.addWidget(self.api_key_input)

        # Base URL (for OpenAI-compatible)
        self.base_url_label = QLabel("API Base URL:")
        self.base_url_input = QLineEdit()
        self.base_url_input.setText("https://api.openai.com/v1")
        self.base_url_input.setToolTip("OpenAI: https://api.openai.com/v1 | Local: http://localhost:8000/v1")
        self.base_url_input.setVisible(False)
        self.base_url_label.setVisible(False)
        self.config_layout.addWidget(self.base_url_label)
        self.config_layout.addWidget(self.base_url_input)

        config_box.setLayout(self.config_layout)
        layout.addWidget(config_box)

        # Test connection button
        self.test_button = QPushButton("🧪 Test Connection")
        self.test_button.clicked.connect(self._test_connection)
        layout.addWidget(self.test_button)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def _on_provider_changed(self):
        """Update UI when provider selection changes - Claude Generated (fixed visibility)"""
        selected = self.provider_group.checkedId()

        # Ollama settings
        self.ollama_host_label.setVisible(selected == 0)
        self.ollama_host_input.setVisible(selected == 0)
        self.ollama_port_label.setVisible(selected == 0)
        self.ollama_port_input.setVisible(selected == 0)

        # OpenAI-compatible settings (selected == 1)
        self.base_url_label.setVisible(selected == 1)
        self.base_url_input.setVisible(selected == 1)

        # Cloud API settings (Gemini=2, Anthropic=3, OpenAI-Compatible=1)
        self.api_key_label.setVisible(selected in [1, 2, 3])
        self.api_key_input.setVisible(selected in [1, 2, 3])

        # Update API Key label text for clarity - Claude Generated
        if selected == 1:
            self.api_key_label.setText("API Key (optional for compatibility mode):")
        elif selected in [2, 3]:
            self.api_key_label.setText("API Key:")

        self.status_label.setText("")

    def _test_connection(self):
        """Test LLM provider connection - Claude Generated"""
        selected = self.provider_group.checkedId()

        if selected == 0:  # Ollama
            host = self.ollama_host_input.text()
            port = self.ollama_port_input.value()
            result = OllamaConnectionValidator.test_native(host, port)
            self.provider_type = "ollama"
            self.base_url = f"http://{host}:{port}"

        elif selected == 1:  # OpenAI-compatible
            base_url = self.base_url_input.text()
            api_key = self.api_key_input.text()
            result = OllamaConnectionValidator.test_openai_compatible(base_url, api_key)
            self.provider_type = "openai_compatible"
            self.base_url = base_url
            self.api_key = api_key

        elif selected == 2:  # Gemini
            api_key = self.api_key_input.text()
            result = APIKeyValidator.validate_gemini(api_key)
            self.provider_type = "gemini"
            self.api_key = api_key

        elif selected == 3:  # Anthropic
            api_key = self.api_key_input.text()
            result = APIKeyValidator.validate_anthropic(api_key)
            self.provider_type = "anthropic"
            self.api_key = api_key

        else:
            result = SetupResult(False, "Unknown provider")

        # Display result
        if result.success:
            self.status_label.setText(f"✅ {result.message}")
            self.status_label.setStyleSheet("color: green;")
            self.available_models = result.data or []
            self.provider_name = self.provider_type  # BUGFIX: Use actual provider type for consistency
        else:
            self.status_label.setText(f"❌ {result.message}")
            self.status_label.setStyleSheet("color: red;")

    def validatePage(self) -> bool:
        """Validate LLM configuration before proceeding - Claude Generated"""
        # Allow proceeding even without testing - user can test later
        if not self.available_models:
            # Warn but don't block
            result = QMessageBox.question(
                self,
                "No Connection Tested",
                "You haven't tested the LLM connection yet.\n\n"
                "Continue anyway? (you can test it after setup)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            return result == QMessageBox.StandardButton.Yes
        return True


class ModelSelectionPage(QWizardPage):
    """Model selection for different pipeline tasks - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("Modell-Auswahl für Pipeline-Schritte")
        self.setSubTitle("Wählen Sie für jeden Schritt das optimale Modell")

        # Store selections {task_type: model_name}
        self.task_model_selections = {}

        layout = QVBoxLayout()

        # Explanation
        explanation = QLabel(
            "Verschiedene Pipeline-Schritte haben unterschiedliche Anforderungen:\n"
            "• Initialisation & Keywords: Benötigen Reasoning-Fähigkeiten\n"
            "• Classification: Benötigt strukturiertes Denken\n"
            "• Vision: Benötigt Bildverständnis\n\n"
            "Sie können für jeden Schritt das optimale Modell wählen.\n"
            "Oder nutzen Sie den Button unten um ein Modell für alle zu verwenden."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Task selection form
        self.task_combos = {}

        # Relevant tasks (skip 'input', 'search', 'dk_search' - keine LLM-Tasks) - Claude Generated
        relevant_tasks = [
            ('initialisation', '🔤 Initialisation', 'Erste Keyword-Generierung'),
            ('keywords', '🔑 Keywords', 'Finale Keyword-Verifikation'),
            ('classification', '📚 Classification', 'DDC/DK/RVK Klassifizierung'),
            ('dk_classification', '📖 DK Classification', 'DK-spezifische Klassifizierung'),
            ('vision', '👁️ Vision', 'Bild-/OCR-Analyse'),
            ('chunked', '📄 Chunked', 'Große Texte in Chunks'),
        ]

        form_layout = QFormLayout()

        for task_key, task_label, task_desc in relevant_tasks:
            combo = QComboBox()
            combo.setMinimumWidth(300)
            combo.setToolTip(task_desc)

            label = QLabel(f"{task_label}:")
            label.setToolTip(task_desc)

            form_layout.addRow(label, combo)
            self.task_combos[task_key] = combo

        layout.addLayout(form_layout)

        # "Use same model for all" button
        self.use_same_btn = QPushButton("🔄 Gleiches Modell für alle Tasks verwenden")
        self.use_same_btn.clicked.connect(self._use_same_for_all)
        layout.addWidget(self.use_same_btn)

        layout.addStretch()
        self.setLayout(layout)

    def initializePage(self):
        """Populate combo boxes with available models - Claude Generated"""
        wizard = self.wizard()
        available_models = getattr(wizard.llm_page, 'available_models', [])

        if not available_models:
            # Show placeholder if no models
            for combo in self.task_combos.values():
                combo.addItem("(Keine Modelle verfügbar)")
            return

        # Populate all combos
        for task_key, combo in self.task_combos.items():
            combo.clear()
            combo.addItems(available_models)

            # Set intelligent defaults based on task
            if 'vision' in task_key:
                # Prefer vision models for vision tasks
                vision_models = [m for m in available_models
                               if 'vision' in m.lower() or 'llava' in m.lower()]
                if vision_models:
                    combo.setCurrentText(vision_models[0])
            else:
                # Use first available model as default
                combo.setCurrentIndex(0)

    def _use_same_for_all(self):
        """Set all tasks to use the same model - Claude Generated"""
        if not self.task_combos:
            return

        # Get first combo's selection
        first_model = list(self.task_combos.values())[0].currentText()

        # Apply to all
        for combo in self.task_combos.values():
            combo.setCurrentText(first_model)

        QMessageBox.information(
            self,
            "Modelle eingestellt",
            f"✅ Alle Tasks verwenden jetzt: {first_model}"
        )

    def validatePage(self) -> bool:
        """Save selections - Claude Generated"""
        # Store selections
        for task_key, combo in self.task_combos.items():
            model = combo.currentText()
            if model and "(Keine Modelle" not in model:
                self.task_model_selections[task_key] = model

        return True


class GNDDatabasePage(QWizardPage):
    """GND Database download page - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("GND Authority Database")
        self.setSubTitle("Optional: Download German National Library keyword data")

        # Track downloaded file for post-wizard import - Claude Generated
        self.downloaded_xml_path = None

        layout = QVBoxLayout()

        # Explanation - Claude Generated (updated)
        explanation = QLabel(
            "Die GND (Gemeinsame Normdatei) ist eine lokale Datenbank für deutsche Schlagwörter.\n"
            "Die Datenbank wird sowohl bei GND-Download als auch bei Lobid-API für interne\n"
            "Verifikation und Abgleich verwendet.\n\n"
            "Optionen:\n"
            "• Download: Vollständige lokale Datenbank (~300 MB, schneller)\n"
            "• Überspringen: Lobid-API lädt Daten bei Bedarf nach (langsamer beim Start)\n\n"
            "Der Import läuft nach Setup im Hintergrund, Sie können ALIMA sofort nutzen."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Download options
        options_box = QGroupBox("Download Options")
        options_layout = QVBoxLayout()

        self.download_radio = QRadioButton("📥 Download GND database from DNB (recommended)")
        self.download_radio.setChecked(True)
        options_layout.addWidget(self.download_radio)

        self.local_radio = QRadioButton("📂 Load from local file")
        options_layout.addWidget(self.local_radio)

        self.skip_radio = QRadioButton("⏭️  Skip for now (use Lobid API)")
        options_layout.addWidget(self.skip_radio)

        options_box.setLayout(options_layout)
        layout.addWidget(options_box)

        # File selector (visible only for local file option)
        self.file_layout = QHBoxLayout()
        file_label = QLabel("GND File:")
        self.file_input = QLineEdit()
        file_button = QPushButton("Browse...")
        file_button.clicked.connect(self._select_file)
        self.file_layout.addWidget(file_label)
        self.file_layout.addWidget(self.file_input)
        self.file_layout.addWidget(file_button)

        layout.addLayout(self.file_layout)
        self._update_file_visibility()

        self.download_radio.toggled.connect(self._update_file_visibility)
        self.local_radio.toggled.connect(self._update_file_visibility)

        # Download status
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Download button
        self.download_button = QPushButton("⬇️  Download GND Database")
        self.download_button.clicked.connect(self._download_gnd)
        layout.addWidget(self.download_button)

        layout.addStretch()
        self.setLayout(layout)

    def _update_file_visibility(self):
        """Show/hide file selector based on selection - Claude Generated"""
        show_file = self.local_radio.isChecked()
        for i in range(self.file_layout.count()):
            widget = self.file_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(show_file)

    def _select_file(self):
        """Select GND file - Claude Generated"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GND Database File",
            "",
            "XML Files (*.xml);;Gzip Files (*.gz);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)

    def _download_gnd(self):
        """Download GND database - Claude Generated"""
        if not self.download_radio.isChecked():
            return

        from PyQt6.QtWidgets import QApplication

        progress = QProgressDialog(
            "Downloading GND database...", "Cancel", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        try:
            def progress_callback(percent):
                progress.setValue(percent)
                QApplication.processEvents()  # Keep UI responsive

            result = GNDDatabaseDownloader.download(progress_callback)

            progress.close()

            if result.success:
                # Save downloaded path for post-wizard import - Claude Generated
                self.downloaded_xml_path = result.data
                self.status_label.setText(
                    f"✅ GND-Datenbank heruntergeladen\n"
                    f"Import läuft nach Setup im Hintergrund"
                )
                self.status_label.setStyleSheet("color: green;")
                self.download_button.setEnabled(False)
            else:
                self.status_label.setText(f"❌ Download failed: {result.message}")
                self.status_label.setStyleSheet("color: red;")

        except Exception as e:
            progress.close()
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def validatePage(self) -> bool:
        """Validate before proceeding - Claude Generated (fixed download UX)"""
        if self.download_radio.isChecked():
            if "✅" not in self.status_label.text():
                # User selected download but hasn't done it yet - offer clear choices
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Icon.Question)
                msgBox.setWindowTitle("GND-Datenbank noch nicht heruntergeladen")
                msgBox.setText("Sie haben 'Download' ausgewählt, aber noch nicht gestartet.")
                msgBox.setInformativeText(
                    "Optionen:\n"
                    "• Jetzt herunterladen: Import läuft nach Setup im Hintergrund (~5-10 Min)\n"
                    "• Überspringen: Lobid-API lädt Daten bei Bedarf nach (langsamer)\n"
                    "• Zurück: Entscheidung überdenken\n\n"
                    "Hinweis: Die Datenbank wird für Verifikation in beiden Fällen verwendet."
                )
                msgBox.setStandardButtons(
                    QMessageBox.StandardButton.Retry |      # "Jetzt herunterladen"
                    QMessageBox.StandardButton.Ignore |     # "Überspringen"
                    QMessageBox.StandardButton.Cancel       # "Zurück"
                )
                msgBox.setDefaultButton(QMessageBox.StandardButton.Retry)

                # Customize button labels
                msgBox.button(QMessageBox.StandardButton.Retry).setText("Jetzt herunterladen")
                msgBox.button(QMessageBox.StandardButton.Ignore).setText("Überspringen")
                msgBox.button(QMessageBox.StandardButton.Cancel).setText("Zurück")

                result = msgBox.exec()

                if result == QMessageBox.StandardButton.Retry:
                    # User wants to download now - trigger download and stay on page
                    self._download_gnd()
                    return False  # Stay on page to show download progress
                elif result == QMessageBox.StandardButton.Ignore:
                    # User explicitly wants to skip despite selecting download option
                    return True  # Proceed to next page without download
                else:  # Cancel
                    # User wants to go back and reconsider
                    return False  # Stay on page
        elif self.local_radio.isChecked():
            if not self.file_input.text():
                QMessageBox.warning(self, "File Required", "Please select a GND file.")
                return False
        elif self.skip_radio.isChecked():
            # Confirm skip action with option to go back - Claude Generated (improved UX, updated text)
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("GND-Download überspringen?")
            msgBox.setText("GND-Download überspringen?")
            msgBox.setInformativeText(
                "Wenn Sie überspringen:\n"
                "• Lobid-API lädt Daten bei Bedarf nach (langsamer beim ersten Zugriff)\n"
                "• Volle Funktionalität bleibt erhalten\n"
                "• Sie können später über Tools → GND-Datenbank importieren nachholen\n\n"
                "Hinweis: Die Datenbank wird auch mit Lobid für Verifikation verwendet."
            )
            msgBox.setStandardButtons(
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.Cancel
            )
            msgBox.setDefaultButton(QMessageBox.StandardButton.Cancel)

            # Customize button text
            msgBox.button(QMessageBox.StandardButton.Yes).setText("Überspringen")
            msgBox.button(QMessageBox.StandardButton.Cancel).setText("Zurück")

            result = msgBox.exec()
            # Yes = skip, Cancel = stay on page (return False to prevent proceeding)
            return result == QMessageBox.StandardButton.Yes

        return True


class SummaryPage(QWizardPage):
    """Configuration summary page - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("Setup Complete!")
        self.setSubTitle("Review your ALIMA configuration")

        layout = QVBoxLayout()

        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        # Completion message
        completion = QLabel(
            "✅ Your ALIMA configuration is ready!\n\n"
            "You can now launch the application and start analyzing library materials."
        )
        completion.setWordWrap(True)
        layout.addWidget(completion)

        layout.addStretch()
        self.setLayout(layout)

    def initializePage(self):
        """Initialize with configuration summary - Claude Generated (enhanced with model selections)"""
        # Get wizard to access configuration pages
        wizard = self.wizard()

        # Build summary from wizard pages (not from config, which isn't set yet)
        gnd_option = "Download from DNB"
        if wizard.gnd_page.skip_radio.isChecked():
            gnd_option = "Skip (use Lobid API)"
        elif wizard.gnd_page.local_radio.isChecked():
            gnd_option = f"Load from file: {wizard.gnd_page.file_input.text()}"

        # Build model selections summary - Claude Generated
        model_summary = ""
        if hasattr(wizard, 'model_page') and wizard.model_page.task_model_selections:
            model_summary = "\n🎯 Modell-Zuordnungen:\n"
            for task, model in wizard.model_page.task_model_selections.items():
                task_label = task.replace('_', ' ').title()
                model_summary += f"   {task_label}: {model}\n"
        else:
            model_summary = "\n🎯 Modell-Zuordnungen: Standard"

        summary = f"""
ALIMA Konfiguration - Zusammenfassung
══════════════════════════════════════

🤖 LLM-Anbieter:
   Typ: {wizard.llm_page.provider_type.title()}
   Name: {wizard.llm_page.provider_name}
   URL: {wizard.llm_page.base_url or "(keine URL)"}
   Verfügbare Modelle: {len(wizard.llm_page.available_models)}
{model_summary}

📚 GND-Datenbank:
   Option: {gnd_option}

✅ Setup abgeschlossen!
   Klicken Sie auf "Fertig stellen" um ALIMA zu starten.
        """
        self.summary_text.setText(summary)
