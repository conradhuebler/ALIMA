#!/usr/bin/env python3
"""Provider configuration dialogs — Claude Generated.

Extracted from ``unified_provider_tab.py`` (F-5 god-file split, June 30) so the
tab module holds only the tab itself:

* :class:`TaskModelSelectionDialog` — pick provider + model for one pipeline task
  (async model detection via the shared ``ModelLoadWorker``).
* :class:`ProviderEditDialog` — add/edit a single LLM provider's connection config.

Both are self-contained ``QDialog``s (no coupling to the tab beyond the Qt parent);
the tab instantiates them and reads results via their getters.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout,
)

from ..utils.config_manager import ProviderDetectionService
from ..utils.config_models import UnifiedProvider
from .workers import ModelLoadWorker
from .styles import get_colors, set_tone


_running_tests: set = set()  # Verbindungstest-Threads leben, bis sie fertig sind


class _ConnectionTestWorker(QThread):
    """Führt einen Verbindungstest außerhalb des UI-Threads aus - Claude Generated"""

    done = pyqtSignal(object)  # SetupResult

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        from ..utils.setup_utils import SetupResult

        try:
            result = self._fn()
        except Exception as e:  # defensiv: der Test selbst fängt schon
            result = SetupResult(False, f"Test fehlgeschlagen: {e}", [])
        self.done.emit(result)


def _endpoint_warnings(base_url: str, api_key: str) -> list:
    """Sanity warnings for a provider's endpoint pair. Never blocks - Claude Generated.

    Two checks, both cheap and both from real mishaps:

    * the base URL is not an http(s) URL — the model list then fails at
      ``LlmService.is_provider_reachable`` and the combo simply stays empty, with
      nothing on screen saying why;
    * the *API key* field parses as a URL while the base URL does not — the two
      were swapped. That is easy to do and hard to see, because the key field is
      masked: a pasted URL shows up as dots.

    Neither message ever echoes the key. Intranet hosts are not flagged (see
    ``net_guard.check_operator_url``).
    """
    from urllib.parse import urlparse

    from src.utils.net_guard import check_operator_url

    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()

    def _is_url(value: str) -> bool:
        try:
            return urlparse(value).scheme in ("http", "https")
        except ValueError:
            return False

    if base_url and not _is_url(base_url) and _is_url(api_key):
        return [
            "Base URL und API Key sehen vertauscht aus: im Schlüsselfeld steht "
            "ein http(s)-URL, im URL-Feld nicht.\n\nDas Schlüsselfeld ist "
            "maskiert, ein hineinkopierter URL erscheint dort nur als Punkte."
        ]

    # Only what actually leaves the model list empty. Plain http is a legitimate
    # intranet setup, so the unencrypted note from check_operator_url is dropped
    # here — it would fire on every save of a working provider.
    warnings = [w for w in check_operator_url(base_url) if "unverschlüsselt" not in w]
    if base_url and not _is_url(base_url):
        # A value that is not a URL is quite often the key pasted into the wrong
        # field — do not echo it into a dialog.
        warnings = [w.replace(base_url, "(Feldinhalt)") for w in warnings]
    return warnings


class TaskModelSelectionDialog(QDialog):
    """Dialog for selecting provider and model for task-specific preferences - Claude Generated"""

    def __init__(self, config_manager, task_name=None, current_model_info=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.task_name = task_name
        self.current_model_info = current_model_info or {}
        self._pending_cache_update = None  # Stores (provider_name, models) for later save
        self.status_label = None  # Will be set in _setup_ui
        # Async model detection (F-7): keep strong refs to in-flight workers until
        # they finish, and remember the provider whose load is current so late
        # results for a switched-away provider are ignored. - Claude Generated
        self._model_workers = set()
        self._pending_provider = None

        self.setWindowTitle("Select Provider and Model")
        self.setModal(True)
        self.resize(450, 350)

        self._setup_ui()
        self._load_providers()
    
    def _setup_ui(self):
        """Setup dialog UI with real provider/model detection - Claude Generated"""
        layout = QVBoxLayout()
        
        # Provider selection
        provider_group = QGroupBox("🌐 Select Provider")
        provider_layout = QFormLayout(provider_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self._load_models)
        provider_layout.addRow("Available Providers:", self.provider_combo)
        layout.addWidget(provider_group)
        
        # Model selection
        model_group = QGroupBox("🎯 Select Model")
        model_layout = QVBoxLayout(model_group)

        model_info_label = QLabel("Choose a model, type to filter/search, or press Enter to validate:")
        model_layout.addWidget(model_info_label)

        # Make model combo editable for quick filtering/typing - Claude Generated
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Don't add custom items
        # Enable substring filtering - Claude Generated
        from PyQt6.QtWidgets import QCompleter
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        # Validate selected model exists when losing focus or pressing Enter - Claude Generated
        self.model_combo.lineEdit().editingFinished.connect(self._validate_model_selection)
        model_layout.addWidget(self.model_combo)

        layout.addWidget(model_group)

        # Status label for model detection feedback - Claude Generated
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        set_tone(self.status_label, "muted")
        self.status_label.setStyleSheet("QLabel { font-style: italic; margin: 5px; }")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("✅ OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("❌ Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _load_providers(self):
        """Load real available providers using detection service - Claude Generated"""
        self.provider_combo.clear()
        
        try:
            detection_service = ProviderDetectionService()
            available_providers = detection_service.get_available_providers()
            
            if not available_providers:
                self.provider_combo.addItem("No providers available")
                return
            
            self.provider_combo.addItems(available_providers)
            
        except Exception as e:
            # Fallback to basic providers if detection fails
            fallback_providers = ["ollama", "gemini", "openai", "anthropic"]
            self.provider_combo.addItems(fallback_providers)
    
    def _load_models(self, provider_name: str):
        """Kick off async model detection (two-tier: live → cached) - Claude Generated

        TIER 1 (live detection) runs off the UI thread via ModelLoadWorker so
        switching the provider no longer freezes the dialog; the result is handled
        in :meth:`_on_models_loaded`, which falls back to TIER 2 (cached) when the
        live list is empty.
        """
        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        if not provider_name or provider_name == "No providers available":
            self._pending_provider = None
            return

        # Loading placeholder while the (network) detection runs in the background.
        self._pending_provider = provider_name
        self.model_combo.addItem("⏳ Lade Modelle…", None)
        self.model_combo.setEnabled(False)
        if self.status_label:
            self.status_label.setText(f"⏳ Detecting models from {provider_name}…")

        detection_service = ProviderDetectionService(self.config_manager)
        worker = ModelLoadWorker(detection_service, provider_name, force=True)
        worker.fetched.connect(self._on_models_loaded)
        worker.finished.connect(lambda w=worker: self._retire_model_worker(w))
        self._model_workers.add(worker)
        worker.start()

    def _retire_model_worker(self, worker) -> None:
        """Drop the strong ref once the worker thread has finished - Claude Generated"""
        self._model_workers.discard(worker)
        worker.deleteLater()

    def _on_models_loaded(self, provider_name: str, detected_models: list):
        """Populate the model combo when detection finishes (main thread) - Claude Generated

        TIER 2 cached fallback runs here when the live list is empty. Late results
        for a provider the user has since switched away from are ignored.
        """
        if provider_name != self._pending_provider:
            return

        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        models_to_display = []
        detection_success = False

        # TIER 1 result (live detection, fetched off-thread)
        if detected_models:
            models_to_display = sorted(detected_models, key=lambda s: s.lower())
            detection_success = True
            if self.status_label:
                self.status_label.setText(f"✅ {len(models_to_display)} models detected from {provider_name}")

        # TIER 2: cached models from config (if live detection returned nothing)
        if not models_to_display and self.config_manager:
            cached_models = self._get_cached_models_from_config(provider_name)
            if cached_models:
                models_to_display = sorted(cached_models, key=lambda s: s.lower())
                if self.status_label:
                    self.status_label.setText(f"⚠️ Provider offline - showing {len(models_to_display)} cached models")

        # Populate dropdown (no visual distinction between detected and cached)
        if models_to_display:
            for model in models_to_display:
                self.model_combo.addItem(model, model)
            self._preselect_current_model()
        else:
            self.model_combo.addItem("No models available", None)
            self.model_combo.setEnabled(False)
            if self.status_label:
                self.status_label.setText(f"❌ No models available for {provider_name}")

        # Store detected models for later caching (only on dialog OK)
        if detection_success:
            self._pending_cache_update = (provider_name, models_to_display)

    def _get_cached_models_from_config(self, provider_name: str) -> list:
        """Get models from config cache for offline use - Claude Generated"""
        if not self.config_manager:
            return []

        try:
            config = self.config_manager.load_config()
            for provider in config.unified_config.providers:
                if provider.name.lower() == provider_name.lower():
                    # Return available_models if populated
                    if provider.available_models:
                        return provider.available_models
                    # Fallback to preferred_model if set
                    if provider.preferred_model:
                        return [provider.preferred_model]
        except Exception as e:
            self.logger.warning(f"Could not read cached models for provider: {e}")

        return []

    def _update_provider_cached_models(self, provider_name: str, models: list) -> None:
        """Update provider's cached model list in config - Claude Generated

        Called only when dialog is accepted (OK button), not during detection.
        """
        if not self.config_manager or not models:
            return

        try:
            config = self.config_manager.load_config()
            for provider in config.unified_config.providers:
                if provider.name.lower() == provider_name.lower():
                    provider.available_models = models
                    self.config_manager.save_config(config)
                    break
        except Exception as e:
            self.logger.warning(f"Could not persist models for provider '{provider_name}': {e}")

    def _preselect_current_model(self) -> None:
        """Pre-select the currently configured model in dropdown - Claude Generated"""
        if not self.current_model_info:
            return

        current_provider = self.current_model_info.get('provider')
        current_model = self.current_model_info.get('model')

        # Check if current provider matches selected provider
        if current_provider and current_provider.lower() == self.provider_combo.currentText().lower():
            # Try to find and select current model
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model:
                    self.model_combo.setCurrentIndex(i)
                    break

    def _validate_model_selection(self) -> None:
        """Validate that typed model name exists in available models - Claude Generated"""
        current_text = self.model_combo.currentText().strip()

        if not current_text:
            return  # Empty is OK (will show as "No models available" handling)

        # Check if text matches any available model (case-insensitive)
        model_found = False
        for i in range(self.model_combo.count()):
            item_text = self.model_combo.itemText(i)
            if item_text.lower() == current_text.lower():
                # Found exact match - select it
                self.model_combo.setCurrentIndex(i)
                model_found = True
                self.status_label.setText(f"✅ Model '{current_text}' verified")
                break

        if not model_found and current_text != "No models available":
            # Model doesn't exist - show error and reset
            self.status_label.setText(
                f"❌ Model '{current_text}' not found in {self.provider_combo.currentText()}"
            )
            set_tone(self.status_label, "error")
            # Clear the input
            self.model_combo.lineEdit().clear()
            set_tone(self.status_label, "muted")

    def accept(self) -> None:
        """Override accept to update cached models before closing - Claude Generated

        Only saves detected models to config when user confirms (OK button).
        """
        # Update config cache if we have pending detected models
        if self._pending_cache_update:
            provider_name, models = self._pending_cache_update
            self._update_provider_cached_models(provider_name, models)

        super().accept()
    
    def get_selected_model(self):
        """Get selected provider and model - Claude Generated"""
        provider = self.provider_combo.currentText()

        if provider == "No providers available":
            return None, None

        model = self.model_combo.currentData()

        if not model:
            return None, None

        return provider, model
    
    def set_default_from_global_preferences(self, provider_preferences):
        """Set dialog defaults from global provider preferences - Claude Generated"""
        try:
            # Set preferred provider as default
            preferred_provider = provider_preferences.preferred_provider
            provider_index = self.provider_combo.findText(preferred_provider)
            if provider_index >= 0:
                self.provider_combo.setCurrentIndex(provider_index)
                self._on_provider_changed()  # Trigger model list update
                
                # Set preferred model for this provider
                preferred_model = provider_preferences.preferred_models.get(preferred_provider)
                if preferred_model:
                    model_index = -1
                    for i in range(self.model_combo.count()):
                        if self.model_combo.itemData(i) == preferred_model:
                            model_index = i
                            break
                    
                    if model_index >= 0:
                        self.model_combo.setCurrentIndex(model_index)
                    else:
                        # Model not found in dropdown, fallback to auto-select
                        self.model_combo.setCurrentIndex(0)  # "(Auto-select)" is always first
                        
        except Exception as e:
            # Silent fallback - don't interrupt user workflow
            pass


class ProviderEditDialog(QDialog):
    """Dialog for editing provider configuration - Claude Generated"""
    
    def __init__(self, provider: Optional[UnifiedProvider] = None, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.is_editing = provider is not None
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Edit Provider" if self.is_editing else "Add Provider")
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        
        if self.is_editing:
            self._load_provider_data()
    
    def _setup_ui(self):
        """Setup the dialog UI - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Provider Type Selection
        type_group = QGroupBox("Provider Type")
        type_layout = QFormLayout(type_group)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["ollama", "openai_compatible", "gemini", "anthropic"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addRow("Type:", self.type_combo)
        
        layout.addWidget(type_group)
        
        # Basic Configuration
        basic_group = QGroupBox("Basic Configuration")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.enabled_checkbox = QCheckBox()
        self.enabled_checkbox.setChecked(True)
        self.preferred_model_edit = QLineEdit()
        self.preferred_model_edit.setPlaceholderText("(leer = automatisch)")
        self.preferred_model_edit.setToolTip("Standard-Modell für diesen Provider (leer lassen für automatische Auswahl)")

        basic_layout.addRow("Name:", self.name_edit)
        basic_layout.addRow("Description:", self.description_edit)
        basic_layout.addRow("Enabled:", self.enabled_checkbox)
        basic_layout.addRow("Preferred Model:", self.preferred_model_edit)
        
        layout.addWidget(basic_group)
        
        # Connection Configuration (dynamic based on type)
        self.connection_group = QGroupBox("Connection Configuration")
        self.connection_layout = QFormLayout(self.connection_group)
        layout.addWidget(self.connection_group)

        # Live-Hinweise zur Eingabe und Verbindungstest - Claude Generated
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setTextFormat(Qt.TextFormat.RichText)
        self.hint_label.hide()
        layout.addWidget(self.hint_label)

        test_row = QHBoxLayout()
        self.test_button = QPushButton("🔌 Verbindung testen")
        self.test_button.setToolTip(
            "Prüft mit den Werten aus diesem Formular, ob der Anbieter antwortet "
            "und Modelle liefert. Nichts wird gespeichert."
        )
        self.test_button.clicked.connect(self._run_connection_test)
        test_row.addWidget(self.test_button)
        self.test_result = QLabel()
        self.test_result.setWordWrap(True)
        test_row.addWidget(self.test_result, 1)
        layout.addLayout(test_row)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initialize connection fields
        self._on_type_changed(self.type_combo.currentText())
    
    def _on_type_changed(self, provider_type: str):
        """Update connection configuration fields based on provider type - Claude Generated"""
        # Clear existing fields
        while self.connection_layout.count():
            child = self.connection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add type-specific fields
        if provider_type == "ollama":
            self.host_edit = QLineEdit("localhost")
            self.port_spinbox = QSpinBox()
            self.port_spinbox.setRange(1, 65535)
            self.port_spinbox.setValue(11434)
            self.ssl_checkbox = QCheckBox()
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.host_edit.setPlaceholderText("localhost  oder  ollama.com")

            self.connection_layout.addRow("Host:", self.host_edit)
            self.connection_layout.addRow("Port:", self.port_spinbox)
            self.connection_layout.addRow("Use SSL:", self.ssl_checkbox)
            self.connection_layout.addRow("API Key (optional):", self.api_key_edit)
            
        elif provider_type == "openai_compatible":
            from ..utils.provider_presets import DISCLAIMER, OPENAI_COMPATIBLE_PRESETS

            self.preset_combo = QComboBox()
            self.preset_combo.addItem("(eigene URL)", None)
            for preset in OPENAI_COMPATIBLE_PRESETS:
                self.preset_combo.addItem(preset.label, preset.id)
            self.preset_combo.setToolTip(DISCLAIMER)
            self.preset_combo.activated.connect(self._on_preset_chosen)

            self.base_url_edit = QLineEdit()
            self.base_url_edit.textChanged.connect(self._sync_preset_combo)
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            
            self.connection_layout.addRow("Vorlage:", self.preset_combo)
            self.connection_layout.addRow("Base URL:", self.base_url_edit)
            self.connection_layout.addRow("API Key:", self.api_key_edit)
            
        elif provider_type in ["gemini", "anthropic"]:
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

            self.connection_layout.addRow("API Key:", self.api_key_edit)

        self._wire_live_feedback(provider_type)

    # ---- Vorlagen für OpenAI-kompatible Anbieter - Claude Generated -------

    def _on_preset_chosen(self, index: int) -> None:
        """Vorlage gewählt: URL setzen; Name und Beschreibung nur, wenn leer."""
        from ..utils.provider_presets import OPENAI_COMPATIBLE_PRESETS

        if index <= 0:
            return
        preset = OPENAI_COMPATIBLE_PRESETS[index - 1]
        self.base_url_edit.setText(preset.base_url)
        if not self.name_edit.text().strip():
            self.name_edit.setText(preset.id)
        if not self.description_edit.text().strip():
            self.description_edit.setText(preset.label)
        self._update_hints()

    def _sync_preset_combo(self, *_args) -> None:
        """Die Vorlagen-Auswahl folgt der URL (Laden, Einfügen, Tippen)."""
        from ..utils.provider_presets import OPENAI_COMPATIBLE_PRESETS, preset_for_url

        combo = getattr(self, "preset_combo", None)
        if combo is None:
            return
        preset = preset_for_url(self.base_url_edit.text())
        index = 0 if preset is None else 1 + OPENAI_COMPATIBLE_PRESETS.index(preset)
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    # ---- Eingabe-Feedback und Verbindungstest - Claude Generated ----------

    def _wire_live_feedback(self, provider_type: str) -> None:
        """Hinweise bei jeder Änderung der Verbindungsfelder neu berechnen."""
        self.test_result.clear()
        self.test_button.setVisible(provider_type in ("ollama", "openai_compatible"))
        for name, signal in (
            ("host_edit", "textChanged"), ("api_key_edit", "textChanged"),
            ("base_url_edit", "textChanged"), ("port_spinbox", "valueChanged"),
            ("ssl_checkbox", "toggled"),
        ):
            widget = getattr(self, name, None)
            if widget is not None and name in self._current_field_names(provider_type):
                getattr(widget, signal).connect(self._update_hints)
        self._update_hints()

    @staticmethod
    def _current_field_names(provider_type: str) -> tuple:
        return {
            "ollama": ("host_edit", "port_spinbox", "ssl_checkbox", "api_key_edit"),
            "openai_compatible": ("base_url_edit", "api_key_edit"),
        }.get(provider_type, ("api_key_edit",))

    def _collect_hints(self) -> list:
        """Liste von (Stufe, Text) für den aktuellen Formularzustand."""
        from ..utils import ollama_endpoint

        ptype = self.type_combo.currentText()
        if ptype == "ollama" and hasattr(self, "host_edit"):
            return [
                (h.level, h.text) for h in ollama_endpoint.hints(
                    self.host_edit.text(), self.port_spinbox.value(),
                    self.ssl_checkbox.isChecked(), self.api_key_edit.text(),
                )
            ]
        if ptype == "openai_compatible" and hasattr(self, "base_url_edit"):
            from ..utils.provider_presets import preset_for_url

            out = [("warn", w) for w in _endpoint_warnings(
                self.base_url_edit.text(), self.api_key_edit.text())]
            preset = preset_for_url(self.base_url_edit.text())
            if preset is not None:
                if preset.key_required and not self.api_key_edit.text().strip():
                    out.append(("warn", f"{preset.label} braucht einen API-Key."))
                if preset.note:
                    out.append(("info", preset.note))
            return out
        return []

    def _update_hints(self, *_args) -> None:
        c = get_colors()
        colors = {"error": c["error"], "warn": c["warning"], "info": c["text_muted"]}
        icons = {"error": "⛔", "warn": "⚠️", "info": "ℹ️"}
        items = self._collect_hints()
        self.hint_label.setText("<br>".join(
            f'<span style="color:{colors[lvl]}">{icons[lvl]} {txt}</span>'
            for lvl, txt in items
        ))
        self.hint_label.setVisible(bool(items))
        # Nur ein fehlender Host sperrt „OK“; alles andere sind Hinweise.
        if getattr(self, "ok_button", None) is not None:
            self.ok_button.setEnabled(not any(lvl == "error" for lvl, _ in items))

    def _run_connection_test(self) -> None:
        from ..utils.ollama_endpoint import clean_api_key
        from ..utils.setup_utils import OllamaConnectionValidator

        ptype = self.type_combo.currentText()
        key = clean_api_key(self.api_key_edit.text())
        if ptype == "ollama":
            host, port, ssl = (self.host_edit.text(), self.port_spinbox.value(),
                               self.ssl_checkbox.isChecked())
            def fn():
                return OllamaConnectionValidator.test_native(host, port, key, use_ssl=ssl)
        elif ptype == "openai_compatible":
            url = self.base_url_edit.text().strip()
            def fn():
                return OllamaConnectionValidator.test_openai_compatible(url, key)
        else:
            return
        self.test_button.setEnabled(False)
        set_tone(self.test_result, "muted")
        self.test_result.setText("⏳ Teste Verbindung …")
        worker = _ConnectionTestWorker(fn)
        worker.done.connect(self._on_test_done)
        worker.finished.connect(lambda w=worker: _running_tests.discard(w))
        _running_tests.add(worker)
        worker.start()

    def _on_test_done(self, result) -> None:
        self.test_button.setEnabled(True)
        ok = bool(result.success)
        set_tone(self.test_result, "success" if ok else "error")
        self.test_result.setText(("✅ " if ok else "❌ ") + result.message)

    def _load_provider_data(self):
        """Load provider data into form fields - Claude Generated"""
        if not self.provider:
            return
        
        self.name_edit.setText(self.provider.name)
        self.description_edit.setText(self.provider.description)
        self.enabled_checkbox.setChecked(self.provider.enabled)
        self.preferred_model_edit.setText(self.provider.preferred_model or "")
        self.type_combo.setCurrentText(self.provider.provider_type)

        # Load connection configuration from provider attributes
        if self.provider.provider_type == "ollama":
            if hasattr(self, 'host_edit'):
                self.host_edit.setText(self.provider.host or "localhost")
            if hasattr(self, 'port_spinbox'):
                self.port_spinbox.setValue(self.provider.port or 11434)
            if hasattr(self, 'ssl_checkbox'):
                self.ssl_checkbox.setChecked(self.provider.use_ssl or False)
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")

        elif self.provider.provider_type == "openai_compatible":
            if hasattr(self, 'base_url_edit'):
                self.base_url_edit.setText(self.provider.base_url or "")
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")

        elif self.provider.provider_type in ["gemini", "anthropic"]:
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")
    
    def accept(self) -> None:
        """Warn about an implausible endpoint before closing - Claude Generated.

        A wrong base URL produces no error anywhere: the provider is stored, the
        model list comes back empty, and nothing on screen connects the two. The
        warning offers a way back to the form and never blocks.
        """
        warnings = _endpoint_warnings(
            getattr(self, "base_url_edit", QLineEdit()).text(),
            getattr(self, "api_key_edit", QLineEdit()).text(),
        )
        if warnings:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Endpunkt prüfen")
            box.setText("\n\n".join(warnings))
            box.setInformativeText(
                "Mit einem ungültigen Endpunkt bleibt die Modell-Liste dieses "
                "Providers leer."
            )
            back = box.addButton("Zurück zum Formular", QMessageBox.ButtonRole.RejectRole)
            box.addButton("Trotzdem übernehmen", QMessageBox.ButtonRole.AcceptRole)
            box.setDefaultButton(back)
            box.exec()
            if box.clickedButton() is back:
                return
        super().accept()

    def get_provider_data(self) -> UnifiedProvider:
        """Extract provider data from form fields - Claude Generated"""
        # Build connection config based on type
        connection_config = {}
        provider_type = self.type_combo.currentText()
        
        from ..utils.ollama_endpoint import clean_api_key, is_ollama_cloud_host, parse_host

        key = clean_api_key(getattr(self, 'api_key_edit', QLineEdit()).text())
        if provider_type == "ollama":
            # Schema/Pfad/Port aus dem Hostfeld werden in die Felder übernommen;
            # Ollama Cloud wird auf HTTPS/443 gesetzt, damit die gespeicherten
            # Werte zu dem passen, was der Client tatsächlich benutzt.
            host, port, ssl = parse_host(
                getattr(self, 'host_edit', QLineEdit()).text(),
                getattr(self, 'port_spinbox', QSpinBox()).value() or 11434,
                getattr(self, 'ssl_checkbox', QCheckBox()).isChecked(),
            )
            if is_ollama_cloud_host(host):
                port, ssl = 443, True
            connection_config = {
                "host": host or "localhost",
                "port": port,
                "use_ssl": ssl,
                "api_key": key,
                "connection_type": "native_client"
            }
        elif provider_type == "openai_compatible":
            connection_config = {
                "base_url": getattr(self, 'base_url_edit', QLineEdit()).text().strip(),
                "api_key": key
            }
        elif provider_type in ["gemini", "anthropic"]:
            connection_config = {"api_key": key}

        return UnifiedProvider(
            name=self.name_edit.text() or f"New {provider_type.title()} Provider",
            provider_type=provider_type,
            enabled=self.enabled_checkbox.isChecked(),
            api_key=connection_config.get("api_key", ""),
            base_url=connection_config.get("base_url", ""),
            preferred_model=self.preferred_model_edit.text().strip(),
            description=self.description_edit.text(),
            # Ollama specific fields
            host=connection_config.get("host", ""),
            port=connection_config.get("port", 11434),
            use_ssl=connection_config.get("use_ssl", False),
            connection_type=connection_config.get("connection_type", "native_client")
        )
