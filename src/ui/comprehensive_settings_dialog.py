#!/usr/bin/env python3
"""
Comprehensive Settings Dialog for ALIMA
Combines database settings, LLM configuration, and system settings.
Claude Generated
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QTextEdit, QLabel, QGroupBox, QScrollArea, QFrame,
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QGuiApplication
import logging
import time
from copy import deepcopy
from typing import Optional

from ..utils.config_manager import ConfigManager, AlimaConfig
from .unified_provider_tab import UnifiedProviderTab
from .plugin_settings_tab import PluginSettingsTab
from .settings import (
    CollapsibleSection,
    SettingsIndex,
    SettingsShell,
    WorkflowList,
    workflow_entries,
)
from ..utils.logging_utils import LOG_LEVEL_CHOICES
from .styles import (
    COLOR_SCHEME_LABELS,
    COLOR_SCHEME_TOOLTIPS,
    COLOR_SCHEMES,
    normalize_color_scheme,
    set_tone,
)


class DatabaseTestWorker(QThread):
    """Worker thread for database connection testing - Claude Generated

    Takes the config to probe rather than reading the manager's, so testing
    unsaved dialog settings does not require publishing them process-wide.
    """
    test_completed = pyqtSignal(bool, str)

    def __init__(self, config_manager: ConfigManager, database_config=None):
        super().__init__()
        self.config_manager = config_manager
        self.database_config = database_config

    def run(self):
        success, message = self.config_manager.test_database_connection(
            self.database_config
        )
        self.test_completed.emit(success, message)


class ComprehensiveSettingsDialog(QDialog):
    """Comprehensive settings dialog combining all ALIMA configurations - Claude Generated"""

    config_changed = pyqtSignal()

    def __init__(self, alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        self.alima_manager = alima_manager  # For ProviderStatusService access - Claude Generated

        # The dialog edits a copy and writes it on Save; Cancel drops it.
        self.config_to_edit = deepcopy(self.config_manager.load_config())

        self.setWindowTitle("ALIMA Settings")
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(520, 420)
        self._apply_initial_window_size()

        self._setup_ui()
        self._load_current_settings()

    def _apply_initial_window_size(self):
        """Size the dialog conservatively so buttons remain reachable on smaller screens."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(900, 700)
            return

        available = screen.availableGeometry()
        target_width = min(900, max(520, available.width() - 80))
        target_height = min(700, max(420, available.height() - 120))
        self.resize(target_width, target_height)
    
    def _setup_ui(self):
        """One sidebar, five pages, no nested navigation - Claude Generated.

        The pages are named after the job someone came to do, and each is a
        composition of ``_build_*_group`` builders. Every control registers with
        ``self.index`` so the search field can reach it, including inside folded
        sections — that registration is what makes folding acceptable.
        """
        layout = QVBoxLayout()

        self.index = SettingsIndex()
        self.shell = SettingsShell(self.index)

        self.unified_provider_tab = UnifiedProviderTab(
            self.config_to_edit.unified_config,
            self.config_to_edit,
            self.config_manager,  # ✅ Pass ConfigManager for persistence operations
            self.alima_manager,
            self
        )
        self.plugin_settings_tab = PluginSettingsTab(self)
        self.plugin_settings_tab.preset_applied.connect(self._on_preset_applied)
        self.database_tab = self._create_database_tab()

        self.unified_provider_tab.config_changed.connect(self.config_changed)

        self._add_models_page()
        self._add_indexing_page()
        self._add_sources_page()
        self._add_storage_page()
        self._add_application_page()

        self._add_wizard_shortcut()
        self.shell.set_footer(
            f"Einstellungen liegen in {self.config_manager.config_file}"
        )

        layout.addWidget(self.shell)
        
        # Buttons
        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.export_preset_button = QPushButton("📤 Als Preset exportieren")
        self.export_preset_button.setToolTip(
            "Gespeicherte Einrichtung als Paket exportieren (Ordner mit alima_presets.json "
            "und den Dateien); ohne Schlüssel und Tokens"
        )
        self.export_preset_button.clicked.connect(self._export_as_preset)

        self.save_button = QPushButton("💾 Save && Close")
        self.save_button.clicked.connect(self._save_and_close)

        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_preset_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_database_tab(self) -> QWidget:
        """Create database configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Database type selection
        type_group = QGroupBox("Database Type")
        type_layout = QFormLayout()
        
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["sqlite", "mysql", "mariadb"])
        self.db_type_combo.currentTextChanged.connect(self._on_db_type_changed)
        type_layout.addRow("Database Type:", self.db_type_combo)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # SQLite configuration
        self.sqlite_group = QGroupBox("SQLite Configuration")
        sqlite_layout = QFormLayout()
        
        self.sqlite_path = QLineEdit()
        sqlite_browse_layout = QHBoxLayout()
        sqlite_browse_layout.addWidget(self.sqlite_path)
        sqlite_browse_button = QPushButton("Browse...")
        sqlite_browse_button.clicked.connect(self._browse_sqlite_path)
        sqlite_browse_layout.addWidget(sqlite_browse_button)
        
        sqlite_layout.addRow("Database Path:", sqlite_browse_layout)
        
        self.sqlite_group.setLayout(sqlite_layout)
        layout.addWidget(self.sqlite_group)
        
        # MySQL configuration
        self.mysql_group = QGroupBox("MySQL/MariaDB Configuration")
        mysql_layout = QFormLayout()
        
        self.mysql_host = QLineEdit()
        mysql_layout.addRow("Host:", self.mysql_host)
        
        self.mysql_port = QSpinBox()
        self.mysql_port.setRange(1, 65535)
        self.mysql_port.setValue(3306)
        mysql_layout.addRow("Port:", self.mysql_port)
        
        self.mysql_database = QLineEdit()
        mysql_layout.addRow("Database:", self.mysql_database)
        
        self.mysql_username = QLineEdit()
        mysql_layout.addRow("Username:", self.mysql_username)
        
        self.mysql_password = QLineEdit()
        self.mysql_password.setEchoMode(QLineEdit.EchoMode.Password)
        mysql_layout.addRow("Password:", self.mysql_password)
        
        self.mysql_charset = QComboBox()
        self.mysql_charset.addItems(["utf8mb4", "utf8", "latin1"])
        mysql_layout.addRow("Charset:", self.mysql_charset)
        
        self.mysql_ssl_disabled = QCheckBox("Disable SSL")
        mysql_layout.addRow("SSL Settings:", self.mysql_ssl_disabled)
        
        self.mysql_group.setLayout(mysql_layout)
        layout.addWidget(self.mysql_group)
        
        # Connection settings
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QFormLayout()
        
        self.connection_timeout = QSpinBox()
        self.connection_timeout.setRange(5, 300)
        self.connection_timeout.setSuffix(" seconds")
        conn_layout.addRow("Connection Timeout:", self.connection_timeout)
        
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Database-specific buttons
        db_button_layout = QHBoxLayout()

        self.db_test_button = QPushButton("🔧 Test Connection")
        self.db_test_button.clicked.connect(self._test_database_connection)

        self.db_reset_button = QPushButton("↺ Reset Database Settings")
        self.db_reset_button.clicked.connect(self._reset_database_to_defaults)

        db_button_layout.addWidget(self.db_test_button)
        db_button_layout.addWidget(self.db_reset_button)
        db_button_layout.addStretch()

        layout.addLayout(db_button_layout)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    
    def _build_source_cache_group(self) -> QGroupBox:
        """Wie Quellantworten zwischengespeichert und zum GND-Pool verdichtet werden."""
        group = QGroupBox("Quell-Cache")
        form = QFormLayout()

        # WP2 raw-first response cache toggles - Claude Generated
        self.enable_response_cache = QCheckBox("Rohantworten der Quellen speichern")
        self.enable_response_cache.setToolTip(
            "Legt API-/HTML-Antworten unverändert im search_response_cache ab "
            "(lobid/swb/catalog/finc + DOI). Hauptschalter; einzelne Plugins "
            "können ihn im Plugin-Formular überschreiben."
        )
        form.addRow("Response-Cache:", self.enable_response_cache)

        self.aggregate_from_raw = QCheckBox("GND-Pool aus den Rohantworten aufbauen")
        self.aggregate_from_raw.setToolTip(
            "Beide Pipelines leiten den GND-Pool (Zähler + Herkunft) über "
            "aggregate_gnd_results aus dem Roh-Cache ab. Abgeschaltet gilt die "
            "ältere Mapping-zuerst-Verdichtung."
        )
        form.addRow("GND-Pool:", self.aggregate_from_raw)

        cache_layout = QHBoxLayout()
        self.cache_dir = QLineEdit()
        cache_layout.addWidget(self.cache_dir)
        cache_browse = QPushButton("Durchsuchen…")
        cache_browse.clicked.connect(self._browse_cache_dir)
        cache_layout.addWidget(cache_browse)
        form.addRow("Cache-Verzeichnis:", cache_layout)

        group.setLayout(form)
        return group

    def _build_workflow_group(self) -> QGroupBox:
        """Die Workflows einer Erschließung: sichtbar, wählbar, editierbar

        Vorher stand hier eine Auswahlliste von Dateinamen. Die Workflows sind
        aber der Gegenstand der Erschließung, nicht ein Implementierungsdetail:
        sie werden mit ihrem eigenen Namen und ihrer eigenen Beschreibung
        aufgelistet, und ein Doppelklick öffnet denselben Workflow-Editor, den
        auch das Menü öffnet. - Claude Generated
        """
        group = QGroupBox("Workflows")
        box = QVBoxLayout()

        self.workflow_list = WorkflowList()
        self.workflow_list.default_changed.connect(self._on_default_workflow_changed)
        self.workflow_list.edit_requested.connect(self._edit_workflow)
        self.workflow_list.new_requested.connect(lambda: self._edit_workflow(""))
        # Gefuellt wird sie von ``_load_current_settings``, nicht schon hier:
        # ein Durchgang durch alle Workflow-YAMLs kostet 200 ms (fast
        # vollstaendig das Parsen der 197 KiB), und zweimal beim Oeffnen des
        # Dialogs ist einmal zu viel. - Claude Generated
        box.addWidget(self.workflow_list)

        group.setLayout(box)
        return group

    def _refresh_workflow_list(self, *, default_stem: Optional[str] = None) -> None:
        """Read the workflow directory again and redraw the list. - Claude Generated"""
        if default_stem is None:
            default_stem = (
                self.workflow_list.default_stem()
                or getattr(self.config_to_edit.system_config, "default_workflow", "")
                or "alima_v51"
            )
        self.workflow_list.refresh(
            workflow_entries(logger=self.logger), default_stem=default_stem
        )

    def _on_default_workflow_changed(self, stem: str) -> None:
        """The star moved. Written on Save like every other setting."""
        self.logger.debug("Standard-Workflow gewählt: %s", stem)

    def _edit_workflow(self, stem: str) -> None:
        """Open the YAML editor for this row; refresh the list afterwards.

        The classic row (``CLASSIC_STEM``) has ``editable=False`` and never
        reaches here — it has no YAML, and the per-step override dialog that
        used to answer "edit" for it (Smart/Advanced/Expert) was removed Oct
        2026: the classic pipeline runs every step on the pipeline default
        already set on the Sprachmodelle page.

        The editor writes immediately and has no part in this dialog's unit of
        work — "Abbrechen" here does not undo a saved workflow. It is a
        separate modal dialog with its own Save, which is how the menu entry
        uses it too. - Claude Generated
        """
        from .workflow_editor_dialog import WorkflowEditorDialog

        editor = WorkflowEditorDialog(self, initial_workflow=stem or None)
        editor.saved.connect(lambda saved_stem: self._refresh_workflow_list())
        editor.exec()
        self._refresh_workflow_list()

    def _build_program_group(self) -> QGroupBox:
        """Die Anwendung selbst: Darstellung, Protokoll, Ablage - Claude Generated"""
        group = QGroupBox("Anwendung")
        form = QFormLayout()

        self.enable_webcam_input = QCheckBox("Kamera-Aufnahmen zulassen")
        self.enable_webcam_input.setToolTip(
            "Abgeschaltet erreicht die Anwendung die Kamera nicht: der Knopf "
            "verschwindet, und der Aufnahmedialog öffnet auch dann nicht, wenn er "
            "auf anderem Weg angestoßen wird."
        )
        form.addRow("Webcam:", self.enable_webcam_input)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setToolTip("Basis-Schriftgröße für alle Bedienelemente (8–16 pt)")
        form.addRow("Schriftgröße:", self.font_size_spin)

        # Same setting as Extras ▸ Farbschema (UIConfig.color_scheme).
        # - Claude Generated
        self.color_scheme_combo = QComboBox()
        for i, name in enumerate(COLOR_SCHEMES):
            self.color_scheme_combo.addItem(COLOR_SCHEME_LABELS[name], name)
            self.color_scheme_combo.setItemData(i, COLOR_SCHEME_TOOLTIPS[name], Qt.ItemDataRole.ToolTipRole)
        self.color_scheme_combo.setToolTip(
            "ALIMA: Farben aus dem Logo. Neutral: Grautöne mit ruhigem Blau. "
            "Desktop: Farben und Hell/Dunkel aus den Systemeinstellungen."
        )
        form.addRow("Farbschema:", self.color_scheme_combo)

        self.log_level = QComboBox()
        for value, label in LOG_LEVEL_CHOICES:
            self.log_level.addItem(label, value)
        self.log_level.setToolTip(
            "Wie ausführlich in alima.log geschrieben wird. Für eine Fehlermeldung "
            "auf \u201eDebug\u201c stellen, das Problem wiederholen und die Datei "
            "mitschicken."
        )
        form.addRow("Protokoll:", self.log_level)

        autosave_layout = QHBoxLayout()
        self.autosave_dir = QLineEdit()
        self.autosave_dir.setToolTip(
            "Verzeichnis für automatisch gespeicherte Pipeline-Ergebnisse und Batch-Output"
        )
        autosave_layout.addWidget(self.autosave_dir)
        autosave_browse = QPushButton("Durchsuchen…")
        autosave_browse.clicked.connect(self._browse_autosave_dir)
        autosave_layout.addWidget(autosave_browse)
        form.addRow("Ergebnis-Ablage:", autosave_layout)

        group.setLayout(form)
        return group

    def _build_indexing_context_group(self) -> QGroupBox:
        """Was der Agent über die eigene Einrichtung und deren Regeln weiß."""
        group = QGroupBox("Haus und Regeln")
        form = QFormLayout()

        self.chat_institution_context = QTextEdit()
        self.chat_institution_context.setMaximumHeight(70)
        self.chat_institution_context.setToolTip(
            "Wird jedem Chat-System-Prompt vorangestellt, um dem Agenten Identität "
            "und Kontext zu geben. Leer = keine institutionelle Rahmung."
        )
        self.chat_institution_context.setPlaceholderText(
            "z.B. Du bist der Chatbot der Universitätsbibliothek XYZ."
        )
        form.addRow("Institutionskontext:", self.chat_institution_context)

        # Personal indexing rules live in their own file, so they get a button
        # rather than a field here. - Claude Generated
        rules_row = QHBoxLayout()
        self.open_rules_btn = QPushButton("Zusatzregeln verwalten…")
        self.open_rules_btn.setToolTip(
            "Persönliche Regeln, die an die Prompts der agentischen Schritte, des "
            "Planers und des Chats angehängt werden. Liegen in rules.yaml neben "
            "der config.json; die klassische Pipeline liest sie nicht."
        )
        self.open_rules_btn.clicked.connect(self._open_rules_dialog)
        rules_row.addWidget(self.open_rules_btn)
        rules_row.addStretch()
        form.addRow("Zusatzregeln:", rules_row)

        group.setLayout(form)
        return group

    def _build_model_switch_group(self) -> QGroupBox:
        """Ob der Chat-Agent sein eigenes Modell wechseln darf - Claude Generated"""
        group = QGroupBox("Modellwechsel durch den Agenten")
        form = QFormLayout()

        self.chat_allow_model_switch = QCheckBox(
            "Der Chat-Agent darf sein Modell selbst wechseln"
        )
        self.chat_allow_model_switch.setToolTip(
            "Betrifft nur den Wechsel aus eigenem Antrieb: der Agent bekommt das "
            "Werkzeug, für den Rest der Sitzung auf ein anderes Modell zu gehen — "
            "etwa auf ein stärkeres für eine schwierige Klassifikation. "
            "Abgeschaltet wird das Werkzeug gar nicht erst angeboten.\n\n"
            "Ein Pipeline-Lauf ist davon nicht betroffen: er nimmt sein Modell "
            "aus der Auswahl im Pipeline-Tab, sonst aus den Standards oben.\n\n"
            "Ein Wechsel gilt ab der nächsten Antwort, wird nie gespeichert und "
            "endet, sobald Du die LLM-Auswahl in der Werkzeugleiste anfasst."
        )
        form.addRow("Selbstständig:", self.chat_allow_model_switch)

        group.setLayout(form)
        return group

    # (value, label) of the two action pickers - Claude Generated
    REPETITION_ACTIONS = (
        ("agent", "Agent entscheidet (Hinweis + Teiltext)"),
        ("retry", "Einmal neu mit anderen Parametern"),
        ("abort", "Abbrechen mit Teilergebnis"),
    )
    CLASSIC_REPETITION_ACTIONS = (
        ("retry", "Neu mit anderen Parametern"),
        ("abort", "Abbrechen mit Teilergebnis"),
    )

    @staticmethod
    def _rate_spin(tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 0.5)
        spin.setDecimals(3)
        spin.setSingleStep(0.005)
        spin.setToolTip(tooltip)
        return spin

    def _build_repetition_group(self) -> QGroupBox:
        """Wiederholungserkennung über den Informationsgehalt - Claude Generated"""
        repetition_group = QGroupBox("🔄 Wiederholungserkennung")
        repetition_layout = QFormLayout()

        scope_note = QLabel(
            "Wirkt auf klassische Läufe, agentische Workflows und den Chat (alle Anbieter). "
            "Gemessen wird, wie viel Neues jeder Abschnitt von 400 Zeichen gegenüber dem "
            "bisher erzeugten Text enthält (komprimierte Bytes je Zeichen). Ein niedriger "
            "Wert heißt: der Text wiederholt sich."
        )
        scope_note.setWordWrap(True)
        set_tone(scope_note, "muted")
        scope_note.setStyleSheet("font-size: 10px;")
        repetition_layout.addRow(scope_note)

        self.repetition_enabled = QCheckBox("Wiederholungen erkennen")
        self.repetition_enabled.setToolTip("Misst den Informationsgehalt der Antwort und des Denkkanals")
        repetition_layout.addRow("Erkennung:", self.repetition_enabled)

        self.repetition_auto_abort = QCheckBox("Bei Schleife abbrechen")
        self.repetition_auto_abort.setToolTip(
            "Aus: nur warnen (Warnleiste, Log), die Generierung läuft weiter"
        )
        repetition_layout.addRow("Abbruch:", self.repetition_auto_abort)

        self.repetition_warn_threshold = self._rate_spin("Unter diesem Wert erscheint eine Warnung")
        self.repetition_abort_threshold = self._rate_spin("Unter diesem Wert wird abgebrochen")
        answer_row = QHBoxLayout()
        answer_row.addWidget(QLabel("Warnung unter"))
        answer_row.addWidget(self.repetition_warn_threshold)
        answer_row.addWidget(QLabel("Abbruch unter"))
        answer_row.addWidget(self.repetition_abort_threshold)
        answer_row.addStretch()
        repetition_layout.addRow("Antwort:", answer_row)

        self.repetition_thinking_warn_threshold = self._rate_spin(
            "Denkkanal (Reasoning-Feld oder <think>): Warnung unter diesem Wert"
        )
        self.repetition_thinking_abort_threshold = self._rate_spin(
            "Denkkanal (Reasoning-Feld oder <think>): Abbruch unter diesem Wert"
        )
        thinking_row = QHBoxLayout()
        thinking_row.addWidget(QLabel("Warnung unter"))
        thinking_row.addWidget(self.repetition_thinking_warn_threshold)
        thinking_row.addWidget(QLabel("Abbruch unter"))
        thinking_row.addWidget(self.repetition_thinking_abort_threshold)
        thinking_row.addStretch()
        repetition_layout.addRow("Denkkanal:", thinking_row)

        # Grace period setting - Claude Generated (2026-02-17)
        self.repetition_grace_period = QDoubleSpinBox()
        self.repetition_grace_period.setRange(0.0, 10.0)
        self.repetition_grace_period.setSingleStep(0.5)
        self.repetition_grace_period.setValue(2.0)
        self.repetition_grace_period.setDecimals(1)
        self.repetition_grace_period.setSuffix(" s")
        self.repetition_grace_period.setToolTip(
            "So lange muss der Wert unter der Abbruchschwelle bleiben, bevor abgebrochen wird\n"
            "0 = sofort"
        )
        repetition_layout.addRow("Gnadenfrist:", self.repetition_grace_period)

        self.repetition_action_combo = QComboBox()
        for value, label in self.REPETITION_ACTIONS:
            self.repetition_action_combo.addItem(label, value)
        self.repetition_action_combo.setToolTip("Was nach einem Abbruch in agentischen Läufen und im Chat geschieht")
        repetition_layout.addRow("Danach (agentisch):", self.repetition_action_combo)

        self.classic_repetition_action_combo = QComboBox()
        for value, label in self.CLASSIC_REPETITION_ACTIONS:
            self.classic_repetition_action_combo.addItem(label, value)
        self.classic_repetition_action_combo.setToolTip("Was nach einem Abbruch in der klassischen Pipeline geschieht")
        repetition_layout.addRow("Danach (klassisch):", self.classic_repetition_action_combo)

        self.repetition_max_retries = QSpinBox()
        self.repetition_max_retries.setRange(0, 3)
        self.repetition_max_retries.setToolTip(
            "Neue Versuche bzw. Agent-Hinweise je Lauf; danach endet der Schritt mit dem Teilergebnis"
        )
        repetition_layout.addRow("Versuche je Lauf:", self.repetition_max_retries)

        repetition_group.setLayout(repetition_layout)
        return repetition_group

    def _build_config_path_group(self) -> QGroupBox:
        """Wo die Konfiguration liegt - Claude Generated"""
        # Where the configuration lives. One file — the project/user/system
        # checkboxes above this used to offer a choice save_config never read.
        # - Claude Generated
        paths_group = QGroupBox("Konfigurationsdatei")
        paths_layout = QVBoxLayout()

        config_info = self.config_manager.get_config_info()
        paths_label = QLabel(config_info["config_file"])
        paths_label.setWordWrap(True)
        paths_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        paths_layout.addWidget(paths_label)

        paths_group.setLayout(paths_layout)
        return paths_group

    def _build_institution_group(self) -> QGroupBox:
        """ISIL and ELN of the library (``SystemConfig.institution_*``).

        One place for the values that used to be fixed to the UB Freiberg; the
        consumers read them through ``src/utils/institution.py``. - Claude Generated
        """
        group = QGroupBox("Einrichtung")
        form = QFormLayout()

        self.institution_isil = QLineEdit()
        self.institution_isil.setPlaceholderText("z. B. DE-105")
        self.institution_isil.setToolTip("ISIL der Bibliothek; leer = nicht gesetzt.")
        form.addRow("ISIL:", self.institution_isil)

        self.institution_eln = QLineEdit()
        self.institution_eln.setPlaceholderText("z. B. 105")
        self.institution_eln.setToolTip(
            "ELN der Bibliothek im K10plus; wird nicht aus der ISIL abgeleitet. "
            "Leer = nicht gesetzt."
        )
        form.addRow("ELN:", self.institution_eln)

        hint = QLabel(
            "Die ISIL bestimmt die Abschlusszeile $A<ISIL> der Schlagwortketten im "
            "WinIBW-Eintrag, die DK-Facette der Bibliothek bei finc und in der "
            "Libero-Webabfrage (udk_raw_<isil>) und den Bestandsfilter der "
            "Libero-Websuche. Die ELN wählt im K10plus das Exemplar der Bibliothek: "
            "lokale Signatur, Notationen und Ketten im Werkzeug search_k10plus und "
            "die lokalen Notationen der DK-Suche über SRU (Preset k10plus). Ohne "
            "ISIL fehlt die Abschlusszeile, und die DK-Auswertung meldet "
            "„Einrichtung nicht gesetzt“."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        group.setLayout(form)
        return group

    # ---- pages -------------------------------------------------------
    #
    # One method per sidebar entry. Each composes group builders and registers
    # its controls with the search index. - Claude Generated

    PAGE_MODELS = "Sprachmodelle"
    PAGE_INDEXING = "Erschließung und Workflows"
    PAGE_SOURCES = "Datenquellen"
    PAGE_STORAGE = "Daten und Speicher"
    PAGE_APPLICATION = "Anwendung"

    def _page(self, *groups) -> QWidget:
        """A page body: group boxes stacked, nothing else."""
        widget = QWidget()
        box = QVBoxLayout(widget)
        for group in groups:
            box.addWidget(group)
        box.addStretch()
        return widget

    def _add_models_page(self) -> None:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.unified_provider_tab, 1)
        box.addWidget(self._build_model_switch_group())
        box.addWidget(self._build_repetition_group())
        self.shell.add_page(self.PAGE_MODELS, page)
        self.index.add(self.PAGE_MODELS, "Modellwechsel durch den Agenten",
                       widget=self.chat_allow_model_switch,
                       synonyms=["switch", "agent"])
        self.index.add(self.PAGE_MODELS, "Wiederholungserkennung",
                       widget=self.repetition_enabled,
                       synonyms=["repetition", "schleife", "loop"])

    def _add_indexing_page(self) -> None:
        self.shell.add_page(
            self.PAGE_INDEXING,
            self._page(
                self._build_workflow_group(),
                self._build_indexing_context_group(),
            ),
        )
        self.index.add(self.PAGE_INDEXING, "Workflows",
                       section="Workflows", widget=self.workflow_list,
                       synonyms=["ablauf", "yaml", "workflow", "standard",
                                 "agentisch", "editor", "bearbeiten"])
        self.index.add(self.PAGE_INDEXING, "Institutionskontext",
                       section="Haus und Regeln",
                       widget=self.chat_institution_context,
                       synonyms=["haus", "bibliothek", "prompt"])
        self.index.add(self.PAGE_INDEXING, "Zusatzregeln",
                       section="Haus und Regeln", widget=self.open_rules_btn,
                       synonyms=["regeln", "rules"])

    def _add_sources_page(self) -> None:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._build_institution_group())
        box.addWidget(self.plugin_settings_tab, 1)
        self.shell.add_page(self.PAGE_SOURCES, page)
        self.index.add(self.PAGE_SOURCES, "Datenquellen",
                       synonyms=["plugin", "lobid", "swb", "katalog", "doi"])
        self.index.add(self.PAGE_SOURCES, "ISIL der Bibliothek",
                       section="Einrichtung", widget=self.institution_isil,
                       synonyms=["isil", "einrichtung", "bibliothek", "sigel",
                                 "institution"])
        self.index.add(self.PAGE_SOURCES, "ELN der Bibliothek",
                       section="Einrichtung", widget=self.institution_eln,
                       synonyms=["eln", "k10plus", "exemplar", "einrichtung"])

    def _add_storage_page(self) -> None:
        cache = CollapsibleSection("Zwischenspeicher", expanded=True)
        cache_box = QVBoxLayout(cache.body())
        cache_box.setContentsMargins(0, 0, 0, 0)
        cache_box.addWidget(self._build_source_cache_group())

        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self.database_tab)
        box.addWidget(cache)
        box.addStretch()
        self.shell.add_page(self.PAGE_STORAGE, page)

        self.index.add(self.PAGE_STORAGE, "Wissensdatenbank",
                       widget=self.db_type_combo,
                       synonyms=["sqlite", "mariadb", "mysql", "datenbank"])
        self.index.add(self.PAGE_STORAGE, "Datei auf diesem Rechner",
                       section="Wissensdatenbank", widget=self.sqlite_path,
                       synonyms=["sqlite", "pfad"])
        self.index.add(self.PAGE_STORAGE, "Zeitlimit",
                       section="Wissensdatenbank", widget=self.connection_timeout,
                       synonyms=["timeout"])
        self.index.add(self.PAGE_STORAGE, "Antworten zwischenspeichern",
                       section="Zwischenspeicher",
                       widget=self.enable_response_cache,
                       reveal=cache.expand, synonyms=["cache", "rohantworten"])
        self.index.add(self.PAGE_STORAGE, "Cache-Verzeichnis",
                       section="Zwischenspeicher", widget=self.cache_dir,
                       reveal=cache.expand, synonyms=["cache", "ordner"])

    def _add_application_page(self) -> None:
        self.shell.add_page(
            self.PAGE_APPLICATION,
            self._page(
                self._build_program_group(),
                self._build_config_path_group(),
                self._build_about_group(),
            ),
        )
        self.index.add(self.PAGE_APPLICATION, "Protokoll", widget=self.log_level,
                       synonyms=["log", "debug", "logdatei", "support"])
        self.index.add(self.PAGE_APPLICATION, "Schriftgröße",
                       widget=self.font_size_spin, synonyms=["font size", "schrift"])
        self.index.add(self.PAGE_APPLICATION, "Farbschema",
                       widget=self.color_scheme_combo,
                       synonyms=["farben", "theme", "design", "dunkel", "hell",
                                 "neutral", "desktop", "color scheme"])
        self.index.add(self.PAGE_APPLICATION, "Kamera-Aufnahmen zulassen",
                       widget=self.enable_webcam_input,
                       synonyms=["webcam", "kamera", "datenschutz"])
        self.index.add(self.PAGE_APPLICATION, "Ergebnis-Ablage",
                       widget=self.autosave_dir,
                       synonyms=["autosave", "ergebnisse", "ordner"])

    def _add_wizard_shortcut(self) -> None:
        """The way into the multi-page setup, under the sidebar - Claude Generated."""
        button = QPushButton("🧭 Einrichtungsassistent…")
        button.setToolTip(
            "Führt durch Anbieter, Modelle, Datenbank und Katalog. Bestehende "
            "Einstellungen bleiben erhalten."
        )
        button.clicked.connect(self._open_setup_wizard)
        self.shell.sidebar_footer.addWidget(button)

    def _open_setup_wizard(self) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "show_setup_wizard"):
            self.accept()
            parent.show_setup_wizard()
        else:
            QMessageBox.information(
                self, "Nicht verfügbar",
                "Der Einrichtungsassistent ist nur aus dem Hauptfenster erreichbar.",
            )

    @staticmethod
    def _scrolled(*groups) -> QWidget:
        """Stack group boxes in a scroll area so nothing is crushed - Claude Generated."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        for group in groups:
            layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(inner)
        outer_layout.addWidget(scroll)
        return outer

    def _create_system_tab(self) -> QWidget:
        """The System tab as a composition of its group builders - Claude Generated."""
        return self._scrolled(
            self._build_program_group(),
            self._build_chat_group(),
            self._build_repetition_group(),
            self._build_config_path_group(),
        )

    def _build_about_group(self) -> QGroupBox:
        """Project identity from the single source of truth - Claude Generated.

        Facts come from ``src/core/about.py``, which ``tests/test_about.py`` pins
        against the README. Was a tab of its own that additionally documented a
        four-level config priority the loader never had.
        """
        from ..core import about as about_mod

        pub = about_mod.PUBLICATION
        group = QGroupBox("Über ALIMA")
        box = QVBoxLayout(group)

        about_label = QLabel(
            f"<b>{about_mod.NAME}</b> — {about_mod.EXPANSION}"
            f"<p>{about_mod.SUMMARY}</p>"
            f"<p><b>Einrichtung:</b> {about_mod.INSTITUTION}<br>"
            f"<b>Lizenz:</b> {about_mod.LICENSE}<br>"
            f"<b>Repository:</b> <a href=\"{about_mod.REPOSITORY}\">"
            f"{about_mod.REPOSITORY}</a></p>"
            f"<p>{pub['citation']}<br>"
            f"<a href=\"{pub['url']}\">{pub['url']}</a>"
            f"{' (Open Access)' if pub['open_access'] else ''}</p>"
        )
        about_label.setWordWrap(True)
        about_label.setOpenExternalLinks(True)
        about_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        box.addWidget(about_label)
        return group

    def _load_current_settings(self):
        """Load current settings into UI elements - Claude Generated"""
        config = self.config_to_edit
        
        # Database settings
        self.db_type_combo.setCurrentText(config.database.db_type)
        self.sqlite_path.setText(config.database.sqlite_path)
        self.mysql_host.setText(config.database.host)
        self.mysql_port.setValue(config.database.port)
        self.mysql_database.setText(config.database.database)
        self.mysql_username.setText(config.database.username)
        self.mysql_password.setText(config.database.password)
        self.mysql_charset.setCurrentText(config.database.charset)
        self.mysql_ssl_disabled.setChecked(config.database.ssl_disabled)
        self.connection_timeout.setValue(config.database.connection_timeout)
        
        # LLM settings are now handled by the unified provider tab
        # Note: Static provider API keys (Gemini, Anthropic) are managed through the unified provider system
        
        # Provider lists are now managed by the unified provider tab
        # Dynamic provider population is handled automatically by the UnifiedProviderTab
        
        # Search-provider + input-source config is edited in the Plugins tab.
        self.plugin_settings_tab.load(config)
        
        # System settings
        self.enable_response_cache.setChecked(
            getattr(config.system_config, "enable_response_cache", True)
        )
        self.aggregate_from_raw.setChecked(
            getattr(config.system_config, "aggregate_from_raw", True)
        )
        lvl_idx = self.log_level.findData(config.system_config.log_level)
        self.log_level.setCurrentIndex(lvl_idx if lvl_idx >= 0 else 1)
        default_workflow = getattr(config.system_config, "default_workflow", "alima_v51") or "alima_v51"
        self._refresh_workflow_list(default_stem=default_workflow)
        self.cache_dir.setText(config.system_config.cache_dir)
        self.autosave_dir.setText(config.system_config.autosave_dir)
        self.institution_isil.setText(getattr(config.system_config, "institution_isil", "") or "")
        self.institution_eln.setText(getattr(config.system_config, "institution_eln", "") or "")

        # Chat-Agent settings - Claude Generated
        chat_cfg = getattr(config, "chat_config", None)
        self.chat_institution_context.setPlainText(
            getattr(chat_cfg, "institution_context", "") if chat_cfg else ""
        )
        self.chat_allow_model_switch.setChecked(
            bool(getattr(chat_cfg, "allow_model_switch", False)) if chat_cfg else False
        )


        # UI settings - Claude Generated
        self.enable_webcam_input.setChecked(config.ui_config.enable_webcam_input)
        self.font_size_spin.setValue(getattr(config.ui_config, "font_size", 10))
        scheme_idx = self.color_scheme_combo.findData(
            normalize_color_scheme(getattr(config.ui_config, "color_scheme", "alima"))
        )
        self.color_scheme_combo.setCurrentIndex(max(0, scheme_idx))

        # Repetition Detection settings - Claude Generated
        from ..utils.config_models import RepetitionDetectionConfig

        rep_config = getattr(config, 'repetition_config', None) or RepetitionDetectionConfig()
        self.repetition_enabled.setChecked(rep_config.enabled)
        self.repetition_auto_abort.setChecked(rep_config.auto_abort)
        self.repetition_warn_threshold.setValue(rep_config.warn_threshold)
        self.repetition_abort_threshold.setValue(rep_config.abort_threshold)
        self.repetition_thinking_warn_threshold.setValue(rep_config.thinking_warn_threshold)
        self.repetition_thinking_abort_threshold.setValue(rep_config.thinking_abort_threshold)
        self.repetition_grace_period.setValue(rep_config.grace_period_seconds)
        self.repetition_action_combo.setCurrentIndex(
            max(0, self.repetition_action_combo.findData(rep_config.repetition_action))
        )
        self.classic_repetition_action_combo.setCurrentIndex(
            max(0, self.classic_repetition_action_combo.findData(rep_config.classic_repetition_action))
        )
        self.repetition_max_retries.setValue(rep_config.max_retries)

        # Update UI based on database type
        self._on_db_type_changed(config.database.db_type)

    def _on_db_type_changed(self, db_type: str):
        """Handle database type change - Claude Generated"""
        if db_type == "sqlite":
            self.sqlite_group.setVisible(True)
            self.mysql_group.setVisible(False)
        else:
            self.sqlite_group.setVisible(False)
            self.mysql_group.setVisible(True)

    def _open_rules_dialog(self):
        """Open the personal-rules dialog. - Claude Generated"""
        from .dialogs.rules_dialog import RulesDialog

        RulesDialog(parent=self).exec()

    def _browse_autosave_dir(self):
        """Browse for autosave directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Autosave-Verzeichnis wählen", self.autosave_dir.text()
        )
        if directory:
            self.autosave_dir.setText(directory)

    def _browse_cache_dir(self):
        """Browse for cache directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Cache-Verzeichnis wählen", self.cache_dir.text()
        )
        if directory:
            self.cache_dir.setText(directory)

    def _browse_sqlite_path(self):
        """Browse for SQLite database path - Claude Generated"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Select SQLite Database", 
            self.sqlite_path.text(),
            "SQLite Database (*.db *.sqlite *.sqlite3);;All Files (*)"
        )
        if file_path:
            self.sqlite_path.setText(file_path)

    def _test_database_connection(self):
        """Test the connection with the settings currently shown - Claude Generated

        Reads only the database widgets. It used to call _get_config_from_ui and
        assign the result to config_manager._config, which pushed every unsaved
        edit — plugins, providers, chat settings — into the running application,
        with no way back short of a restart.
        """
        from ..utils.config_models import DatabaseConfig

        probe = DatabaseConfig(
            db_type=self.db_type_combo.currentText(),
            sqlite_path=self.sqlite_path.text(),
            host=self.mysql_host.text(),
            port=self.mysql_port.value(),
            database=self.mysql_database.text(),
            username=self.mysql_username.text(),
            password=self.mysql_password.text(),
            charset=self.mysql_charset.currentText(),
            ssl_disabled=self.mysql_ssl_disabled.isChecked(),
            connection_timeout=self.connection_timeout.value(),
        )

        self.db_test_button.setEnabled(False)
        self.db_test_button.setText("Testing...")

        self.db_test_worker = DatabaseTestWorker(self.config_manager, probe)
        self.db_test_worker.test_completed.connect(self._on_db_test_completed)
        self.db_test_worker.start()
    
    @pyqtSlot(bool, str)
    def _on_db_test_completed(self, success: bool, message: str):
        """Handle database test completion - Claude Generated"""
        self.db_test_button.setEnabled(True)
        self.db_test_button.setText("🔧 Test Connection")

        if success:
            QMessageBox.information(self, "Database Test", f"✅ {message}")
        else:
            QMessageBox.warning(self, "Database Test", f"❌ {message}")

    def _reset_database_to_defaults(self):
        """Reset only database settings to defaults - Claude Generated"""
        reply = QMessageBox.question(
            self,
            "Reset Database Settings",
            "Are you sure you want to reset database settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset only database-related UI elements to defaults
            self.db_type_combo.setCurrentText("sqlite")
            self.sqlite_path.setText("alima_knowledge.db")
            self.mysql_host.setText("localhost")
            self.mysql_port.setValue(3306)
            self.mysql_database.setText("")
            self.mysql_username.setText("")
            self.mysql_password.setText("")
            self.mysql_charset.setCurrentText("utf8mb4")
            self.mysql_ssl_disabled.setChecked(False)
            self.connection_timeout.setValue(30)

            # Trigger UI updates
            self._on_db_type_changed("sqlite")

            QMessageBox.information(self, "Reset Complete", "Database settings have been reset to defaults.")

    def _get_config_from_ui(self) -> AlimaConfig:
        """Extract configuration from UI elements - Claude Generated (Refactoring)"""
        # Use the Unit of Work copy and update it with UI values
        config = self.config_to_edit

        # Harvest the provider tab's default-provider/model selectors into
        # unified_config now (they have no live writeback — see unified_provider_tab).
        try:
            self.unified_provider_tab._update_config_from_ui()
        except Exception as e:
            self.logger.warning(f"Could not sync provider tab defaults: {e}")
        
        # Database configuration — mutate, never rebuild. A reconstructed dataclass
        # resets every field this dialog has no widget for (gnd_local_path did
        # exactly that until Sep 2026); mutating leaves them alone by construction.
        # - Claude Generated
        db = config.database_config
        db.db_type = self.db_type_combo.currentText()
        db.sqlite_path = self.sqlite_path.text()
        db.host = self.mysql_host.text()
        db.port = self.mysql_port.value()
        db.database = self.mysql_database.text()
        db.username = self.mysql_username.text()
        db.password = self.mysql_password.text()
        db.charset = self.mysql_charset.currentText()
        db.ssl_disabled = self.mysql_ssl_disabled.isChecked()
        db.connection_timeout = self.connection_timeout.value()
        
        # Unified provider configuration - use current configuration as-is
        # Provider settings are managed through the unified provider system
        # No need to reconstruct LLMConfig - unified_config is already properly managed
        

        # Per-plugin enable/config now lives in config.plugins (applied below,
        # after all legacy sections are built). - Claude Generated

        # System configuration — only what this dialog has a widget for. Everything
        # else (DOI mirror fields, the wizard flags, prompts_path,
        # enable_code_plugins from the Plugins tab, url_fetch_allowlist /
        # url_fetch_max_bytes with no widget at all) keeps its loaded value because
        # it is not touched. - Claude Generated
        sys_cfg = config.system_config
        sys_cfg.log_level = self.log_level.currentData()
        sys_cfg.cache_dir = self.cache_dir.text()
        sys_cfg.autosave_dir = self.autosave_dir.text()
        sys_cfg.default_workflow = self.workflow_list.default_stem()
        sys_cfg.enable_response_cache = self.enable_response_cache.isChecked()
        sys_cfg.aggregate_from_raw = self.aggregate_from_raw.isChecked()
        sys_cfg.institution_isil = self.institution_isil.text().strip()
        sys_cfg.institution_eln = self.institution_eln.text().strip()

        # Chat-Agent configuration — preserve all other ChatConfig fields (they
        # have no UI here; e.g. the persisted default provider/model combo,
        # autonomous_pipeline) and only update institution_context. - Claude Generated
        if getattr(config, "chat_config", None) is None:
            from ..utils.config_models import ChatConfig
            config.chat_config = ChatConfig()
        config.chat_config.institution_context = self.chat_institution_context.toPlainText().strip()
        config.chat_config.allow_model_switch = self.chat_allow_model_switch.isChecked()

        # UI configuration — ui_language and ptheta_banner_seen have no widget and
        # were reset on every save while this rebuilt the dataclass. - Claude Generated
        from ..utils.config_models import RepetitionDetectionConfig

        config.ui_config.enable_webcam_input = self.enable_webcam_input.isChecked()
        config.ui_config.font_size = self.font_size_spin.value()
        config.ui_config.color_scheme = normalize_color_scheme(self.color_scheme_combo.currentData())

        # Repetition Detection — segment_chars and smoothing_segments have no
        # widget and keep their loaded values. abort never above warn (the
        # dataclass enforces the same on load). - Claude Generated
        if getattr(config, "repetition_config", None) is None:
            config.repetition_config = RepetitionDetectionConfig()
        rep = config.repetition_config
        rep.enabled = self.repetition_enabled.isChecked()
        rep.auto_abort = self.repetition_auto_abort.isChecked()
        rep.warn_threshold = self.repetition_warn_threshold.value()
        rep.abort_threshold = min(self.repetition_abort_threshold.value(), rep.warn_threshold)
        rep.thinking_warn_threshold = self.repetition_thinking_warn_threshold.value()
        rep.thinking_abort_threshold = min(
            self.repetition_thinking_abort_threshold.value(), rep.thinking_warn_threshold
        )
        rep.grace_period_seconds = self.repetition_grace_period.value()
        rep.repetition_action = self.repetition_action_combo.currentData() or "agent"
        rep.classic_repetition_action = self.classic_repetition_action_combo.currentData() or "retry"
        rep.max_retries = self.repetition_max_retries.value()

        # Task preferences are already up-to-date in config_to_edit from UnifiedProviderTab - Claude Generated (Refactoring)

        # Plugin instances are authoritative and edited only in the Plugins tab
        # (the Catalog tab + DOI System entries were removed). Apply them; the
        # derive-on-save in ConfigManager re-mirrors instances → the DOI
        # SystemConfig fields, the last remaining mirror. - Claude Generated
        try:
            self.plugin_settings_tab.apply_to(config)
        except Exception as e:
            self.logger.warning(f"Could not apply plugin settings: {e}")

        return config
    
    def _export_as_preset(self):
        """Export the saved setup as an institution package (folder with
        ``alima_presets.json`` and its files) - Claude Generated

        Same export as the button in the preset group of the Plugins tab
        (``preset_export.export_preset``): no API keys or tokens; a database password only
        when the export dialog asks for it.
        """
        self.plugin_settings_tab.start_preset_export()

    def _on_preset_applied(self, steps, saved) -> None:
        """A preset was applied or reverted in the Plugins tab and saved.

        The tab has updated its instances already; here the system values it
        set (ISIL, ELN, standard workflow) reach this dialog's copy and its
        widgets, and the copy takes the saved ledger, so "Save" does not undo
        the preset. - Claude Generated
        """
        from ..utils.preset_apply import replay_steps

        cfg = self.config_to_edit
        system_steps = [s for s in steps if s["path"] and s["path"][0] == "system_config"]
        replay_steps(cfg, system_steps)
        cfg.applied_presets = deepcopy(getattr(saved, "applied_presets", []) or [])
        keys = {s["path"][1] for s in system_steps}
        if "institution_isil" in keys:
            self.institution_isil.setText(cfg.system_config.institution_isil or "")
        if "institution_eln" in keys:
            self.institution_eln.setText(cfg.system_config.institution_eln or "")
        if "default_workflow" in keys:
            self._refresh_workflow_list(default_stem=cfg.system_config.default_workflow)
        self.config_changed.emit()

    def reject(self):
        """Cancel: undo the provider edits that were written on the way.

        Provider add/edit/remove persists immediately, because the model list is
        fetched through LlmService from the *saved* config — a provider held only
        in this dialog's copy can never load its models. So Cancel has to roll
        that back rather than simply drop the copy. - Claude Generated
        """
        try:
            if self.unified_provider_tab.restore_providers_at_open():
                self.config_changed.emit()
        except Exception as e:
            self.logger.error(f"Provider-Änderungen konnten nicht zurückgenommen werden: {e}")
        super().reject()

    def _save_and_close(self):
        """Save configuration and close dialog - Claude Generated

        Timed end to end: the whole chain — reading the widgets, writing the
        file, and the ``config_changed`` handler that refreshes every tab — runs
        synchronously inside this button click, so the UI is frozen for its
        duration. The phases go to DEBUG and only the total to INFO, so a normal
        run stays quiet and a slow one can still be taken apart with
        ``--log-level DEBUG``.
        """
        _log = logging.getLogger(__name__)
        _t_start = time.monotonic()
        try:
            # Task preferences are now handled by the unified provider tab automatically
            
            # Provider preferences are handled by the unified provider tab and Unit of Work pattern - Claude Generated (Refactoring)
            
            # Get configuration from UI
            _t = time.monotonic()
            config = self._get_config_from_ui()
            _log.debug(f"⏱ Settings: Widgets auslesen {time.monotonic() - _t:.2f}s")
            
            # Save configuration
            _t = time.monotonic()
            success = self.config_manager.save_config(config)
            _log.debug(f"⏱ Settings: save_config {time.monotonic() - _t:.2f}s")
            if not success:
                QMessageBox.critical(self, "Save Error", "Failed to save configuration!")
                return

            # Emit signal and close. The handler runs synchronously here — its
            # own per-step timings are logged by ``_refresh_components``.
            _t = time.monotonic()
            self.config_changed.emit()
            _log.debug(f"⏱ Settings: config_changed-Handler {time.monotonic() - _t:.2f}s")
            self.accept()
            _log.info(f"⏱ Settings: gesamt {time.monotonic() - _t_start:.2f}s")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving configuration: {str(e)}")



if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = ComprehensiveSettingsDialog()
    dialog.show()
    sys.exit(app.exec())
