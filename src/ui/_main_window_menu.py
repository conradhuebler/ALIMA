"""Menu-bar + self-update/checkout mixin for MainWindow. Claude Generated.

Extracted from ``main_window.py`` (F-5 god-file split): ``create_menu_bar`` and
the git checkout / update-process methods. Method bodies are moved verbatim.
``MainWindowMenuMixin`` is mixed into ``MainWindow`` (which provides the actions/
slots the menu binds and the widgets these drive); not a standalone window.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from ..core.gitupdate import GitUpdateWorker
from .commit_selector_dialog import CommitSelectorDialog
from .styles import get_font_size


class MainWindowMenuMixin:
    """Menu bar + update/checkout (verbatim from MainWindow)."""

    def create_menu_bar(self):
        """Erstellt die Menüleiste - Claude Generated (Reorganized)"""
        menubar = self.menuBar()

        # ========== Datei-Menü (Workflow-fokussiert) ==========
        file_menu = menubar.addMenu("&Datei")

        # Analyse-Zustand laden - Claude Generated
        load_state_action = file_menu.addAction("📂 &Analyse-Zustand laden...")
        load_state_action.triggered.connect(self.load_analysis_state_from_file)

        # Analyse-Zustand speichern - Claude Generated (Refactored to use unified persistence)
        save_state_action = file_menu.addAction("💾 Analyse-Zustand &speichern...")
        save_state_action.triggered.connect(self.export_current_gui_state)

        file_menu.addSeparator()

        # Beenden-Aktion
        exit_action = file_menu.addAction("🚪 &Beenden")
        exit_action.triggered.connect(self.close)

        # ========== Extras/Tools-Menü (Datenbank und Debug) ==========
        tools_menu = menubar.addMenu("E&xtras")

        # Dark/light theme toggle — Claude Generated
        self._theme_action = tools_menu.addAction("🌙 Dunkles Design")
        self._theme_action.triggered.connect(self._toggle_theme)

        # Font size submenu — Claude Generated
        font_menu = tools_menu.addMenu("🔠 Schriftgröße")
        for pt in [8, 9, 10, 11, 12, 13, 14, 16]:
            action = font_menu.addAction(f"{pt} pt")
            action.setCheckable(True)
            action.setChecked(pt == get_font_size())
            action.triggered.connect(lambda checked, size=pt: self._set_font_size_from_menu(size))
        self._font_size_menu = font_menu

        tools_menu.addSeparator()

        # GND-Datenbank importieren (moved from Datei)
        import_action = tools_menu.addAction("📥 &GND-Datenbank importieren...")
        import_action.triggered.connect(self.import_gnd_database)

        # Database viewer action - Claude Generated
        db_viewer_action = tools_menu.addAction("📊 &Datenbank-Viewer")
        db_viewer_action.triggered.connect(self.show_database_viewer)

        # Clear search cache action - Claude Generated
        clear_cache_action = tools_menu.addAction("🗑️ Such-&Cache leeren...")
        clear_cache_action.triggered.connect(self.clear_search_cache)

        # Cleanup malformed entries action - Claude Generated (Ultra-Deep Fix)
        cleanup_action = tools_menu.addAction("🧹 &Malformed Einträge bereinigen...")
        cleanup_action.triggered.connect(self.cleanup_malformed_entries)

        tools_menu.addSeparator()

        # Batch processing actions - Claude Generated
        batch_process_action = tools_menu.addAction("📦 &Batch Processing...")
        batch_process_action.triggered.connect(self.show_batch_processing_dialog)

        batch_review_action = tools_menu.addAction("📋 Batch-Ergebnisse &laden...")
        batch_review_action.triggered.connect(self.load_batch_results)

        tools_menu.addSeparator()

        # P-γ: Single-step execution dialog
        single_step_action = tools_menu.addAction("🎯 Run Single &Step…")
        single_step_action.setShortcut("Ctrl+Shift+S")
        single_step_action.triggered.connect(self.show_single_step_dialog)

        tools_menu.addSeparator()

        # Erschließungsvergleich - Claude Generated
        compare_states_action = tools_menu.addAction("🔍 Erschließungs&vergleich...")
        compare_states_action.triggered.connect(self._open_comparison_tab)

        # ========== Ansicht-Menü ==========
        view_menu = menubar.addMenu("&Ansicht")

        # P-δ.5a: chat_dock retired. Chat input lives inside PipelineChatPanel
        # in the Pipeline tab — no separate menu action needed.

        # Agentic-Kontext-Dock auf Wunsch anzeigen - Claude Generated
        show_agentic_action = view_menu.addAction("🤖 Agentic &Kontext")
        show_agentic_action.triggered.connect(self._show_agentic_dock)

        view_menu.addSeparator()

        # ========== Bearbeiten-Menü ==========
        edit_menu = menubar.addMenu("&Bearbeiten")

        # Einstellungen-Aktion
        settings_action = edit_menu.addAction("⚙️ &Einstellungen")
        settings_action.triggered.connect(self.show_settings)

        # Prompt-Konfiguration-Aktion
        prompt_config_action = edit_menu.addAction("📝 &Prompt-Konfiguration")
        prompt_config_action.triggered.connect(self.show_prompt_editor)

        # Workflow-Editor-Aktion - Claude Generated
        workflow_editor_action = edit_menu.addAction("📋 &Workflow-Editor")
        workflow_editor_action.triggered.connect(self.show_workflow_editor)

        # ========== Update-Menü ==========
        update_menu = menubar.addMenu("&Updates")

        # Nach Updates suchen
        check_update_action = update_menu.addAction("🔄 Nach &Updates suchen")
        check_update_action.triggered.connect(self.check_for_updates)

        # NEUE OPTION: Zu spezifischem Commit wechseln
        specific_commit_action = update_menu.addAction(
            "🎯 Zu &spezifischem Commit wechseln"
        )
        specific_commit_action.triggered.connect(self.checkout_specific_commit)

        # ========== Hilfe-Menü ==========
        help_menu = menubar.addMenu("&Hilfe")

        # Über-Dialog
        about_action = help_menu.addAction("ℹ️ Ü&ber")
        about_action.triggered.connect(self.show_about)

        # Hilfe-Dialog
        help_action = help_menu.addAction("❓ &Hilfe")
        help_action.triggered.connect(self.show_help)

    def checkout_specific_commit(self):
        """Öffnet einen Dialog zur Auswahl eines spezifischen Commits"""
        dialog = CommitSelectorDialog(self)
        if dialog.exec():
            target_commit = dialog.get_selected_commit()
            if not target_commit:
                return

            reply = QMessageBox.question(
                self,
                "Zu spezifischem Commit wechseln",
                f"Möchten Sie wirklich zu '{target_commit}' wechseln? Dies kann zu Programminstabilität führen, "
                "wenn der ausgewählte Commit nicht mit der aktuellen Version kompatibel ist.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Starte den Update-Prozess mit spezifischem Commit
                self._start_update_process(target_commit=target_commit)

    def check_for_updates(self):
        """Prüft auf Updates und installiert sie bei Bedarf"""
        self.logger.info("Prüfe auf Updates...")
        self._start_update_process()

    def _start_update_process(self, target_commit=None):
        """Startet den Update-Prozess mit optionalem Ziel-Commit"""
        # Erstelle den Progress-Dialog
        if target_commit:
            message = f"Wechsle zu Commit: {target_commit}..."
        else:
            message = "Prüfe auf Updates..."

        progress = QProgressDialog(message, "Abbrechen", 0, 0, self)
        progress.setWindowTitle("Software-Update")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setCancelButton(None)  # Entferne den Abbrechen-Button
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()

        # Erstelle und starte den Worker
        self.update_worker = GitUpdateWorker(target_commit=target_commit)
        self.update_worker.update_progress.connect(
            lambda msg: progress.setLabelText(msg)
        )
        self.update_worker.update_finished.connect(
            lambda success, msg: self.update_completed(success, msg, progress)
        )
        self.update_worker.start()

    def update_completed(self, success, message, progress_dialog):
        """Wird aufgerufen, wenn der Update-Prozess abgeschlossen ist"""
        progress_dialog.close()

        if success:
            QMessageBox.information(self, "Update Status", message)
            if "bereits auf dem neuesten Stand" not in message:
                reply = QMessageBox.question(
                    self,
                    "Neustart erforderlich",
                    "Für die Anwendung der Updates ist ein Neustart erforderlich. Jetzt neu starten?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.save_settings()
                    # Starte das Programm neu
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
        else:
            QMessageBox.warning(self, "Update-Fehler", message)
