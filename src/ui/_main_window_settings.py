"""Settings / theme / dialog-launcher mixin for MainWindow. Claude Generated.

Extracted from ``main_window.py`` (F-5 god-file split): settings persistence,
theme + font-size apply/toggle, config-change refresh, and the dialog launchers
(settings, prompt/workflow editor, DB viewer, cache cleanup, batch + single-step,
batch-results load). Method bodies are moved verbatim. ``MainWindowSettingsMixin``
is mixed into ``MainWindow``; not a standalone window.
"""
from __future__ import annotations

import logging

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.config_manager import ConfigManager
from ..utils.pipeline_defaults import get_autosave_dir
from .comprehensive_settings_dialog import ComprehensiveSettingsDialog
from .styles import (
    DEFAULT_FONT_FAMILY,
    get_font_size,
    get_main_stylesheet,
    set_dark_mode,
    set_font_size,
)
from .tablewidget import DatabaseViewerDialog


class MainWindowSettingsMixin:
    """Settings/theme + dialog launchers (verbatim from MainWindow)."""

    def show_settings(self):
        """Öffnet den umfassenden Einstellungsdialog - Claude Generated"""
        try:
            dialog = ComprehensiveSettingsDialog(alima_manager=self.alima_manager, parent=self)
            dialog.config_changed.connect(self._on_config_changed)
            if dialog.exec():
                # Einstellungen wurden gespeichert
                self.load_settings()
                # Aktualisiere alle Komponenten mit neuer Konfiguration
                self._refresh_components()
                # Runtime plugin toggle: rebuild the chat agent's tool registry so
                # enabled/disabled plugins + endpoint changes take effect without a
                # restart. - Claude Generated
                self._refresh_plugin_tools()
        except Exception as e:
            self.logger.error(f"Fehler beim Öffnen der Einstellungen: {e}")
            QMessageBox.critical(
                self, "Fehler", f"Fehler beim Öffnen der Einstellungen: {str(e)}"
            )

    def _refresh_plugin_tools(self):
        """Rebuild the embedded chat panel's MCP tool registry after a settings
        save, so plugin enable/disable + config changes apply at runtime. - Claude Generated"""
        try:
            panel = getattr(getattr(self, "pipeline_tab", None), "stream_widget", None)
            reg = getattr(panel, "mcp_registry", None)
            if reg is not None and hasattr(reg, "refresh"):
                reg.refresh()
                self.logger.info("Chat tool registry refreshed after settings change")
        except Exception as e:
            self.logger.warning(f"Could not refresh plugin tools: {e}")
        # WP-K5: SearchTab-Quellen-Checkboxen folgen Plugin-Änderungen jetzt
        # ohne Neustart - Claude Generated
        try:
            tab = getattr(self, "search_tab", None)
            if tab is not None and hasattr(tab, "refresh_sources"):
                tab.refresh_sources()
        except Exception as e:
            self.logger.warning(f"Could not refresh SearchTab sources: {e}")

    def load_settings(self):
        """Lädt die gespeicherten Einstellungen"""
        # Fenster-Geometrie
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self.tabs.setCurrentIndex(0)

        # Restore pipeline splitter states - Claude Generated
        if hasattr(self, 'pipeline_tab'):
            self.pipeline_tab.restore_splitter_state(self.settings)

        # Load dark mode preference — Claude Generated
        from src.alima_gui import is_system_dark_mode
        app = QApplication.instance()
        saved = self.settings.value("dark_mode")
        if saved is not None:
            dark = saved == "true" or saved is True
        else:
            dark = is_system_dark_mode(app)
        self.apply_theme(dark)

        # Load font size preference — Claude Generated
        saved_fs = self.settings.value("font_size", None)
        if saved_fs is None and self.config_manager:
            try:
                saved_fs = self.config_manager.load_config().ui_config.font_size
            except Exception:
                saved_fs = 10
        if saved_fs is not None:
            self.apply_font_size(int(saved_fs))

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dark_mode", self._dark_mode)

        # Save pipeline splitter states - Claude Generated
        if hasattr(self, 'pipeline_tab'):
            self.pipeline_tab.save_splitter_state(self.settings)

    def apply_theme(self, dark: bool):
        """Switch between dark and light theme — Claude Generated"""
        from src.alima_gui import _apply_dark_app_palette, _apply_light_app_palette
        self._dark_mode = dark
        set_dark_mode(dark)
        app = QApplication.instance()
        if dark:
            _apply_dark_app_palette(app)
        else:
            _apply_light_app_palette(app)
        if app:
            app.setStyleSheet(get_main_stylesheet())
        self.setStyleSheet(get_main_stylesheet())
        self._reapply_tab_stylesheets()
        label = "☀️ Helles Design" if dark else "🌙 Dunkles Design"
        if hasattr(self, '_theme_action'):
            self._theme_action.setText(label)
        self.settings.setValue("dark_mode", dark)

    def apply_font_size(self, pt: int) -> None:
        """Set global font size and refresh all stylesheets.

        Root cause of earlier "font doesn't apply" bug: many widgets call
        setFont(QFont(...)) directly, which beats the stylesheet cascade.
        Fix: (1) QApplication.setFont() for all widgets that rely on Qt defaults,
        (2) refresh_styles() walk so widgets with explicit setFont() rebuild
        their fonts via get_scaled_font(). — Claude Generated
        """
        set_font_size(pt)
        app = QApplication.instance()
        if app:
            app.setFont(QFont(DEFAULT_FONT_FAMILY, pt))
            app.setStyleSheet(get_main_stylesheet())
        self.setStyleSheet(get_main_stylesheet())
        self._reapply_tab_stylesheets()
        self._walk_refresh_styles(self)
        self.settings.setValue("font_size", pt)
        if self.config_manager:
            try:
                config = self.config_manager.load_config()
                config.ui_config.font_size = pt
                self.config_manager.save_config(config)
            except Exception as e:
                self.logger.debug(f"Could not persist font_size to config: {e}")

    def _walk_refresh_styles(self, widget) -> None:
        """Recursively call refresh_styles() on all child widgets that define it. — Claude Generated"""
        try:
            for child in widget.findChildren(QWidget):
                fn = getattr(child, "refresh_styles", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as e:
                        self.logger.debug(
                            f"refresh_styles on {child.__class__.__name__} failed: {e}"
                        )
        except Exception as e:
            self.logger.debug(f"_walk_refresh_styles failed: {e}")

    def _reapply_tab_stylesheets(self):
        """Re-apply stylesheets to all tabs after theme change — Claude Generated"""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if tab and hasattr(tab, 'refresh_styles'):
                try:
                    tab.refresh_styles()
                except Exception as e:
                    self.logger.debug(f"refresh_styles on tab {i} failed: {e}")

    def _toggle_theme(self):
        """Toggle between dark and light theme — Claude Generated"""
        self.apply_theme(not self._dark_mode)

    def _set_font_size_from_menu(self, pt: int) -> None:
        """Apply font size from menu and update checkmarks. — Claude Generated"""
        self.apply_font_size(pt)
        if hasattr(self, "_font_size_menu"):
            for action in self._font_size_menu.actions():
                try:
                    action.setChecked(action.text() == f"{pt} pt")
                except Exception:
                    pass  # menu action already deleted — cosmetic only - Claude Generated

    def _on_config_changed(self):
        """Handle configuration changes from comprehensive settings dialog - Claude Generated"""
        self.logger.info("Configuration changed, refreshing components")
        # Apply font size if it changed in the settings dialog — Claude Generated
        try:
            new_fs = self.config_manager.load_config().ui_config.font_size
            if new_fs != get_font_size():
                self.apply_font_size(new_fs)
        except Exception:
            logging.getLogger(__name__).debug("font-size refresh failed", exc_info=True)
        self._refresh_components()

    def _refresh_components(self):
        """Refresh all components with new configuration - Claude Generated"""
        try:
            # 1. Reload LLM service configuration and reinitialize providers
            if hasattr(self, 'llm_service'):
                self.llm_service.reload_providers()

            # 1b. Reload the shared provider-detection service used by the
            # provider/model pickers. It wraps its own LlmService whose client map
            # stays stale otherwise, so a newly added provider's models wouldn't
            # appear until restart (the list refreshes, the models don't). Must run
            # before the per-tab refresh in step 6. - Claude Generated
            try:
                self.config_manager.get_provider_detection_service().reload()
            except Exception:
                self.logger.debug("detection service reload failed", exc_info=True)

            # 2. Refresh provider status to update reachability and available models
            if hasattr(self, 'llm_service'):
                self.llm_service.refresh_all_provider_status()

            # 3. Reload pipeline configuration to use updated provider preferences
            if hasattr(self, 'pipeline_manager'):
                self.pipeline_manager.reload_config()

            # 4. Update tabs with new provider information
            self.update_tabs_with_provider_info()

            # 5. Update global status bar with new provider and cache info
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.update_provider_info()
                self.global_status_bar.update_cache_status()

            # 6. Notify tabs about configuration changes (custom handlers)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if hasattr(tab, 'on_config_changed'):
                    tab.on_config_changed()

            self.logger.info("Configuration refreshed (providers, pipeline, tabs, status bar)")

            # Show user feedback
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.show_temporary_message(
                    "✅ Einstellungen erfolgreich übernommen",
                    5000
                )

        except Exception as e:
            self.logger.error(f"Error refreshing components: {e}", exc_info=True)
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.show_temporary_message(
                    f"⚠️ Fehler beim Übernehmen der Einstellungen: {str(e)}",
                    10000
                )

    # Claude Generated - DELETED: export_results() and export_current_analysis()
    # export_results: Inconsistent tab-specific export removed
    # export_current_analysis: Obsolete - now use analysis_review_tab.current_analysis directly

    def import_results(self):
        """Importiert gespeicherte Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, "import_results"):
            current_tab.import_results()
        else:
            self.global_status_bar.show_temporary_message("Import nicht verfügbar für diesen Tab", 3000)

    def show_about(self):
        """Zeigt den Über-Dialog"""
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("Über AlIma")
        layout = QVBoxLayout(about_dialog)

        # Über-Text
        about_text = QLabel(
            "AlIma - Sacherschließung mit LLMs\nVersion 1.0\n\n"
            "Entwickelt von Conrad Hübler\n"
            "TU Freiberg\n"
            "Lizenz: LGPL-3.0 license \n"
            "GitHub: https://github.com/conradhuebler/ALIMA"
        )
        layout.addWidget(about_text)

        # Schließen-Button
        close_button = QPushButton("Schließen")
        close_button.clicked.connect(about_dialog.close)
        layout.addWidget(close_button)

        about_dialog.exec()

    def show_help(self):
        """Zeigt den Hilfe-Dialog"""

    def show_prompt_editor(self):
        """Öffnet den Prompt-Editor-Dialog"""
        from .prompt_editor_dialog import PromptEditorDialog

        editor = PromptEditorDialog(self)
        editor.exec()

    def show_workflow_editor(self):
        """Öffnet den Workflow-YAML-Editor - Claude Generated"""
        from .workflow_editor_dialog import WorkflowEditorDialog

        editor = WorkflowEditorDialog(self)
        editor.exec()

    def show_database_viewer(self):
        """Open database viewer dialog - Claude Generated"""
        try:
            # Get current database configuration
            config_manager = ConfigManager()
            database_config = config_manager.get_database_config()

            # Create and execute modal dialog (automatic memory management)
            dialog = DatabaseViewerDialog(database_config, self)
            dialog.exec()  # Modal dialog with automatic cleanup

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen des Datenbank-Viewers:\n{str(e)}"
            )

    def clear_search_cache(self):
        """Clear search mappings cache (with confirmation) - Claude Generated"""
        try:
            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Such-Cache leeren?",
                "Alle gespeicherten Suchergebnisse werden gelöscht.\n\n"
                "GND-Einträge und Klassifikationen bleiben erhalten.\n\n"
                "Fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Get singleton instance and clear cache - Claude Generated (Enhanced error handling)
                try:
                    from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                    uk_manager = UnifiedKnowledgeManager.get_instance()
                    success, message = uk_manager.clear_search_cache()

                    if success:
                        QMessageBox.information(
                            self,
                            "Erfolg",
                            message
                        )
                        self.logger.info(f"Search cache cleared successfully: {message}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Warnung",
                            message
                        )
                        self.logger.warning(f"Cache clear warning: {message}")

                except Exception as cache_error:
                    # FIX: Better error handling for cache clear operations - Claude Generated
                    error_msg = f"Fehler beim Leeren des Caches:\n{str(cache_error)}\n\nVersuchen Sie später erneut."
                    QMessageBox.critical(self, "Fehler", error_msg)
                    self.logger.error(f"Cache clear failed with exception: {cache_error}", exc_info=True)

        except Exception as e:
            # FIX: Catch outer exceptions (dialog, etc.) - Claude Generated
            error_msg = f"Unerwarteter Fehler:\n{str(e)}"
            QMessageBox.critical(self, "Fehler", error_msg)
            self.logger.error(f"Unexpected error in clear_search_cache: {e}", exc_info=True)

    def cleanup_malformed_entries(self):
        """Clean up malformed classification entries (count>0 but no titles) - Claude Generated (Ultra-Deep Fix)"""
        try:
            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Malformed Einträge bereinigen?",
                "Entfernt DK-Einträge mit count>0 aber ohne Titel.\n\n"
                "Diese Einträge können Live-Suchen blockieren.\n\n"
                "Fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Get singleton instance and cleanup - Claude Generated
                try:
                    from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                    uk_manager = UnifiedKnowledgeManager.get_instance()
                    success, message = uk_manager.cleanup_malformed_classifications()

                    if success:
                        QMessageBox.information(
                            self,
                            "Erfolg",
                            message
                        )
                        self.logger.info(f"Malformed entries cleaned: {message}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Warnung",
                            message
                        )
                        self.logger.warning(f"Cleanup warning: {message}")

                except Exception as cleanup_error:
                    error_msg = f"Fehler beim Bereinigen:\n{str(cleanup_error)}\n\nVersuchen Sie später erneut."
                    QMessageBox.critical(self, "Fehler", error_msg)
                    self.logger.error(f"Cleanup failed with exception: {cleanup_error}", exc_info=True)

        except Exception as e:
            error_msg = f"Unerwarteter Fehler:\n{str(e)}"
            QMessageBox.critical(self, "Fehler", error_msg)
            self.logger.error(f"Unexpected error in cleanup_malformed_entries: {e}", exc_info=True)

    def show_batch_processing_dialog(self):
        """Open batch processing dialog (non-modal singleton) - Claude Generated"""
        try:
            # Reuse existing dialog or create new one - Claude Generated
            if self._batch_dialog is None:
                from .batch_processing_dialog import BatchProcessingDialog
                self._batch_dialog = BatchProcessingDialog(
                    alima_manager=self.alima_manager,
                    cache_manager=self.cache_manager,
                    config_manager=self.config_manager,
                    logger=self.logger,
                    pipeline_tab=self.pipeline_tab,
                    parent=self
                )

            # Show non-modal (user can continue working while batch runs)
            self._batch_dialog.show()
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen des Batch Processing Dialogs:\n{str(e)}"
            )
            self.logger.error(f"Failed to open batch processing dialog: {e}")

    def show_single_step_dialog(self):
        """Open the P-γ SingleStepDialog (modal). Claude Generated."""
        try:
            from .dialogs.single_step_dialog import SingleStepDialog

            dlg = SingleStepDialog(
                llm_service=self.llm_service,
                alima_manager=self.alima_manager,
                parent=self,
            )
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen des Single-Step-Dialogs:\n{e}",
            )
            self.logger.error(f"Failed to open SingleStepDialog: {e}")

    def load_batch_results(self):
        """Load and review batch processing results - Claude Generated"""
        try:
            # Ask user to select output directory
            directory = QFileDialog.getExistingDirectory(
                self,
                "Batch-Ergebnisse laden",
                str(get_autosave_dir(self.config_manager)),
                QFileDialog.Option.ShowDirsOnly
            )

            if not directory:
                return

            # Find all JSON files in directory
            from pathlib import Path
            json_files = list(Path(directory).glob("*.json"))

            # Filter out the .batch_state.json file
            json_files = [f for f in json_files if f.name != ".batch_state.json"]

            if not json_files:
                QMessageBox.warning(
                    self,
                    "Keine Ergebnisse",
                    f"Keine JSON-Dateien in {directory} gefunden."
                )
                return

            # Switch to analysis review tab
            for i in range(self.tabs.count()):
                if isinstance(self.tabs.widget(i), type(self.analysis_review_tab)):
                    self.tabs.setCurrentIndex(i)
                    break

            # Load batch directory into analysis review tab - Claude Generated
            self.analysis_review_tab.load_batch_directory(directory)

            self.update_status(f"Loaded {len(json_files)} batch result(s) from {directory}")

            QMessageBox.information(
                self,
                "Batch-Ergebnisse geladen",
                f"Geladen: {len(json_files)} Ergebnis-Dateien aus {directory}\n\n"
                f"Verwenden Sie die Batch-Tabelle um einzelne Ergebnisse anzuzeigen."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Laden der Batch-Ergebnisse:\n{str(e)}"
            )
            self.logger.error(f"Failed to load batch results: {e}")

