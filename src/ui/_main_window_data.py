"""Data-I/O + GND-import mixin for MainWindow. Claude Generated.

Extracted from ``main_window.py`` (F-5 god-file split): GND/lobid import +
background-import progress, file extraction/download, analysis-state load/save +
tab population, comparison-tab open, and app restart. Method bodies are moved
verbatim. ``MainWindowDataMixin`` is mixed into ``MainWindow``; not a standalone
window.
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
)

from ..core.gndparser import GNDParser
from ..utils.pipeline_defaults import get_autosave_dir


class MainWindowDataMixin:
    """Data I/O + GND import (verbatim from MainWindow)."""

    def import_gnd_database(self):
        """Importiert die GND-Datenbank aus einer lokalen Datei oder lädt sie von DNB herunter"""
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QFileDialog
        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtGui import QCursor
        import os

        # Ask user for source
        reply = QMessageBox.question(
            self,
            "GND-Datenbank Import",
            "Möchten Sie eine lokale Datei auswählen oder die aktuelle GND-Datenbank von DNB herunterladen und speichern?",
            QMessageBox.StandardButton.Open
            | QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        xml_file_path = None

        # Set wait cursor
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

        try:
            if reply == QMessageBox.StandardButton.Open:
                # Restore cursor for file dialog
                QApplication.restoreOverrideCursor()

                # Select local file
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    "GND-Datenbank auswählen",
                    str(Path.home() / "Downloads"),
                    "XML/GZ-Dateien (*.xml *.xml.gz *.gz)",
                )

                if filename:
                    # Set wait cursor again
                    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
                    xml_file_path = self._handle_file_extraction(filename)

            elif reply == QMessageBox.StandardButton.Save:
                # Download from DNB
                xml_file_path = self._download_and_extract_gnd()

            else:
                # Cancel
                QApplication.restoreOverrideCursor()
                return

            if xml_file_path and os.path.exists(xml_file_path):
                try:
                    # Create progress dialog for parsing
                    progress = QProgressDialog(
                        "Importiere GND-Datenbank...", "Abbrechen", 0, 0, self
                    )
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.show()

                    # Process events to show progress dialog
                    QApplication.processEvents()

                    parser = GNDParser(self.cache_manager)
                    self.logger.info(f"Importiere GND-Datenbank: {xml_file_path}")
                    
                    # Console Progress Output - Claude Generated
                    self.logger.info("🔄 Starte GND-Datenbank Import...")
                    self.logger.info(f"📁 Datei: {xml_file_path}")

                    # Connect parser progress signals if available
                    if hasattr(parser, "progress_updated"):
                        parser.progress_updated.connect(progress.setValue)
                        parser.status_updated.connect(progress.setLabelText)

                        # Also connect to console output - Claude Generated
                        def console_progress(value):
                            if value > 0:
                                self.logger.info(f"📊 Fortschritt: {value}%")

                        def console_status(status):
                            self.logger.info(f"ℹ️ Status: {status}")

                        parser.progress_updated.connect(console_progress)
                        parser.status_updated.connect(console_status)

                    self.logger.info("⚙️ Verarbeite XML-Daten...")
                    parser.process_file(xml_file_path)
                    self.logger.info("✅ GND-Import erfolgreich abgeschlossen!")

                    progress.close()
                    QMessageBox.information(
                        self, "Erfolg", "GND-Datenbank erfolgreich importiert!"
                    )

                except Exception as e:
                    progress.close()
                    QMessageBox.critical(
                        self, "Fehler", f"Fehler beim Importieren: {str(e)}"
                    )
            else:
                QMessageBox.warning(
                    self, "Fehler", "Keine gültige Datei zum Importieren gefunden."
                )

        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()

    def load_analysis_state_from_file(self):
        """
        Öffnet einen Datei-Dialog, um einen JSON-Analyse-Zustand zu laden
        und die UI damit zu befüllen - Claude Generated
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from ..utils.pipeline_utils import PipelineJsonManager
        from ..core.data_models import KeywordAnalysisState

        # Datei-Dialog öffnen
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Analyse-Zustand laden",
            str(get_autosave_dir(self.config_manager)),
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_name:
            return  # Benutzer hat abgebrochen

        try:
            # 1. JSON-Datei laden und parsen
            self.logger.info(f"Loading analysis state from: {file_name}")
            state = PipelineJsonManager.load_analysis_state(file_name)

            # 2. Daten an die Tabs verteilen
            self.populate_all_tabs_from_state(state)

            # 3. Erfolgsmeldung
            self.global_status_bar.show_temporary_message("✅ Analyse-Zustand erfolgreich geladen.", 5000)

            # 4. Zur Pipeline-Ansicht wechseln für Übersicht
            self.tabs.setCurrentWidget(self.pipeline_tab)

            self.logger.info("Analysis state successfully loaded and distributed to tabs")

        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Analyse-Zustands: {e}")
            QMessageBox.critical(
                self,
                "Ladefehler",
                f"Die Datei konnte nicht geladen werden:\n\n{str(e)}"
            )

    def populate_all_tabs_from_state(self, state):
        """
        Verteilt die Daten aus einem KeywordAnalysisState-Objekt
        an alle relevanten UI-Tabs - Claude Generated

        Uses on_pipeline_results_ready() for core distribution to avoid duplication.
        """
        from ..core.data_models import KeywordAnalysisState

        self.logger.info("Distributing analysis state data to all tabs...")

        try:
            # Use the centralized distribution logic for base tabs - Claude Generated
            self.on_pipeline_results_ready(state)

            # Add loaded analysis to tab-local history - Claude Generated
            self.abstract_tab.add_external_analysis_to_history(state)
            self.analyse_keywords.add_external_analysis_to_history(state)
            self.logger.info("Added loaded analysis to tab histories")

            # Additional loading-specific UI enhancements - Claude Generated

            # 1. 🚀 Pipeline Tab - Show loaded state indicators
            if hasattr(self.pipeline_tab, 'unified_input') and state.original_abstract:
                self.pipeline_tab.unified_input.set_text_directly(
                    state.original_abstract,
                    "Geladen aus JSON"
                )
                if hasattr(self.pipeline_tab, 'show_loaded_state_indicator'):
                    self.pipeline_tab.show_loaded_state_indicator(state)
                self.logger.info("✅ Pipeline tab: loaded state indicators shown")

            # 5. 📊 Analyse-Review Tab - already populated by on_pipeline_results_ready() above

            # 6. 📚 UB-Katalog Tab - Keywords for library catalog search
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                # Type-safe join - Claude Generated (Fix for string parsing bug)
                final_kw = state.final_llm_analysis.extracted_gnd_keywords
                final_keywords = ", ".join(final_kw) if isinstance(final_kw, list) else str(final_kw)
                self.ub_catalog_tab.update_keywords(final_keywords)
                self.logger.info("✅ UB-Katalog tab populated with final keywords")

            # 7. 🖼️ Bilderkennung Tab - Show OCR details if input was image
            # Note: Currently we don't have image source info in KeywordAnalysisState
            # This could be enhanced in future versions

            # 8. Show summary in status bar
            total_keywords = len(state.final_llm_analysis.extracted_gnd_keywords) if state.final_llm_analysis else len(state.initial_keywords)
            summary_message = f"📁 Geladen: {total_keywords} Schlagwörter aus {len(state.search_results)} Suchvorgängen"
            self.global_status_bar.show_temporary_message(summary_message, 10000)

            self.logger.info(f"🎯 Analysis state distribution complete: {summary_message}")

        except Exception as e:
            self.logger.error(f"Error distributing analysis state: {e}")
            raise  # Re-raise to be handled by calling method

    # Claude Generated - DELETED: collect_current_gui_state()
    # This UI-scraping method is now obsolete and removed.
    # Reason: Single Source of Truth architecture - we now use analysis_review_tab.current_analysis directly.
    # All export operations now access the canonical data object instead of scraping UI elements.

    def export_current_gui_state(self):
        """
        Exportiert den aktuellen Analyse-Zustand aus dem Analyse-Review-Tab.
        Claude Generated - Refactored to use Single Source of Truth (analysis_review_tab.current_analysis)
        """
        from ..utils.pipeline_utils import AnalysisPersistence

        # 1. Prüfe, ob der Analyse-Review-Tab ein gültiges Ergebnis hat
        if self.analysis_review_tab and self.analysis_review_tab.current_analysis:
            # 2. Das "echte" Datenobjekt direkt holen (Single Source of Truth)
            state_to_save = self.analysis_review_tab.current_analysis
            self.logger.info("Exporting current analysis state from AnalysisReviewTab.")

            # 3. Use working_title for filename if available - Claude Generated
            if hasattr(state_to_save, 'working_title') and state_to_save.working_title:
                default_filename = f"{state_to_save.working_title}.json"
                self.logger.info(f"Using working_title for export: {default_filename}")
            else:
                default_filename = f"analysis_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.logger.info(f"No working_title, using timestamp: {default_filename}")

            # 4. Den bewährten Speicher-Dialog mit dem korrekten Objekt aufrufen
            file_path = AnalysisPersistence.save_with_dialog(
                state=state_to_save,
                parent_widget=self,
                default_filename=default_filename
            )

            if file_path:
                self.global_status_bar.show_temporary_message("✅ Analyse-Zustand erfolgreich exportiert.", 5000)
                self.logger.info(f"Analysis state successfully exported to: {file_path}")
        else:
            # 4. Fehlerbehandlung, wenn keine Daten zum Speichern da sind
            self.logger.warning("Export triggered, but no analysis state available in Review-Tab.")
            QMessageBox.information(
                self,
                "Keine Daten zum Speichern",
                "Es ist keine abgeschlossene Analyse vorhanden, die gespeichert werden könnte.\n\n"
                "Bitte führen Sie zuerst eine Analyse durch oder laden Sie ein Ergebnis in den 'Analyse-Review'-Tab."
            )

    def _open_comparison_tab(self):
        """Show and switch to the comparison tab - Claude Generated"""
        self.tabs.setTabVisible(self._comparison_tab_idx, True)
        self.tabs.setCurrentIndex(self._comparison_tab_idx)

    def _show_comparison_tab(self):
        """Signal handler: make comparison tab visible and switch to it - Claude Generated"""
        self.tabs.setTabVisible(self._comparison_tab_idx, True)
        self.tabs.setCurrentIndex(self._comparison_tab_idx)

    def _handle_file_extraction(self, filename: str) -> str:
        """Behandelt die Extraktion von gz-Dateien"""
        import gzip
        import tempfile
        from pathlib import Path
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt

        if filename.endswith(".gz"):
            # Extract gz file
            try:
                # Create progress dialog for extraction
                progress = QProgressDialog("Entpacke Datei...", "Abbrechen", 0, 0, self)
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QApplication.processEvents()

                # Create temporary file for extracted content
                temp_dir = tempfile.mkdtemp()
                extracted_filename = Path(filename).stem  # Remove .gz extension
                temp_xml_path = os.path.join(temp_dir, extracted_filename)

                self.logger.info(f"Extrahiere {filename} nach {temp_xml_path}")

                with gzip.open(filename, "rb") as gz_file:
                    with open(temp_xml_path, "wb") as xml_file:
                        # Read in chunks to allow for progress updates
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            if progress.wasCanceled():
                                progress.close()
                                return None

                            chunk = gz_file.read(chunk_size)
                            if not chunk:
                                break
                            xml_file.write(chunk)
                            QApplication.processEvents()

                progress.close()
                return temp_xml_path

            except Exception as e:
                if "progress" in locals():
                    progress.close()
                self.logger.error(f"Fehler beim Extrahieren der gz-Datei: {str(e)}")
                QMessageBox.critical(
                    self, "Fehler", f"Fehler beim Extrahieren: {str(e)}"
                )
                return None
        else:
            # File is already XML
            return filename

    def _download_and_extract_gnd(self) -> str:
        """Lädt die GND-Datenbank von DNB herunter und extrahiert sie"""
        import requests
        import gzip
        import tempfile
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt

        url = "https://data.dnb.de/GND/authorities-gnd-sachbegriff_dnbmarc.mrc.xml.gz"

        try:
            # Create progress dialog
            progress = QProgressDialog(
                "Lade GND-Datenbank herunter...", "Abbrechen", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            # Download file
            self.logger.info(f"Lade GND-Datenbank herunter von: {url}")
            
            # Console Progress Output - Claude Generated
            self.logger.info("🌐 Starte DNB-Download...")
            self.logger.info(f"📡 URL: {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                self.logger.info(f"📦 Dateigröße: {total_size / (1024*1024):.1f} MB")

            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            last_console_percent = 0

            self.logger.info("⬇️ Download läuft...")
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if progress.wasCanceled():
                        self.logger.info("❌ Download abgebrochen")
                        progress.close()
                        return None

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        download_percent = int((downloaded / total_size) * 50)
                        progress.setValue(download_percent)  # 50% for download

                        # Console progress every 10% - Claude Generated
                        console_percent = (downloaded / total_size) * 100
                        if console_percent - last_console_percent >= 10:
                            self.logger.info(f"📊 Download: {console_percent:.0f}%")
                            last_console_percent = console_percent

                    QApplication.processEvents()

            progress.setLabelText("Entpacke Datenbank...")
            progress.setValue(50)
            QApplication.processEvents()

            # Extract gz file
            self.logger.info("📦 Entpacke GZ-Datei...")
            self.logger.info(f"Entpacke {temp_gz_path} nach {temp_xml_path}")
            with gzip.open(temp_gz_path, "rb") as gz_file:
                with open(temp_xml_path, "wb") as xml_file:
                    xml_file.write(gz_file.read())

            self.logger.info("✅ Download und Entpackung abgeschlossen")
            progress.setValue(100)
            progress.close()

            # Clean up gz file
            os.remove(temp_gz_path)

            self.logger.info(
                f"GND-Datenbank erfolgreich heruntergeladen und entpackt: {temp_xml_path}"
            )
            return temp_xml_path

        except requests.RequestException as e:
            if "progress" in locals():
                progress.close()
            self.logger.error(f"Fehler beim Herunterladen: {str(e)}")
            QMessageBox.critical(
                self, "Download-Fehler", f"Fehler beim Herunterladen: {str(e)}"
            )
            return None
        except Exception as e:
            if "progress" in locals():
                progress.close()
            self.logger.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")
            QMessageBox.critical(self, "Fehler", f"Fehler beim Verarbeiten: {str(e)}")
            return None

    def import_lobid_dnb_data(self):
        """Importiert DNB/GND-Daten über LobidSuggester mit Progress - Claude Generated"""
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QVBoxLayout, QDialog, QLabel, QTextEdit, QPushButton
        from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
        from PyQt6.QtGui import QCursor, QFont
        from ..core.search.providers.lobid.suggester import LobidSuggester
        from pathlib import Path
        import time
        
        class LobidImportWorker(QThread):
            """Worker thread for Lobid DNB import - Claude Generated"""
            progress_updated = pyqtSignal(str)  # Progress message
            finished_successfully = pyqtSignal(int)  # Number of entries imported
            error_occurred = pyqtSignal(str)  # Error message
            
            def __init__(self, force_download=False, debug=False):
                super().__init__()
                self.force_download = force_download
                self.debug = debug
                
            def run(self):
                try:
                    self.progress_updated.emit("🔄 Initialisiere Lobid-Suggester...")

                    # Factory-built provider → the same suggester construction
                    # (and data_dir) the search path uses. The former direct
                    # LobidSuggester(data_dir="data/lobid") prepared a directory
                    # the factory-built search suggester never read. - Claude Generated
                    from ..core.search.factory import build_provider
                    from ..core.search.service import resolve_gnd_instances, underlying_suggester

                    instances = resolve_gnd_instances(["lobid"])
                    if not instances:
                        self.error_occurred.emit(
                            "Lobid-Instanz ist deaktiviert (Plugin-Einstellungen)"
                        )
                        return
                    start_time = time.time()
                    # Suggester construction may already download missing GND
                    # data (prepare(False) at init) — include it in the timing.
                    provider = build_provider(instances[0])
                    lobid_suggester = underlying_suggester(provider)
                    data_dir = lobid_suggester.data_dir

                    self.progress_updated.emit(f"📁 Datenverzeichnis: {data_dir}")
                    self.progress_updated.emit(f"🔄 Erzwungener Download: {self.force_download}")

                    if self.force_download or not (data_dir / "subjects.json").exists():
                        self.progress_updated.emit("⬇️ Lade GND-Sachbegriffe von DNB herunter...")
                        
                    # Use the prepare method which handles download and processing
                    lobid_suggester.prepare(force_gnd_download=self.force_download)
                    
                    elapsed = time.time() - start_time
                    self.progress_updated.emit(f"✅ DNB-Import abgeschlossen in {elapsed:.2f} Sekunden")
                    
                    # Show some statistics
                    if lobid_suggester.gnd_subjects:
                        subject_count = len(lobid_suggester.gnd_subjects)
                        self.progress_updated.emit(f"📊 Importierte {subject_count:,} GND-Sachbegriff-Einträge")
                        
                        # Show sample entries
                        sample_entries = list(lobid_suggester.gnd_subjects.items())[:3]
                        self.progress_updated.emit("📋 Beispiel-Einträge:")
                        for gnd_id, title in sample_entries:
                            self.progress_updated.emit(f"   {gnd_id}: {title}")
                        
                        self.finished_successfully.emit(subject_count)
                    else:
                        self.error_occurred.emit("Keine GND-Sachbegriffe gefunden")
                        
                except Exception as e:
                    self.error_occurred.emit(f"Import-Fehler: {str(e)}")
        
        # Ask user for import options
        reply = QMessageBox.question(
            self,
            "Lobid DNB Import",
            "Möchten Sie die GND-Sachbegriffe von DNB herunterladen?\n\n"
            "Dies kann einige Minuten dauern, da die Daten heruntergeladen und verarbeitet werden müssen.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Check if data already exists (same default dir a factory-built
        # suggester resolves — no construction needed for the path). - Claude Generated
        subjects_file = LobidSuggester.default_data_dir() / "subjects.json"
        force_download = False
        
        if subjects_file.exists():
            force_reply = QMessageBox.question(
                self,
                "Daten bereits vorhanden",
                f"GND-Daten wurden bereits gefunden in:\n{subjects_file}\n\n"
                "Möchten Sie die Daten trotzdem neu herunterladen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            force_download = (force_reply == QMessageBox.StandardButton.Yes)
        
        # Create progress dialog
        class ImportProgressDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Lobid DNB Import")
                self.setModal(True)
                self.resize(600, 400)
                
                layout = QVBoxLayout()
                
                self.status_label = QLabel("Starte Import...")
                font = QFont()
                font.setBold(True)
                self.status_label.setFont(font)
                layout.addWidget(self.status_label)
                
                self.progress_text = QTextEdit()
                self.progress_text.setReadOnly(True)
                from .styles import get_scaled_font
                self.progress_text.setFont(get_scaled_font(size_delta=-1, monospace=True))
                layout.addWidget(self.progress_text)
                
                self.cancel_button = QPushButton("Abbrechen")
                self.cancel_button.clicked.connect(self.reject)
                layout.addWidget(self.cancel_button)
                
                self.setLayout(layout)
                
            def add_progress_message(self, message: str):
                self.progress_text.append(message)
                # Auto-scroll to bottom
                scrollbar = self.progress_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
            def set_status(self, status: str):
                self.status_label.setText(status)
        
        # Create and show progress dialog
        progress_dialog = ImportProgressDialog(self)
        progress_dialog.show()
        
        # Create and start worker thread
        self.import_worker = LobidImportWorker(force_download=force_download, debug=True)
        
        # Connect worker signals
        self.import_worker.progress_updated.connect(progress_dialog.add_progress_message)
        self.import_worker.progress_updated.connect(progress_dialog.set_status)
        
        def on_import_finished(entry_count):
            progress_dialog.cancel_button.setText("Schließen")
            progress_dialog.add_progress_message(f"\n🎉 Import erfolgreich abgeschlossen!")
            progress_dialog.set_status(f"✅ {entry_count:,} Einträge importiert")
            
            # Update cache statistics if available
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.update_status()
                
        def on_import_error(error_message):
            progress_dialog.cancel_button.setText("Schließen")
            progress_dialog.add_progress_message(f"\n❌ Fehler: {error_message}")
            progress_dialog.set_status("❌ Import fehlgeschlagen")
            
            QMessageBox.critical(self, "Import-Fehler", f"Fehler beim Import:\n{error_message}")
        
        self.import_worker.finished_successfully.connect(on_import_finished)
        self.import_worker.error_occurred.connect(on_import_error)
        
        # Handle cancel button
        def on_cancel():
            if self.import_worker.isRunning():
                progress_dialog.set_status("🛑 Import wird abgebrochen...")
                progress_dialog.add_progress_message("🛑 Benutzer hat Import abgebrochen")
                self.import_worker.terminate()
                self.import_worker.wait(3000)  # Wait max 3 seconds
            progress_dialog.accept()
            
        progress_dialog.rejected.connect(on_cancel)
        
        # Start the import
        self.import_worker.start()
        
        # Show dialog and wait for completion
        progress_dialog.exec()

    def restart_application(self):
        """Startet ALIMA neu - Claude Generated"""
        try:
            self.logger.info("Restarting application...")
            self.save_settings()

            # Restart Python process
            python = sys.executable
            os.execl(python, python, *sys.argv)

        except Exception as e:
            self.logger.error(f"Application restart failed: {e}")
            QMessageBox.warning(
                self,
                "Neustart fehlgeschlagen",
                "Die Anwendung konnte nicht automatisch neu gestartet werden.\n"
                "Bitte starten Sie ALIMA manuell neu."
            )

    def check_pending_gnd_import(self):
        """Check if GND was downloaded in wizard and offer background import - Claude Generated"""
        import json
        from pathlib import Path

        marker_file = Path(tempfile.gettempdir()) / "alima_gnd_pending.json"

        if not marker_file.exists():
            return

        try:
            marker_data = json.loads(marker_file.read_text())
            xml_path = marker_data.get('xml_path')

            if not xml_path or not Path(xml_path).exists():
                marker_file.unlink()
                return

            # Ask user if they want to start background import
            reply = QMessageBox.question(
                self,
                "GND-Import",
                "Eine GND-Datenbank wurde im Setup heruntergeladen.\n\n"
                "Möchten Sie den Import jetzt im Hintergrund starten?\n"
                "(Sie können währenddessen weiterarbeiten)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._start_background_gnd_import(xml_path)

            # Remove marker regardless of choice
            marker_file.unlink()

        except Exception as e:
            self.logger.error(f"Error checking pending GND import: {str(e)}")
            if marker_file.exists():
                marker_file.unlink()

    def _start_background_gnd_import(self, xml_file_path: str):
        """Start non-blocking GND import in background (StatusBar-based) - Claude Generated"""
        try:
            from ..utils.gnd_import_worker import GNDImportWorker

            self.logger.info(f"Starting background GND import: {xml_file_path}")

            # Create status bar widgets - Claude Generated (non-blocking UI)
            self.gnd_import_status_label = QLabel("🔄 GND-Import: Starte...")

            self.gnd_import_progress_bar = QProgressBar()
            self.gnd_import_progress_bar.setMaximum(100)
            self.gnd_import_progress_bar.setMinimumWidth(200)
            self.gnd_import_progress_bar.setMaximumHeight(20)

            self.gnd_import_cancel_btn = QPushButton("Abbrechen")
            self.gnd_import_cancel_btn.setMaximumWidth(80)
            self.gnd_import_cancel_btn.setMaximumHeight(20)

            # Add to status bar
            statusbar = self.statusBar()
            statusbar.addWidget(self.gnd_import_status_label)
            statusbar.addWidget(self.gnd_import_progress_bar)
            statusbar.addWidget(self.gnd_import_cancel_btn)

            # Create worker
            self.gnd_import_worker = GNDImportWorker(xml_file_path, self.cache_manager)

            # Connect signals
            self.gnd_import_worker.progress_updated.connect(self._update_gnd_progress)
            self.gnd_import_worker.status_updated.connect(self._update_gnd_status)
            self.gnd_import_worker.finished_successfully.connect(self._on_gnd_import_complete)
            self.gnd_import_worker.error_occurred.connect(self._on_gnd_import_error)

            # Connect cancel button
            self.gnd_import_cancel_btn.clicked.connect(self.gnd_import_worker.cancel)

            # Show notification
            QMessageBox.information(
                self,
                "GND-Import gestartet",
                "✅ GND-Import läuft im Hintergrund.\n\n"
                "Sie können ALIMA normal nutzen.\n"
                "Fortschritt wird in der Statusleiste angezeigt.",
                QMessageBox.StandardButton.Ok
            )

            # Start worker
            self.gnd_import_worker.start()

        except Exception as e:
            self.logger.error(f"Error starting GND import: {str(e)}", exc_info=True)
            QMessageBox.warning(
                self,
                "GND-Import Fehler",
                f"Fehler beim Starten des GND-Imports:\n{str(e)}"
            )

    def _update_gnd_progress(self, current: int, total: int):
        """Update GND import progress bar - Claude Generated"""
        if total > 0 and hasattr(self, 'gnd_import_progress_bar'):
            percent = int((current / total) * 100)
            self.gnd_import_progress_bar.setValue(percent)
            if hasattr(self, 'gnd_import_status_label'):
                self.gnd_import_status_label.setText(
                    f"🔄 GND-Import: {current:,} / {total:,} ({percent}%)"
                )

    def _update_gnd_status(self, status_msg: str):
        """Update GND import status message - Claude Generated"""
        if hasattr(self, 'gnd_import_status_label'):
            self.gnd_import_status_label.setText(f"🔄 {status_msg}")

    def _on_gnd_import_complete(self, count: int):
        """Handle successful GND import completion - Claude Generated"""
        self.logger.info(f"GND import completed successfully: {count:,} entries")

        # Remove status bar widgets - Claude Generated
        statusbar = self.statusBar()
        if hasattr(self, 'gnd_import_status_label'):
            statusbar.removeWidget(self.gnd_import_status_label)
            statusbar.removeWidget(self.gnd_import_progress_bar)
            statusbar.removeWidget(self.gnd_import_cancel_btn)

            # Cleanup references
            self.gnd_import_status_label.deleteLater()
            self.gnd_import_progress_bar.deleteLater()
            self.gnd_import_cancel_btn.deleteLater()

        # Show success message in status bar
        statusbar.showMessage(
            f"✅ GND-Import abgeschlossen: {count:,} Einträge importiert",
            10000  # Show for 10 seconds
        )

    def _on_gnd_import_error(self, error_msg: str):
        """Handle GND import error - Claude Generated"""
        self.logger.error(f"GND import failed: {error_msg}")

        # Remove status bar widgets - Claude Generated
        statusbar = self.statusBar()
        if hasattr(self, 'gnd_import_status_label'):
            statusbar.removeWidget(self.gnd_import_status_label)
            statusbar.removeWidget(self.gnd_import_progress_bar)
            statusbar.removeWidget(self.gnd_import_cancel_btn)

            # Cleanup references
            self.gnd_import_status_label.deleteLater()
            self.gnd_import_progress_bar.deleteLater()
            self.gnd_import_cancel_btn.deleteLater()

        # Show error in status bar
        statusbar.showMessage(f"❌ GND-Import fehlgeschlagen: {error_msg}", 15000)

    # In der MainWindow Klasse - füge folgende Methoden hinzu

