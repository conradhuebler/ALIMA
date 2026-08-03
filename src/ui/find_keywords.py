"""SearchTab — GND-Schlagwortsuche + Pipeline-Nachbearbeitung.

Zwei Rollen in einem Tab:

* **Standalone-Suche**: Begriffe gegen die aktivierten GND-Quellen-Plugins
  suchen (derselbe Provider-Service wie die Pipeline, inkl. Mapping-first- und
  WP2-Raw-Cache). Läuft seit dem Aug-2026-Umbau in einem Worker-Thread.
* **Pipeline-Nachbearbeitung**: `update_data(analysis_state)` zeigt die
  Pipeline-Suchergebnisse mit Auswahl-/Cache-Transparenz; Doppelklick toggelt
  die Auswahl, `selection_changed` trägt Änderungen zurück.

Umbau Aug 2026 (WP-K5): asynchrone Suche (``GndSearchWorker``), Quellen-
Checkboxen live aus dem Plugin-System (``refresh_sources`` — vorher nur beim
Start gebaut), Häufigkeit zeigt ``display_count`` statt des Pool-``count``
(Count-Landmine: der ist bei Cache-Hits immer 1), Klassifikations-Spalte aus
dem kanonischen Pool-Vokabular, Pipeline-Mapping-Ansicht tatsächlich im Layout
(war gebaut, aber nie angehängt), tote Signale/Methoden/Legacy-Zweige entfernt.

Externer Vertrag (MainWindow): Konstruktor-Signatur, ``update_data``,
``update_search_field``, ``display_search_results``, ``refresh_styles``,
``refresh_sources``, Signal ``selection_changed``;
``_gnd_source_ids`` ist unbound testbar (tests/test_search_plugins.py).
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QTextBrowser,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QHeaderView,
    QSplitter,
    QProgressBar,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from typing import Any, Dict, List, Optional
import logging
import re
from pathlib import Path

from .workers import DNBSyncWorker, GndSearchWorker
from .styles import (
    get_main_stylesheet,
    get_button_styles,
    get_status_label_styles,
    get_scaled_font,
    LAYOUT,
    COLORS,
)


# ── Pure helpers (Qt-frei, testbar) ──────────────────────────────────────────

def extract_search_terms(text: str) -> List[str]:
    """Suchbegriffe aus Freitext: Anführungszeichen = exakte Phrase, sonst
    Komma-getrennt. - Claude Generated"""
    quoted_pattern = r'"([^"]+)"'
    quoted = re.findall(quoted_pattern, text)
    remaining = re.sub(quoted_pattern, "", text)
    return quoted + [t.strip() for t in remaining.split(",") if t.strip()]


def determine_relation(keyword: str, search_term: str) -> int:
    """0 = exakt, 1 = ähnlich (Teilstring), 2 = verschieden. - Claude Generated"""
    kw, st = keyword.lower(), search_term.lower()
    if kw == st:
        return 0
    if st in kw or kw in st:
        return 1
    return 2


def preferred_display_count(entry: Dict[str, Any]) -> int:
    """Die echte Häufigkeit für die Anzeige: ``display_count`` wenn vorhanden,
    sonst ``count``. Der Pool-``count`` ist ein Ranking-Platzhalter und bei
    Cache-Hits immer 1 (Count-Landmine, F-4) — ihn anzuzeigen war der
    "Häufigkeit zeigt 1"-Bug dieses Tabs. - Claude Generated"""
    try:
        return int(entry.get("display_count") or entry.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def format_classifications_compact(
    classifications: Any, max_per_system: int = 2
) -> str:
    """Kanonisches ``{SYSTEM: [entries]}`` als kompakte Zellen-Zeile,
    z.B. ``DDC 551.48 · RVK WI 5000, WI 4800 (+1)``. - Claude Generated"""
    from ..utils.classification_systems import KNOWN_SYSTEMS, codes_for_system

    parts = []
    for system in KNOWN_SYSTEMS:
        codes = codes_for_system(classifications, system)
        if not codes:
            continue
        shown = ", ".join(codes[:max_per_system])
        more = f" (+{len(codes) - max_per_system})" if len(codes) > max_per_system else ""
        parts.append(f"{system} {shown}{more}")
    return " · ".join(parts)


_RELATION_SYMBOLS = ["=", "≈", "≠"]
_RELATION_COLORS = [QColor("#4caf50"), QColor("#ff9800"), QColor("#9e9e9e")]

_CACHE_STATUS_ICONS = {"cache": "💾", "new": "🌐", "outdated": "⚠️"}
_CACHE_STATUS_TOOLTIPS = {
    "cache": "Aus Datenbank-Cache",
    "new": "Neu von der Quelle geholt",
    "outdated": "Cache älter als 90 Tage",
}

# Tabellenspalten (eine Wahrheit für alle Füller + den Doppelklick-Toggle)
_COLS = ["Begriff", "GND-ID", "Häufigkeit", "Ähnlichkeit", "Klassifikation", "Status"]
_COL_TERM, _COL_GND, _COL_COUNT, _COL_REL, _COL_CLS, _COL_STATUS = range(len(_COLS))


class SearchTab(QWidget):
    """GND-Suche (standalone) + Pipeline-Ergebnis-Nachbearbeitung."""

    selection_changed = pyqtSignal(dict)  # {'modified': {...}, 'manual': [...]}

    def __init__(
        self,
        cache_manager,
        parent=None,
        config_file: Path = Path.home() / ".alima_config.json",
        alima_manager=None,
        pipeline_manager=None,
        ub_catalog_tab=None,
    ):
        super().__init__(parent)
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        # Vereinheitlichung (Aug 4): der UB-Katalog ist ein Ergebnis-Reiter
        # dieses Tabs mit GETEILTER Sucheingabe — der frühere
        # SearchTabUnified-Combo-Umschalter ist damit weg.
        self.ub_catalog_tab = ub_catalog_tab

        # Pipeline-Nachbearbeitungszustand
        self.original_pipeline_state = None
        self.modified_selections: Dict[str, str] = {}
        self.manual_additions: List[Dict[str, Any]] = []
        self.cache_status: Dict[str, str] = {}
        # Pool-Einträge der letzten Suche je GND-ID (für die Detail-Ansicht)
        self._entries_by_gnd: Dict[str, Dict[str, Any]] = {}

        self._search_worker: Optional[GndSearchWorker] = None
        self._manual_worker: Optional[GndSearchWorker] = None
        self.sync_worker: Optional[DNBSyncWorker] = None

        self.init_ui()

    # ── UI-Aufbau ────────────────────────────────────────────────────────────

    def init_ui(self):
        self.setStyleSheet(get_main_stylesheet())
        btn_styles = get_button_styles()

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"]
        )

        # Kontrolleiste
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)
        self.status_label = QLabel("Aktueller Status: Bereit")
        self.status_label.setStyleSheet(get_status_label_styles()["info"])
        control_bar.addWidget(self.status_label)
        control_bar.addStretch(1)
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.progressBar.setRange(0, 0)  # busy indicator — Dauer ist netzabhängig
        self.progressBar.setFixedWidth(200)
        control_bar.addWidget(self.progressBar)
        layout.addLayout(control_bar)

        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Suchbereich
        search_widget = QWidget()
        search_box_layout = QVBoxLayout(search_widget)
        search_box_layout.setContentsMargins(0, 0, 0, 0)
        search_group = QGroupBox("Schlagwortsuche")
        search_layout = QVBoxLayout(search_group)
        search_layout.setSpacing(LAYOUT["inner_spacing"])
        search_layout.setContentsMargins(10, 20, 10, 10)

        search_header = QLabel("Suchbegriffe:")
        search_header.setFont(get_scaled_font(bold=True))
        search_layout.addWidget(search_header)

        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText(
            "Suchbegriffe (durch Komma getrennt oder in Anführungszeichen für exakte Phrasen)"
        )
        self.search_input.setMaximumHeight(100)
        self.search_input.setFont(get_scaled_font(size_delta=+1))
        search_layout.addWidget(self.search_input)

        # Quellen-Zeile — wird von refresh_sources() live neu gebaut
        options_frame = QFrame()
        options_frame.setStyleSheet(
            f"background-color: {COLORS['background_dark']}; border-radius: 6px; padding: 4px;"
        )
        self.sources_layout = QHBoxLayout(options_frame)
        sources_label = QLabel("GND-Quellen (Plugins):")
        sources_label.setFont(get_scaled_font(size_delta=-1, bold=True))
        self.sources_layout.addWidget(sources_label)
        self.sources_layout.addStretch(1)
        self.source_checkboxes: Dict[str, QCheckBox] = {}
        self.refresh_sources()
        # UB-Katalog ist keine GND-Quelle, sondern die DK/RVK-Suche im Bestand —
        # eigene Checkbox NACH den Plugins, von refresh_sources unberührt.
        self.ub_catalog_checkbox = None
        if self.ub_catalog_tab is not None:
            self.ub_catalog_checkbox = QCheckBox("UB-Katalog (DK/RVK)")
            self.ub_catalog_checkbox.setToolTip(
                "Zusätzlich die DK/RVK-Suche im UB-Katalog ausführen "
                "(Ergebnis im Reiter „UB-Katalog“; deutlich langsamer)"
            )
            self.sources_layout.insertWidget(
                self.sources_layout.count() - 1, self.ub_catalog_checkbox
            )
        search_layout.addWidget(options_frame)

        # Ein Button, zwei Zustände: startet die Suche bzw. bricht die
        # laufende ab (Operator-Wunsch Aug 4). - Claude Generated
        self._btn_styles = btn_styles
        self.search_button = QPushButton("Suche starten")
        self.search_button.setStyleSheet(btn_styles["primary"])
        self.search_button.clicked.connect(self._on_search_button)
        self.search_button.setShortcut("Ctrl+Return")
        search_layout.addWidget(self.search_button)

        search_box_layout.addWidget(search_group)
        main_splitter.addWidget(search_widget)

        # Ergebnisbereich: Reiter statt des früheren Combo-Umschalters —
        # GND-Schlagwörter, UB-Katalog (eingebettet, geteilte Eingabe) und das
        # Pipeline-Mapping (versteckt bis Pipeline-Daten da sind). - Claude Generated
        self.results_tabs = QTabWidget()

        results_container = QWidget()
        results_container_layout = QVBoxLayout(results_container)
        results_container_layout.setContentsMargins(0, 0, 0, 0)
        results_group = QGroupBox("Suchergebnisse")
        results_box_layout = QVBoxLayout(results_group)
        results_box_layout.setSpacing(LAYOUT["inner_spacing"])
        results_box_layout.setContentsMargins(10, 20, 10, 10)

        upper_splitter = QSplitter(Qt.Orientation.Horizontal)

        table_frame = QWidget()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_header = QLabel("Gefundene Schlagwörter:")
        table_header.setFont(get_scaled_font(bold=True))
        table_layout.addWidget(table_header)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(len(_COLS))
        self.results_table.setHorizontalHeaderLabels(_COLS)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.results_table)
        upper_splitter.addWidget(table_frame)

        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_header = QLabel("GND-Details:")
        details_header.setFont(get_scaled_font(bold=True))
        details_layout.addWidget(details_header)
        self.details_display = QTextBrowser()
        self.details_display.setOpenExternalLinks(True)
        details_layout.addWidget(self.details_display)

        self.update_button = QPushButton("🔄 DNB-Sync Selected")
        self.update_button.setStyleSheet(btn_styles["secondary"])
        self.update_button.setToolTip(
            "Aktualisiert alle ausgewählten (✅) GND-Einträge mit aktuellen Daten von der DNB. "
            "Holt DDC-Klassifikationen und GND-Systematiken."
        )
        self.update_button.clicked.connect(self.update_selected_entry)
        details_layout.addWidget(self.update_button)

        upper_splitter.addWidget(details_widget)
        upper_splitter.setSizes([600, 400])
        results_box_layout.addWidget(upper_splitter)

        # Manuelle Nachsuche
        self.manual_search_group = QGroupBox("➕ Manuelle Nachsuche")
        self.manual_search_group.setVisible(False)
        manual_layout = QHBoxLayout(self.manual_search_group)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        manual_layout.addWidget(QLabel("Zusätzlicher Suchbegriff:"))
        self.manual_search_input = QTextEdit()
        self.manual_search_input.setPlaceholderText("Begriff für manuelle Suche eingeben...")
        self.manual_search_input.setMaximumHeight(60)
        manual_layout.addWidget(self.manual_search_input)
        self.manual_search_button = QPushButton("Suchen")
        self.manual_search_button.setStyleSheet(btn_styles["primary"])
        self.manual_search_button.clicked.connect(self.perform_manual_search)
        manual_layout.addWidget(self.manual_search_button)
        results_box_layout.addWidget(self.manual_search_group)

        # Aktionen
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        self.toggle_manual_button = QPushButton("🔧 Manuelle Nachsuche")
        self.toggle_manual_button.setCheckable(True)
        self.toggle_manual_button.setStyleSheet(btn_styles["secondary"])
        self.toggle_manual_button.setToolTip(
            "Aktiviert manuelle Suche für zusätzliche GND-Schlagwörter, "
            "die die Pipeline nicht gefunden hat"
        )
        self.toggle_manual_button.toggled.connect(self.manual_search_group.setVisible)
        actions_layout.addWidget(self.toggle_manual_button)
        actions_layout.addStretch(1)

        self.save_changes_button = QPushButton("💾 Änderungen Speichern")
        self.save_changes_button.setStyleSheet(btn_styles["accent"])
        self.save_changes_button.setEnabled(False)
        self.save_changes_button.setToolTip(
            "Speichert Ihre Änderungen an der GND-Auswahl zurück in den Analyse-Status. "
            "Änderungen werden dann in anderen Tabs sichtbar."
        )
        self.save_changes_button.clicked.connect(self.save_changes)
        actions_layout.addWidget(self.save_changes_button)
        results_box_layout.addLayout(actions_layout)

        results_container_layout.addWidget(results_group)
        self.results_tabs.addTab(results_container, "🔑 GND-Schlagwörter")

        # UB-Katalog als eingebetteter Ergebnis-Reiter (geteilte Sucheingabe)
        self._ub_tab_idx = -1
        if self.ub_catalog_tab is not None:
            panel = getattr(self.ub_catalog_tab, "ub_search_panel", None)
            if panel is not None and hasattr(panel, "set_embedded"):
                panel.set_embedded(lambda: self.search_input.toPlainText())
            self._ub_tab_idx = self.results_tabs.addTab(
                self.ub_catalog_tab, "📚 UB-Katalog (DK/RVK)"
            )

        # Pipeline-Mapping: versteckter Reiter, nur im Pipeline-Modus sichtbar
        self.transparency_text = QTextEdit()
        self.transparency_text.setReadOnly(True)
        self.transparency_text.setPlaceholderText(
            "Mapping-Details werden nach Pipeline-Ausführung hier angezeigt..."
        )
        self._mapping_tab_idx = self.results_tabs.addTab(
            self.transparency_text, "📋 Pipeline-Mapping"
        )
        self.results_tabs.setTabVisible(self._mapping_tab_idx, False)

        main_splitter.addWidget(self.results_tabs)
        main_splitter.setSizes([300, 700])
        layout.addWidget(main_splitter)

        self.results_table.itemSelectionChanged.connect(self.show_details)
        self.results_table.itemDoubleClicked.connect(self.on_result_double_clicked)

    def refresh_styles(self):
        """Re-apply styles after theme change — Claude Generated"""
        self.setStyleSheet(get_main_stylesheet())
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(get_status_label_styles()["info"])

    # ── Quellen (Plugin-System, live) ────────────────────────────────────────

    def _gnd_source_ids(self):
        """Aktivierte + verfügbare GND-Quellen-Typen (Plugins-Tab-Gate) für die
        Quellen-Checkboxen.

        ``None`` (Config nicht lesbar) → Fallback lobid+swb; eine *leere* Liste
        heißt dagegen „Operator hat alle GND-Quellen deaktiviert" und bleibt leer —
        beides zu vermengen würde gegen ein explizites Disable suchen.
        - Claude Generated"""
        try:
            from ..core.search.factory import enabled_gnd_provider_ids

            ids = enabled_gnd_provider_ids(available_only=True)
            return ids if ids is not None else ["lobid", "swb"]
        except Exception as e:
            self.logger.warning(f"Quellenliste nicht ladbar, Fallback lobid+swb: {e}")
            return ["lobid", "swb"]

    def refresh_sources(self):
        """Quellen-Checkboxen aus dem Plugin-System neu bauen (WP-K5-Fix:
        vorher nur einmalig in init_ui → Enable/Disable griff erst nach
        Neustart). Angehakt-Zustand bekannter Quellen bleibt erhalten; neue
        Quellen starten mit dem lobid/swb-Default. Wird nach jedem
        Settings-Save aufgerufen (``_refresh_plugin_tools``). - Claude Generated"""
        previous = {pid: cb.isChecked() for pid, cb in self.source_checkboxes.items()}
        for cb in self.source_checkboxes.values():
            self.sources_layout.removeWidget(cb)
            cb.deleteLater()
        self.source_checkboxes = {}

        default_checked = {"lobid", "swb"}
        # direkt nach dem Label einfügen — die UB-Katalog-Checkbox und der
        # Stretch bleiben dahinter stehen
        insert_at = 1
        for provider_id in self._gnd_source_ids():
            try:
                from ..core.search.registry import get_provider

                cls = get_provider(provider_id)
                label = getattr(cls, "label", provider_id)
                tooltip = cls.doc().description if hasattr(cls, "doc") else ""
            except Exception:
                label, tooltip = provider_id, ""
            checkbox = QCheckBox(label)
            checkbox.setChecked(previous.get(provider_id, provider_id in default_checked))
            if tooltip:
                checkbox.setToolTip(tooltip)
            self.sources_layout.insertWidget(insert_at, checkbox)
            insert_at += 1
            self.source_checkboxes[provider_id] = checkbox
        self.logger.debug(f"Suchquellen aktualisiert: {sorted(self.source_checkboxes)}")

    # ── Standalone-Suche (asynchron, abbrechbar) ─────────────────────────────

    def _on_search_button(self):
        """Start/Abbrechen-Umschalter des Such-Buttons - Claude Generated"""
        if self._search_worker is not None and self._search_worker.isRunning():
            self._cancel_search()
        else:
            self.perform_search()

    def _cancel_search(self):
        """Laufende Suche abbrechen: der Worker verwirft sein Ergebnis
        (die HTTP-Anfrage selbst läuft im Hintergrund aus — StoppableWorker
        prüft an den Signalpunkten, kann aber keinen Request unterbrechen).
        - Claude Generated"""
        if self._search_worker is not None:
            self._search_worker.request_stop()
        self._search_ui_idle()
        self._set_status("Suche abgebrochen.", "info")

    def _search_ui_idle(self):
        self.search_button.setText("Suche starten")
        self.search_button.setStyleSheet(self._btn_styles["primary"])
        self.progressBar.setVisible(False)

    def _search_ui_running(self):
        self.search_button.setText("⏹ Abbrechen")
        self.search_button.setStyleSheet(self._btn_styles["secondary"])
        self.progressBar.setVisible(True)

    def perform_search(self):
        text = self.search_input.toPlainText().strip()
        if not text:
            self._set_status("Bitte geben Sie mindestens einen Suchbegriff ein.", "warning")
            return
        search_terms = extract_search_terms(text)
        if not search_terms:
            self._set_status("Keine gültigen Suchbegriffe gefunden.", "warning")
            return
        ub_wanted = bool(
            self.ub_catalog_checkbox is not None and self.ub_catalog_checkbox.isChecked()
        )
        if not self.source_checkboxes and not ub_wanted:
            self._set_status("Keine GND-Quelle aktiv — im Plugins-Tab aktivieren.", "warning")
            return

        provider_ids = [pid for pid, cb in self.source_checkboxes.items() if cb.isChecked()]
        if not provider_ids and not ub_wanted and self.source_checkboxes:
            fallback = next(iter(self.source_checkboxes))
            self.logger.warning(f"Keine Suchquelle ausgewählt, verwende {fallback}.")
            provider_ids = [fallback]

        # Standalone-Suche verlässt den Pipeline-Modus: Mapping-Reiter aus,
        # Doppelklick-Toggle deaktivieren (bezöge sich auf veralteten Zustand)
        self.results_tabs.setTabVisible(self._mapping_tab_idx, False)
        self.original_pipeline_state = None

        if provider_ids:
            self.results_table.setRowCount(0)
            self.details_display.clear()
            self._entries_by_gnd = {}
            self._search_ui_running()
            self._set_status(
                f"Suche nach {len(search_terms)} Begriff(en) in {', '.join(provider_ids)}...",
                "info",
            )
            self._search_worker = GndSearchWorker(search_terms, provider_ids)
            self._search_worker.finished_with_results.connect(self._on_search_finished)
            self._search_worker.search_failed.connect(self._on_search_failed)
            self._search_worker.start()

        # UB-Katalog-Suche (eigener Worker im eingebetteten Panel, geteilte
        # Eingabe) parallel bzw. allein auslösen - Claude Generated
        if ub_wanted and self.ub_catalog_tab is not None:
            panel = getattr(self.ub_catalog_tab, "ub_search_panel", None)
            if panel is not None:
                panel.start_search()
            if not provider_ids:
                self._set_status("UB-Katalog-Suche läuft...", "info")

        # Auf den Reiter wechseln, der gleich Ergebnisse zeigt
        if ub_wanted and not provider_ids and self._ub_tab_idx >= 0:
            self.results_tabs.setCurrentIndex(self._ub_tab_idx)
        elif provider_ids:
            self.results_tabs.setCurrentIndex(0)

    def _on_search_finished(self, results: dict, errors):
        self._search_ui_idle()
        rows, skipped = self._rows_from_results(results)
        self.display_results(rows)
        msg = f"Suche abgeschlossen — {len(rows)} Ergebnisse"
        if skipped:
            msg += f" ({skipped} ohne GND-ID übersprungen)"
        if errors:
            msg += f" — Quellen fehlgeschlagen: {', '.join(sorted(errors))}"
        self._set_status(msg, "warning" if errors else "success")

    def _on_search_failed(self, message: str):
        self._search_ui_idle()
        self._set_status(f"Fehler: {message}", "error")

    def _rows_from_results(self, results: dict):
        """Kanonische Service-Ergebnisse → sortierte Anzeige-Zeilen.

        Ein Eintrag ist ``{count, gnd_ids, classifications, display_count?}``
        (Suggester-Vertrag v2); die frühere Legacy-List-Branch ist entfernt —
        der Unified-Service liefert dieses Format nie. - Claude Generated
        """
        rows = []
        skipped = 0
        for search_term, term_results in (results or {}).items():
            if not isinstance(term_results, dict):
                self.logger.warning(
                    f"Unerwartetes Ergebnisformat für '{search_term}': {type(term_results).__name__}"
                )
                continue
            for keyword, entry in term_results.items():
                gnd_id = next(iter(entry.get("gnd_ids", [])), "")
                if not gnd_id:
                    skipped += 1
                    continue
                self._entries_by_gnd[gnd_id] = entry
                rows.append({
                    "gnd_id": gnd_id,
                    "term": keyword,
                    "count": preferred_display_count(entry),
                    "relation": determine_relation(keyword, search_term),
                    "search_term": search_term,
                    "classifications": format_classifications_compact(
                        entry.get("classifications")
                    ),
                })
        rows.sort(key=lambda r: (r["relation"], -r["count"]))
        return rows, skipped

    def display_results(self, rows: List[Dict[str, Any]]):
        """Zeigt Standalone-Suchzeilen in der Tabelle an. - Claude Generated"""
        self.results_table.setRowCount(0)
        self.cache_status = self._cache_status_for_ids([r["gnd_id"] for r in rows])
        for r in rows:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            term_item = QTableWidgetItem(r["term"])
            term_item.setToolTip(f"Suchbegriff: {r['search_term']}")
            self.results_table.setItem(row, _COL_TERM, term_item)

            gnd_item = QTableWidgetItem(r["gnd_id"])
            gnd_item.setToolTip("Klicken für Details")
            self.results_table.setItem(row, _COL_GND, gnd_item)

            count_item = QTableWidgetItem(str(r["count"]))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, _COL_COUNT, count_item)

            rel_item = QTableWidgetItem(_RELATION_SYMBOLS[r["relation"]])
            rel_item.setForeground(_RELATION_COLORS[r["relation"]])
            rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, _COL_REL, rel_item)

            self.results_table.setItem(
                row, _COL_CLS, QTableWidgetItem(r["classifications"])
            )

            status = self.cache_status.get(r["gnd_id"], "new")
            status_item = QTableWidgetItem(_CACHE_STATUS_ICONS[status])
            status_item.setToolTip(_CACHE_STATUS_TOOLTIPS[status])
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(row, _COL_STATUS, status_item)

    # ── Cache-Status (geteilt von Standalone + Pipeline-Ansicht) ─────────────

    def _cache_status_for_ids(self, gnd_ids) -> Dict[str, str]:
        """Batch-Cache-Status je GND-ID: 'cache' (<90 Tage), 'outdated',
        'new'. Eine SQL-Query statt N Einzel-Lookups. - Claude Generated"""
        from datetime import datetime

        all_ids = {gid for gid in gnd_ids if gid}
        if not all_ids:
            return {}
        status: Dict[str, str] = {}
        try:
            db = getattr(self.cache_manager, "db_manager", None)
            if db is None:
                return {gid: "new" for gid in all_ids}
            id_list = list(all_ids)
            placeholders = ",".join(["?"] * len(id_list))
            rows = db.fetch_all(
                f"SELECT gnd_id, updated_at FROM gnd_entries WHERE gnd_id IN ({placeholders})",
                id_list,
            )
            cached = {row.get("gnd_id"): row.get("updated_at") for row in rows if row.get("gnd_id")}
            now = datetime.now()
            for gid in all_ids:
                updated_at = cached.get(gid)
                if not updated_at:
                    status[gid] = "new"
                    continue
                try:
                    updated_str = str(updated_at)
                    if "T" in updated_str:
                        updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    else:
                        updated_dt = datetime.strptime(updated_str, "%Y-%m-%d %H:%M:%S")
                    age_days = (now - updated_dt.replace(tzinfo=None)).days
                    status[gid] = "outdated" if age_days >= 90 else "cache"
                except (ValueError, AttributeError):
                    status[gid] = "cache"
        except Exception as e:  # DB nicht erreichbar — Anzeige degradiert zu 'new'
            self.logger.warning(f"Batch cache status query failed: {e}")
            return {gid: "new" for gid in all_ids}
        return status

    # ── Details ──────────────────────────────────────────────────────────────

    def show_details(self):
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        gnd_item = self.results_table.item(row, _COL_GND)
        if gnd_item is None:
            return
        gnd_id = gnd_item.text()

        parts = [f"<h3>Details für GND-ID: {gnd_id}</h3>"]
        parts.append(
            f'<p><a href="https://lobid.org/gnd/{gnd_id}">lobid.org/gnd/{gnd_id}</a></p>'
        )

        # Pool-Eintrag der aktuellen Suche (Klassifikationen aus den Quellen)
        entry = self._entries_by_gnd.get(gnd_id)
        if entry:
            cls_line = format_classifications_compact(
                entry.get("classifications"), max_per_system=10
            )
            if cls_line:
                parts.append(f"<p><b>Klassifikationen (Suche):</b> {cls_line}</p>")
            display_count = preferred_display_count(entry)
            if display_count:
                parts.append(f"<p><b>Häufigkeit:</b> {display_count}</p>")

        if gnd_id and self.cache_manager.gnd_entry_exists(gnd_id):
            gnd_entry = self.cache_manager.get_gnd_entry_by_id(gnd_id) or {}
            parts.append(f"<p><b>Titel:</b> {gnd_entry.get('title', 'N/A')}</p>")
            if gnd_entry.get("description"):
                parts.append(f"<p><b>Beschreibung:</b> {gnd_entry.get('description')}</p>")
            for field, label in (("ddcs", "DDC-Klassifikationen"), ("dks", "DK-Klassifikationen"), ("synonyms", "Synonyme")):
                values = [v for v in (gnd_entry.get(field) or "").split(";") if v.strip()]
                if values:
                    items = "".join(f"<li>{v}</li>" for v in values)
                    parts.append(f"<p><b>{label}:</b><ul>{items}</ul></p>")
            parts.append(
                "<p style='font-size: 0.9em; color: #666;'>"
                f"<b>Erstellt am:</b> {gnd_entry.get('created_at', 'N/A')}<br>"
                f"<b>Zuletzt aktualisiert:</b> {gnd_entry.get('updated_at', 'N/A')}</p>"
            )
        elif not entry:
            parts.append("<p>Kein lokaler Datenbank-Eintrag vorhanden.</p>")

        self.details_display.setHtml(
            "<html><body style='font-family: Arial, sans-serif;'>"
            + "".join(parts)
            + "</body></html>"
        )

    # ── DNB-Sync (bestehender Worker-Pfad) ───────────────────────────────────

    def update_selected_entry(self):
        """Batch DNB sync for all selected/displayed entries - Claude Generated"""
        selected_gnd_ids = []
        for row in range(self.results_table.rowCount()):
            begriff_item = self.results_table.item(row, _COL_TERM)
            if begriff_item and begriff_item.text().startswith("✅"):
                selected_gnd_ids.append(self.results_table.item(row, _COL_GND).text())

        if not selected_gnd_ids:
            self._set_status("Keine ausgewählten Einträge zum Synchronisieren", "warning")
            return

        self.update_button.setEnabled(False)
        self.update_button.setText(f"Synchronisiere {len(selected_gnd_ids)} Einträge...")
        self.progressBar.setVisible(True)
        self._set_status(f"Starte DNB-Sync für {len(selected_gnd_ids)} Einträge...", "info")

        self.sync_worker = DNBSyncWorker(selected_gnd_ids, self.cache_manager)
        self.sync_worker.finished.connect(self.on_sync_finished)
        self.sync_worker.start()

    def on_sync_finished(self, success_count, error_count):
        """Handle DNB sync completion - Claude Generated"""
        self.update_button.setEnabled(True)
        self.update_button.setText("🔄 DNB-Sync Selected")
        self.progressBar.setVisible(False)
        if error_count == 0:
            self._set_status(f"DNB-Sync erfolgreich: {success_count} Einträge aktualisiert", "success")
        else:
            self._set_status(
                f"DNB-Sync abgeschlossen: {success_count} erfolgreich, {error_count} Fehler",
                "warning",
            )
        if self.results_table.selectedItems():
            self.show_details()

    # ── Pipeline-Nachbearbeitung ────────────────────────────────────────────

    def update_search_field(self, keywords):
        """Aktualisiert das Suchfeld mit den gegebenen Schlüsselwörtern"""
        self.search_input.setText(keywords)

    def display_search_results(self, results: Dict) -> None:
        """Viewer für Pipeline-Suchergebnisse im Roh-Dict-Format - Claude Generated"""
        self.logger.info(f"Displaying pipeline search results: {len(results)} terms")
        self.details_display.clear()
        self._entries_by_gnd = {}
        rows, skipped = self._rows_from_results(results)
        self.display_results(rows)
        msg = f"Pipeline-Ergebnisse angezeigt — {len(rows)} Ergebnisse"
        if skipped:
            msg += f" ({skipped} ohne GND-ID)"
        self._set_status(msg, "success")

    @pyqtSlot(object)
    def update_data(self, analysis_state):
        """Pipeline-Ergebnisse mit Auswahl-/Cache-Transparenz anzeigen - Claude Generated"""
        if not analysis_state:
            self._set_status("Keine Pipeline-Ergebnisse verfügbar", "warning")
            self.transparency_text.setHtml(
                "<p style='color: orange;'><b>Keine Pipeline-Daten geladen</b></p>"
                "<p>Führen Sie die Pipeline aus, um GND-Suchergebnisse zu sehen.</p>"
            )
            return

        try:
            self.original_pipeline_state = analysis_state
            self.modified_selections = {}
            self.manual_additions = []
            self.save_changes_button.setEnabled(False)
            # Pipeline-Modus: Mapping-Reiter einblenden; finale Keywords in die
            # geteilte Sucheingabe (Nachsuche in GND wie UB-Katalog) - Claude Generated
            self.results_tabs.setTabVisible(self._mapping_tab_idx, True)
            if analysis_state.final_llm_analysis:
                final_kw = analysis_state.final_llm_analysis.extracted_gnd_keywords or []
                if final_kw:
                    self.search_input.setText(", ".join(final_kw))

            initial_keywords = analysis_state.initial_keywords or []
            search_results = analysis_state.search_results or []
            final_keywords = []
            if analysis_state.final_llm_analysis:
                final_keywords = analysis_state.final_llm_analysis.extracted_gnd_keywords or []

            if not search_results:
                self._set_status("Pipeline hat keine GND-Einträge gefunden", "warning")
                self.transparency_text.setHtml(
                    "<p style='color: orange;'><b>Keine Suchergebnisse</b></p>"
                    "<p>Die Pipeline hat keine GND-Schlagwörter gefunden. "
                    "Verwenden Sie die manuelle Nachsuche unten.</p>"
                )
                self.results_table.setRowCount(0)
                return

            all_ids = [
                gnd_id
                for sr in search_results
                for gnd_id in sr.results.keys()
            ]
            self.cache_status = self._cache_status_for_ids(all_ids)
            self._build_mapping_table(initial_keywords, search_results, final_keywords)
            self._display_pipeline_results(search_results, final_keywords)

            total_entries = sum(len(sr.results) for sr in search_results)
            self._set_status(
                f"✅ Pipeline-Ergebnisse: {total_entries} GND-Einträge "
                f"({len(final_keywords)} ausgewählt) | 💡 Doppelklick zum Umschalten",
                "success",
            )
        except Exception as e:
            self.logger.error(f"Error in update_data: {e}", exc_info=True)
            self._set_status(f"Fehler beim Laden der Pipeline-Ergebnisse: {str(e)}", "error")

    def _build_mapping_table(self, initial_keywords: List[str], search_results: List,
                             final_keywords: List[str]) -> None:
        """Init-Keyword → GND-Einträge Mapping als HTML - Claude Generated"""
        mapping_text = "<h3>Pipeline-Mapping (Initial Keywords → GND-Einträge)</h3>"
        mapping_text += (
            "<p style='color: gray; font-size: 0.9em;'>"
            "Zeigt welche Pipeline-Keywords zu welchen GND-Einträgen führten:</p>"
        )
        if not initial_keywords:
            mapping_text += "<p style='color: orange;'><b>Keine Initial-Keywords gefunden</b></p>"
            self.transparency_text.setHtml(mapping_text)
            return

        for init_kw in initial_keywords:
            matching = [sr for sr in search_results if sr.search_term == init_kw]
            if not matching:
                continue
            mapping_text += (
                f"<p style='margin-top: 12px;'><b>🔍 \"{init_kw}\"</b>:</p>"
                "<ul style='margin-left: 20px;'>"
            )
            for sr in matching:
                for gnd_id, entry_data in sr.results.items():
                    label = entry_data.get("label", "N/A")
                    if gnd_id in final_keywords:
                        status = "✅ <span style='color: green; font-weight: bold;'>[Verwendet]</span>"
                    else:
                        status = "<span style='color: gray;'>[Nicht verwendet]</span>"
                    mapping_text += f"<li>{label} (GND:{gnd_id}) {status}</li>"
            mapping_text += "</ul>"

        self.transparency_text.setHtml(mapping_text)

    def _display_pipeline_results(self, search_results: List, final_keywords: List[str]) -> None:
        """Pipeline-Ergebnistabelle mit Transparenz-Overlays - Claude Generated"""
        self.results_table.setRowCount(0)
        self._entries_by_gnd = {}

        for search_result in search_results:
            init_keyword = search_result.search_term
            for gnd_id, entry_data in search_result.results.items():
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._entries_by_gnd[gnd_id] = entry_data
                is_selected = gnd_id in final_keywords

                begriff = entry_data.get("label", entry_data.get("title", "N/A"))
                begriff_item = QTableWidgetItem(begriff)
                if is_selected:
                    begriff_item.setFont(get_scaled_font(bold=True))
                    begriff_item.setForeground(QColor("#2e7d32"))
                    begriff_item.setText(f"✅ {begriff}")
                self.results_table.setItem(row, _COL_TERM, begriff_item)

                self.results_table.setItem(row, _COL_GND, QTableWidgetItem(gnd_id))

                count_item = QTableWidgetItem(str(preferred_display_count(entry_data)))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, _COL_COUNT, count_item)

                relation = determine_relation(begriff, init_keyword)
                rel_item = QTableWidgetItem(_RELATION_SYMBOLS[relation])
                rel_item.setForeground(_RELATION_COLORS[relation])
                rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                rel_item.setToolTip(f"Initial-Keyword: {init_keyword}")
                self.results_table.setItem(row, _COL_REL, rel_item)

                self.results_table.setItem(
                    row,
                    _COL_CLS,
                    QTableWidgetItem(
                        format_classifications_compact(entry_data.get("classifications"))
                    ),
                )

                status = self.cache_status.get(gnd_id, "new")
                status_item = QTableWidgetItem(_CACHE_STATUS_ICONS[status])
                status_item.setToolTip(_CACHE_STATUS_TOOLTIPS[status])
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, _COL_STATUS, status_item)

                if is_selected:
                    self._highlight_row(row, QColor("#e8f5e9"))

        self.logger.info(f"Displayed {self.results_table.rowCount()} pipeline results")

    def _highlight_row(self, row: int, color: QColor):
        for col in range(len(_COLS)):
            item = self.results_table.item(row, col)
            if item:
                item.setBackground(color)

    # ── Manuelle Nachsuche (asynchron, gleicher Worker) ──────────────────────

    def perform_manual_search(self):
        """Manuelle GND-Nachsuche; Treffer werden mit 🔧 markiert - Claude Generated"""
        search_term = self.manual_search_input.toPlainText().strip()
        if not search_term:
            self._set_status("Bitte einen Suchbegriff eingeben", "warning")
            return

        # Bevorzugt lobid/swb (billig, keine DK-Lookups), sonst die erste aktive
        # Quelle. Ohne aktive Quelle nicht suchen statt eine zu erfinden.
        manual_ids = [
            pid for pid in ("lobid", "swb") if pid in self.source_checkboxes
        ] or list(self.source_checkboxes)[:1]
        if not manual_ids:
            self._set_status("Keine GND-Quelle aktiv — im Plugins-Tab aktivieren.", "warning")
            return

        self.manual_search_button.setEnabled(False)
        self.progressBar.setVisible(True)
        self._set_status(f"Manuelle Suche: {search_term}", "info")

        self._manual_worker = GndSearchWorker(extract_search_terms(search_term), manual_ids)
        self._manual_worker.finished_with_results.connect(self._on_manual_finished)
        self._manual_worker.search_failed.connect(self._on_manual_failed)
        self._manual_worker.start()

    def _on_manual_finished(self, all_results: dict, errors):
        self.manual_search_button.setEnabled(True)
        self.progressBar.setVisible(False)

        existing_ids = {
            self.results_table.item(row, _COL_GND).text()
            for row in range(self.results_table.rowCount())
            if self.results_table.item(row, _COL_GND)
        }
        added_count = 0
        for term, results in (all_results or {}).items():
            if not isinstance(results, dict):
                continue
            for keyword, data in results.items():
                gnd_id = next(iter(data.get("gnd_ids", set())), None)
                if not gnd_id or gnd_id in existing_ids:
                    continue
                existing_ids.add(gnd_id)
                self._entries_by_gnd[gnd_id] = data

                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                begriff_item = QTableWidgetItem(f"🔧 {keyword}")
                begriff_item.setForeground(QColor("#1976d2"))
                self.results_table.setItem(row, _COL_TERM, begriff_item)
                self.results_table.setItem(row, _COL_GND, QTableWidgetItem(gnd_id))
                count_item = QTableWidgetItem(str(preferred_display_count(data)))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, _COL_COUNT, count_item)
                rel_item = QTableWidgetItem("🔧")
                rel_item.setToolTip("Manuell hinzugefügt")
                rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, _COL_REL, rel_item)
                self.results_table.setItem(
                    row,
                    _COL_CLS,
                    QTableWidgetItem(format_classifications_compact(data.get("classifications"))),
                )
                status_item = QTableWidgetItem("🆕")
                status_item.setToolTip("Manuelle Addition")
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, _COL_STATUS, status_item)
                self._highlight_row(row, QColor("#e3f2fd"))

                self.manual_additions.append(
                    {"gnd_id": gnd_id, "label": keyword, "search_term": term}
                )
                added_count += 1

        if added_count > 0:
            self._set_status(f"Manuelle Suche: {added_count} neue Einträge hinzugefügt", "success")
            self.save_changes_button.setEnabled(True)
        else:
            msg = "Manuelle Suche: Keine neuen Einträge gefunden"
            if errors:
                msg += f" — Quellen fehlgeschlagen: {', '.join(sorted(errors))}"
            self._set_status(msg, "info")
        self.manual_search_input.clear()

    def _on_manual_failed(self, message: str):
        self.manual_search_button.setEnabled(True)
        self.progressBar.setVisible(False)
        self._set_status(f"Fehler bei manueller Suche: {message}", "error")

    # ── Auswahl-Änderungen ───────────────────────────────────────────────────

    def save_changes(self):
        """Änderungen an der GND-Auswahl zurück in den Analyse-Status - Claude Generated"""
        if not (self.modified_selections or self.manual_additions):
            return
        self.selection_changed.emit(
            {"modified": self.modified_selections, "manual": self.manual_additions}
        )
        self._set_status(
            f"Änderungen gespeichert: {len(self.modified_selections)} geändert, "
            f"{len(self.manual_additions)} manuell hinzugefügt",
            "success",
        )
        self.save_changes_button.setEnabled(False)

    def on_result_double_clicked(self, item):
        """Doppelklick toggelt den Auswahl-Status (nur Pipeline-Modus) - Claude Generated"""
        if not self.original_pipeline_state:
            return

        row = item.row()
        gnd_id = self.results_table.item(row, _COL_GND).text()
        begriff_item = self.results_table.item(row, _COL_TERM)
        begriff_text = begriff_item.text()

        final_keywords = []
        if self.original_pipeline_state.final_llm_analysis:
            final_keywords = (
                self.original_pipeline_state.final_llm_analysis.extracted_gnd_keywords or []
            )
        was_originally_selected = gnd_id in final_keywords
        is_currently_selected = begriff_text.startswith("✅")

        if is_currently_selected:
            begriff_item.setText(begriff_text.replace("✅ ", ""))
            begriff_item.setFont(get_scaled_font())
            begriff_item.setForeground(QColor("#000000"))
            for col in range(len(_COLS)):
                cell_item = self.results_table.item(row, col)
                if cell_item and cell_item.background().color() != QColor("#e3f2fd"):
                    cell_item.setBackground(QColor("#ffffff"))
            if was_originally_selected:
                self.modified_selections[gnd_id] = "deselected"
            else:
                self.modified_selections.pop(gnd_id, None)
        else:
            begriff_item.setText(f"✅ {begriff_text.replace('🔧 ', '')}")
            begriff_item.setFont(get_scaled_font(bold=True))
            begriff_item.setForeground(QColor("#2e7d32"))
            self._highlight_row(row, QColor("#e8f5e9"))
            if not was_originally_selected:
                self.modified_selections[gnd_id] = "selected"
            else:
                self.modified_selections.pop(gnd_id, None)

        self.save_changes_button.setEnabled(
            bool(self.modified_selections or self.manual_additions)
        )

    # ── Statuszeile ──────────────────────────────────────────────────────────

    def _set_status(self, message: str, kind: str = "info"):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(get_status_label_styles()[kind])
