"""UI-construction + result-rendering mixin for PipelineTab. Claude Generated.

Extracted from ``pipeline_tab.py`` (F-5 god-file split): the per-step widget
builders (input/initialisation/search/keywords/dk_search/dk_classification), the
GND-hit table rendering/marking/filtering, the DK-search result display +
classification formatting, and splitter-state persistence. Method bodies are
moved verbatim. ``PipelineTabUiMixin`` is mixed into ``PipelineTab`` (which
provides ``__init__``, ``setup_ui`` and the widgets/state these build against);
it is not a standalone widget.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline_manager import PipelineStep
from ..utils.pipeline_utils import PipelineResultFormatter
from .unified_input_widget import UnifiedInputWidget


class PipelineTabUiMixin:
    """Step-widget builders + result rendering (verbatim from PipelineTab)."""

    def create_input_step_widget(self) -> QWidget:
        """Create unified input step widget - Claude Generated"""
        # Create unified input widget
        self.unified_input = UnifiedInputWidget(
            llm_service=self.llm_service, alima_manager=self.alima_manager
        )

        # Connect signals
        self.unified_input.text_ready.connect(self.on_input_text_ready)
        self.unified_input.input_cleared.connect(self.on_input_cleared)

        return self.unified_input

    def on_input_text_ready(self, text: str, source_info: str):
        """Handle ready input text - Claude Generated"""
        self.logger.info(f"Input text ready: {len(text)} chars from {source_info}")

        # Update the input step
        input_step = self._get_step_by_id("input")
        if input_step:
            input_step.output_data = {
                "text": text,
                "source_info": source_info,
                "timestamp": datetime.now().isoformat(),
            }
            input_step.status = "completed"

            # Update step widget
            if "input" in self.step_widgets:
                self.step_widgets["input"].update_step_data(input_step)

        # Store text for pipeline
        self.current_input_text = text
        self.current_source_info = source_info
        # Capture source type/data from input widget - Claude Generated
        self.current_input_type = getattr(self.unified_input, 'current_source_type', 'text')
        self.current_input_source = getattr(self.unified_input, 'current_source_data', '')

    def on_input_cleared(self):
        """Handle input clearing - Claude Generated"""
        self.current_input_text = ""
        self.current_source_info = ""
        self.current_input_type = "text"  # Reset source tracking - Claude Generated
        self.current_input_source = ""

        # Reset input step
        input_step = self._get_step_by_id("input")
        if input_step:
            input_step.status = "pending"
            input_step.output_data = None

            if "input" in self.step_widgets:
                self.step_widgets["input"].update_step_data(input_step)

    def _get_step_by_id(self, step_id: str) -> Optional[PipelineStep]:
        """Get step by ID - Claude Generated"""
        for step_widget in self.step_widgets.values():
            if step_widget.step.step_id == step_id:
                return step_widget.step
        return None

    def _create_text_result_widget(
        self, label_text: str, placeholder: str, min_height: int = 80, max_height: int = 300
    ) -> tuple[QWidget, QTextEdit]:
        """
        Helper method to create standardized text result widgets.
        Returns tuple of (widget, text_edit) for consistent layout.
        Claude Generated
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)  # Reduziert von default - Claude Generated
        layout.setContentsMargins(0, 0, 0, 0)  # Keine extra margins

        # Label kompakter gestylt
        label = QLabel(label_text)
        label.setStyleSheet(
            "font-weight: bold; color: #555; padding: 2px;"
        )
        label.setMaximumHeight(18)  # Explizite Height
        label.setWordWrap(False)  # Keine Zeilenumbrüche

        # Results area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        #text_edit.setMinimumHeight(min_height)
        #text_edit.setMaximumHeight(max_height)
        text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        text_edit.setPlaceholderText(placeholder)

        layout.addWidget(label, 0)  # Stretch 0
        layout.addWidget(text_edit, 1)  # Stretch 1

        return widget, text_edit

    def create_initialisation_step_widget(self) -> QWidget:
        """Create initialisation step widget - Claude Generated"""
        widget, self.initialisation_result = self._create_text_result_widget(
            label_text="Extrahierte freie Schlagworte:",
            placeholder="Freie Schlagworte werden hier angezeigt..."
        )
        return widget

    def create_search_step_widget(self) -> QWidget:
        """Create search step widget — sortable GND-hit table, 3-tier view - Claude Generated

        Shows every catalog hit with a GND-ID (Begriff / GND-ID / Häufigkeit /
        Auswahl). Three tiers: full pool (~1000 hits) → chunk-selected
        (selection_chunks survivors, ☑) → final verified keywords (✅).
        Filterable by free text and by tier.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        label = QLabel("GND-Suchergebnisse:")
        label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        label.setMaximumHeight(18)
        header.addWidget(label, 0)
        self.search_tier_stats = QLabel("")
        self.search_tier_stats.setStyleSheet("color: #777; padding: 2px;")
        self.search_tier_stats.setMaximumHeight(18)
        header.addWidget(self.search_tier_stats, 0)
        header.addStretch(1)
        self.search_filter_input = QLineEdit()
        self.search_filter_input.setPlaceholderText("Filter: Begriff / GND-ID…")
        self.search_filter_input.setClearButtonEnabled(True)
        self.search_filter_input.setMaximumWidth(220)
        self.search_filter_input.textChanged.connect(self._filter_gnd_hits)
        header.addWidget(self.search_filter_input, 0)
        self.search_tier_filter = QComboBox()
        self.search_tier_filter.addItems(
            ["Alle (Pool)", "Chunk-Auswahl", "Finale Auswahl"]
        )
        self.search_tier_filter.setToolTip(
            "Pool = alle GND-Treffer der Suche;\n"
            "Chunk-Auswahl = vom LLM im Chunking als relevant gefiltert;\n"
            "Finale Auswahl = verifizierte finale Schlagwörter"
        )
        self.search_tier_filter.currentIndexChanged.connect(self._filter_gnd_hits)
        header.addWidget(self.search_tier_filter, 0)
        layout.addLayout(header, 0)

        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(4)
        self.search_results_table.setHorizontalHeaderLabels(
            ["Begriff", "GND-ID", "Häufigkeit", "Auswahl"]
        )
        self.search_results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.search_results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.search_results_table.setSortingEnabled(True)
        self.search_results_table.verticalHeader().setVisible(False)
        hh = self.search_results_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.search_results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.search_results_table, 1)

        # Raw state for re-filtering / re-marking selection - Claude Generated
        self.search_raw_rows = []
        self.search_chunk_ids = set()
        self.search_chunk_labels = set()
        self.search_final_ids = set()
        self.search_final_labels = set()
        return widget

    def _populate_gnd_hits(self, search_results, selected=None, reset_marks=False) -> None:
        """Fill the GND-Recherche table from search_results (+ optional selection).

        ``search_results`` may be the classic dict form, a List[SearchResult], or
        agentic ``gnd_entries`` — PipelineResultFormatter.flatten_gnd_hits handles
        all three. ``selected`` marks the FINAL keyword tier; pass None to leave
        the current marks untouched. ``reset_marks=True`` clears both tiers
        (new pipeline run, fresh pool). - Claude Generated
        """
        if not hasattr(self, "search_results_table"):
            return
        self.search_raw_rows = PipelineResultFormatter.flatten_gnd_hits(search_results)
        if reset_marks:
            self.search_chunk_ids = set()
            self.search_chunk_labels = set()
            self.search_final_ids = set()
            self.search_final_labels = set()
        if selected is not None:
            self.search_final_ids, self.search_final_labels = (
                PipelineResultFormatter.extract_selected_gnd_keys(selected)
            )
        self._render_gnd_hits_table()

    def _mark_gnd_selection(self, selected, tier: str = "final") -> None:
        """Mark GND rows for a selection tier - Claude Generated

        tier="chunk": survivors of the chunked relevance filter
        (selection_chunks → selected_keywords); tier="final": final/verified
        keywords. Both tiers stay marked independently for the 3-tier view.
        """
        if not hasattr(self, "search_results_table") or not self.search_raw_rows:
            return
        ids, labels = PipelineResultFormatter.extract_selected_gnd_keys(selected)
        if tier == "chunk":
            self.search_chunk_ids, self.search_chunk_labels = ids, labels
        else:
            self.search_final_ids, self.search_final_labels = ids, labels
        self._render_gnd_hits_table()

    def _gnd_row_tier(self, row) -> int:
        """Tier of a pool row: 2=final, 1=chunk-selected, 0=pool-only - Claude Generated"""
        gid = row["gnd_id"]
        label = row["begriff"].lower()
        if gid in self.search_final_ids or label in self.search_final_labels:
            return 2
        if gid in self.search_chunk_ids or label in self.search_chunk_labels:
            return 1
        return 0

    def _render_gnd_hits_table(self) -> None:
        """Render search_raw_rows into the table with 3-tier highlighting - Claude Generated

        Render order is critical for the user-set filter (combo + free-text) to
        survive tier re-renders triggered by ``selection`` /
        ``verify_keywords`` in the agentic pipeline. ``setRowCount(0)`` and
        ``setSortingEnabled(True)`` both implicitly drop the per-row
        ``setRowHidden`` state, so the filter must be applied ATOMICALLY at the
        very end (after sorting is on) — anything earlier produces a brief
        flash where the pool rows appear unfiltered. - Claude Generated
        """
        table = getattr(self, "search_results_table", None)
        if table is None:
            return
        # Block the tier combo while we rebuild — otherwise a spurious
        # ``currentIndexChanged`` from a transient row-rebuild can re-enter
        # ``_filter_gnd_hits`` and clobber the just-applied state. - Claude Generated
        if hasattr(self, "search_tier_filter"):
            self.search_tier_filter.blockSignals(True)
        try:
            table.setSortingEnabled(False)
            table.setRowCount(0)
            have_marks = bool(
                self.search_chunk_ids or self.search_chunk_labels
                or self.search_final_ids or self.search_final_labels
            )
            n_chunk = 0
            n_final = 0

            for row in self.search_raw_rows:
                tier = self._gnd_row_tier(row)
                if tier == 2:
                    n_final += 1
                    n_chunk += 1  # final keywords passed the chunk filter too
                elif tier == 1:
                    n_chunk += 1
                r = table.rowCount()
                table.insertRow(r)

                begriff_item = QTableWidgetItem(row["begriff"])
                gnd_item = QTableWidgetItem(row["gnd_id"])
                count_item = QTableWidgetItem()
                count_item.setData(Qt.ItemDataRole.DisplayRole, int(row.get("count", 0)))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if tier == 2:
                    sel_text = "✅ Final"
                elif tier == 1:
                    sel_text = "☑ Chunk"
                else:
                    sel_text = "" if have_marks else "—"
                sel_item = QTableWidgetItem(sel_text)
                sel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if tier == 2:
                    for it in (begriff_item, gnd_item, count_item, sel_item):
                        it.setForeground(QColor("#2e7d32"))
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                elif tier == 1:
                    for it in (begriff_item, gnd_item, count_item, sel_item):
                        it.setForeground(QColor("#1565c0"))
                if row.get("search_terms"):
                    begriff_item.setToolTip(
                        "Gefunden über: " + ", ".join(row["search_terms"])
                    )
                # Stash the tier for the visibility filter.
                begriff_item.setData(Qt.ItemDataRole.UserRole, tier)

                table.setItem(r, 0, begriff_item)
                table.setItem(r, 1, gnd_item)
                table.setItem(r, 2, count_item)
                table.setItem(r, 3, sel_item)

            table.setSortingEnabled(True)
            if hasattr(self, "search_tier_stats"):
                self.search_tier_stats.setText(
                    f"· Pool: {len(self.search_raw_rows)} · Chunk: {n_chunk} · Final: {n_final}"
                )
            # _filter_gnd_hits MUST be the very last action — it applies the
            # user-set tier + free-text filter atomically over the freshly
            # inserted rows. Calling it earlier (e.g. before
            # setSortingEnabled) leaves a window where the pool appears
            # unfiltered between snapshot bursts. - Claude Generated
            self._filter_gnd_hits()
        finally:
            if hasattr(self, "search_tier_filter"):
                self.search_tier_filter.blockSignals(False)

    def _filter_gnd_hits(self) -> None:
        """Apply tier filter (Alle/Chunk/Final) + free-text filter - Claude Generated"""
        table = getattr(self, "search_results_table", None)
        if table is None:
            return
        min_tier = 0
        if hasattr(self, "search_tier_filter"):
            min_tier = self.search_tier_filter.currentIndex()  # 0/1/2
        needle = ""
        if hasattr(self, "search_filter_input"):
            needle = self.search_filter_input.text().strip().lower()
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            gnd_item = table.item(r, 1)
            tier = int(item.data(Qt.ItemDataRole.UserRole) or 0) if item else 0
            hide = tier < min_tier
            if not hide and needle:
                begriff = item.text().lower() if item else ""
                gid = gnd_item.text().lower() if gnd_item else ""
                hide = needle not in begriff and needle not in gid
            table.setRowHidden(r, hide)

    def create_keywords_step_widget(self) -> QWidget:
        """Create keywords step widget (Verbale Erschließung) - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Finale Schlagworte ────────────────────────────────────────────────
        kw_label = QLabel("Finale GND-Schlagworte:")
        kw_label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        kw_label.setMaximumHeight(18)
        layout.addWidget(kw_label, 0)

        self.keywords_result = QTextEdit()
        self.keywords_result.setReadOnly(True)
        self.keywords_result.setPlaceholderText("Finale Schlagworte werden hier angezeigt...")
        self.keywords_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.keywords_result, 1)

        # ── Schlagwortketten ─────────────────────────────────────────────────
        chains_label = QLabel("Schlagwortketten (mit Verifikation):")
        chains_label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        chains_label.setMaximumHeight(18)
        layout.addWidget(chains_label, 0)

        self.keyword_chains_result = QTextEdit()
        self.keyword_chains_result.setReadOnly(True)
        self.keyword_chains_result.setPlaceholderText("Schlagwortketten erscheinen nach Abschluss der Verschlagwortung...")
        self.keyword_chains_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.keyword_chains_result, 2)

        return widget

    def create_dk_search_step_widget(self) -> QWidget:
        """Create DK search step with splitter for controls/results - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ═══ Splitter zwischen Controls und Results ═══
        self.dk_search_splitter = QSplitter(Qt.Orientation.Vertical)
        self.dk_search_splitter.setChildrenCollapsible(True)  # Allow collapse

        # Top: Controls (Config + Filter)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(8)

        # Config Section (kompakter)
        config_header = QLabel("⚙️ Katalog-Such-Konfiguration")
        config_header.setStyleSheet("font-weight: bold; color: #555;")
        controls_layout.addWidget(config_header)

        # Kompakte Grid-Layout statt 3 separate Rows
        config_grid = QGridLayout()
        config_grid.setSpacing(8)

        # Row 0: Max Results + Frequency (nebeneinander)
        config_grid.addWidget(QLabel("Max. Ergebnisse:"), 0, 0)
        self.dk_search_max_results = QSpinBox()
        self.dk_search_max_results.setRange(5, 100)
        from ..utils.pipeline_defaults import DEFAULT_DK_MAX_RESULTS
        self.dk_search_max_results.setValue(DEFAULT_DK_MAX_RESULTS)
        self.dk_search_max_results.setToolTip("Max. Katalog-Suchergebnisse pro Keyword")
        config_grid.addWidget(self.dk_search_max_results, 0, 1)

        config_grid.addWidget(QLabel("Min. Häufigkeit:"), 0, 2)
        self.dk_frequency_threshold = QSpinBox()
        self.dk_frequency_threshold.setRange(1, 50)
        from ..utils.pipeline_defaults import DEFAULT_DK_FREQUENCY_THRESHOLD
        self.dk_frequency_threshold.setValue(DEFAULT_DK_FREQUENCY_THRESHOLD)
        self.dk_frequency_threshold.setToolTip("Nur Klassifikationen mit >= N Vorkommen")
        config_grid.addWidget(self.dk_frequency_threshold, 0, 3)

        config_grid.setColumnStretch(4, 1)  # Push to left
        #controls_layout.addLayout(config_grid)

        # Row 1: Force Update Checkbox
        from PyQt6.QtWidgets import QCheckBox
        self.force_update_checkbox = QCheckBox("Katalog-Cache ignorieren")
        self.force_update_checkbox.setToolTip(
            "Erzwingt Live-Suche im Katalog und ignoriert gecachte Ergebnisse."
        )
        self.force_update_checkbox.setChecked(False)
        #controls_layout.addWidget(self.force_update_checkbox)

        # Filter Section (kompakter)
        filter_header = QLabel("🔍 Ergebnisse filtern")
        filter_header.setStyleSheet("font-weight: bold; color: #555;")
        controls_layout.addWidget(filter_header)

        # Filter Grid
        filter_grid = QGridLayout()
        filter_grid.setSpacing(8)

        # Row 0: Search + Clear + Mode + Count
        filter_grid.addWidget(QLabel("Suchen:"), 0, 0)
        self.dk_search_filter_input = QLineEdit()
        self.dk_search_filter_input.setPlaceholderText("Filter eingeben...")
        self.dk_search_filter_input.textChanged.connect(self._filter_dk_search_results)
        filter_grid.addWidget(self.dk_search_filter_input, 0, 1, 1, 2)  # Span 2 cols

        clear_filter_btn = QPushButton("×")
        clear_filter_btn.setMaximumWidth(30)
        clear_filter_btn.setToolTip("Filter löschen")
        clear_filter_btn.clicked.connect(lambda: self.dk_search_filter_input.clear())
        filter_grid.addWidget(clear_filter_btn, 0, 3)

        filter_grid.addWidget(QLabel("Modus:"), 0, 4)
        self.dk_filter_mode = QComboBox()
        self.dk_filter_mode.addItems(["Alle", "Titel", "Klassifikationscodes", "Keywords"])
        self.dk_filter_mode.setStyleSheet(
            "QComboBox { padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }"
            "QComboBox QAbstractItemView { background-color: #2b2b2b; color: #ccc; "
            "selection-background-color: #005fcc; selection-color: white; border: 1px solid #ccc; }"
        )
        self.dk_filter_mode.currentTextChanged.connect(self._filter_dk_search_results)
        filter_grid.addWidget(self.dk_filter_mode, 0, 5)

        self.dk_filter_count_label = QLabel("")
        self.dk_filter_count_label.setStyleSheet("color: #666;")
        filter_grid.addWidget(self.dk_filter_count_label, 0, 6)

        filter_grid.setColumnStretch(7, 1)  # Push to left
        controls_layout.addLayout(filter_grid)

        controls_layout.addStretch()  # Push controls to top
        self.dk_search_splitter.addWidget(controls_widget)

        # Bottom: Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(5, 5, 5, 5)

        results_header = QLabel("📊 Katalog-Suchergebnisse")
        results_header.setStyleSheet("font-weight: bold; color: #555;")
        results_layout.addWidget(results_header)

        self.dk_search_raw_data = []  # Store for filtering

        self.dk_search_results = QTextEdit()
        self.dk_search_results.setReadOnly(True)
        self.dk_search_results.setMinimumHeight(80)
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_search_results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_search_results.setPlaceholderText(
            "Katalog-Suchergebnisse für DK/RVK-Klassifikationen..."
        )
        results_layout.addWidget(self.dk_search_results)
        self.dk_search_splitter.addWidget(results_widget)

        # Splitter ratio: 25% controls, 75% results
        self.dk_search_splitter.setStretchFactor(0, 1)
        self.dk_search_splitter.setStretchFactor(1, 3)
        self.dk_search_splitter.setSizes([120, 360])  # Initial

        layout.addWidget(self.dk_search_splitter)
        # ═══ END Splitter ═══

        return widget

    def create_dk_classification_step_widget(self) -> QWidget:
        """Create DK classification step widget with splitter between input/results - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ═══ Splitter zwischen Input und Results ═══
        self.dk_classification_splitter = QSplitter(Qt.Orientation.Vertical)
        self.dk_classification_splitter.setChildrenCollapsible(False)

        # Top: Input Summary
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(5, 5, 5, 5)

        # Header statt GroupBox
        input_header = QLabel("📥 Eingangsdaten für LLM-Klassifikation")
        input_header.setStyleSheet("font-weight: bold; color: #555;")
        input_layout.addWidget(input_header)

        self.dk_input_summary = QTextEdit()
        self.dk_input_summary.setReadOnly(True)
        self.dk_input_summary.setMinimumHeight(60)  # Reduziert von 80 - Claude Generated
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_input_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_input_summary.setPlaceholderText(
            "Zusammenfassung der Katalog-Suchergebnisse für LLM..."
        )
        input_layout.addWidget(self.dk_input_summary)
        self.dk_classification_splitter.addWidget(input_widget)

        # Bottom: Results Display
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(5, 5, 5, 5)

        # Header statt GroupBox
        results_header = QLabel("✅ Finale DK/RVK-Klassifikationen")
        results_header.setStyleSheet("font-weight: bold; color: #555;")
        results_layout.addWidget(results_header)

        self.dk_classification_results = QTextEdit()
        self.dk_classification_results.setReadOnly(True)
        self.dk_classification_results.setMinimumHeight(60)  # Reduziert von 80 - Claude Generated
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_classification_results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_classification_results.setPlaceholderText(
            "Finale DK/RVK-Klassifikationen vom LLM werden hier angezeigt...\n"
            "Format: DK 666.76, RVK Q12, RVK QC 130, ..."
        )
        results_layout.addWidget(self.dk_classification_results)
        self.dk_classification_splitter.addWidget(results_widget)

        # Splitter ratio: 30% input, 70% results (results wichtiger)
        self.dk_classification_splitter.setStretchFactor(0, 3)
        self.dk_classification_splitter.setStretchFactor(1, 7)
        self.dk_classification_splitter.setSizes([100, 250])  # Initial

        layout.addWidget(self.dk_classification_splitter)
        # ═══ END Splitter ═══

        # Statistics Label (bleibt)
        self.dk_compact_stats = QLabel()
        self.dk_compact_stats.setWordWrap(True)
        self.dk_compact_stats.setTextFormat(Qt.TextFormat.RichText)
        self.dk_compact_stats.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.dk_compact_stats)

        return widget

    def save_splitter_state(self, settings):
        """Save splitter positions to QSettings - Claude Generated"""
        if hasattr(self, 'main_splitter'):
            settings.setValue("pipeline/main_splitter", self.main_splitter.saveState())

    def restore_splitter_state(self, settings):
        """Restore splitter positions from QSettings - Claude Generated"""
        state = settings.value("pipeline/main_splitter")
        if state and hasattr(self, 'main_splitter'):
            self.main_splitter.restoreState(state)

    def _filter_dk_search_results(self):
        """Filter displayed DK search results based on search input - Claude Generated"""
        if not hasattr(self, 'dk_search_raw_data') or not self.dk_search_raw_data:
            return

        filter_text = self.dk_search_filter_input.text().strip().lower()
        filter_mode = self.dk_filter_mode.currentText()

        # Ohne Filter: alle Ergebnisse anzeigen
        if not filter_text:
            self._display_dk_search_results(self.dk_search_raw_data)
            self.dk_filter_count_label.setText("")
            return

        # Filter anwenden
        filtered_results = []
        for result in self.dk_search_raw_data:
            dk_code = result.get("dk", "").lower()
            titles = [t.lower() for t in result.get("titles", [])]
            keywords = [k.lower() for k in result.get("keywords", [])]

            match = False
            if filter_mode == "Alle":
                match = (filter_text in dk_code or
                        any(filter_text in title for title in titles) or
                        any(filter_text in kw for kw in keywords))
            elif filter_mode == "Titel":
                match = any(filter_text in title for title in titles)
            elif filter_mode == "Klassifikationscodes":
                match = filter_text in dk_code
            elif filter_mode == "Keywords":
                match = any(filter_text in kw for kw in keywords)

            if match:
                filtered_results.append(result)

        self._display_dk_search_results(filtered_results)
        self.dk_filter_count_label.setText(
            f"Zeige {len(filtered_results)} von {len(self.dk_search_raw_data)} Ergebnissen"
        )

    def _display_dk_search_results(self, results: List[Dict[str, Any]]):
        """Display DK search results with formatting - Claude Generated

        Formatting delegates to ``PipelineResultFormatter.format_dk_search_results_text``
        (shared with the Agentic-Chat); this method keeps only the widget-side
        empty-state handling.
        """
        if not results:
            self.dk_search_results.setPlainText(
                "Keine Ergebnisse gefunden" if hasattr(self, 'dk_search_filter_input')
                and self.dk_search_filter_input.text()
                else "Keine DK/RVK-Klassifikationen gefunden"
            )
            return

        text = PipelineResultFormatter.format_dk_search_results_text(results)
        if not text:
            # Non-empty input but nothing displayable (no titles/counts) —
            # never blank the widget silently - Claude Generated
            self.logger.warning(
                f"_display_dk_search_results: {len(results)} Einträge, aber keine "
                "darstellbaren Titel/Häufigkeiten — Anzeige nicht geleert"
            )
            self.dk_search_results.setPlainText(
                f"({len(results)} Katalog-Einträge ohne darstellbare Titel/Häufigkeit)"
            )
            return
        self.dk_search_results.setPlainText(text)

    def _format_dk_classifications_with_titles(
        self,
        dk_classifications: List[str],
        dk_search_results: List[Dict[str, Any]],
        max_titles_per_code: int = 5
    ) -> str:
        """Format final classifications with catalog titles using HTML - Claude Generated

        Delegates to the shared ``PipelineResultFormatter`` (single source of
        truth, also used by the Agentic-Chat). The shared formatter returns an
        HTML fragment; the Pipeline-Tab wraps it in a body with the Arial base
        font for ``QTextEdit.setHtml``.
        """
        fragment = PipelineResultFormatter.format_dk_classifications_html(
            dk_classifications, dk_search_results, max_titles_per_code
        )
        if not dk_classifications:
            return fragment
        return (
            "<html><body style='font-family: Arial, sans-serif;'>"
            f"{fragment}</body></html>"
        )

    @staticmethod
    def _split_classification_code(classification: str) -> tuple[str, str]:
        """Split a prefixed classification string into (system, code)."""
        return PipelineResultFormatter.split_classification_code(classification)


