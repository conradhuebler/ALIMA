"""Unified Search Tab — P-θ.3. Claude Generated.

Thin wrapper around the existing :class:`SearchTab` (GND/SWB/Lobid) and
:class:`UBCatalogTab` widgets. A single source picker at the top swaps
between them via :class:`QStackedWidget`.

Both backends keep their own internal renderers (per-source renderer
approach per WP10 P-θ.3 lock-in). External references to ``search_tab``,
``ub_catalog_tab``, and ``ub_search_tab`` on MainWindow remain valid —
the widgets are only re-parented into this wrapper.

Signal forwarding:
    * ``SearchTab.search_completed`` / ``selection_changed`` keep emitting
      from the inner widget — the existing main-window wiring still works.
    * ``UBCatalogTab.search_completed`` likewise feeds the unified DK tab.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class SearchTabUnified(QWidget):
    """Source-picker + stacked panels for GND/SWB/Lobid and UB-Katalog."""

    SOURCE_GND = "gnd"
    SOURCE_UB_CATALOG = "ub_catalog"

    def __init__(
        self,
        search_tab: QWidget,
        ub_catalog_tab: QWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.search_tab = search_tab
        self.ub_catalog_tab = ub_catalog_tab

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Source-picker bar
        top_bar = QHBoxLayout()
        source_label = QLabel("🔍 Quelle:")
        source_label.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(source_label)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(220)
        self.source_combo.addItem(
            "GND / SWB / Lobid (Schlagwort-Recherche)", self.SOURCE_GND
        )
        self.source_combo.addItem(
            "UB-Katalog / DK (Bibliotheksbestand)", self.SOURCE_UB_CATALOG
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        top_bar.addWidget(self.source_combo)
        top_bar.addStretch(1)
        outer.addLayout(top_bar)

        # Stacked panels — keep both widgets alive, swap visibility only.
        self.stack = QStackedWidget()
        self.stack.addWidget(self.search_tab)       # idx 0 → SOURCE_GND
        self.stack.addWidget(self.ub_catalog_tab)   # idx 1 → SOURCE_UB_CATALOG
        outer.addWidget(self.stack, stretch=1)

    # --------------------------------------------------------------
    def _on_source_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        data = self.source_combo.currentData()
        self.logger.debug(f"SearchTabUnified switched to source: {data}")

    def set_source(self, source: str) -> None:
        """Programmatically select a source (``"gnd"`` or ``"ub_catalog"``)."""
        for idx in range(self.source_combo.count()):
            if self.source_combo.itemData(idx) == source:
                self.source_combo.setCurrentIndex(idx)
                return
