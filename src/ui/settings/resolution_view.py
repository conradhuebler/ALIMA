"""The table that says which model runs where, and why.

It is the first thing on the "Sprachmodelle" page on purpose: it is not an input
but a *consequence*. Four settings can decide the model, and which one applied
used to be invisible — "ich stelle X ein und bekomme Y" had no single cause and
no surface that could answer it.

Deliberately no reachability probe on open: four providers is four network calls
before the dialog is usable, and a dead endpoint hangs it. The table shows the
configured resolution; the button next to it checks reachability on demand.
- Claude Generated
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .effective_models import effective_rows


class ResolutionView(QGroupBox):
    """Four rows: general, pipeline, agentic, vision."""

    HEADERS = ("Wofür", "Wirksames Modell", "Woher")

    #: Emitted when the user asks for a reachability check.
    check_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Was wofür benutzt wird", parent)
        box = QVBoxLayout(self)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        box.addWidget(self.table)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray; font-size: 10px;")
        box.addWidget(self.note)

    # -- content ---------------------------------------------------------

    def refresh(self, unified_config: Any) -> None:
        """Rebuild from the configuration. Never touches the network."""
        try:
            rows = effective_rows(unified_config)
        except Exception:
            rows = []

        self.table.setRowCount(len(rows))
        notes = []
        muted = self.palette().color(QPalette.ColorRole.PlaceholderText)

        for r, row in enumerate(rows):
            model = f"{row.provider} | {row.model}" if row.configured else "—"
            if row.configured and not row.model:
                model = f"{row.provider} | (kein Modell)"
            source = row.source_label
            if row.is_fallback:
                source = f"{source} (geerbt)"

            for c, text in enumerate((row.task, model, source)):
                item = QTableWidgetItem(text)
                if not row.configured or (c == 2 and row.is_fallback):
                    item.setForeground(QBrush(muted))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(r, c, item)

            if row.note:
                notes.append(row.note)

        self.table.resizeRowsToContents()
        self.note.setText("  ".join(notes))
        self.note.setVisible(bool(notes))

    def row_texts(self) -> list:
        """The rendered rows, for tests and for a quick eyeball. - Claude Generated"""
        out = []
        for r in range(self.table.rowCount()):
            out.append(tuple(
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ))
        return out
