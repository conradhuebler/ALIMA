"""The settings dialog frame: one sidebar, five pages, no nested navigation.

A tab bar invites a second row when it runs out of space, and that is exactly
what happened here before. A sidebar does not: there is room above it for the
search field, below it for the way into the setup wizard, and inside it for a
count per page. - Claude Generated
"""

from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,

    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .search import SettingsIndex

class SettingsShell(QWidget):
    """Sidebar + page stack + search + footer."""

    page_changed = pyqtSignal(int)

    def __init__(self, index: SettingsIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._pages: List[Tuple[str, QWidget]] = []

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Einstellung suchen…")
        self.search_field.setClearButtonEnabled(True)
        self.search_field.textChanged.connect(self._on_search)

        self.nav = QListWidget()
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setMaximumWidth(230)
        self.nav.currentRowChanged.connect(self._on_nav)

        self.results = QListWidget()
        self.results.setVisible(False)
        self.results.setMaximumWidth(230)
        self.results.itemActivated.connect(self._on_result)
        self.results.itemClicked.connect(self._on_result)

        self.stack = QStackedWidget()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.search_field)
        left_layout.addWidget(self.nav, 1)
        left_layout.addWidget(self.results, 1)

        self.sidebar_footer = QVBoxLayout()
        left_layout.addLayout(self.sidebar_footer)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        self.footer = QLabel("")
        self.footer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.footer.setStyleSheet("color: gray;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter, 1)
        outer.addWidget(self.footer)

    # -- pages -----------------------------------------------------------

    def add_page(self, title: str, widget: QWidget, *, scrollable: bool = True) -> QWidget:
        """Register a page. Scrolling inside a page replaces nesting."""
        if scrollable:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setWidget(widget)
            shown: QWidget = area
        else:
            shown = widget
        self._pages.append((title, widget))
        self.stack.addWidget(shown)
        QListWidgetItem(title, self.nav)
        if self.nav.currentRow() < 0:
            self.nav.setCurrentRow(0)
        return widget

    def set_page_badge(self, title: str, badge: str) -> None:
        """A count next to a page name, e.g. "Datenquellen  12/16"."""
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            if item.text().split("  ")[0] == title:
                item.setText(f"{title}  {badge}" if badge else title)
                return

    def show_page(self, title: str) -> None:
        for row, (name, _w) in enumerate(self._pages):
            if name == title:
                self.nav.setCurrentRow(row)
                return

    def set_footer(self, text: str) -> None:
        self.footer.setText(text)

    # -- search ----------------------------------------------------------

    def _on_search(self, text: str) -> None:
        hits = self._index.search(text)
        self.results.clear()
        showing = bool(text.strip())
        self.results.setVisible(showing)
        self.nav.setVisible(not showing)
        if not showing:
            return
        if not hits:
            item = QListWidgetItem("keine Treffer")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.results.addItem(item)
            return
        for entry in hits:
            item = QListWidgetItem(f"{entry.label}\n{entry.path()}")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.results.addItem(item)

    def _on_result(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        if entry is None:
            return
        self.show_page(entry.page)
        if entry.reveal is not None:
            entry.reveal()
        widget = getattr(entry, "widget", None)
        if widget is not None:
            try:
                widget.setFocus(Qt.FocusReason.OtherFocusReason)
            except Exception:
                pass

    def _on_nav(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            self.page_changed.emit(row)
