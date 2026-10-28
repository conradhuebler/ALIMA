"""The configured providers as a plain list.

An accordion was tried first and was wrong: five entries that one wants to see at
once do not benefit from folding, and the fold hid mostly static text behind an
interaction. A list shows every provider at a glance; double-click edits, the
same as everywhere else.

Per-model chunking thresholds used to be edited here. They are still read —
``model_capabilities.get_chunking_threshold`` consults them between an explicit
call argument and the pattern match on the model name — but they are an escape
hatch for a model the automatic detection gets wrong, and no installation has
ever needed one. The escape hatch stays reachable by hand in ``config.json``
(the dialog footer names the path); it no longer costs a section here.
- Claude Generated
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ..styles import set_tone

def _endpoint(provider: Any) -> str:
    url = getattr(provider, "base_url", "") or ""
    if url:
        return url
    host = getattr(provider, "host", "") or ""
    port = getattr(provider, "port", None)
    return f"{host}:{port}" if host else "(Vorgabe)"


def _needs_key(provider: Any) -> bool:
    return (
        getattr(provider, "provider_type", "") in ("gemini", "anthropic")
        and not getattr(provider, "api_key", "")
    )


class ProviderList(QWidget):
    """The providers, all visible. Double-click edits."""

    COLUMNS = ("Aktiv / Name", "Typ", "Endpunkt", "Modelle", "Zustand")

    edit_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    models_requested = pyqtSignal(str)
    add_requested = pyqtSignal()
    enabled_toggled = pyqtSignal(str, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

        # A tree with one level: a list that has real columns. Joining the fields
        # with separators into one string looked like formatting and was not —
        # nothing lined up. - Claude Generated
        self.list = QTreeWidget()
        self.list.setColumnCount(len(self.COLUMNS))
        self.list.setHeaderLabels(list(self.COLUMNS))
        self.list.setRootIsDecorated(False)
        self.list.setAlternatingRowColors(True)
        self.list.setUniformRowHeights(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.list.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.itemChanged.connect(self._on_item_changed)
        self._populating = False
        self.list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        box.addWidget(self.list)

        hint = QLabel("Häkchen schaltet einen Anbieter ein oder aus. Doppelklick bearbeitet ihn.")
        set_tone(hint, "muted")
        hint.setStyleSheet("font-size: 10px;")
        box.addWidget(hint)

        row = QHBoxLayout()
        self.add_button = QPushButton("➕ Hinzufügen…")
        self.add_button.clicked.connect(self.add_requested)
        self.edit_button = QPushButton("Bearbeiten…")
        self.edit_button.clicked.connect(lambda: self._emit_for_selection(self.edit_requested))
        self.remove_button = QPushButton("Entfernen")
        self.remove_button.clicked.connect(lambda: self._emit_for_selection(self.remove_requested))
        self.models_button = QPushButton("Modelle abfragen")
        self.models_button.setToolTip(
            "Fragt die Modellliste dieses einen Anbieters neu ab."
        )
        self.models_button.clicked.connect(lambda: self._emit_for_selection(self.models_requested))
        for button in (self.add_button, self.edit_button,
                       self.remove_button, self.models_button):
            row.addWidget(button)
        row.addStretch()
        box.addLayout(row)

        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())
        self._update_buttons()

    # -- content ---------------------------------------------------------

    def refresh(self, providers: List[Any], *,
                models: Dict[str, List[str]],
                suggestions: Optional[List[Any]] = None) -> None:
        """``suggestions``: vorkonfigurierte, noch nicht übernommene Anbieter.

        Sie erscheinen abgehakt-aus am Ende; das Anhaken meldet sich über
        ``enabled_toggled`` wie bei jedem anderen Eintrag. - Claude Generated
        """
        selected = self.selected_name()
        self._populating = True
        try:
            self._fill(providers, models, suggestions or [])
        finally:
            self._populating = False
        self._fit_height()
        if selected:
            self.select(selected)

    def _fill(self, providers, models, suggestions) -> None:
        self.list.clear()
        self._models = dict(models or {})
        for provider, is_suggestion in (
            [(p, False) for p in providers] + [(p, True) for p in suggestions]
        ):
            name = getattr(provider, "name", "?")
            known = list(self._models.get(name) or getattr(provider, "available_models", []) or [])
            state = []
            if is_suggestion:
                state.append("vorkonfiguriert")
            elif not getattr(provider, "enabled", True):
                state.append("aus")
            if _needs_key(provider):
                state.append("kein Schlüssel")
            item = QTreeWidgetItem([
                name,
                getattr(provider, "provider_type", ""),
                _endpoint(provider),
                str(len(known)),
                ", ".join(state),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                0,
                Qt.CheckState.Checked
                if (getattr(provider, "enabled", True) and not is_suggestion)
                else Qt.CheckState.Unchecked,
            )
            item.setTextAlignment(3, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if state:
                muted = self.palette().placeholderText()
                for c in range(len(self.COLUMNS)):
                    item.setForeground(c, muted)
            self.list.addTopLevelItem(item)

        if not providers:
            empty = QTreeWidgetItem([
                "Noch kein Anbieter — ohne einen kann ALIMA kein Sprachmodell aufrufen.",
                "", "", "", "",
            ])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addTopLevelItem(empty)

    def _on_item_changed(self, item, column: int) -> None:
        if self._populating or column != 0:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name:
            self.enabled_toggled.emit(
                name, item.checkState(0) == Qt.CheckState.Checked
            )

    def _fit_height(self) -> None:
        """Show every provider without scrolling; they are few and all relevant."""
        rows = max(self.list.topLevelItemCount(), 1)
        row_height = max(self.list.sizeHintForRow(0), 20) if rows else 20
        header = self.list.header()
        total = (
            rows * row_height
            + max(header.height(), header.sizeHint().height())
            + 2 * self.list.frameWidth()
            + 4
        )
        self.list.setMinimumHeight(total)
        self.list.setMaximumHeight(total)

    # -- selection -------------------------------------------------------

    def selected_name(self) -> str:
        item = self.list.currentItem()
        return (item.data(0, Qt.ItemDataRole.UserRole) or "") if item else ""

    def select(self, name: str) -> None:
        for row in range(self.list.topLevelItemCount()):
            item = self.list.topLevelItem(row)
            if item.data(0, Qt.ItemDataRole.UserRole) == name:
                self.list.setCurrentItem(item)
                return

    def names(self) -> List[str]:
        out = []
        for row in range(self.list.topLevelItemCount()):
            name = self.list.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole)
            if name:
                out.append(name)
        return out

    def row_texts(self) -> List[tuple]:
        return [
            tuple(
                self.list.topLevelItem(r).text(c) for c in range(len(self.COLUMNS))
            )
            for r in range(self.list.topLevelItemCount())
        ]

    def _emit_for_selection(self, signal) -> None:
        name = self.selected_name()
        if name:
            signal.emit(name)

    def _on_double_click(self, item, _column: int = 0) -> None:
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name:
            self.edit_requested.emit(name)

    def _update_buttons(self, *_args) -> None:
        has = bool(self.selected_name())
        for button in (self.edit_button, self.remove_button, self.models_button):
            button.setEnabled(has)

    # -- model limits ----------------------------------------------------

