"""The table that says which model runs where, and why.

It is the first thing on the "Sprachmodelle" page, and it is both the answer and
the control: each row shows what is in effect, where that comes from, and lets
you change it. Four settings can decide the model, and which one applied used to
be invisible — "ich stelle X ein und bekomme Y" had no single cause and no
surface that could answer it. Showing the resolution in a table *and* offering
the same four values again as separate pickers below would just double it.

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
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .effective_models import effective_rows


class ResolutionView(QGroupBox):
    """Four rows: general, pipeline, agentic, vision."""

    HEADERS = ("Wofür", "Anbieter", "Modell", "Woher", "Ändern")

    #: (scope, provider, model) — the host persists it.
    selection_changed = pyqtSignal(str, str, str)

    #: Scopes in the order they are shown, with the caption of their picker.
    EMPTY_LABELS = {
        "general": None,  # always concrete: everything else falls back to it
        "pipeline": "(erbt den allgemeinen)",
        "agentic": "(erbt den allgemeinen)",
        "vision": "(erbt den allgemeinen)",
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Was wofür benutzt wird", parent)
        box = QVBoxLayout(self)
        self._selectors: dict = {}
        self._provider_names: list = []
        self._loading = False

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        # Model names vary wildly in length (e.g. "a1" vs. a full Ollama tag
        # like "deepseek-v4-flash:cloud"), so ResizeToContents here made the
        # whole table visibly resize every time a different model was
        # selected - correct behaviour for that resize mode, wrong column for
        # it. A fixed width + ellipsis (default item behaviour) + a tooltip
        # for the full name keeps every column stable regardless of which
        # model is in effect. - Claude Generated
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(2, 200)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setShowGrid(False)
        box.addWidget(self.table)

        # Nothing else goes in here. A wrapping note label below the table was
        # the one element that grew when the dialog was resized, and it grew
        # into empty space on a page whose content is fixed-height. - Claude Generated
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    # -- content ---------------------------------------------------------

    def refresh(self, unified_config: Any) -> None:
        """Rebuild from the configuration. Never touches the network.

        The provider list for the pickers is derived here rather than pushed in
        from outside: the pickers are created on first refresh, so a caller that
        set the providers beforehand set them on nothing and the provider combo
        came up empty while the model combo worked. - Claude Generated
        """
        try:
            rows = effective_rows(unified_config)
        except Exception:
            rows = []

        try:
            self._provider_names = [
                p.name for p in unified_config.get_enabled_providers()
            ]
        except Exception:
            self._provider_names = []

        self.table.setRowCount(len(rows))
        muted = self.palette().color(QPalette.ColorRole.PlaceholderText)

        self._loading = True
        for r, row in enumerate(rows):
            provider = row.provider if row.configured else "—"
            model = (row.model or "(kein Modell)") if row.configured else ""
            source = row.source_label
            if row.is_fallback:
                source = f"{source} (geerbt)"

            for c, text in enumerate((row.task, provider, model, source)):
                item = QTableWidgetItem(text)
                if not row.configured or (c == 3 and row.is_fallback):
                    item.setForeground(QBrush(muted))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if c == 2:
                    # The model column has a fixed width now, so a long name
                    # is elided (default item behaviour) - the tooltip is
                    # where the full value stays reachable. - Claude Generated
                    item.setToolTip(text)
                self.table.setItem(r, c, item)

            self.table.setCellWidget(r, 4, self._selector_for(row, unified_config))
        self._loading = False

        self._fit_height()

    def _fit_height(self) -> None:
        """Uniform rows that fit their picker, and no scrollbar - Claude Generated.

        ``resizeRowsToContents`` measures the text cells and ignores the widget in
        the last column, which produced rows of 35/35/35/26 px with a 25 px combo
        squeezed into the last one. The picker decides the height instead.
        A four-row table that then scrolls is the worst of both: these rows are
        the content, not a sample of it.
        """
        needed = 0
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 4)
            if widget is not None:
                needed = max(needed, widget.sizeHint().height(),
                             widget.minimumSizeHint().height())
        row_height = max(needed + 8, 30)
        for r in range(self.table.rowCount()):
            self.table.setRowHeight(r, row_height)

        # The header reports 0 before the first layout pass, which made the table
        # one header shorter than its rows and clipped the last picker. Fall back
        # to its size hint, and leave room for a horizontal scrollbar. - Claude Generated
        header = self.table.horizontalHeader()
        header_height = max(header.height(), header.sizeHint().height())
        total = (
            header_height
            + 2 * self.table.frameWidth()
            + row_height * self.table.rowCount()
            + self.table.horizontalScrollBar().sizeHint().height()
        )
        self.table.setMinimumHeight(total)
        self.table.setMaximumHeight(total)

    def _selector_for(self, row, unified_config):
        """The picker that sets this row's own default - Claude Generated.

        ``general`` has no empty option: everything else falls back to it, so it
        must stay concrete. The others may be left empty and then inherit, which
        the "Woher" column reports as such.
        """
        from ..provider_model_selector import ProviderModelSelector

        empty_label = self.EMPTY_LABELS.get(row.scope)
        selector = self._selectors.get(row.scope)
        if selector is None:
            selector = ProviderModelSelector(
                allow_empty=empty_label is not None,
                empty_provider_label=empty_label or "",
                # Non-editable: an editable QComboBox sizes itself from its
                # current text, not from the full item list, so the column
                # visibly resized every time a different (shorter/longer)
                # model name was picked. A plain dropdown's size hint comes
                # from the widest item in the list, which doesn't change
                # when the selection does. The exact configured model is
                # shown regardless, in the separate "Modell" text column.
                # - Claude Generated
                editable_model=False,
            )
            selector.selectionChanged.connect(
                lambda p, m, scope=row.scope: self._on_pick(scope, p, m)
            )
            self._selectors[row.scope] = selector

        selector.set_providers(list(self._provider_names), refresh=False)
        own_provider, own_model = self._own_choice(row.scope, unified_config)
        selector.set_selection(own_provider, own_model)
        return selector

    @staticmethod
    def _own_choice(scope: str, unified_config):
        """What this scope has set *itself*, not what it inherits."""
        attr = {
            "general": ("preferred_provider", "preferred_model"),
            "pipeline": ("pipeline_default_provider", "pipeline_default_model"),
            "agentic": ("agentic_default_provider", "agentic_default_model"),
            "vision": ("vision_default_provider", "vision_default_model"),
        }[scope]
        return (
            getattr(unified_config, attr[0], "") or "",
            getattr(unified_config, attr[1], "") or "",
        )

    def _on_pick(self, scope: str, provider: str, model: str) -> None:
        if self._loading:
            return
        self.selection_changed.emit(scope, provider or "", model or "")

    def row_texts(self) -> list:
        """The rendered rows, for tests and for a quick eyeball. - Claude Generated"""
        out = []
        for r in range(self.table.rowCount()):
            out.append(tuple(
                self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(self.table.columnCount())
            ))
        return out
