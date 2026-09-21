"""A named, foldable section - Claude Generated."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """A section whose title stays visible when its body is folded away.

    The title says what is inside ("Modellgrenzen", "Feinheiten der Verbindung"),
    never "Erweitert": a folded section is only acceptable because the search
    field can reach into it, and a generic title defeats that.

    ``subtitle`` carries a count or state so a folded section can still report
    what it holds ("1 Abweichung", "2 freigegeben").
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, *, subtitle: str = "", expanded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title

        self._button = QToolButton()
        self._button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.clicked.connect(self._on_clicked)

        self._body = QFrame()
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self._body.setVisible(expanded)

        # Take only the height actually needed. Without this a folded section
        # keeps the vertical stretch its body asked for, which is what made the
        # provider cards balloon and run into each other. - Claude Generated
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(self._button)
        outer.addWidget(self._body)

        self.set_subtitle(subtitle)

    # -- content ---------------------------------------------------------

    def body(self) -> QFrame:
        """The frame to put a layout on."""
        return self._body

    def set_subtitle(self, subtitle: str) -> None:
        """State a folded section can still show, e.g. a count."""
        self._subtitle = subtitle or ""
        self._button.setText(
            f"{self._title}  ({self._subtitle})" if self._subtitle else self._title
        )

    # -- folding ---------------------------------------------------------

    def is_expanded(self) -> bool:
        return self._button.isChecked()

    def expand(self) -> None:
        """Open it. Used by the search when a hit lies inside."""
        if not self._button.isChecked():
            self._button.setChecked(True)
            self._on_clicked(True)

    def _on_clicked(self, checked: bool) -> None:
        self._button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(checked)
        self.updateGeometry()
        self.toggled.emit(checked)
