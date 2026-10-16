"""The workflows an Erschließung can run, listed and editable in place.

The setting used to be a single combo box of file stems — "alima_v51_105
(v5.1.105)" — which named neither what the workflow does nor where it lives, and
offered no way to look inside it. The workflows are the substance of an
Erschließung; the file name is an implementation detail.

Each workflow names itself (``name:``) and describes itself (``description:``)
in its own YAML. That is what the list shows. Double-click opens the existing
``WorkflowEditorDialog`` on that file, the same editor the menu opens.

The classic pipeline is listed too although it is not a file. It is what runs
when nothing agentic is chosen, so leaving it out would make the list read as if
there were no such option. Its row carries no version, no YAML, and
(``editable=False``) no "Bearbeiten" action — the per-step provider/model
override dialog that used to live behind it (Smart/Advanced/Expert,
``PipelineConfigDialog``) was removed Oct 2026: the classic pipeline runs
every step on the one pipeline default already set on the Sprachmodelle page,
and a second, dialog-shaped place to set the same value was one picker too
many.

Shipped workflows live in the project's ``workflows/`` directory and are not
written in place: the editor's "Speichern" puts a copy under
``~/.config/alima/workflows/``, which the loader searches as well. The "Ablage"
column says which of the two a row is, because it decides what editing does.
- Claude Generated
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
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

#: The value ``SystemConfig.default_workflow`` carries for the non-agentic path.
CLASSIC_STEM = "__classic__"

#: Cap on the description line. The longest in the shipped set has 877
#: characters and would otherwise be the tallest thing on the page.
SUMMARY_LIMIT = 180


@dataclass
class WorkflowEntry:
    """One row: what it is called, what it does, where it lives."""

    stem: str
    name: str
    version: str = ""
    steps: int = 0
    location: str = ""
    summary: str = ""
    editable: bool = True


CLASSIC_ENTRY = WorkflowEntry(
    stem=CLASSIC_STEM,
    name="Klassische Pipeline (nicht agentisch)",
    steps=5,
    location="eingebaut",
    summary="Fünf feste Schritte ohne Agenten: Eingabe, Initialisierung, "
            "Suche, Schlagworte, Klassifikation. Läuft auf dem Pipeline-"
            "Standard (Sprachmodelle-Seite) — kein eigener Dialog.",
    editable=False,
)


def summarise(workflow: Any) -> str:
    """Version, step count and the first sentence of the description.

    Only the first sentence: the descriptions are prose of up to 877 characters,
    and a wrapping line that long would be the tallest element of the page.
    - Claude Generated
    """
    steps = len(getattr(workflow, "steps", []) or [])
    head = f"v{getattr(workflow, 'version', '?')} · " + (
        "1 Schritt" if steps == 1 else f"{steps} Schritte"
    )
    text = " ".join((getattr(workflow, "description", "") or "").split())
    if not text:
        return head

    if len(text) > SUMMARY_LIMIT:
        stop = text.find(". ")
        if 0 < stop + 1 <= SUMMARY_LIMIT:
            text = text[: stop + 1]
        else:
            text = text[:SUMMARY_LIMIT].rsplit(" ", 1)[0] + " …"
    return f"{head} · {text}"


def _user_dir() -> Path:
    return Path.home() / ".config" / "alima" / "workflows"


def workflow_entries(*, include_classic: bool = True,
                     logger: Any = None) -> List[WorkflowEntry]:
    """Every readable v4 workflow, in the order the loader finds them.

    A file that does not parse is skipped rather than shown as broken: the
    settings dialog must open on any workflow directory. ``strict=False`` keeps
    a workflow readable whose step type is registered by a plugin later.
    - Claude Generated
    """
    from ...core.agents.workflow_loader import discover_workflow_files, load_workflow
    from ...utils.error_visibility import log_caught

    entries: List[WorkflowEntry] = [CLASSIC_ENTRY] if include_classic else []
    user_dir = _user_dir()
    seen: set = set()
    for path in discover_workflow_files():
        if path.stem in seen:
            continue
        seen.add(path.stem)
        try:
            workflow = load_workflow(path, strict=False)
        except Exception as exc:
            # debug: a v3 YAML in the directory is not a fault, it is simply
            # not one of the workflows selectable here. log_caught still raises
            # defect-shaped errors to ERROR. - Claude Generated
            log_caught(logger, exc, f"Workflow {path.name} nicht lesbar",
                       expected_level="debug")
            continue
        try:
            own = path.parent.resolve() == user_dir.resolve()
        except OSError:
            own = False
        entries.append(WorkflowEntry(
            stem=path.stem,
            name=workflow.name,
            version=workflow.version,
            steps=len(workflow.steps),
            location="Benutzer" if own else "Projekt",
            summary=summarise(workflow),
        ))
    return entries


class WorkflowList(QWidget):
    """The workflows, all visible. Double-click edits, ⭐ marks the default."""

    COLUMNS = ("", "Workflow", "Version", "Schritte", "Ablage")

    #: Beyond this the list scrolls instead of the page.
    MAX_VISIBLE_ROWS = 10

    default_changed = pyqtSignal(str)
    edit_requested = pyqtSignal(str)
    new_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._default = ""
        self._entries: List[WorkflowEntry] = []

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)

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
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        box.addWidget(self.list)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray;")
        self.summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        box.addWidget(self.summary)

        row = QHBoxLayout()
        self.default_button = QPushButton("⭐ Als Standard")
        self.default_button.setToolTip(
            "Dieser Workflow läuft, wenn im Pipeline-Tab oder per CLI keiner "
            "gewählt wird (dasselbe Feld wie der ⭐ im Pipeline-Tab)."
        )
        self.default_button.clicked.connect(self._make_default)
        self.edit_button = QPushButton("Bearbeiten…")
        self.edit_button.clicked.connect(self._edit_selected)
        self.new_button = QPushButton("➕ Neu…")
        self.new_button.clicked.connect(self.new_requested)
        for button in (self.default_button, self.edit_button, self.new_button):
            row.addWidget(button)
        row.addStretch()
        box.addLayout(row)

        self.list.currentItemChanged.connect(lambda *_: self._on_selection())
        self._on_selection()

    # -- content ---------------------------------------------------------

    def refresh(self, entries: List[WorkflowEntry], *, default_stem: str) -> None:
        selected = self.selected_stem()
        self._entries = list(entries)
        self._default = default_stem or ""
        self.list.clear()

        bold = QFont()
        bold.setBold(True)
        # Two files may carry the same ``name:``; then the file name decides and
        # has to be visible. - Claude Generated
        taken: dict = {}
        for entry in self._entries:
            taken[entry.name] = taken.get(entry.name, 0) + 1

        for entry in self._entries:
            label = entry.name
            if taken[label] > 1:
                label = f"{label} [{entry.stem}]"
            item = QTreeWidgetItem([
                "⭐" if entry.stem == self._default else "",
                label,
                f"v{entry.version}" if entry.version else "",
                str(entry.steps) if entry.steps else "",
                entry.location,
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.stem)
            item.setTextAlignment(
                3, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            item.setToolTip(1, entry.summary)
            if entry.stem == self._default:
                item.setFont(1, bold)
            self.list.addTopLevelItem(item)

        self._fit_height()
        self.select(selected or self._default)
        self._on_selection()

    def _fit_height(self) -> None:
        rows = max(self.list.topLevelItemCount(), 1)
        row_height = max(self.list.sizeHintForRow(0), 20)
        header = self.list.header()
        total = (
            min(rows, self.MAX_VISIBLE_ROWS) * row_height
            + max(header.height(), header.sizeHint().height())
            + 2 * self.list.frameWidth()
            + 4
        )
        self.list.setMinimumHeight(total)
        self.list.setMaximumHeight(total)

    # -- selection -------------------------------------------------------

    def selected_stem(self) -> str:
        item = self.list.currentItem()
        return (item.data(0, Qt.ItemDataRole.UserRole) or "") if item else ""

    def select(self, stem: str) -> None:
        for row in range(self.list.topLevelItemCount()):
            item = self.list.topLevelItem(row)
            if item.data(0, Qt.ItemDataRole.UserRole) == stem:
                self.list.setCurrentItem(item)
                return

    def default_stem(self) -> str:
        """What the dialog writes to ``SystemConfig.default_workflow``."""
        return self._default

    def entry(self, stem: str) -> Optional[WorkflowEntry]:
        for entry in self._entries:
            if entry.stem == stem:
                return entry
        return None

    def row_texts(self) -> List[tuple]:
        return [
            tuple(self.list.topLevelItem(r).text(c) for c in range(len(self.COLUMNS)))
            for r in range(self.list.topLevelItemCount())
        ]

    # -- actions ---------------------------------------------------------

    def _on_selection(self) -> None:
        entry = self.entry(self.selected_stem())
        self.summary.setText(entry.summary if entry else "")
        self.edit_button.setEnabled(bool(entry and entry.editable))
        self.default_button.setEnabled(
            bool(entry) and entry.stem != self._default
        )

    def _make_default(self) -> None:
        stem = self.selected_stem()
        if not stem or stem == self._default:
            return
        self.refresh(self._entries, default_stem=stem)
        self.default_changed.emit(stem)

    def _edit_selected(self) -> None:
        entry = self.entry(self.selected_stem())
        if entry and entry.editable:
            self.edit_requested.emit(entry.stem)

    def _on_double_click(self, item, _column: int = 0) -> None:
        stem = item.data(0, Qt.ItemDataRole.UserRole)
        entry = self.entry(stem)
        if entry and entry.editable:
            self.edit_requested.emit(stem)
