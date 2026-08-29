"""Workflow YAML Editor Dialog - Claude Generated.

Structured form editor for the agentic v4 workflow YAML files
(``workflows/*.yaml``). It lets the operator view, edit and create workflows
from the Qt6 GUI without touching the workflow engine.

Design constraints (see docs / CLAUDE.md):
    * The workflow engine (loader, executor, registry, steps) is consumed
      read-only — nothing here mutates it.
    * Round-trip via ``ruamel.yaml`` preserves the comments / section headers
      in the shipped YAML files. Only fields the user actually edits are
      mutated in place; untouched keys (and their comments) survive.
    * Before a file is written it is validated with the *real* execution
      loader ``load_workflow(path, strict=True)``. A workflow that the engine
      cannot load is never written — this is exactly what "save so it can be
      loaded" means (``pipeline_manager`` calls ``find_workflow_file`` +
      ``load_workflow(strict=True)`` at execution time).
    * Saved files go to the user override dir ``~/.config/alima/workflows/``
      which is already part of ``DEFAULT_SEARCH_PATHS`` → auto-discovered.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml as _pyyaml

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .styles import get_scaled_font

logger = logging.getLogger(__name__)

# ruamel is imported lazily so the rest of the app never hard-depends on it.
try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarstring import LiteralScalarString

    _RUAMEL_OK = True
except Exception:  # noqa: BLE001
    _RUAMEL_OK = False


_NEW_TEMPLATE = """\
name: "Neuer Workflow"
version: "4.0"
description: ""

settings:
  temperature: 0.5
  max_tokens: 32768

steps:
  - id: extraction
    type: llm_agent
    description: ""
    enabled: true
    inputs:
      abstract: "${abstract}"
    outputs:
      extracted_keywords: "response.keywords"
    system_prompt: |
      Du bist ein hilfreicher Assistent.
    user_prompt: |
      {abstract}
"""


# ──────────────────────────────────────────────────────────────────────────
# Scalar <-> cell-text helpers (table values are edited as YAML text)
# ──────────────────────────────────────────────────────────────────────────
def _val_to_cell(value: Any) -> str:
    """Render a YAML value as editable cell text - Claude Generated."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _pyyaml.safe_dump(
            value, default_flow_style=True, allow_unicode=True
        ).strip()
    except Exception:  # noqa: BLE001
        return str(value)


def _cell_to_val(text: str) -> Any:
    """Parse cell text back into a YAML value - Claude Generated."""
    text = text.strip()
    if text == "":
        return None  # empty cell == YAML null (matches yaml.safe_load(""))
    try:
        return _pyyaml.safe_load(text)
    except Exception:  # noqa: BLE001
        return text


def _norm(value: Any) -> Any:
    """Canonicalise a value for change-detection (None/"" both → None) - Claude Generated."""
    if value is None or value == "":
        return None
    return value  # ruamel containers subclass list/dict → compare == to plain


def _flowify(value: Any) -> Any:
    """Render assigned lists inline (`[a, b]`) like the shipped files - Claude Generated."""
    if _RUAMEL_OK and isinstance(value, list) and not isinstance(value, CommentedSeq):
        seq = CommentedSeq(value)
        seq.fa.set_flow_style()
        return seq
    return value


def _registries() -> Tuple[list, list]:
    """Return (step_types, tool_fns) from the engine registries - Claude Generated."""
    step_types: list = []
    tool_fns: list = []
    try:
        import src.core.agents  # noqa: F401  (registers built-in step types)
        import src.core.agents.deterministic_functions  # noqa: F401  (registers fns)
        from src.core.agents.registry import list_steps, list_tool_fns

        step_types = list_steps()
        tool_fns = list_tool_fns()
    except Exception as e:  # noqa: BLE001
        logger.warning("Workflow registries unavailable: %s", e)
    if not step_types:
        step_types = ["llm_agent", "deterministic", "reflection"]
    return step_types, tool_fns


# ──────────────────────────────────────────────────────────────────────────
# Reusable key/value table for mapping blocks (inputs/outputs/llm/config/...)
# ──────────────────────────────────────────────────────────────────────────
class _KeyValueTable(QWidget):
    """Two-column (key -> YAML value) editor - Claude Generated."""

    changed = pyqtSignal()

    def __init__(self, key_header: str = "Key", val_header: str = "Wert", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([key_header, val_header])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(80)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Zeile")
        add_btn.clicked.connect(lambda: self._add_row("", ""))
        rm_btn = QPushButton("− Zeile")
        rm_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _add_row(self, key: str, val: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(key))
        self.table.setItem(r, 1, QTableWidgetItem(val))
        self.changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if rows:
            self.changed.emit()

    def load(self, mapping: Any) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                self._add_row(str(k), _val_to_cell(v))
        self.table.blockSignals(False)

    def dump(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in range(self.table.rowCount()):
            key_item = self.table.item(r, 0)
            if key_item is None:
                continue
            key = key_item.text().strip()
            if not key:
                continue
            val_item = self.table.item(r, 1)
            out[key] = _cell_to_val(val_item.text() if val_item else "")
        return out


# ──────────────────────────────────────────────────────────────────────────
# Main dialog
# ──────────────────────────────────────────────────────────────────────────
class WorkflowEditorDialog(QDialog):
    """Structured editor for v4 workflow YAML files - Claude Generated."""

    saved = pyqtSignal(str)  # emits the saved workflow stem

    def __init__(self, parent=None, initial_workflow: Optional[str] = None):
        super().__init__(parent)
        self._doc: Optional[Any] = None  # ruamel CommentedMap
        self._path: Optional[Path] = None
        self._dirty = False
        self._loading = False
        self._suppress_list = False
        self._current: Optional[tuple] = None
        self._yaml = self._make_yaml() if _RUAMEL_OK else None
        self._step_types, self._tool_fns = _registries()

        self._init_ui()

        if not _RUAMEL_OK:
            QMessageBox.critical(
                self,
                "ruamel.yaml fehlt",
                "Der Workflow-Editor benötigt 'ruamel.yaml'.\n\n"
                "Installation:  .venv/bin/pip install ruamel.yaml",
            )
            self._set_loaded(False)
            return

        if initial_workflow:
            self._open_named(initial_workflow)
        else:
            self._set_loaded(False)

    # ----- setup -----------------------------------------------------------
    @staticmethod
    def _make_yaml():
        y = YAML()  # default typ='rt' (round-trip, keeps comments)
        y.preserve_quotes = True
        y.width = 4096  # avoid reflowing long lines
        y.indent(mapping=2, sequence=4, offset=2)  # match shipped workflow style
        # Render None as explicit `null` (ruamel defaults to blank) so a no-op
        # round-trip of the shipped files is zero-diff.
        y.representer.add_representer(
            type(None),
            lambda r, d: r.represent_scalar("tag:yaml.org,2002:null", "null"),
        )
        return y

    def _init_ui(self) -> None:
        self.setWindowTitle("Workflow-Editor")
        self.resize(1100, 760)
        root = QVBoxLayout(self)

        # toolbar
        tb = QHBoxLayout()
        self.new_btn = QPushButton("Neu")
        self.new_btn.clicked.connect(self.new_config)
        self.open_btn = QPushButton("Öffnen…")
        self.open_btn.clicked.connect(self.open_config)
        self.save_btn = QPushButton("Speichern")
        self.save_btn.clicked.connect(self.save_config)
        self.save_as_btn = QPushButton("Speichern unter…")
        self.save_as_btn.clicked.connect(self.save_config_as)
        for b in (self.new_btn, self.open_btn, self.save_btn, self.save_as_btn):
            tb.addWidget(b)
        tb.addStretch()
        self.target_label = QLabel(f"Ziel: {self._user_dir()}")
        self.target_label.setStyleSheet("color: #888;")
        tb.addWidget(self.target_label)
        root.addLayout(tb)

        # splitter: navigation | forms
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # left: vertical split — workflow settings (top) over steps list (bottom)
        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.setMinimumWidth(240)

        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        self.settings_btn = QPushButton("⚙ Workflow-Einstellungen")
        self.settings_btn.setCheckable(True)
        self.settings_btn.clicked.connect(self._show_settings_panel)
        top_l.addWidget(self.settings_btn)
        top_l.addStretch()
        left_split.addWidget(top)

        bottom = QWidget()
        bot_l = QVBoxLayout(bottom)
        bot_l.setContentsMargins(0, 0, 0, 0)
        bot_l.addWidget(QLabel("Steps"))
        self.steps_list = QListWidget()
        self.steps_list.currentRowChanged.connect(self._on_step_row_changed)
        bot_l.addWidget(self.steps_list, 1)

        step_btns = QHBoxLayout()
        self.add_step_btn = QPushButton("+ Step")
        self.add_step_btn.clicked.connect(self._add_step)
        self.rm_step_btn = QPushButton("− Step")
        self.rm_step_btn.clicked.connect(self._remove_step)
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(lambda: self._move_step(-1))
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(lambda: self._move_step(1))
        for b in (self.add_step_btn, self.rm_step_btn, self.up_btn, self.down_btn):
            step_btns.addWidget(b)
        bot_l.addLayout(step_btns)
        left_split.addWidget(bottom)

        left_split.setStretchFactor(0, 0)
        left_split.setStretchFactor(1, 1)
        left_split.setSizes([60, 600])
        splitter.addWidget(left_split)

        # right: stacked forms
        self.stack = QStackedWidget()
        self.placeholder_panel = QLabel(
            "Kein Workflow geladen.\nNutze »Neu« oder »Öffnen…«."
        )
        self.placeholder_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.settings_panel = self._build_settings_form()
        self.step_panel = self._build_step_form()
        self.stack.addWidget(self.placeholder_panel)
        self.stack.addWidget(self.settings_panel)
        self.stack.addWidget(self.step_panel)
        splitter.addWidget(self.stack)
        splitter.setSizes([280, 820])

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50;")
        root.addWidget(self.status_label)

    @staticmethod
    def _group(title: str, widget: QWidget) -> QGroupBox:
        g = QGroupBox(title)
        lay = QVBoxLayout(g)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.addWidget(widget)
        return g

    def _build_settings_form(self) -> QWidget:
        inner = QWidget()
        v = QVBoxLayout(inner)
        form = QFormLayout()
        self.wf_name = QLineEdit()
        self.wf_version = QLineEdit()
        self.wf_name.textChanged.connect(self._mark_dirty)
        self.wf_version.textChanged.connect(self._mark_dirty)
        form.addRow("name", self.wf_name)
        form.addRow("version", self.wf_version)
        v.addLayout(form)

        self.wf_desc = QTextEdit()
        self.wf_desc.setAcceptRichText(False)
        self.wf_desc.setMaximumHeight(80)
        self.wf_desc.textChanged.connect(self._mark_dirty)
        v.addWidget(self._group("description", self.wf_desc))

        self.settings_table = _KeyValueTable("Einstellung", "Wert")
        self.settings_table.changed.connect(self._mark_dirty)
        self.meta_table = _KeyValueTable("Schlüssel", "Wert")
        self.meta_table.changed.connect(self._mark_dirty)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._group("settings", self.settings_table))
        split.addWidget(self._group("meta_agent", self.meta_table))
        split.setSizes([220, 180])
        v.addWidget(split, 1)
        return inner

    def _build_step_form(self) -> QWidget:
        """Tabbed step editor — keeps each concern uncluttered, gives prompts room - Claude Generated."""
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.step_tabs = tabs

        # --- Tab: Allgemein ---
        general = QWidget()
        gform = QFormLayout(general)
        self.s_id = QLineEdit()
        self.s_type = QComboBox()
        self.s_type.setEditable(True)
        self.s_type.addItems(self._step_types)
        self.s_type.currentTextChanged.connect(self._on_type_changed)
        self.s_desc = QLineEdit()
        self.s_enabled = QCheckBox("enabled")
        self.s_depends = QLineEdit()
        self.s_depends.setPlaceholderText("kommagetrennt, z. B. extraction, search")
        self.s_when = QLineEdit()
        self.s_when.setPlaceholderText('z. B. ${extra.dk_prompt_text} != ""')
        for w in (self.s_id, self.s_desc, self.s_depends, self.s_when):
            w.textChanged.connect(self._mark_dirty)
        self.s_enabled.toggled.connect(self._mark_dirty)
        gform.addRow("id", self.s_id)
        gform.addRow("type", self.s_type)
        gform.addRow("description", self.s_desc)
        gform.addRow("", self.s_enabled)
        gform.addRow("depends_on", self.s_depends)
        gform.addRow("when", self.s_when)
        tabs.addTab(general, "Allgemein")

        # --- Tab: Ein-/Ausgaben ---
        self.s_inputs = _KeyValueTable("Input", "Wert/Pfad")
        self.s_outputs = _KeyValueTable("Output", "Ergebnis-Pfad")
        for t in (self.s_inputs, self.s_outputs):
            t.changed.connect(self._mark_dirty)
        io_split = QSplitter(Qt.Orientation.Vertical)
        io_split.addWidget(self._group("inputs", self.s_inputs))
        io_split.addWidget(self._group("outputs", self.s_outputs))
        io_split.setSizes([200, 200])
        tabs.addTab(io_split, "Ein-/Ausgaben")

        # --- Tab: LLM (llm_agent only) ---
        self.s_llm = _KeyValueTable("llm", "Wert")
        self.s_tools = _KeyValueTable("tools", "Wert (explicit/preset)")
        self.s_chunking = _KeyValueTable("chunking", "Wert")
        for t in (self.s_llm, self.s_tools, self.s_chunking):
            t.changed.connect(self._mark_dirty)
        llm_split = QSplitter(Qt.Orientation.Vertical)
        llm_split.addWidget(self._group("llm", self.s_llm))
        llm_split.addWidget(self._group("tools (Keys: explicit, preset)", self.s_tools))
        llm_split.addWidget(self._group("chunking", self.s_chunking))
        tabs.addTab(llm_split, "LLM")

        # --- Tab: Prompts (llm_agent only) ---
        mono = get_scaled_font(monospace=True)
        self.s_sysprompt = QTextEdit()
        self.s_sysprompt.setAcceptRichText(False)
        self.s_sysprompt.setFont(mono)
        self.s_sysprompt.textChanged.connect(self._mark_dirty)
        self.s_userprompt = QTextEdit()
        self.s_userprompt.setAcceptRichText(False)
        self.s_userprompt.setFont(mono)
        self.s_userprompt.textChanged.connect(self._mark_dirty)
        pr_split = QSplitter(Qt.Orientation.Vertical)
        pr_split.addWidget(self._group("system_prompt", self.s_sysprompt))
        pr_split.addWidget(self._group("user_prompt", self.s_userprompt))
        pr_split.setSizes([420, 260])
        tabs.addTab(pr_split, "Prompts")

        # --- Tab: Funktion (deterministic only) ---
        func = QWidget()
        fl = QVBoxLayout(func)
        fl.addWidget(QLabel("function"))
        self.s_function = QComboBox()
        self.s_function.setEditable(True)
        self.s_function.addItem("")
        self.s_function.addItems(self._tool_fns)
        self.s_function.currentTextChanged.connect(self._mark_dirty)
        fl.addWidget(self.s_function)
        self.s_config = _KeyValueTable("config", "Wert")
        self.s_config.changed.connect(self._mark_dirty)
        fl.addWidget(self._group("config", self.s_config), 1)
        tabs.addTab(func, "Funktion")

        self._tab_idx = {
            "llm": tabs.indexOf(llm_split),
            "prompts": tabs.indexOf(pr_split),
            "func": tabs.indexOf(func),
        }
        return tabs

    # ----- dirty / ui state ------------------------------------------------
    def _mark_dirty(self, *_) -> None:
        if self._loading:
            return
        self._dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self._path.name if self._path else "(neu)"
        star = " *" if self._dirty else ""
        self.setWindowTitle(f"Workflow-Editor — {name}{star}")

    def _set_loaded(self, loaded: bool) -> None:
        for b in (
            self.save_btn,
            self.save_as_btn,
            self.add_step_btn,
            self.rm_step_btn,
            self.up_btn,
            self.down_btn,
            self.steps_list,
            self.settings_btn,
        ):
            b.setEnabled(loaded)
        if not loaded:
            self.stack.setCurrentWidget(self.placeholder_panel)

    def _on_type_changed(self, _text: str) -> None:
        self._apply_type_visibility()
        self._mark_dirty()

    def _apply_type_visibility(self) -> None:
        t = self.s_type.currentText().strip()
        is_llm = t == "llm_agent"
        self.step_tabs.setTabVisible(self._tab_idx["llm"], is_llm)
        self.step_tabs.setTabVisible(self._tab_idx["prompts"], is_llm)
        self.step_tabs.setTabVisible(self._tab_idx["func"], t == "deterministic")

    # ----- navigation (settings button + steps list) ----------------------
    def _steps(self) -> list:
        if self._doc is None:
            return []
        steps = self._doc.get("steps")
        return steps if isinstance(steps, list) else []

    def _step_label(self, i: int, step: Any) -> str:
        label = f"{i + 1}. {step.get('id', '?')}  ·  {step.get('type', '?')}"
        if not step.get("enabled", True):
            label += "  (aus)"
        return label

    def _rebuild_steps_list(self) -> None:
        self._suppress_list = True
        self.steps_list.clear()
        for i, step in enumerate(self._steps()):
            self.steps_list.addItem(self._step_label(i, step))
        self._suppress_list = False

    def _refresh_step_labels(self) -> None:
        steps = self._steps()
        for i in range(min(self.steps_list.count(), len(steps))):
            self.steps_list.item(i).setText(self._step_label(i, steps[i]))

    def _highlight_step(self, idx: int) -> None:
        """Select a list row without triggering a (re)commit - Claude Generated."""
        self._suppress_list = True
        self.steps_list.setCurrentRow(idx)
        self._suppress_list = False
        self.settings_btn.setChecked(False)

    def _highlight_settings(self) -> None:
        self._suppress_list = True
        self.steps_list.setCurrentRow(-1)
        self._suppress_list = False
        self.settings_btn.setChecked(True)

    def _show_settings_panel(self) -> None:
        """Settings button handler - Claude Generated."""
        if self._doc is None:
            self.settings_btn.setChecked(False)
            return
        self._commit_current()
        self._refresh_step_labels()
        self._highlight_settings()
        self._current = ("settings",)
        self._show_settings()
        self.stack.setCurrentWidget(self.settings_panel)

    def _on_step_row_changed(self, row: int) -> None:
        if self._suppress_list or self._doc is None or row < 0:
            return
        self._commit_current()
        self._refresh_step_labels()
        self.settings_btn.setChecked(False)
        self._current = ("step", row)
        self._show_step(row)
        self.stack.setCurrentWidget(self.step_panel)

    # ----- populate forms --------------------------------------------------
    def _show_settings(self) -> None:
        self._loading = True
        d = self._doc
        self.wf_name.setText(str(d.get("name", "")))
        self.wf_version.setText(str(d.get("version", "4.0")))
        self.wf_desc.setPlainText(str(d.get("description", "") or ""))
        self.settings_table.load(d.get("settings") if isinstance(d.get("settings"), dict) else {})
        self.meta_table.load(d.get("meta_agent") if isinstance(d.get("meta_agent"), dict) else {})
        self._loading = False

    def _show_step(self, i: int) -> None:
        steps = self._steps()
        if not (0 <= i < len(steps)):
            return
        step = steps[i]
        self._loading = True
        self.s_id.setText(str(step.get("id", "")))
        self.s_type.setCurrentText(str(step.get("type", "llm_agent")))
        self.s_desc.setText(str(step.get("description", "") or ""))
        self.s_enabled.setChecked(bool(step.get("enabled", True)))
        self.s_depends.setText(", ".join(step.get("depends_on", []) or []))
        self.s_when.setText(str(step.get("when", step.get("condition", "")) or ""))
        self.s_inputs.load(step.get("inputs") if isinstance(step.get("inputs"), dict) else {})
        self.s_outputs.load(step.get("outputs") if isinstance(step.get("outputs"), dict) else {})
        self.s_llm.load(step.get("llm") if isinstance(step.get("llm"), dict) else {})
        self.s_tools.load(step.get("tools") if isinstance(step.get("tools"), dict) else {})
        self.s_chunking.load(step.get("chunking") if isinstance(step.get("chunking"), dict) else {})
        self.s_function.setCurrentText(str(step.get("function", "") or ""))
        self.s_config.load(step.get("config") if isinstance(step.get("config"), dict) else {})
        self.s_sysprompt.setPlainText(str(step.get("system_prompt", "") or ""))
        self.s_userprompt.setPlainText(str(step.get("user_prompt", "") or ""))
        self._loading = False
        self._apply_type_visibility()

    # ----- commit forms into the round-trip doc ----------------------------
    def _commit_current(self) -> None:
        if self._doc is None or self._current is None:
            return
        if self._current[0] == "settings":
            self._commit_settings()
        elif self._current[0] == "step":
            self._commit_step(self._current[1])

    def _commit_settings(self) -> None:
        d = self._doc
        self._set_scalar(d, "name", self.wf_name.text().strip() or "Workflow")
        self._set_scalar(d, "version", self.wf_version.text().strip() or "4.0")
        self._set_prompt(d, "description", self.wf_desc.toPlainText())
        self._merge_map(d, "settings", self.settings_table.dump())
        self._merge_map(d, "meta_agent", self.meta_table.dump())

    def _commit_step(self, i: int) -> None:
        steps = self._steps()
        if not (0 <= i < len(steps)):
            return
        step = steps[i]
        self._set_scalar(step, "id", self.s_id.text().strip() or step.get("id", "step"))
        self._set_scalar(step, "type", self.s_type.currentText().strip() or "llm_agent")
        self._set_or_del(step, "description", self.s_desc.text())
        if bool(step.get("enabled", True)) != self.s_enabled.isChecked():
            step["enabled"] = self.s_enabled.isChecked()
        deps = [d.strip() for d in self.s_depends.text().split(",") if d.strip()]
        if deps != list(step.get("depends_on", []) or []):  # unchanged → keep style
            if deps:
                seq = CommentedSeq(deps)
                seq.fa.set_flow_style()  # match the conventional inline `[a, b]`
                step["depends_on"] = seq
            else:
                step.pop("depends_on", None)
        self._set_or_del(step, "when", self.s_when.text())
        step.pop("condition", None)  # normalise to `when:`
        self._merge_map(step, "inputs", self.s_inputs.dump())
        self._merge_map(step, "outputs", self.s_outputs.dump())

        stype = step["type"]
        if stype == "llm_agent":
            self._merge_map(step, "llm", self.s_llm.dump())
            self._merge_map(step, "tools", self.s_tools.dump())
            self._merge_map(step, "chunking", self.s_chunking.dump())
            self._set_prompt(step, "system_prompt", self.s_sysprompt.toPlainText())
            self._set_prompt(step, "user_prompt", self.s_userprompt.toPlainText())
            for k in ("function", "config"):
                step.pop(k, None)
        elif stype == "deterministic":
            self._set_or_del(step, "function", self.s_function.currentText())
            self._merge_map(step, "config", self.s_config.dump())
            for k in ("llm", "tools", "chunking", "system_prompt", "user_prompt"):
                step.pop(k, None)

    @staticmethod
    def _set_scalar(mapping: Any, key: str, value: Any) -> None:
        """Assign only if changed — preserves ruamel's original style/quotes - Claude Generated."""
        if key in mapping and str(mapping.get(key)) == str(value):
            return
        mapping[key] = value

    def _set_or_del(self, mapping: Any, key: str, value: str) -> None:
        if value is None or str(value).strip() == "":
            mapping.pop(key, None)
        else:
            self._set_scalar(mapping, key, value)

    def _set_prompt(self, step: Any, key: str, text: str) -> None:
        text = text.rstrip("\n")
        if text.strip() == "":
            step.pop(key, None)
            return
        if key in step and str(step.get(key)).rstrip("\n") == text:
            return  # unchanged → keep original node/style
        if "\n" in text:
            step[key] = LiteralScalarString(text + "\n")
        else:
            step[key] = text

    def _merge_map(self, parent: Any, key: str, new_dict: Dict[str, Any]) -> None:
        """Update an existing sub-map in place (keeps comments) or drop it - Claude Generated."""
        if not new_dict:
            # Preserve an already-empty block (e.g. `tools: []`, `inputs: {}`)
            # rather than churning it away; only drop a previously-filled one.
            if key in parent and not parent.get(key):
                return
            parent.pop(key, None)
            return
        existing = parent.get(key)
        if not isinstance(existing, dict):
            parent[key] = new_dict
            return
        for k, v in new_dict.items():
            # Compare by normalised value so an unchanged `""`/`null` or an
            # unchanged list is not rewritten (which would churn style).
            if k in existing and _norm(existing.get(k)) == _norm(v):
                continue  # unchanged → preserve original node/style
            existing[k] = _flowify(v)
        for k in [k for k in existing if k not in new_dict]:
            del existing[k]

    # ----- step list ops ---------------------------------------------------
    def _selected_step_index(self) -> Optional[int]:
        if self._current and self._current[0] == "step":
            return self._current[1]
        return None

    def _new_step_map(self):
        m = self._yaml.load("id: new_step\ntype: llm_agent\nenabled: true\n")
        existing = {s.get("id") for s in self._steps()}
        base, n, nid = "new_step", 1, "new_step"
        while nid in existing:
            n += 1
            nid = f"new_step_{n}"
        m["id"] = nid
        return m

    def _add_step(self) -> None:
        if self._doc is None:
            return
        self._commit_current()
        steps = self._doc.get("steps")
        if not isinstance(steps, list):
            self._doc["steps"] = self._yaml.load("[]")
            steps = self._doc["steps"]
        steps.append(self._new_step_map())
        idx = len(steps) - 1
        self._dirty = True
        self._update_title()
        self._rebuild_steps_list()
        self._current = ("step", idx)
        self._show_step(idx)
        self.stack.setCurrentWidget(self.step_panel)
        self._highlight_step(idx)

    def _remove_step(self) -> None:
        i = self._selected_step_index()
        if i is None:
            QMessageBox.information(self, "Step entfernen", "Bitte links einen Step auswählen.")
            return
        steps = self._steps()
        if not (0 <= i < len(steps)):
            return
        sid = steps[i].get("id", "?")
        if QMessageBox.question(self, "Step entfernen", f"Step '{sid}' löschen?") != QMessageBox.StandardButton.Yes:
            return
        del self._doc["steps"][i]
        self._current = None
        self._dirty = True
        self._update_title()
        self._rebuild_steps_list()
        self._current = ("settings",)
        self._show_settings()
        self.stack.setCurrentWidget(self.settings_panel)
        self._highlight_settings()

    def _move_step(self, delta: int) -> None:
        i = self._selected_step_index()
        if i is None:
            return
        self._commit_current()
        steps = self._doc["steps"]
        j = i + delta
        if not (0 <= j < len(steps)):
            return
        steps[i], steps[j] = steps[j], steps[i]
        self._dirty = True
        self._update_title()
        self._rebuild_steps_list()
        self._current = ("step", j)
        self._show_step(j)
        self.stack.setCurrentWidget(self.step_panel)
        self._highlight_step(j)

    # ----- file ops --------------------------------------------------------
    @staticmethod
    def _user_dir() -> Path:
        return Path.home() / ".config" / "alima" / "workflows"

    @staticmethod
    def _project_dir() -> Path:
        return Path("workflows")

    def new_config(self) -> None:
        if not self._prompt_to_save():
            return
        self._doc = self._yaml.load(_NEW_TEMPLATE)
        self._path = None
        self._dirty = True
        self._set_loaded(True)
        self._rebuild_steps_list()
        self._current = ("settings",)
        self._show_settings()
        self.stack.setCurrentWidget(self.settings_panel)
        self._highlight_settings()
        self._update_title()
        self.status_label.setText("Neuer Workflow (noch nicht gespeichert).")

    def open_config(self) -> None:
        if not self._prompt_to_save():
            return
        start = str(self._project_dir().resolve()) if self._project_dir().exists() else str(Path.home())
        fn, _ = QFileDialog.getOpenFileName(
            self, "Workflow öffnen", start, "YAML (*.yaml *.yml)"
        )
        if fn:
            self._open_path(Path(fn))

    def _open_named(self, name: str) -> None:
        try:
            from src.core.agents.workflow_loader import find_workflow_file

            path = find_workflow_file(name)
        except Exception:  # noqa: BLE001
            path = None
        if path is not None:
            self._open_path(Path(path))
        else:
            self._set_loaded(False)
            self.status_label.setText(f"Workflow '{name}' nicht gefunden.")

    def _open_path(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                self._doc = self._yaml.load(f)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Fehler beim Öffnen", f"{path}\n\n{e}")
            return
        if not isinstance(self._doc, dict):
            QMessageBox.critical(self, "Ungültige Datei", f"{path}: keine YAML-Mapping-Struktur.")
            self._doc = None
            return
        self._path = path
        self._dirty = False
        self._set_loaded(True)
        self._rebuild_steps_list()
        self._current = ("settings",)
        self._show_settings()
        self.stack.setCurrentWidget(self.settings_panel)
        self._highlight_settings()
        self._update_title()
        in_user = path.parent.resolve() == self._user_dir().resolve()
        self.status_label.setText(
            f"Geladen: {path}" + ("" if in_user else "  (»Speichern« legt eine Kopie im User-Verzeichnis an)")
        )

    def save_config(self) -> None:
        if self._doc is None:
            return
        self._commit_current()
        if self._path is not None and self._path.parent.resolve() == self._user_dir().resolve():
            self._write(self._path)
        else:
            self.save_config_as()

    def save_config_as(self) -> None:
        if self._doc is None:
            return
        self._commit_current()
        default_stem = self._path.stem if self._path else _slug(str(self._doc.get("name", "workflow")))
        if (self._project_dir() / f"{default_stem}.yaml").exists():
            default_stem = f"{default_stem}_custom"
        stem, ok = QInputDialog.getText(
            self, "Speichern unter", "Dateiname (ohne .yaml):", text=default_stem
        )
        if not ok or not stem.strip():
            return
        stem = Path(stem.strip()).stem  # drop any extension the user typed
        dest = self._user_dir() / f"{stem}.yaml"
        if dest.exists():
            if QMessageBox.question(
                self, "Überschreiben?", f"{dest} existiert bereits. Überschreiben?"
            ) != QMessageBox.StandardButton.Yes:
                return
        self._write(dest)

    def _doc_to_text(self) -> str:
        buf = io.StringIO()
        self._yaml.dump(self._doc, buf)
        return buf.getvalue()

    def _validate_text(self, text: str) -> Tuple[bool, str]:
        """Validate with the REAL execution loader - Claude Generated."""
        try:
            from src.core.agents.workflow_loader import load_workflow
        except Exception as e:  # noqa: BLE001
            return False, f"Loader nicht verfügbar: {e}"
        tf = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        try:
            tf.write(text)
            tf.flush()
            tf.close()
            load_workflow(tf.name, strict=True)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass

    def _write(self, dest: Path) -> None:
        text = self._doc_to_text()
        ok, err = self._validate_text(text)
        if not ok:
            QMessageBox.critical(
                self,
                "Ungültiges Workflow-YAML",
                "Die Datei wurde NICHT gespeichert, weil der Workflow-Loader sie "
                "(strict) nicht laden kann:\n\n" + err,
            )
            return
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Fehler beim Speichern", f"{dest}\n\n{e}")
            return
        self._path = dest
        self._dirty = False
        self._update_title()
        self.status_label.setText(f"✓ Gespeichert & validiert: {dest}")
        self.saved.emit(dest.stem)

        shadow = self._project_dir() / f"{dest.stem}.yaml"
        if shadow.exists() and shadow.resolve() != dest.resolve():
            QMessageBox.warning(
                self,
                "Hinweis: Vorrang bei der Ausführung",
                f"Gespeichert unter:\n{dest}\n\n"
                f"Achtung: »workflows/{dest.stem}.yaml« existiert bereits. Bei der "
                f"Ausführung wird die Projekt-Kopie bevorzugt (die Suche durchsucht "
                f"»workflows/« zuerst), deine bearbeitete Version wird also NICHT "
                f"verwendet.\n\nSpeichere unter einem anderen Namen, um sie zu nutzen.",
            )

    # ----- close handling --------------------------------------------------
    def _prompt_to_save(self) -> bool:
        """Return True if it's safe to proceed (discard/save), False to cancel."""
        if not self._dirty or self._doc is None:
            return True
        reply = QMessageBox.question(
            self,
            "Ungespeicherte Änderungen",
            "Es gibt ungespeicherte Änderungen. Speichern?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self.save_config()
            return not self._dirty  # save may have been cancelled
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._prompt_to_save():
            event.accept()
        else:
            event.ignore()

    def reject(self) -> None:  # noqa: D401
        if self._prompt_to_save():
            super().reject()


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in text.strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "workflow"
