"""SingleStepDialog — P-γ. Claude Generated.

Modal dialog that lets the user run a single workflow step (or a cascade of
prerequisite steps) against a SharedContext loaded from disk or seeded from
the current pipeline state.

UI layout (top → bottom):
    1. Workflow picker + Step picker.
    2. "Load state…" row.
    3. Form panel (one row per non-static input, built by StepFormBuilder).
    4. Stream output + final renderer output.
    5. Action bar: Cancel · Run step · Run prerequisites + step.

The dialog never blocks the GUI thread — step execution runs inside
:class:`SingleStepRunWorker` (QThread).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.agents.shared_context import SharedContext
from src.core.agents.state_bridge import load_state_file
from src.core.agents.sub_agents import create_caching_registry
from src.core.agents.workflow_executor import WorkflowExecutor
from src.core.agents.workflow_loader import (
    DEFAULT_SEARCH_PATHS,
    WorkflowDef,
    load_workflow,
)
from src.ui.forms.form_field import FormField
from src.ui.forms.step_form_builder import (
    build_step_form,
    find_missing_prerequisites,
)


class SingleStepRunWorker(QThread):
    """Worker thread that runs a list of step_ids sequentially.

    Emits stream tokens as they arrive and a final result payload (the
    step_results dict of the last successfully-executed step) when done.
    """

    stream = pyqtSignal(str)
    step_started = pyqtSignal(str)  # step_id
    step_completed = pyqtSignal(str, dict)  # step_id, step_results
    finished_ok = pyqtSignal(dict)  # final context.step_results
    failed = pyqtSignal(str)  # error message

    def __init__(
        self,
        *,
        workflow: WorkflowDef,
        context: SharedContext,
        step_chain: List[str],
        llm_service: Any,
        tool_registry: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.workflow = workflow
        self.context = context
        self.step_chain = step_chain
        self.llm_service = llm_service
        self.tool_registry = tool_registry

    def run(self) -> None:
        try:
            executor = WorkflowExecutor(
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                stream_callback=self._on_stream,
            )
            for step_id in self.step_chain:
                self.step_started.emit(step_id)
                report = executor.run(
                    self.workflow,
                    self.context,
                    only_step=step_id,
                    stop_on_error=True,
                )
                if not report.success:
                    self.failed.emit(
                        report.error or f"step '{step_id}' failed"
                    )
                    return
                self.step_completed.emit(step_id, self.context.step_results)
            self.finished_ok.emit(self.context.step_results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _on_stream(self, token: str, *_args: Any) -> None:
        try:
            self.stream.emit(token)
        except Exception:
            pass


class SingleStepDialog(QDialog):
    """Run a single workflow step (with optional cascade)."""

    def __init__(
        self,
        *,
        llm_service: Any,
        alima_manager: Any,
        parent: Optional[QWidget] = None,
        initial_context: Optional[SharedContext] = None,
        initial_workflow: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setWindowTitle("Run Single Step")
        self.setModal(True)
        self.resize(900, 700)

        self.llm_service = llm_service
        self.alima_manager = alima_manager
        self.context = initial_context or SharedContext()
        self.workflow: Optional[WorkflowDef] = None
        self._form_fields: List[FormField] = []
        self._worker: Optional[SingleStepRunWorker] = None

        config_manager = (
            getattr(alima_manager, "config_manager", None)
            if alima_manager
            else None
        )
        self.tool_registry = create_caching_registry(
            config_manager=config_manager
        )

        self._build_ui()
        self._populate_workflows()
        if initial_workflow:
            idx = self.workflow_combo.findData(initial_workflow)
            if idx >= 0:
                self.workflow_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Row 1 — workflow + step combos
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Workflow:"))
        self.workflow_combo = QComboBox()
        self.workflow_combo.setMinimumWidth(220)
        top_row.addWidget(self.workflow_combo)
        top_row.addSpacing(20)
        top_row.addWidget(QLabel("Step:"))
        self.step_combo = QComboBox()
        self.step_combo.setMinimumWidth(220)
        top_row.addWidget(self.step_combo)
        top_row.addStretch(1)
        self.load_button = QPushButton("📂 Load state…")
        top_row.addWidget(self.load_button)
        outer.addLayout(top_row)

        # Row 2 — provider + model
        prov_row = QHBoxLayout()
        prov_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumWidth(160)
        prov_row.addWidget(self.provider_combo)
        prov_row.addSpacing(20)
        prov_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        prov_row.addWidget(self.model_combo)
        prov_row.addStretch(1)
        outer.addLayout(prov_row)

        # State info label
        self.state_info = QLabel("Context: <empty>")
        self.state_info.setStyleSheet("color: #555; padding: 2px;")
        outer.addWidget(self.state_info)

        # Splitter: form panel (top) vs output (bottom)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Form panel
        form_container = QWidget()
        self.form_layout = QFormLayout(form_container)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignTop)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setWidget(form_container)
        splitter.addWidget(form_scroll)

        # Output panel
        out_container = QWidget()
        out_layout = QVBoxLayout(out_container)
        out_layout.addWidget(QLabel("Stream / Output:"))
        self.stream_view = QPlainTextEdit()
        self.stream_view.setReadOnly(True)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.stream_view.setFont(mono)
        out_layout.addWidget(self.stream_view, stretch=3)
        out_layout.addWidget(QLabel("Result:"))
        self.result_view = QTextBrowser()
        out_layout.addWidget(self.result_view, stretch=2)
        splitter.addWidget(out_container)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

        # Action bar
        self.cascade_button = QPushButton("▶▶ Run prerequisites + step")
        self.run_button = QPushButton("▶ Run step")
        self.run_button.setDefault(True)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.addButton(
            self.cascade_button, QDialogButtonBox.ButtonRole.ActionRole
        )
        button_box.addButton(
            self.run_button, QDialogButtonBox.ButtonRole.AcceptRole
        )
        button_box.rejected.connect(self.reject)
        outer.addWidget(button_box)

        # Wire signals
        self.workflow_combo.currentIndexChanged.connect(self._on_workflow_changed)
        self.step_combo.currentIndexChanged.connect(self._refresh_form)
        self.load_button.clicked.connect(self._on_load_state)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.cascade_button.clicked.connect(self._on_cascade_clicked)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)

        self._populate_providers()

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------
    def _populate_providers(self) -> None:
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        providers: List[str] = []
        try:
            if self.llm_service is not None:
                providers = list(self.llm_service.get_available_providers() or [])
        except Exception as exc:
            self.logger.warning(f"Could not list providers: {exc}")
        for name in providers:
            self.provider_combo.addItem(name, name)
        self.provider_combo.blockSignals(False)
        if self.provider_combo.count() > 0:
            self._on_provider_changed(0)

    def _on_provider_changed(self, _index: int) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        provider = self.provider_combo.currentData()
        if not provider or self.llm_service is None:
            self.model_combo.blockSignals(False)
            return
        try:
            models = list(self.llm_service.get_available_models(provider) or [])
        except Exception as exc:
            self.logger.warning(f"Could not list models for {provider}: {exc}")
            models = []
        for name in models:
            self.model_combo.addItem(name, name)
        self.model_combo.blockSignals(False)

    def _populate_workflows(self) -> None:
        self.workflow_combo.blockSignals(True)
        self.workflow_combo.clear()
        seen: set = set()
        for base in DEFAULT_SEARCH_PATHS:
            if not base.exists() or not base.is_dir():
                continue
            for path in sorted(base.glob("*.yaml")):
                key = path.resolve()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    wf = load_workflow(path, strict=False)
                except Exception:
                    continue
                self.workflow_combo.addItem(
                    f"{path.stem} (v{wf.version})", path.stem
                )
        self.workflow_combo.blockSignals(False)
        if self.workflow_combo.count() > 0:
            self._on_workflow_changed(0)

    def _on_workflow_changed(self, _index: int) -> None:
        name = self.workflow_combo.currentData()
        if not name:
            return
        try:
            from src.core.agents.workflow_loader import find_workflow_file
            path = find_workflow_file(name)
            if path is None:
                raise FileNotFoundError(f"workflow '{name}' not found")
            self.workflow = load_workflow(path, strict=False)
        except Exception as exc:
            self.logger.error(f"Failed to load workflow '{name}': {exc}")
            QMessageBox.warning(
                self, "Workflow load failed", f"Could not load '{name}':\n{exc}"
            )
            return

        self.step_combo.blockSignals(True)
        self.step_combo.clear()
        for cfg in self.workflow.steps:
            done = "✓ " if cfg.id in (self.context.step_results or {}) else ""
            self.step_combo.addItem(f"{done}{cfg.id} ({cfg.type})", cfg.id)
        self.step_combo.blockSignals(False)
        if self.step_combo.count() > 0:
            self.step_combo.setCurrentIndex(0)
        self._refresh_form()

    def _refresh_form(self) -> None:
        # Clear existing rows
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)
        self._form_fields = []

        if not self.workflow:
            return
        step_id = self.step_combo.currentData()
        if not step_id:
            return

        fields = build_step_form(self.workflow, step_id, self.context)
        self._form_fields = fields

        for f in fields:
            widget = self._build_field_widget(f)
            label_text = f"{f.name} ({f.kind})"
            if f.missing:
                label_text += f"  — fehlt; writer: {f.writer_step_id or '?'}"
            label = QLabel(label_text)
            if f.missing:
                label.setStyleSheet("color: #b00; font-weight: bold;")
            self.form_layout.addRow(label, widget)

        # Update state info
        populated = sum(
            1
            for k in ("abstract", "extracted_keywords", "gnd_entries")
            if getattr(self.context, k, None)
        )
        results = len(self.context.step_results or {})
        self.state_info.setText(
            f"Context: {populated}/3 core fields filled · {results} step results"
        )

        # Enable cascade button only when there are missing prereqs
        missing = any(f.missing for f in fields if f.kind == "derived")
        self.cascade_button.setEnabled(missing)

    def _build_field_widget(self, field: FormField) -> QWidget:
        """Build a widget for a single FormField. Stores reference on the field."""
        if field.widget == "text":
            box = QTextEdit()
            box.setPlaceholderText(f"${{ {field.expr} }}")
            if field.value:
                box.setPlainText(str(field.value))
            box.setMinimumHeight(80)
            field._widget = box  # type: ignore[attr-defined]
            return box

        # readonly / list / json / raw_json
        view = QPlainTextEdit()
        view.setReadOnly(True)
        if field.missing:
            view.setPlaceholderText("— missing —")
        elif field.value is None:
            view.setPlainText("")
        else:
            try:
                view.setPlainText(
                    json.dumps(field.value, ensure_ascii=False, indent=2)
                    if not isinstance(field.value, str)
                    else field.value
                )
            except TypeError:
                view.setPlainText(repr(field.value))
        view.setMaximumHeight(140)
        field._widget = view  # type: ignore[attr-defined]
        return view

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_load_state(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load state JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            ctx, kind = load_state_file(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Load failed", f"Could not parse {path}:\n{exc}"
            )
            return
        self.context = ctx
        self.state_info.setText(
            f"Loaded {Path(path).name} ({kind}) — abstract: "
            f"{len(self.context.abstract)} chars"
        )
        self._refresh_form()

    def _stash_user_fill_inputs(self) -> None:
        """Write user-edited values from form widgets back into the context."""
        for f in self._form_fields:
            if f.kind != "user_fill":
                continue
            widget = getattr(f, "_widget", None)
            if widget is None:
                continue
            text = (
                widget.toPlainText()
                if hasattr(widget, "toPlainText")
                else None
            )
            if text is None:
                continue
            # Only support bare ${root} expressions for now.
            expr = f.expr.strip()
            if not (expr.startswith("${") and expr.endswith("}")):
                continue
            root = expr[2:-1].split(".", 1)[0]
            if hasattr(self.context, root):
                cur_val = getattr(self.context, root)
                if isinstance(cur_val, list):
                    parsed = [
                        line.strip() for line in text.splitlines() if line.strip()
                    ]
                    setattr(self.context, root, parsed)
                else:
                    setattr(self.context, root, text)

    def _chunking_confirm(self, step_ids: List[str]) -> bool:
        """If any step in ``step_ids`` enables chunking, ask the user to confirm."""
        if not self.workflow:
            return True
        chunked: List[str] = []
        for sid in step_ids:
            cfg = next((s for s in self.workflow.steps if s.id == sid), None)
            if cfg and (cfg.raw.get("chunking", {}) or {}).get("enabled"):
                chunked.append(sid)
        if not chunked:
            return True
        names = ", ".join(chunked)
        reply = QMessageBox.question(
            self,
            "Chunking enabled",
            f"Step(s) {names} use chunking — execution may issue many "
            f"LLM calls. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_run_clicked(self) -> None:
        if not self.workflow:
            return
        step_id = self.step_combo.currentData()
        if not step_id:
            return
        self._stash_user_fill_inputs()
        if not self._chunking_confirm([step_id]):
            return
        self._launch_worker([step_id])

    def _on_cascade_clicked(self) -> None:
        if not self.workflow:
            return
        step_id = self.step_combo.currentData()
        if not step_id:
            return
        self._stash_user_fill_inputs()
        chain = find_missing_prerequisites(self.workflow, step_id, self.context)
        if len(chain) > 1:
            reply = QMessageBox.question(
                self,
                "Confirm cascade",
                "About to run the following steps in order:\n\n  → "
                + "\n  → ".join(chain)
                + "\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not self._chunking_confirm(chain):
            return
        self._launch_worker(chain)

    def _launch_worker(self, chain: List[str]) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self, "Already running", "A step is currently executing."
            )
            return
        if not self.workflow:
            return

        provider = self.provider_combo.currentData()
        model = self.model_combo.currentData()
        if not provider or not model:
            QMessageBox.warning(
                self,
                "Provider missing",
                "Select a provider and model before running.",
            )
            return
        self.context.provider = provider
        self.context.model = model
        if getattr(self.context, "prompt_service", None) is None:
            ps = getattr(self.alima_manager, "prompt_service", None)
            if ps is not None:
                self.context.prompt_service = ps

        self.run_button.setEnabled(False)
        self.cascade_button.setEnabled(False)
        self.stream_view.clear()
        self.result_view.clear()
        self.stream_view.appendPlainText(
            f"▶ Chain: {' → '.join(chain)}  ·  {provider}/{model}\n"
        )

        self._worker = SingleStepRunWorker(
            workflow=self.workflow,
            context=self.context,
            step_chain=chain,
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            parent=self,
        )
        self._worker.stream.connect(self._on_stream_token)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_completed.connect(self._on_step_completed)
        self._worker.finished_ok.connect(self._on_chain_done)
        self._worker.failed.connect(self._on_chain_failed)
        self._worker.finished.connect(self._reset_buttons)
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    def _on_stream_token(self, token: str) -> None:
        # Append without injecting an extra newline.
        cur = self.stream_view.toPlainText()
        self.stream_view.setPlainText(cur + token)
        # Auto-scroll
        sb = self.stream_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_step_started(self, step_id: str) -> None:
        self.stream_view.appendPlainText(f"\n── ▶ {step_id} ──")

    def _on_step_completed(self, step_id: str, _results: dict) -> None:
        self.stream_view.appendPlainText(f"── ✓ {step_id} done ──")

    def _on_chain_done(self, step_results: dict) -> None:
        last = self.step_combo.currentData()
        payload = (step_results or {}).get(last, {})
        self._render_result(last, payload)
        self._refresh_form()

    def _on_chain_failed(self, error: str) -> None:
        QMessageBox.critical(self, "Step failed", error)
        self.stream_view.appendPlainText(f"\n❌ {error}")

    def _reset_buttons(self) -> None:
        self.run_button.setEnabled(True)
        # cascade button gets re-enabled by _refresh_form() based on form state

    def _render_result(self, step_id: str, payload: Any) -> None:
        slot = f"slot:{step_id}"
        try:
            from src.ui.renderers import get_renderer

            cls = get_renderer(slot)
            renderer = cls()
            html = renderer.render_html(payload)
            self.result_view.setHtml(html)
        except Exception:
            try:
                self.result_view.setPlainText(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
            except TypeError:
                self.result_view.setPlainText(repr(payload))
