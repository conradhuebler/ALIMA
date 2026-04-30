"""Agentic Context Widget - Claude Generated.

Dynamic, workflow-agnostic display of SharedContext state during v4
workflow execution. Panels are built from the active ``WorkflowDef`` at
run-start (no hardcoded step set), so new workflows are rendered
automatically.

Each step gets a collapsible panel (default closed). The header always
shows status icon, step id/type and duration. The body renders the
SharedContext snapshot generically: known fields (``extracted_keywords``,
``gnd_entries``, ``selected_keywords``, ``keyword_chains``,
``dk_classifications``, ``rvk_classifications``, ``missing_concepts``)
use typed renderers; all ``extra.*`` entries are rendered via a generic
chip/table/JSON formatter. Panels auto-expand on ``running`` or ``error``.
"""

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


STATUS_ICON = {
    "pending": "▷",
    "running": "▶",
    "completed": "✓",
    "error": "✗",
}

STATUS_COLOR = {
    "pending": "#888888",
    "running": "#8be9fd",
    "completed": "#50fa7b",
    "error": "#ff5555",
}

# Known SharedContext fields that get typed rendering (rest falls back to
# generic JSON-ish printer).
TYPED_FIELDS = (
    "extracted_keywords",
    "gnd_entries",
    "selected_keywords",
    "keyword_chains",
    "dk_classifications",
    "rvk_classifications",
    "missing_concepts",
)


class AgenticStepPanel(QFrame):
    """Collapsible panel rendering one workflow step's context slice."""

    def __init__(
        self,
        step_id: str,
        step_type: str = "",
        description: str = "",
        output_paths: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.step_id = step_id
        self.step_type = step_type
        self.description = description
        # Output paths this step writes (e.g. ["extra.titles", "extra.catalog_hits"]).
        # Panel renders ONLY these fields to avoid showing earlier steps' data.
        self.output_paths: List[str] = list(output_paths or [])
        self._user_toggled = False  # Once user clicks, stop auto-expanding
        # Status fields need to exist before _build() → _refresh_header_text
        # dereferences them.
        self._status = "pending"
        self._duration: Optional[float] = None
        self._error: Optional[str] = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1e1e1e; border: 1px solid #333; border-radius: 4px; }"
        )
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_btn.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            " color: #f8f8f2; font-weight: bold; padding: 2px 4px; text-align: left; }"
            "QToolButton:hover { background: #2a2a2a; }"
        )
        self.toggle_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.toggle_btn.toggled.connect(self._on_toggled)
        self._refresh_header_text()
        layout.addWidget(self.toggle_btn)

        # Body fills the available widget space; internal scrollbar handles
        # overflow. No max-height cap so tall content uses the full panel
        # width and the outer QScrollArea handles vertical overflow.
        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setFont(QFont("Consolas", 9))
        self.body.setStyleSheet(
            "QTextEdit { background: #181818; color: #f8f8f2;"
            " border: 1px solid #2a2a2a; border-radius: 2px; }"
        )
        self.body.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.body.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.body.setMinimumHeight(100)
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.body.setVisible(False)
        layout.addWidget(self.body, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, snap: Dict[str, Any]) -> None:
        self._status = snap.get("_step_status", "running")
        self._duration = snap.get("_step_duration")
        self._error = snap.get("_step_error")
        self._refresh_header_text()

        self.body.setHtml(self._render_body(snap))

        # Auto-expand on running/error the first time (until user toggles)
        if not self._user_toggled and self._status in ("running", "error"):
            self.toggle_btn.setChecked(True)

    def reset(self) -> None:
        self._status = "pending"
        self._duration = None
        self._error = None
        self._user_toggled = False
        self.toggle_btn.setChecked(False)
        self.body.clear()
        self._refresh_header_text()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        self._user_toggled = True
        self.body.setVisible(checked)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _refresh_header_text(self) -> None:
        icon = STATUS_ICON.get(self._status, "▷")
        color = STATUS_COLOR.get(self._status, "#888888")
        parts = [f"<span style='color:{color}; font-size:14px;'>{icon}</span>"]
        parts.append(
            f"<span style='color:#f8f8f2; font-weight:bold;'>{self._esc(self.step_id)}</span>"
        )
        if self.step_type:
            parts.append(
                f"<span style='color:#6272a4;'>({self._esc(self.step_type)})</span>"
            )
        if self._duration is not None:
            parts.append(
                f"<span style='color:#bd93f9;'>{self._duration:.1f}s</span>"
            )
        if self._status == "error" and self._error:
            parts.append(
                f"<span style='color:#ff5555;'>❗ {self._esc(self._error[:60])}</span>"
            )
        # QToolButton text doesn't render HTML reliably → strip to plain
        plain = [
            f"{STATUS_ICON.get(self._status, '▷')} {self.step_id}",
        ]
        if self.step_type:
            plain.append(f"({self.step_type})")
        if self._duration is not None:
            plain.append(f"{self._duration:.1f}s")
        if self._status == "error" and self._error:
            plain.append(f"❗ {self._error[:60]}")
        self.toggle_btn.setText("  ".join(plain))
        # Color-tint arrow/text via stylesheet override per status
        self.toggle_btn.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            f" color: {color}; font-weight: bold; padding: 2px 4px; text-align: left; }}"
            "QToolButton:hover { background: #2a2a2a; }"
        )

    # --- Rendering --------------------------------------------------------

    def _render_body(self, snap: Dict[str, Any]) -> str:
        blocks: List[str] = []

        if self.description:
            blocks.append(
                f"<div style='color:#6272a4; font-style:italic; margin-bottom:4px;'>"
                f"{self._esc(self.description)}</div>"
            )

        if self._status == "error" and self._error:
            blocks.append(
                f"<div style='color:#ff5555; font-weight:bold;'>Fehler: "
                f"{self._esc(self._error)}</div>"
            )

        # Render ONLY the fields this step writes, so downstream panels
        # don't re-display the upstream output. If no output_paths were
        # declared, fall back to showing the whole snapshot (legacy).
        if self.output_paths:
            for path in self.output_paths:
                val = self._resolve_snap_path(snap, path)
                if val in (None, "", [], {}):
                    continue
                label, local = self._label_for_path(path)
                blocks.append(self._render_field(label, val) if local in TYPED_FIELDS else self._render_generic(label, val))
        else:
            for name in TYPED_FIELDS:
                val = snap.get(name)
                if val:
                    html = self._render_field(name, val)
                    if html:
                        blocks.append(html)
            extra = snap.get("extra") or {}
            if isinstance(extra, dict):
                for key, val in extra.items():
                    if val in (None, "", [], {}):
                        continue
                    blocks.append(self._render_generic(f"extra.{key}", val))

        if not blocks:
            return "<i style='color:#888'>— keine Daten —</i>"
        return "".join(blocks)

    @staticmethod
    def _resolve_snap_path(snap: Dict[str, Any], path: str) -> Any:
        """Resolve dotted path like 'extra.titles' against snapshot dict."""
        parts = path.split(".")
        cur: Any = snap
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur

    @staticmethod
    def _label_for_path(path: str) -> tuple:
        """Return (display_label, last_segment)."""
        last = path.split(".")[-1]
        return path, last

    def _render_field(self, name: str, val: Any) -> str:
        if name in ("extracted_keywords", "missing_concepts"):
            return self._chips(name, val)
        if name in ("gnd_entries", "selected_keywords", "dk_classifications", "rvk_classifications"):
            return self._dict_list(name, val)
        if name == "keyword_chains":
            return self._chains(val)
        return self._render_generic(name, val)

    def _chips(self, label: str, val: Any) -> str:
        if not isinstance(val, list):
            return self._render_generic(label, val)
        if not val:
            return ""
        # Detect truncation sentinel
        truncated = 0
        items = val
        if items and isinstance(items[-1], dict) and "_truncated" in items[-1]:
            truncated = items[-1]["_truncated"]
            items = items[:-1]
        chips = "  ".join(
            f"<span style='background:#44475a; color:#f8f8f2; padding:1px 6px;"
            f" border-radius:3px;'>{self._esc(k)}</span>"
            for k in items
        )
        trunc_text = f" <span style='color:#ff79c6;'>+{truncated} more</span>" if truncated else ""
        return (
            f"<div style='color:#8be9fd; margin-top:4px;'>{self._esc(label)} "
            f"({len(items)}{trunc_text})</div><div>{chips}</div>"
        )

    # Field ordering + header keys for known record types. Unknown dicts
    # fall back to the order keys appear in.
    HEADER_KEYS = ("title", "keyword", "rsn", "code", "input_title", "label")
    ID_KEYS = ("gnd_id", "gnd_ids", "rsn", "id")

    def _dict_list(self, label: str, val: Any) -> str:
        if not isinstance(val, list) or not val:
            return ""
        # Detect truncation sentinel
        truncated = 0
        items = val
        if items and isinstance(items[-1], dict) and "_truncated" in items[-1]:
            truncated = items[-1]["_truncated"]
            items = items[:-1]

        # Derive canonical key union so missing fields per record are visible.
        canonical_keys: List[str] = []
        seen = set()
        for item in items:
            if isinstance(item, dict):
                for k in item.keys():
                    if k not in seen:
                        seen.add(k)
                        canonical_keys.append(k)
        trunc_text = f" <span style='color:#ff79c6;'>+{truncated} more</span>" if truncated else ""
        rows: List[str] = [
            f"<div style='color:#8be9fd; margin-top:6px; font-weight:bold;'>"
            f"{self._esc(label)} ({len(items)}{trunc_text})</div>"
        ]
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                rows.append(
                    f"<div style='margin-left:6px;'>#{idx}: {self._esc(item)}</div>"
                )
                continue
            rows.append(self._render_record(idx, item, canonical_keys))
        return "".join(rows)

    def _render_record(
        self,
        idx: int,
        item: Dict[str, Any],
        canonical_keys: Optional[List[str]] = None,
    ) -> str:
        """Render one dict as a labeled card with header + field lines.

        Missing keys (present in canonical_keys but not in this item) render
        as "— fehlt" so gaps are visible rather than silently hidden.
        """
        # Pick header: first non-empty value for a recognised header key.
        header_val = ""
        for hk in self.HEADER_KEYS:
            v = item.get(hk)
            if v:
                header_val = str(v)
                break
        # Secondary identifier line (gnd_id, rsn, etc.) for quick scanning.
        id_parts: List[str] = []
        for ik in self.ID_KEYS:
            v = item.get(ik)
            if v:
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                id_parts.append(f"{ik}={v}")

        header_html = (
            f"<div style='color:#50fa7b; font-weight:bold;'>"
            f"#{idx} {self._esc(header_val) if header_val else '(ohne Titel)'}"
            f"</div>"
        )
        subheader_html = (
            f"<div style='color:#f1fa8c; font-size:10px; margin-bottom:2px;'>"
            f"{self._esc(' · '.join(id_parts))}</div>"
            if id_parts
            else ""
        )

        # Iterate canonical keys (union over all records); fall back to
        # item's own keys if no canonical set was provided.
        keys = canonical_keys or list(item.keys())
        # Skip keys already shown in header/subheader
        skip = set(self.HEADER_KEYS) | set(self.ID_KEYS)

        lines: List[str] = []
        for k in keys:
            if k in skip:
                continue
            raw = item.get(k)
            missing = raw in (None, "", [], {})
            if missing:
                v_html = "<span style='color:#6272a4; font-style:italic;'>— fehlt</span>"
            else:
                if isinstance(raw, list):
                    v_str = ", ".join(str(x) for x in raw)
                elif isinstance(raw, dict):
                    v_str = ", ".join(f"{kk}={vv}" for kk, vv in raw.items())
                else:
                    v_str = str(raw)
                v_html = (
                    f"<span style='color:#f8f8f2; white-space:pre-wrap;'>"
                    f"{self._esc(v_str)}</span>"
                )
            lines.append(
                f"<div style='margin-left:10px;'>"
                f"<span style='color:#bd93f9'>{self._esc(k)}:</span> {v_html}"
                f"</div>"
            )

        return (
            f"<div style='margin-top:6px; padding:4px 6px;"
            f" border-left:3px solid #44475a; background:#161616;"
            f" border-radius:2px;'>"
            f"{header_html}{subheader_html}{''.join(lines)}</div>"
        )

    def _chains(self, val: Any) -> str:
        if not isinstance(val, list) or not val:
            return ""
        # Detect truncation sentinel
        truncated = 0
        items = val
        if items and isinstance(items[-1], dict) and "_truncated" in items[-1]:
            truncated = items[-1]["_truncated"]
            items = items[:-1]
        trunc_text = f" <span style='color:#ff79c6;'>+{truncated} more</span>" if truncated else ""
        rows: List[str] = [
            f"<div style='color:#8be9fd; margin-top:4px;'>keyword_chains "
            f"({len(items)}{trunc_text})</div>"
        ]
        for ch in items:
            if not isinstance(ch, dict):
                continue
            chain = ch.get("chain") or []
            reason = ch.get("reason", "")
            rows.append(
                f"<div><span style='color:#f8f8f2'>• "
                f"{self._esc(' → '.join(str(c) for c in chain))}</span>"
                f" <span style='color:#6272a4'>{self._esc(reason)}</span></div>"
            )
        return "".join(rows)

    def _render_generic(self, label: str, val: Any) -> str:
        if isinstance(val, list):
            # List of dicts → table-ish; list of strings → chips
            if val and isinstance(val[0], dict):
                return self._dict_list(label, val)
            return self._chips(label, [str(v) for v in val])
        if isinstance(val, dict):
            rows = [
                f"<div><span style='color:#f1fa8c'>{self._esc(k)}:</span> "
                f"<span style='color:#f8f8f2; white-space:pre-wrap;'>{self._esc(v)}</span></div>"
                for k, v in val.items()
            ]
            return (
                f"<div style='color:#8be9fd; margin-top:4px;'>{self._esc(label)}"
                f"</div>{''.join(rows)}"
            )
        return (
            f"<div><span style='color:#8be9fd'>{self._esc(label)}:</span> "
            f"<span style='color:#f8f8f2; white-space:pre-wrap;'>{self._esc(val)}</span></div>"
        )

    @staticmethod
    def _esc(text: Any) -> str:
        s = str(text) if text is not None else ""
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


class AgenticContextWidget(QWidget):
    """Dynamic container rendering one panel per workflow step."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("QWidget { background: #121212; }")
        self.panels: Dict[str, AgenticStepPanel] = {}
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self.header_label = QLabel("🤖 Agentic Context")
        self.header_label.setStyleSheet(
            "color: #8be9fd; font-weight: bold; font-size: 11px; padding: 2px 4px;"
        )
        outer.addWidget(self.header_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(4)
        self._inner_layout.addStretch()
        scroll.setWidget(self._inner)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_panels(self, workflow_def: Any) -> None:
        """Rebuild panels from a parsed ``WorkflowDef``.

        Accepts either a WorkflowDef instance (preferred) or a dict with a
        ``steps`` key, for testability. Call once at run start.
        """
        self.clear_panels()

        steps = self._extract_steps(workflow_def)
        if not steps:
            return

        # Insert panels before the stretch item
        stretch_index = self._inner_layout.count() - 1
        for s in steps:
            if not self._step_enabled(s):
                continue
            step_id = self._step_field(s, "id")
            step_type = self._step_field(s, "type") or ""
            description = self._step_field(s, "description") or ""
            if not step_id:
                continue
            outputs = self._step_field(s, "outputs") or {}
            output_paths = list(outputs.keys()) if isinstance(outputs, dict) else []
            panel = AgenticStepPanel(
                step_id, step_type, description, output_paths=output_paths
            )
            self.panels[step_id] = panel
            self._inner_layout.insertWidget(stretch_index, panel)
            stretch_index += 1

        name = getattr(workflow_def, "name", None) or (
            workflow_def.get("name") if isinstance(workflow_def, dict) else None
        )
        if name:
            self.header_label.setText(f"🤖 Agentic Context — {name}")
        else:
            self.header_label.setText("🤖 Agentic Context")

    def clear_panels(self) -> None:
        for panel in list(self.panels.values()):
            self._inner_layout.removeWidget(panel)
            panel.deleteLater()
        self.panels.clear()

    @pyqtSlot(str, dict)
    def on_context_updated(self, step_name: str, snapshot: Dict[str, Any]) -> None:
        panel = self.panels.get(step_name)
        if panel is None:
            return
        panel.apply(snapshot)

    def reset(self) -> None:
        for panel in self.panels.values():
            panel.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_steps(workflow_def: Any) -> List[Any]:
        if workflow_def is None:
            return []
        steps = getattr(workflow_def, "steps", None)
        if steps is not None:
            return list(steps)
        if isinstance(workflow_def, dict):
            return list(workflow_def.get("steps") or [])
        return []

    @staticmethod
    def _step_enabled(step: Any) -> bool:
        enabled = getattr(step, "enabled", None)
        if enabled is None and isinstance(step, dict):
            enabled = step.get("enabled", True)
        return True if enabled is None else bool(enabled)

    @staticmethod
    def _step_field(step: Any, name: str) -> Optional[str]:
        val = getattr(step, name, None)
        if val is None and isinstance(step, dict):
            val = step.get(name)
        return val
