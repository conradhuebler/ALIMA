"""Agentic Context Widget - Claude Generated.

Displays SharedContext state during agentic MetaAgent pipeline execution.
Mirrors the visual language of PipelineStepWidget (status icons, dark monospace
palette) but renders the dynamic context each SubAgent has produced.

Consumes snapshot dicts emitted by MetaAgent.context_callback (via
PipelineWorker.agentic_context_updated). Each snapshot merges
SharedContext.to_dict() with per-step meta:
    _step_name, _step_status, _step_duration, _step_quality,
    _step_iteration, _step_agent, _cache_stats.
"""

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
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

STEP_ORDER = ["extraction", "search", "selection", "classification"]
STEP_LABEL = {
    "extraction": "1. Extraction — Keywords aus Abstract",
    "search": "2. Search — GND-Pool aufbauen",
    "selection": "3. Selection — GND-Verifikation",
    "classification": "4. Classification — DK/RVK",
}


class AgenticStepPanel(QFrame):
    """Single panel rendering one SubAgent's current context slice."""

    def __init__(self, step_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.step_name = step_name
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1e1e1e; border: 1px solid #333; border-radius: 4px; }"
        )
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)

        self.status_label = QLabel(STATUS_ICON["pending"])
        self.status_label.setStyleSheet(
            f"color: {STATUS_COLOR['pending']}; font-size: 16px; font-weight: bold;"
        )
        header.addWidget(self.status_label)

        self.title_label = QLabel(STEP_LABEL.get(self.step_name, self.step_name))
        self.title_label.setStyleSheet(
            "color: #f8f8f2; font-weight: bold; font-size: 12px;"
        )
        header.addWidget(self.title_label)

        header.addStretch()

        self.meta_label = QLabel("")
        self.meta_label.setStyleSheet("color: #bd93f9; font-size: 11px;")
        header.addWidget(self.meta_label)

        layout.addLayout(header)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("Quality %p%")
        self.quality_bar.setFixedHeight(10)
        self.quality_bar.setStyleSheet(
            "QProgressBar { background: #282a36; border: 1px solid #444; border-radius: 2px;"
            " color: #f8f8f2; font-size: 9px; text-align: center; }"
            "QProgressBar::chunk { background: #50fa7b; }"
        )
        self.quality_bar.setVisible(False)
        layout.addWidget(self.quality_bar)

        self.missing_label = QLabel("")
        self.missing_label.setStyleSheet(
            "color: #ff9800; font-size: 11px; font-weight: bold;"
        )
        self.missing_label.setWordWrap(True)
        self.missing_label.setVisible(False)
        layout.addWidget(self.missing_label)

        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setFont(QFont("Consolas", 9))
        self.body.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #f8f8f2; border: none; }"
        )
        self.body.setMaximumHeight(140)
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.body)

    def reset(self) -> None:
        self.update_status("pending")
        self.meta_label.setText("")
        self.quality_bar.setVisible(False)
        self.quality_bar.setValue(0)
        self.missing_label.setVisible(False)
        self.missing_label.setText("")
        self.body.clear()

    def update_status(self, status: str) -> None:
        icon = STATUS_ICON.get(status, "▷")
        color = STATUS_COLOR.get(status, "#888888")
        self.status_label.setText(icon)
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 16px; font-weight: bold;"
        )

    def apply(self, snap: Dict[str, Any]) -> None:
        status = snap.get("_step_status", "running")
        self.update_status(status)

        meta_parts: List[str] = []
        agent = snap.get("_step_agent")
        if agent:
            meta_parts.append(agent)
        iteration = snap.get("_step_iteration", 0) or 0
        if iteration:
            meta_parts.append(f"Iter {iteration}")
        duration = snap.get("_step_duration")
        if duration is not None:
            meta_parts.append(f"{duration:.1f}s")
        cache = snap.get("_cache_stats") or {}
        if cache.get("total_hits") or cache.get("total_misses"):
            rate = cache.get("hit_rate", 0.0) * 100
            meta_parts.append(f"cache {rate:.0f}%")
        self.meta_label.setText("  |  ".join(meta_parts))

        quality = snap.get("_step_quality")
        if quality is not None:
            self.quality_bar.setVisible(True)
            self.quality_bar.setValue(int(max(0.0, min(1.0, quality)) * 100))

        self.body.setHtml(self._render_body(snap))

        if self.step_name == "selection":
            missing = snap.get("missing_concepts") or []
            if missing:
                preview = ", ".join(missing[:5])
                more = "…" if len(missing) > 5 else ""
                self.missing_label.setText(
                    f"⚠ {len(missing)} fehlende Konzepte: {preview}{more}"
                )
                self.missing_label.setVisible(True)
            else:
                self.missing_label.setVisible(False)

    def _render_body(self, snap: Dict[str, Any]) -> str:
        if self.step_name == "extraction":
            kws = snap.get("extracted_keywords") or []
            if not kws:
                return "<i style='color:#888'>—</i>"
            chips = "  ".join(
                f"<span style='background:#44475a; color:#f8f8f2; padding:1px 6px;"
                f" border-radius:3px;'>{self._esc(k)}</span>"
                for k in kws
            )
            return f"<div>{chips}</div>"

        if self.step_name == "search":
            entries = snap.get("gnd_entries") or []
            header = (
                f"<div style='color:#8be9fd'>{len(entries)} GND-Einträge</div>"
            )
            if not entries:
                return header
            rows = []
            for e in entries[:10]:
                title = self._esc(e.get("title", ""))
                gid = self._esc(e.get("gnd_id", ""))
                rows.append(
                    f"<div><span style='color:#f1fa8c'>{gid}</span> "
                    f"<span style='color:#f8f8f2'>{title}</span></div>"
                )
            if len(entries) > 10:
                rows.append(
                    f"<div style='color:#888'>… +{len(entries) - 10} weitere</div>"
                )
            return header + "".join(rows)

        if self.step_name == "selection":
            sel = snap.get("selected_keywords") or []
            chains = snap.get("keyword_chains") or []
            header = (
                f"<div style='color:#8be9fd'>{len(sel)} ausgewählt"
                f" · {len(chains)} Ketten</div>"
            )
            if not sel:
                return header
            rows = []
            for kw in sel[:12]:
                title = self._esc(kw.get("title", ""))
                gid = self._esc(kw.get("gnd_id", ""))
                conf = kw.get("confidence")
                conf_s = (
                    f" <span style='color:#50fa7b'>{int(conf * 100)}%</span>"
                    if isinstance(conf, (int, float))
                    else ""
                )
                rows.append(
                    f"<div><span style='color:#f1fa8c'>{gid}</span> "
                    f"<span style='color:#f8f8f2'>{title}</span>{conf_s}</div>"
                )
            if len(sel) > 12:
                rows.append(
                    f"<div style='color:#888'>… +{len(sel) - 12} weitere</div>"
                )
            return header + "".join(rows)

        if self.step_name == "classification":
            dks = snap.get("dk_classifications") or []
            rvks = snap.get("rvk_classifications") or []
            rows = []
            if dks:
                rows.append(
                    f"<div style='color:#8be9fd'>DK ({len(dks)})</div>"
                )
                for cls in dks[:8]:
                    code = self._esc(cls.get("code", ""))
                    title = self._esc(cls.get("title", ""))
                    conf = cls.get("confidence")
                    color = self._conf_color(conf)
                    rows.append(
                        f"<div><span style='color:{color}; font-weight:bold'>{code}</span> "
                        f"<span style='color:#f8f8f2'>{title}</span></div>"
                    )
            if rvks:
                rows.append(
                    f"<div style='color:#8be9fd; margin-top:4px'>RVK ({len(rvks)})</div>"
                )
                for cls in rvks[:8]:
                    code = self._esc(cls.get("code", ""))
                    title = self._esc(cls.get("title", ""))
                    rows.append(
                        f"<div><span style='color:#ffb86c; font-weight:bold'>{code}</span> "
                        f"<span style='color:#f8f8f2'>{title}</span></div>"
                    )
            if not rows:
                return "<i style='color:#888'>—</i>"
            return "".join(rows)

        return "<i style='color:#888'>—</i>"

    @staticmethod
    def _conf_color(conf: Any) -> str:
        if not isinstance(conf, (int, float)):
            return "#f8f8f2"
        if conf >= 0.75:
            return "#50fa7b"
        if conf >= 0.5:
            return "#8be9fd"
        if conf >= 0.25:
            return "#f1fa8c"
        return "#ff5555"

    @staticmethod
    def _esc(text: Any) -> str:
        s = str(text) if text is not None else ""
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


class AgenticContextWidget(QWidget):
    """Container rendering the four agentic SubAgent panels."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("QWidget { background: #121212; }")
        self.panels: Dict[str, AgenticStepPanel] = {}
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        header = QLabel("🤖 Agentic Context")
        header.setStyleSheet(
            "color: #8be9fd; font-weight: bold; font-size: 11px; padding: 2px 4px;"
        )
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(6)

        for step in STEP_ORDER:
            panel = AgenticStepPanel(step)
            self.panels[step] = panel
            inner_layout.addWidget(panel)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    @pyqtSlot(str, dict)
    def on_context_updated(self, step_name: str, snapshot: Dict[str, Any]) -> None:
        panel = self.panels.get(step_name)
        if panel is None:
            return
        panel.apply(snapshot)

    def reset(self) -> None:
        for panel in self.panels.values():
            panel.reset()
