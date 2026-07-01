"""Category-grouped plugin settings tab - Claude Generated.

Replaces the checkbox-only ``SearchProviderSelectorWidget`` ("nur ticks"). Each
plugin *instance* gets its own panel: enable, label, primary flag, an agent
``usage_hint``, and a form auto-generated from the plugin type's declared
``config_fields`` (so finc-URL, Libero-token, DOI contact-email … live *with* the
plugin, not scattered in the config). Instances can be added, duplicated and
removed, and several instances of one type may coexist (e.g. two finc endpoints).

The tab reads/writes ``AlimaConfig.plugins`` directly and is authoritative on
save; the legacy Catalog/System tabs keep working via
``plugin_migration.sync_instances_from_mirrors``. See ``docs/plugin_system.md``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.plugins import CHOICE, INT, BOOL, SECRET, get_category, list_categories
from src.utils.config_models import PluginInstanceConfig

# Human labels for the known categories (fallback = the raw name).
_CATEGORY_LABELS = {
    "search_provider": "🔎 Suchquellen",
    "input_source": "📥 Input-Quellen",
}


def _ensure_categories() -> None:
    import src.core.search  # noqa: F401
    import src.utils.input_sources  # noqa: F401


class _CategoryPanel(QWidget):
    """Instance list + per-instance form for one plugin category."""

    def __init__(self, category: str, parent=None):
        super().__init__(parent)
        self.category = category
        self._adapter = get_category(category)
        self._instances: List[PluginInstanceConfig] = []
        self._field_widgets: Dict[str, QWidget] = {}
        self._current_row = -1

        root = QHBoxLayout(self)

        # Left: instance list + actions
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.list)

        self.type_combo = QComboBox()
        for t in self._adapter.list_types():
            self.type_combo.addItem(self._adapter.type_meta(t).label, t)
        left.addWidget(self.type_combo)

        btns = QHBoxLayout()
        add = QPushButton("Hinzufügen")
        dup = QPushButton("Duplizieren")
        rem = QPushButton("Entfernen")
        add.clicked.connect(self._add_instance)
        dup.clicked.connect(self._duplicate_instance)
        rem.clicked.connect(self._remove_instance)
        for b in (add, dup, rem):
            btns.addWidget(b)
        left.addLayout(btns)
        root.addLayout(left, 1)

        # Right: form for the selected instance
        self.form_box = QGroupBox("Instanz")
        self.form_layout = QFormLayout(self.form_box)
        root.addWidget(self.form_box, 2)

    # -- data -----------------------------------------------------------------
    def load(self, instances: List[PluginInstanceConfig]) -> None:
        self._instances = [copy.deepcopy(i) for i in instances]
        self._refresh_list()
        if self._instances:
            self.list.setCurrentRow(0)
        else:
            self._build_form(None)

    def collect(self) -> List[PluginInstanceConfig]:
        """Flush the visible form and return the edited instances."""
        self._flush_form()
        return [copy.deepcopy(i) for i in self._instances]

    # -- list ops -------------------------------------------------------------
    def _refresh_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for inst in self._instances:
            flag = " ★" if inst.is_primary else ""
            state = "" if inst.enabled else "  (deaktiviert)"
            QListWidgetItem(f"{inst.display_label()} [{inst.provider_id}]{flag}{state}", self.list)
        self.list.blockSignals(False)

    def _unique_id(self, base: str) -> str:
        existing = {i.instance_id for i in self._instances}
        if base not in existing:
            return base
        n = 2
        while f"{base}-{n}" in existing:
            n += 1
        return f"{base}-{n}"

    def _add_instance(self) -> None:
        type_id = self.type_combo.currentData()
        if not type_id:
            return
        meta = self._adapter.type_meta(type_id)
        settings = {f.key: f.default for f in meta.config_fields}
        has_primary = any(i.provider_id == type_id and i.is_primary for i in self._instances)
        inst = PluginInstanceConfig(
            instance_id=self._unique_id(type_id),
            category=self.category,
            provider_id=type_id,
            label=meta.label,
            enabled=True,
            is_primary=not has_primary,
            settings=settings,
        )
        self._flush_form()
        self._instances.append(inst)
        self._refresh_list()
        self.list.setCurrentRow(len(self._instances) - 1)

    def _duplicate_instance(self) -> None:
        if not (0 <= self._current_row < len(self._instances)):
            return
        self._flush_form()
        src = self._instances[self._current_row]
        clone = copy.deepcopy(src)
        clone.instance_id = self._unique_id(src.instance_id)
        clone.is_primary = False
        clone.label = f"{src.display_label()} (Kopie)"
        self._instances.append(clone)
        self._refresh_list()
        self.list.setCurrentRow(len(self._instances) - 1)

    def _remove_instance(self) -> None:
        if not (0 <= self._current_row < len(self._instances)):
            return
        del self._instances[self._current_row]
        self._current_row = -1
        self._refresh_list()
        if self._instances:
            self.list.setCurrentRow(0)
        else:
            self._build_form(None)

    def _on_row_changed(self, row: int) -> None:
        self._flush_form()
        self._current_row = row
        self._build_form(self._instances[row] if 0 <= row < len(self._instances) else None)

    # -- form -----------------------------------------------------------------
    def _clear_form(self) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self._field_widgets = {}

    def _build_form(self, inst: Optional[PluginInstanceConfig]) -> None:
        self._clear_form()
        if inst is None:
            self.form_layout.addRow(QLabel("Keine Instanz ausgewählt."))
            return

        meta = self._adapter.type_meta(inst.provider_id)

        # Natural-language self-description (what it does + input/output).
        doc = getattr(meta, "doc", None)
        if doc is not None and (doc.description or doc.input or doc.output):
            doc_lbl = QLabel(doc.as_text())
            doc_lbl.setWordWrap(True)
            doc_lbl.setStyleSheet(
                "background:#f0f0f0; color:#333; padding:6px; border-radius:4px;"
            )
            self.form_layout.addRow(doc_lbl)

        self._enabled_cb = QCheckBox("Aktiviert")
        self._enabled_cb.setChecked(inst.enabled)
        self.form_layout.addRow(self._enabled_cb)

        self._primary_cb = QCheckBox("Primär (klassischer Pfad)")
        self._primary_cb.setChecked(inst.is_primary)
        self.form_layout.addRow(self._primary_cb)

        self._label_edit = QLineEdit(inst.label)
        self.form_layout.addRow("Name", self._label_edit)

        self._hint_edit = QTextEdit(inst.usage_hint)
        self._hint_edit.setPlaceholderText("Optionaler Hinweis für den Agenten, wann diese Instanz genutzt werden soll.")
        self._hint_edit.setMaximumHeight(64)
        self.form_layout.addRow("Agent-Hinweis", self._hint_edit)

        meta = self._adapter.type_meta(inst.provider_id)
        for fld in meta.config_fields:
            w = self._make_field_widget(fld, inst.settings.get(fld.key, fld.default))
            self._field_widgets[fld.key] = w
            label = f"{fld.label} *" if fld.required else fld.label
            self.form_layout.addRow(label, w)
            if fld.help:
                hint = QLabel(fld.help)
                hint.setWordWrap(True)
                hint.setStyleSheet("color: gray; font-size: 10px;")
                self.form_layout.addRow("", hint)

    def _make_field_widget(self, fld, value) -> QWidget:
        if fld.kind == BOOL:
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        if fld.kind == INT:
            w = QSpinBox()
            w.setRange(0, 10_000_000)
            try:
                w.setValue(int(value or 0))
            except (TypeError, ValueError):
                w.setValue(0)
            return w
        if fld.kind == CHOICE:
            w = QComboBox()
            for c in (fld.choices or []):
                w.addItem(str(c), c)
            idx = w.findData(value)
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w
        w = QLineEdit("" if value is None else str(value))
        if fld.kind == SECRET:
            w.setEchoMode(QLineEdit.EchoMode.Password)
        return w

    def _read_field_widget(self, fld, w):
        if fld.kind == BOOL:
            return w.isChecked()
        if fld.kind == INT:
            return w.value()
        if fld.kind == CHOICE:
            return w.currentData()
        return w.text()

    def _flush_form(self) -> None:
        """Write the visible form back into the current instance."""
        if not (0 <= self._current_row < len(self._instances)):
            return
        if not hasattr(self, "_enabled_cb"):
            return
        inst = self._instances[self._current_row]
        inst.enabled = self._enabled_cb.isChecked()
        inst.is_primary = self._primary_cb.isChecked()
        inst.label = self._label_edit.text()
        inst.usage_hint = self._hint_edit.toPlainText()
        meta = self._adapter.type_meta(inst.provider_id)
        for fld in meta.config_fields:
            w = self._field_widgets.get(fld.key)
            if w is not None:
                inst.settings[fld.key] = self._read_field_widget(fld, w)
        # Enforce single primary per type.
        if inst.is_primary:
            for other in self._instances:
                if other is not inst and other.provider_id == inst.provider_id:
                    other.is_primary = False
        self._refresh_list()


class PluginSettingsTab(QWidget):
    """Top-level plugins tab: one sub-tab per registered plugin category."""

    def __init__(self, parent=None):
        super().__init__(parent)
        _ensure_categories()
        layout = QVBoxLayout(self)
        info = QLabel(
            "Plugins bündeln ihre eigene Konfiguration (Endpunkte, Tokens …). Jede "
            "Instanz kann aktiviert, als primär markiert und mit einem Agent-Hinweis "
            "versehen werden. Mehrere Instanzen eines Typs sind möglich."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._tabs = QTabWidget()
        self._panels: Dict[str, _CategoryPanel] = {}
        for cat in list_categories():
            panel = _CategoryPanel(cat)
            self._panels[cat] = panel
            self._tabs.addTab(panel, _CATEGORY_LABELS.get(cat, cat))
        layout.addWidget(self._tabs)

    def load(self, config) -> None:
        """Populate every category panel from ``config.plugins``."""
        plugins = list(getattr(config, "plugins", []) or [])
        for cat, panel in self._panels.items():
            panel.load([p for p in plugins if p.category == cat])

    def apply_to(self, config) -> None:
        """Write the edited instances back into ``config.plugins`` (authoritative)."""
        collected: List[PluginInstanceConfig] = []
        managed = set(self._panels)
        for panel in self._panels.values():
            collected.extend(panel.collect())
        # Preserve instances of any category this tab doesn't manage.
        other = [p for p in (getattr(config, "plugins", []) or []) if p.category not in managed]
        config.plugins = collected + other
