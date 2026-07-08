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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
    "lookup": "🔖 Lookups (API)",
}


def _ensure_categories() -> None:
    import src.core.search  # noqa: F401
    import src.utils.input_sources  # noqa: F401
    import src.utils.lookups  # noqa: F401


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
            w = self._make_field_widget(
                fld, inst.settings.get(fld.key, fld.default), instance_id=inst.instance_id
            )
            self._field_widgets[fld.key] = w
            label = f"{fld.label} *" if fld.required else fld.label
            self.form_layout.addRow(label, w)
            if fld.help:
                hint = QLabel(fld.help)
                hint.setWordWrap(True)
                hint.setStyleSheet("color: gray; font-size: 10px;")
                self.form_layout.addRow("", hint)

    def _make_field_widget(self, fld, value, instance_id: str = "") -> QWidget:
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
            # Surface an active env-var override (runtime wins over the stored
            # value; the stored value stays editable). - Claude Generated
            import os

            from src.core.plugins.schema import env_var_name

            var = env_var_name(instance_id, fld.key)
            if os.environ.get(var):
                w.setPlaceholderText(f"überschrieben durch Umgebungsvariable {var}")
                w.setToolTip(
                    f"Zur Laufzeit gewinnt die Umgebungsvariable {var}; der hier "
                    "gespeicherte Wert wird dann ignoriert."
                )
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


class _ExportBundleDialog(QDialog):
    """Collect bundle meta + a checklist of which search instances to export - Claude Generated."""

    def __init__(self, instances, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bundle exportieren")
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("z. B. ub-freiberg")
        self.label_edit = QLineEdit()
        self.institution_edit = QLineEdit()
        form.addRow("Bundle-ID:", self.id_edit)
        form.addRow("Label:", self.label_edit)
        form.addRow("Institution:", self.institution_edit)
        v.addLayout(form)
        v.addWidget(QLabel("Zu exportierende Suchquellen (Secrets werden entfernt):"))
        self.list = QListWidget()
        for iid, label in instances:
            item = QListWidgetItem(f"{iid}  —  {label}" if label else iid)
            item.setData(Qt.ItemDataRole.UserRole, iid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(item)
        v.addWidget(self.list)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected_instance_ids(self) -> List[str]:
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out


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

        # Directory-plugin rescan with the Tier-2 approval dialog (config load
        # runs headless = deny; this button is the interactive path). The
        # enable_code_plugins master gate lives here too (was config.json-only).
        # - Claude Generated
        scan_row = QHBoxLayout()
        self._code_plugins_cb = QCheckBox("Code-Plugins (Tier 2) erlauben")
        self._code_plugins_cb.setToolTip(
            "Master-Schalter für Code-Plugins aus ~/.config/alima/plugins/.\n"
            "Zusätzlich erfordert jedes Code-Plugin eine einzelne Freigabe\n"
            "(Security-Scan + Hash). Freigegebene Plugins laufen in-process\n"
            "mit vollen Rechten — kein Sandbox. Deklarative Plugins (nur\n"
            "plugin.toml) laden unabhängig von diesem Schalter."
        )
        scan_row.addWidget(self._code_plugins_cb)
        self._scan_btn = QPushButton("📂 Plugin-Verzeichnis scannen…")
        self._scan_btn.setToolTip(
            "Scannt ~/.config/alima/plugins/ erneut; nicht freigegebene "
            "Code-Plugins werden mit Scan-Findings zur Freigabe angezeigt."
        )
        self._scan_btn.clicked.connect(self._rescan_plugins)
        scan_row.addWidget(self._scan_btn)
        scan_row.addStretch(1)
        layout.addLayout(scan_row)

        layout.addWidget(self._build_bundle_group())

    def load(self, config) -> None:
        """Populate every category panel from ``config.plugins``."""
        plugins = list(getattr(config, "plugins", []) or [])
        for cat, panel in self._panels.items():
            panel.load([p for p in plugins if p.category == cat])
        self._code_plugins_cb.setChecked(
            bool(getattr(getattr(config, "system_config", None), "enable_code_plugins", False))
        )

    def apply_to(self, config) -> None:
        """Write the edited instances back into ``config.plugins`` (authoritative)."""
        collected: List[PluginInstanceConfig] = []
        managed = set(self._panels)
        for panel in self._panels.values():
            collected.extend(panel.collect())
        # Preserve instances of any category this tab doesn't manage.
        other = [p for p in (getattr(config, "plugins", []) or []) if p.category not in managed]
        config.plugins = collected + other
        if getattr(config, "system_config", None) is not None:
            config.system_config.enable_code_plugins = self._code_plugins_cb.isChecked()
        self._warn_on_operator_urls(collected)

    # ---- Institutional bundles (plugins + advisory profile) --------------
    def _build_bundle_group(self) -> QGroupBox:
        """Install / list / remove / export institutional bundles - Claude Generated."""
        box = QGroupBox("📦 Bundles — Einrichtungs-Deployment")
        v = QVBoxLayout(box)
        hint = QLabel(
            "Plugins + beratendes Config-Profil gebündelt installieren oder die "
            "aktuelle Einrichtung exportieren. Secrets werden nur deklariert, nie "
            "mitgeliefert (per-User via GUI/ENV nachtragen)."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self._bundle_list = QListWidget()
        v.addWidget(self._bundle_list)

        row = QHBoxLayout()
        for text, slot in (
            ("Installieren…", self._install_bundle_clicked),
            ("Entfernen", self._remove_bundle_clicked),
            ("Exportieren…", self._export_bundle_clicked),
            ("Aktualisieren", self._refresh_bundles),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        v.addLayout(row)
        self._refresh_bundles()
        return box

    def _refresh_bundles(self) -> None:
        from src.utils import bundle as bundle_mod

        self._bundle_list.clear()
        try:
            bundles = bundle_mod.list_bundles()
        except Exception as exc:  # never let a bad config break the tab
            self._bundle_list.addItem(f"(Fehler beim Laden: {exc})")
            return
        if not bundles:
            self._bundle_list.addItem("(keine Bundles installiert)")
            return
        for b in bundles:
            plugins = ", ".join(b["plugins"]) or "—"
            item = QListWidgetItem(f"{b['id']}  v{b['version']}  [{b['label']}] — {plugins}")
            item.setData(Qt.ItemDataRole.UserRole, b["id"])
            self._bundle_list.addItem(item)

    def _format_install_report(self, report) -> str:
        lines = [f"Bundle '{report.bundle_id}' v{report.version} installiert."]
        for pid, status, sev in report.plugins:
            extra = "" if sev in ("none", "") else f"  [Scan: {sev}]"
            lines.append(f"  {'✅' if status == 'loaded' else '❌'} {pid} ({status}){extra}")
        if report.profile_keys:
            lines.append(f"Profil (beratend): {', '.join(report.profile_keys)}")
        if report.integrity_mismatches:
            lines.append(f"⚠️ Integritäts-Abweichung: {', '.join(report.integrity_mismatches)}")
        unmet = [s for s in report.required_secrets if not s["satisfied"]]
        for s in report.required_secrets:
            state = "gesetzt" if s["satisfied"] else "FEHLT"
            lines.append(f"Secret [{state}]: {s['plugin']}.{s['key']} (env {s['env_var']})")
        if unmet:
            lines.append("→ Offene Secrets im Plugin-Formular oben oder per ENV nachtragen.")
        for w in report.warnings:
            lines.append(f"⚠️ {w}")
        return "\n".join(lines)

    def _install_bundle_clicked(self) -> None:
        from src.utils import bundle as bundle_mod
        from src.utils.config_manager import ConfigManager

        path, _ = QFileDialog.getOpenFileName(
            self, "Bundle wählen (.zip)", "", "Bundle (*.zip);;Alle Dateien (*)"
        )
        if not path:
            return
        confirm = QMessageBox.question(
            self, "Bundle installieren?",
            f"Bundle installieren?\n\n{path}\n\nEnthaltene Code-Plugins werden "
            "freigegeben und laufen in-process mit vollen Rechten. Nur bei "
            "vertrauenswürdiger Quelle fortfahren.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            report = bundle_mod.install_bundle(path)
        except Exception as exc:
            QMessageBox.critical(self, "Bundle-Installation", f"Fehlgeschlagen:\n{exc}")
            return
        self.load(ConfigManager().load_config())  # reflect newly seeded instances
        self._refresh_bundles()
        QMessageBox.information(self, "Bundle installiert", self._format_install_report(report))

    def _remove_bundle_clicked(self) -> None:
        from src.utils import bundle as bundle_mod
        from src.utils.config_manager import ConfigManager

        item = self._bundle_list.currentItem()
        bid = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not bid:
            QMessageBox.information(self, "Bundle entfernen", "Kein Bundle ausgewählt.")
            return
        if QMessageBox.question(
            self, "Bundle entfernen?",
            f"Bundle '{bid}' entfernen?\nInstanzen, Freigaben und Plugin-Dateien "
            "werden zurückgesetzt (eigene Änderungen an anderen Keys bleiben).",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            bundle_mod.remove_bundle(bid)
        except Exception as exc:
            QMessageBox.critical(self, "Bundle entfernen", f"Fehlgeschlagen:\n{exc}")
            return
        self.load(ConfigManager().load_config())
        self._refresh_bundles()
        QMessageBox.information(self, "Bundle entfernt", f"'{bid}' entfernt.")

    def _export_bundle_clicked(self) -> None:
        from src.utils import bundle as bundle_mod
        from src.utils.config_manager import ConfigManager

        config = ConfigManager().load_config()
        instances = [
            (p.instance_id, p.label) for p in config.plugins
            if p.category == "search_provider" and getattr(p, "enabled", True)
        ]
        if not instances:
            QMessageBox.information(
                self, "Bundle exportieren", "Keine aktivierten Suchquellen zum Export."
            )
            return
        dlg = _ExportBundleDialog(instances, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        bid = dlg.id_edit.text().strip()
        selected = dlg.selected_instance_ids()
        if not bid:
            QMessageBox.information(self, "Bundle exportieren", "Bundle-ID fehlt.")
            return
        if not selected:
            QMessageBox.information(self, "Bundle exportieren", "Keine Quelle ausgewählt.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Bundle speichern", f"{bid}.zip", "Bundle (*.zip)"
        )
        if not path:
            return
        try:
            out = bundle_mod.export_bundle(
                path, bundle_id=bid, label=dlg.label_edit.text().strip(),
                institution=dlg.institution_edit.text().strip(), instance_ids=selected,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Bundle exportieren", f"Fehlgeschlagen:\n{exc}")
            return
        QMessageBox.information(
            self, "Bundle exportiert",
            f"Exportiert:\n{out}\n\nExportiert wird der zuletzt gespeicherte Stand "
            "(Einstellungen vorher speichern). Secrets wurden entfernt und nur "
            "deklariert — Endpunkte/IDs vor Verteilung prüfen.",
        )

    def _rescan_plugins(self) -> None:
        """Re-run directory discovery with the interactive approval gate - Claude Generated."""
        from src.utils.config_manager import ConfigManager
        from src.utils.plugin_discovery import discover_plugins

        cm = ConfigManager()
        config = cm.load_config()
        # The checkbox governs the scan live (persisted on settings save).
        enable_code = self._code_plugins_cb.isChecked()
        result = discover_plugins(
            config,
            plugins_dir=cm.plugins_dir,
            approve_cb=self._approval_dialog,
            enable_code_plugins=enable_code,
            persist=lambda: cm.save_config(config),
        )
        if result is None:
            QMessageBox.information(
                self, "Plugin-Verzeichnis",
                f"Kein Plugin-Verzeichnis gefunden:\n{cm.plugins_dir}",
            )
            return
        self.load(config)  # reflect freshly merged instances/types
        lines = []
        for p in result.plugins:
            mark = {"loaded": "✅", "blocked": "⛔", "denied": "🚫", "error": "❌"}.get(p.status, "•")
            detail = f" — {p.detail}" if p.detail else ""
            lines.append(f"{mark} {p.manifest.id} [{p.status}]{detail}")
        if not lines:
            lines = ["(keine Plugins im Verzeichnis)"]
        if not enable_code and any(p.status == "blocked" for p in result.plugins):
            lines.append("")
            lines.append("Hinweis: Code-Plugins sind deaktiviert (enable_code_plugins).")
        QMessageBox.information(self, "Plugin-Scan", "\n".join(lines))

    def _approval_dialog(self, manifest, findings, digest) -> bool:
        """Tier-2 consent dialog: manifest + severity-sorted findings + hash - Claude Generated.

        Default is decline; approval pins the SHA-256 (any file change re-prompts).
        """
        by_sev = {"high": 0, "medium": 0, "low": 0}
        for f in findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        summary = ", ".join(f"{n}× {sev}" for sev, n in by_sev.items() if n) or "keine Findings"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Code-Plugin freigeben?")
        box.setText(
            f"<b>{manifest.label}</b> (id: <code>{manifest.id}</code>, "
            f"Kategorie: {manifest.category})<br>"
            f"Quelle: <code>{manifest.source_dir}</code><br><br>"
            f"Security-Scan: <b>{summary}</b><br><br>"
            "⚠️ Nach Freigabe läuft dieses Plugin <b>in-process mit vollen "
            "Rechten</b> — der Scan ist eine Abschreckung, <b>kein Sandbox</b>. "
            "Nur freigeben, wenn Sie der Quelle vertrauen.<br>"
            "Jede Dateiänderung erfordert eine erneute Freigabe (Hash-Pinning)."
        )
        detail_lines = [str(f) for f in findings] or ["(keine Findings)"]
        detail_lines += ["", f"SHA-256: {digest}"]
        box.setDetailedText("\n".join(detail_lines))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.button(QMessageBox.StandardButton.Yes).setText("Freigeben")
        box.button(QMessageBox.StandardButton.No).setText("Ablehnen")
        return box.exec() == QMessageBox.StandardButton.Yes

    def _warn_on_operator_urls(self, instances: List[PluginInstanceConfig]) -> None:
        """Sanity warnings for configured base URLs (never blocks the save) - Claude Generated.

        Uses ``net_guard.check_operator_url`` posture (a): scheme/parseability
        only — intranet endpoints are legitimate and are NOT flagged.
        """
        from src.core.plugins.schema import URL
        from src.utils.net_guard import check_operator_url

        warnings: List[str] = []
        for inst in instances:
            if not getattr(inst, "enabled", True):
                continue
            try:
                fields = get_category(inst.category).config_fields(inst.provider_id)
            except Exception:
                continue
            for fld in fields:
                if fld.kind != URL:
                    continue
                value = (inst.settings or {}).get(fld.key)
                for msg in check_operator_url(str(value or "")):
                    warnings.append(f"[{inst.label or inst.instance_id}] {msg}")
        if warnings:
            QMessageBox.warning(
                self,
                "Plugin-URLs prüfen",
                "Gespeichert. Hinweise zu konfigurierten Endpunkten:\n\n"
                + "\n".join(f"• {w}" for w in warnings),
            )
