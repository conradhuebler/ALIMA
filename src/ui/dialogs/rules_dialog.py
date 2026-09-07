"""RulesDialog — persönliche Zusatzregeln verwalten. Claude Generated.

The visible counterpart to ``src/core/user_rules.py``: everything the chat's
``propose_rule`` stores shows up here, and everything here can be edited,
silenced, deleted, exported and imported by hand.

Two things the dialog says out loud, because both are easy to get wrong:

* A rule is prose in a prompt. Whether a model follows it — and whether it
  judges a prose condition the way the cataloguer would — is not guaranteed.
* Rules live in the user's config, not in the repository. The same workflow
  therefore behaves differently on another machine; that is the point, and it
  is why every run records which rules it used.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.user_rules import RuleStore, UserRule, available_scope_steps

_INTRO = (
    "Zusatzregeln werden an die Systemprompts der agentischen Schritte, des "
    "Planers und des Chats angehängt — je nach Geltungsbereich. Die Bedingung "
    "ist Prosa und wird vom Modell beurteilt, nicht ausgewertet: ob eine Regel "
    "greift, entscheidet das Modell, und ob es ihr folgt, ebenso. Die "
    "klassische Pipeline liest diese Regeln nicht."
)


class RuleEditDialog(QDialog):
    """Edit one rule: text, prose condition, scope, active. - Claude Generated"""

    def __init__(self, rule: Optional[UserRule] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Regel bearbeiten" if rule else "Neue Regel")
        self.setMinimumWidth(560)
        self._rule = rule

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "Eine klare Anweisung, z.B. 'Formschlagwörter gehören nicht in core_keywords.'"
        )
        self.text_edit.setMaximumHeight(90)
        form.addRow("Regel:", self.text_edit)

        self.when_edit = QLineEdit()
        self.when_edit.setPlaceholderText("optional, z.B. 'bei Überblickswerken'")
        self.when_edit.setToolTip(
            "Prosa-Bedingung. Sie wird der Regel vorangestellt und vom Modell "
            "beurteilt — es gibt keine Auswertung im Code."
        )
        form.addRow("Bedingung:", self.when_edit)

        self.workflows_edit = QLineEdit()
        self.workflows_edit.setPlaceholderText("* (alle), oder z.B. alima_v51*")
        self.workflows_edit.setToolTip("Glob-Muster, mehrere durch Komma getrennt.")
        form.addRow("Workflows:", self.workflows_edit)

        # Steps are picked, not typed: a rule scoped to a step id that does not
        # exist never fires, and nothing would say so. The list comes from the
        # workflow on disk. - Claude Generated
        self.steps_list = QListWidget()
        self.steps_list.setMaximumHeight(150)
        self.steps_list.setToolTip(
            "An welchen Stellen die Regel gelesen wird. Eine Regel über die "
            "fertige Ausgabe gehört zu 'reflection' — dem letzten Schritt eines "
            "Laufs. Jeder Haken kostet Token in genau diesem Prompt."
        )
        self._step_items = {}
        for step_id, desc in available_scope_steps():
            label = f"{step_id} — {desc}" if desc else step_id
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, step_id)
            item.setToolTip(desc or step_id)
            self.steps_list.addItem(item)
            self._step_items[step_id] = item
        form.addRow("Schritte:", self.steps_list)

        self.steps_extra = QLineEdit()
        self.steps_extra.setPlaceholderText("zusätzliche Glob-Muster, z.B. selection*")
        self.steps_extra.setToolTip(
            "Für Muster, die keinem einzelnen Schritt entsprechen. Mehrere durch "
            "Komma getrennt; werden zu den Haken oben addiert."
        )
        form.addRow("", self.steps_extra)

        self.enabled_box = QCheckBox("aktiv — gilt ab dem nächsten Lauf")
        form.addRow("", self.enabled_box)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Warum diese Regel? Wird als Herkunft gespeichert.")
        form.addRow("Notiz:", self.note_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if rule is not None:
            self.text_edit.setPlainText(rule.text)
            self.when_edit.setText(rule.applies_when)
            self.workflows_edit.setText(", ".join(rule.workflows))
            self._preselect_steps(rule.steps)
            self.enabled_box.setChecked(rule.enabled)
            self.note_edit.setText(str(rule.origin.get("note", "") or ""))
        else:
            self.enabled_box.setChecked(True)

    def _preselect_steps(self, steps: List[str]) -> None:
        """Tick the known steps; anything else goes into the glob field."""
        extra = []
        for pattern in steps:
            item = self._step_items.get(pattern)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                extra.append(pattern)
        self.steps_extra.setText(", ".join(extra))

    def _selected_steps(self) -> List[str]:
        picked = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._step_items.values()
            if item.checkState() == Qt.CheckState.Checked
        ]
        picked += self._split(self.steps_extra.text())
        # No pick at all means "everywhere" — same default as the store.
        return picked or ["*"]

    @staticmethod
    def _split(value: str) -> List[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    def result_rule(self) -> Optional[UserRule]:
        """The edited rule, or None when the text is empty."""
        text = self.text_edit.toPlainText().strip()
        if not text:
            return None
        base = self._rule
        origin = dict(base.origin) if base else {"source": "manual"}
        note = self.note_edit.text().strip()
        if note:
            origin["note"] = note
        return UserRule(
            id=base.id if base else "",
            text=text,
            applies_when=self.when_edit.text().strip(),
            workflows=self._split(self.workflows_edit.text()) or ["*"],
            steps=self._selected_steps(),
            enabled=self.enabled_box.isChecked(),
            origin=origin,
        )


class RulesDialog(QDialog):
    """List, edit, enable/disable, delete, export and import rules."""

    _COLUMNS = ("Aktiv", "Regel", "Bedingung", "Gilt für", "Herkunft")

    def __init__(self, parent: Optional[QWidget] = None, store: Optional[RuleStore] = None):
        super().__init__(parent)
        self.setWindowTitle("Persönliche Zusatzregeln")
        self.resize(940, 520)
        self.store = store or RuleStore()
        self._rules: List[UserRule] = []

        layout = QVBoxLayout(self)

        intro = QLabel(_INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.path_label = QLabel(str(self.store.path))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        self.table = QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self._COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(lambda *_: self._edit_selected())
        layout.addWidget(self.table)

        actions = QHBoxLayout()
        self.new_btn = QPushButton("Neu…")
        self.edit_btn = QPushButton("Bearbeiten…")
        self.delete_btn = QPushButton("Löschen")
        self.export_btn = QPushButton("Exportieren…")
        self.import_btn = QPushButton("Importieren…")
        self.export_btn.setToolTip("Schreibt die Regeln samt Herkunft in eine teilbare Datei.")
        self.import_btn.setToolTip(
            "Liest eine Regeldatei ein. Die Herkunft bleibt erhalten; importierte "
            "Regeln sind zunächst inaktiv."
        )
        for btn in (self.new_btn, self.edit_btn, self.delete_btn):
            actions.addWidget(btn)
        actions.addStretch()
        actions.addWidget(self.export_btn)
        actions.addWidget(self.import_btn)
        layout.addLayout(actions)

        self.new_btn.clicked.connect(self._new_rule)
        self.edit_btn.clicked.connect(self._edit_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.export_btn.clicked.connect(self._export)
        self.import_btn.clicked.connect(self._import)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.reload()

    # -- table ---------------------------------------------------------

    def reload(self) -> None:
        self._rules = self.store.load()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._rules))
        for row, rule in enumerate(self._rules):
            active = QTableWidgetItem()
            active.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            active.setCheckState(
                Qt.CheckState.Checked if rule.enabled else Qt.CheckState.Unchecked
            )
            active.setData(Qt.ItemDataRole.UserRole, rule.id)
            self.table.setItem(row, 0, active)

            text_item = QTableWidgetItem(rule.text)
            text_item.setToolTip(f"{rule.id}\n\n{rule.text}")
            self.table.setItem(row, 1, text_item)
            self.table.setItem(row, 2, QTableWidgetItem(rule.applies_when))
            self.table.setItem(row, 3, QTableWidgetItem(rule.scope_label()))
            origin_item = QTableWidgetItem(rule.origin_label())
            origin_item.setToolTip(
                "\n".join(f"{k}: {v}" for k, v in sorted(rule.origin.items())) or "—"
            )
            self.table.setItem(row, 4, origin_item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def _selected_rule(self) -> Optional[UserRule]:
        row = self.table.currentRow()
        if 0 <= row < len(self._rules):
            return self._rules[row]
        return None

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Checkbox toggled → persist immediately (reversible, so no prompt)."""
        if item.column() != 0:
            return
        rule_id = item.data(Qt.ItemDataRole.UserRole)
        if not rule_id:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        if not self.store.set_enabled(str(rule_id), enabled):
            QMessageBox.warning(self, "Regeln", "Die Regel konnte nicht gespeichert werden.")
        self.reload()

    # -- actions -------------------------------------------------------

    def _new_rule(self) -> None:
        dialog = RuleEditDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        rule = dialog.result_rule()
        if rule is None:
            QMessageBox.information(self, "Regeln", "Ohne Regeltext wird nichts gespeichert.")
            return
        stored = self.store.add(
            rule.text,
            applies_when=rule.applies_when,
            workflows=rule.workflows,
            steps=rule.steps,
            enabled=rule.enabled,
            origin=rule.origin,
        )
        if stored is None:
            QMessageBox.warning(self, "Regeln", "Die Regel konnte nicht gespeichert werden.")
        self.reload()

    def _edit_selected(self) -> None:
        rule = self._selected_rule()
        if rule is None:
            return
        dialog = RuleEditDialog(rule=rule, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        edited = dialog.result_rule()
        if edited is None:
            QMessageBox.information(self, "Regeln", "Ohne Regeltext wird nichts gespeichert.")
            return
        if not self.store.update(edited):
            QMessageBox.warning(self, "Regeln", "Die Regel konnte nicht gespeichert werden.")
        self.reload()

    def _delete_selected(self) -> None:
        rule = self._selected_rule()
        if rule is None:
            return
        answer = QMessageBox.question(
            self,
            "Regel löschen",
            f"Diese Regel samt Herkunft löschen?\n\n{rule.text}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.store.remove(rule.id):
            QMessageBox.warning(self, "Regeln", "Die Regel konnte nicht gelöscht werden.")
        self.reload()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Regeln exportieren", "alima_rules.yaml", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        ok, count = self.store.export(path)
        if not ok:
            QMessageBox.warning(self, "Regeln", "Export fehlgeschlagen.")
            return
        QMessageBox.information(
            self, "Regeln", f"{count} Regel(n) nach {path} geschrieben — Herkunft unverändert."
        )

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Regeln importieren", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        ok, added = self.store.import_file(path, activate=False)
        if not ok:
            QMessageBox.warning(self, "Regeln", "Import fehlgeschlagen.")
            return
        self.reload()
        QMessageBox.information(
            self,
            "Regeln",
            f"{len(added)} Regel(n) übernommen. Sie sind zunächst inaktiv — "
            "setze den Haken bei den Regeln, die gelten sollen.",
        )
