"""Git commit/branch selector dialog. Claude Generated.

Extracted verbatim from ``main_window.py`` (F-5 god-file split). Standalone
``QDialog`` used by ``MainWindow.checkout_specific_commit`` (now in
``MainWindowMenuMixin``).
"""
from __future__ import annotations

import datetime
import os
import subprocess

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class CommitSelectorDialog(QDialog):
    def __init__(self, parent=None, repo_path=None):
        super().__init__(parent)
        self.repo_path = repo_path or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.selected_commit = None
        # Mindestdatum für Commits: 26. Mai 2025
        self.min_allowed_date = datetime.datetime(
            2025,
            5,
            26,
            14,
            7,
            16,
            tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
        )

        self.init_ui()
        self.load_commits()

    def init_ui(self):
        self.setWindowTitle("Commit auswählen")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)

        # Informationstext
        info_label = QLabel(
            "Wählen Sie einen spezifischen Commit oder Branch/Tag aus, zu dem gewechselt werden soll. "
            f"Aus Sicherheitsgründen werden nur Commits ab dem {self.min_allowed_date.strftime('%d.%m.%Y')} angezeigt."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-weight: bold; color: #CC0000;")
        layout.addWidget(info_label)

        # Auswahl zwischen Commit, Branch oder Tag
        self.selection_type = QComboBox()
        self.selection_type.addItems(["Branch", "Tag", "Commit"])
        self.selection_type.currentIndexChanged.connect(self.update_selection_list)
        layout.addWidget(self.selection_type)

        # Auswahlliste für Branches, Tags oder Commits
        self.commit_list = QComboBox()
        self.commit_list.setEditable(True)  # Erlaubt manuelle Eingabe
        layout.addWidget(self.commit_list)

        # Commit-Details anzeigen
        self.details_label = QLabel("Commit-Details:")
        layout.addWidget(self.details_label)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(200)
        layout.addWidget(self.details_text)

        self.commit_list.currentTextChanged.connect(self.show_commit_details)

        # Buttons
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("Auswählen")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)  # Standardmäßig deaktiviert

        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def load_commits(self):
        self.update_selection_list()

    def check_commit_date(self, commit_ref):
        """
        Überprüft, ob ein Commit nach dem Mindestdatum erstellt wurde.

        Returns:
            tuple: (bool, datetime) - (Ist gültig?, Commit-Datum)
        """
        try:
            # Commit-Datum abrufen
            result = subprocess.run(
                ["git", "show", "-s", "--format=%aD", commit_ref],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            commit_date_str = result.stdout.strip()

            # Konvertiere das Datum-String in ein datetime-Objekt
            commit_date = datetime.datetime.strptime(
                commit_date_str, "%a, %d %b %Y %H:%M:%S %z"
            )

            # Prüfe, ob das Commit-Datum nach dem Mindestdatum liegt
            is_valid = commit_date >= self.min_allowed_date

            return is_valid, commit_date

        except Exception:
            return False, None

    def get_filtered_commits(self, command, extract_func=None):
        """
        Führt einen Git-Befehl aus und filtert die Ergebnisse nach Datum.

        Args:
            command: Git-Befehlszeile als Liste
            extract_func: Optional Funktion zur Extraktion des Commit-Refs

        Returns:
            list: Liste der gültigen Commits/Branches/Tags
        """
        try:
            result = subprocess.run(
                command, cwd=self.repo_path, check=True, capture_output=True, text=True
            )
            all_items = result.stdout.strip().split("\n")

            valid_items = []
            for item in all_items:
                if not item.strip():
                    continue

                # Falls nötig, extrahiere die Commit-Referenz
                if extract_func:
                    commit_ref = extract_func(item)
                else:
                    commit_ref = item

                # Prüfe das Datum
                is_valid, _ = self.check_commit_date(commit_ref)
                if is_valid:
                    valid_items.append(item)

            return valid_items

        except subprocess.CalledProcessError:
            return []

    def update_selection_list(self):
        selection_type = self.selection_type.currentText()
        self.commit_list.clear()

        try:
            if selection_type == "Branch":
                # Lokale Branches laden und nach Datum filtern
                local_branches = self.get_filtered_commits(
                    ["git", "branch", "--format=%(refname:short)"]
                )

                # Remote Branches laden und nach Datum filtern
                remote_branches = self.get_filtered_commits(
                    ["git", "branch", "-r", "--format=%(refname:short)"]
                )

                all_branches = local_branches + remote_branches
                self.commit_list.addItems(
                    sorted([branch for branch in all_branches if branch.strip()])
                )

            elif selection_type == "Tag":
                # Tags laden und nach Datum filtern
                tags = self.get_filtered_commits(["git", "tag"])
                self.commit_list.addItems(sorted([tag for tag in tags if tag.strip()]))

            elif selection_type == "Commit":
                # Letzte 50 Commits laden und nach Datum filtern
                commits = self.get_filtered_commits(
                    ["git", "log", "-50", "--pretty=format:%h - %s (%an, %ar)"],
                    lambda x: x.split(" - ")[0].strip(),
                )
                self.commit_list.addItems(
                    [commit for commit in commits if commit.strip()]
                )

        except Exception as e:
            self.details_text.setText(f"Fehler beim Laden der Auswahl: {str(e)}")

        # Initialzustand setzen
        if self.commit_list.count() > 0:
            self.commit_list.setCurrentIndex(0)
        else:
            self.details_text.setText(
                f"Keine Einträge gefunden, die nach dem Mindestdatum ({self.min_allowed_date.strftime('%d.%m.%Y')}) erstellt wurden."
            )

    def show_commit_details(self, commit_ref):
        if not commit_ref:
            self.details_text.clear()
            self.ok_button.setEnabled(False)
            return

        # Bei Commits nur den Hash extrahieren
        if self.selection_type.currentText() == "Commit" and " - " in commit_ref:
            commit_ref = commit_ref.split(" - ")[0].strip()

        try:
            # Prüfe das Datum
            is_valid, commit_date = self.check_commit_date(commit_ref)

            if not is_valid:
                self.details_text.setText(
                    f"FEHLER: Der ausgewählte Commit wurde vor dem erlaubten Mindestdatum "
                    f"({self.min_allowed_date.strftime('%d.%m.%Y')}) erstellt.\n\n"
                    f"Commit-Datum: {commit_date.strftime('%d.%m.%Y %H:%M') if commit_date else 'Unbekannt'}"
                )
                self.ok_button.setEnabled(False)
                return

            # Commit-Details anzeigen
            result = subprocess.run(
                ["git", "show", "--stat", commit_ref],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            details = result.stdout.strip()

            date_info = f"Commit-Datum: {commit_date.strftime('%d.%m.%Y %H:%M')}\n\n"
            self.details_text.setText(date_info + details)

            # Aktiviere den OK-Button
            self.ok_button.setEnabled(True)

        except subprocess.CalledProcessError:
            self.details_text.setText(f"Ungültige Referenz: {commit_ref}")
            self.ok_button.setEnabled(False)
        except Exception as e:
            self.details_text.setText(
                f"Fehler beim Anzeigen der Commit-Details: {str(e)}"
            )
            self.ok_button.setEnabled(False)

    def get_selected_commit(self):
        commit_ref = self.commit_list.currentText()

        # Bei Commits nur den Hash extrahieren
        if self.selection_type.currentText() == "Commit" and " - " in commit_ref:
            commit_ref = commit_ref.split(" - ")[0].strip()

        # Führe eine letzte Prüfung durch, um sicherzustellen, dass das Datum passt
        is_valid, _ = self.check_commit_date(commit_ref)
        if not is_valid:
            return None

        return commit_ref


