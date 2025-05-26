# Füge diese Imports hinzu
import sys
import os
import subprocess
import datetime
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
)


# Diese Klasse fügen wir als Worker für den Update-Prozess hinzu
class GitUpdateWorker(QThread):
    update_progress = pyqtSignal(str)
    update_finished = pyqtSignal(bool, str)

    def __init__(self, repo_path=None, target_commit=None):
        super().__init__()
        self.repo_path = repo_path or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.target_commit = (
            target_commit  # Spezifischer Commit-Hash oder Branch/Tag-Name
        )
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

    def run(self):
        try:
            # Prüfe, ob Git installiert ist
            self.update_progress.emit("Prüfe Git-Installation...")
            try:
                subprocess.run(["git", "--version"], check=True, capture_output=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                self.update_finished.emit(
                    False, "Git ist nicht installiert oder nicht im PATH verfügbar."
                )
                return

            # Prüfe, ob das Verzeichnis ein Git-Repository ist
            self.update_progress.emit("Prüfe Git-Repository...")
            if not os.path.exists(os.path.join(self.repo_path, ".git")):
                self.update_finished.emit(
                    False, "Das Programm-Verzeichnis ist kein Git-Repository."
                )
                return

            # Aktuellen Commit speichern für möglichen Rollback
            current_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            # Fetch der neuesten Änderungen
            self.update_progress.emit("Hole neueste Updates...")
            subprocess.run(["git", "fetch", "--all"], cwd=self.repo_path, check=True)

            if self.target_commit:
                # Zielcommit ist angegeben - prüfe, ob gültig
                self.update_progress.emit(f"Prüfe Ziel-Commit: {self.target_commit}...")

                # Prüfe, ob der Commit existiert
                try:
                    subprocess.run(
                        ["git", "rev-parse", "--verify", self.target_commit],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    self.update_finished.emit(
                        False,
                        f"Der angegebene Commit '{self.target_commit}' existiert nicht.",
                    )
                    return

                # Überprüfe das Datum des Commits
                try:
                    commit_date_str = subprocess.run(
                        ["git", "show", "-s", "--format=%aD", self.target_commit],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Konvertiere das Datum-String in ein datetime-Objekt
                    commit_date = datetime.datetime.strptime(
                        commit_date_str, "%a, %d %b %Y %H:%M:%S %z"
                    )

                    # Prüfe, ob das Commit-Datum nach dem Mindestdatum liegt
                    if commit_date < self.min_allowed_date:
                        self.update_finished.emit(
                            False,
                            f"Der Commit '{self.target_commit}' vom {commit_date.strftime('%d.%m.%Y %H:%M')} "
                            f"ist zu alt. Aus Sicherheitsgründen sind nur Commits ab dem "
                            f"{self.min_allowed_date.strftime('%d.%m.%Y %H:%M')} erlaubt.",
                        )
                        return

                except subprocess.CalledProcessError as e:
                    self.update_finished.emit(
                        False, f"Fehler beim Prüfen des Commit-Datums: {str(e)}"
                    )
                    return
                except ValueError as e:
                    self.update_finished.emit(
                        False, f"Fehler beim Verarbeiten des Datums: {str(e)}"
                    )
                    return

                # Wechsle zum Zielcommit
                self.update_progress.emit(f"Wechsle zu Commit: {self.target_commit}...")

                # Status speichern, um zu prüfen, ob es ungespeicherte Änderungen gibt
                git_status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

                if git_status:
                    self.update_finished.emit(
                        False,
                        "Es gibt ungespeicherte Änderungen. Bitte speichern oder verwerfen Sie diese zuerst.",
                    )
                    return

                try:
                    # Checkout zum Zielcommit
                    subprocess.run(
                        ["git", "checkout", self.target_commit],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True,
                    )

                    self.update_finished.emit(
                        True,
                        f"Erfolgreich zu Commit '{self.target_commit}' gewechselt.",
                    )

                except subprocess.CalledProcessError as e:
                    # Rollback bei Fehler
                    self.update_progress.emit(
                        "Fehler beim Checkout. Führe Rollback durch..."
                    )
                    subprocess.run(
                        ["git", "checkout", current_commit],
                        cwd=self.repo_path,
                        check=True,
                    )
                    error_msg = f"Fehler beim Checkout zum Ziel-Commit: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}"
                    self.update_finished.emit(False, error_msg)

            else:
                # Normale Update-Logik für den aktuellen Branch
                # Prüfe, ob Updates verfügbar sind
                current_branch = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

                result = subprocess.run(
                    ["git", "rev-list", f"HEAD..origin/{current_branch}", "--count"],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commit_count = int(result.stdout.strip())

                if commit_count == 0:
                    self.update_finished.emit(
                        True, "Das Programm ist bereits auf dem neuesten Stand."
                    )
                    return

                # Prüfe vor dem Pull, ob die neuen Commits den Datums-Sicherheitscheck bestehen
                latest_remote_commit = subprocess.run(
                    ["git", "rev-parse", f"origin/{current_branch}"],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

                # Überprüfe das Datum des neuesten remote Commits
                try:
                    commit_date_str = subprocess.run(
                        ["git", "show", "-s", "--format=%aD", latest_remote_commit],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                    # Konvertiere das Datum-String in ein datetime-Objekt
                    commit_date = datetime.datetime.strptime(
                        commit_date_str, "%a, %d %b %Y %H:%M:%S %z"
                    )

                    # Prüfe, ob das Commit-Datum nach dem Mindestdatum liegt
                    if commit_date < self.min_allowed_date:
                        self.update_finished.emit(
                            False,
                            f"Der neueste Remote-Commit vom {commit_date.strftime('%d.%m.%Y %H:%M')} "
                            f"ist zu alt. Aus Sicherheitsgründen sind nur Commits ab dem "
                            f"{self.min_allowed_date.strftime('%d.%m.%Y %H:%M')} erlaubt.",
                        )
                        return

                except Exception as e:
                    self.update_finished.emit(
                        False, f"Fehler beim Prüfen des Commit-Datums: {str(e)}"
                    )
                    return

                # Pull der neuesten Änderungen
                self.update_progress.emit("Installiere Updates...")

                try:
                    # Pull der neuesten Änderungen
                    subprocess.run(["git", "pull"], cwd=self.repo_path, check=True)
                    self.update_finished.emit(
                        True,
                        f"Update erfolgreich! {commit_count} neue Änderungen wurden installiert.",
                    )

                except subprocess.CalledProcessError as e:
                    # Bei Fehler: Rollback
                    self.update_progress.emit(
                        "Update fehlgeschlagen. Führe Rollback durch..."
                    )
                    subprocess.run(
                        ["git", "reset", "--hard", current_commit],
                        cwd=self.repo_path,
                        check=True,
                    )
                    error_msg = f"Git-Fehler: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}"
                    self.update_finished.emit(False, error_msg)

        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Git-Fehler: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}"
            )
            self.update_finished.emit(False, error_msg)
        except Exception as e:
            self.update_finished.emit(False, f"Update-Fehler: {str(e)}")
