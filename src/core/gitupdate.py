# Füge diese Imports hinzu
import sys
import os
import subprocess
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QProgressDialog


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
            try:
                # Run 'git rev-parse --is-inside-work-tree' to check if it's a valid git repo
                result = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=self.repo_path,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                if result.returncode != 0 or result.stdout.strip() != "true":
                    self.update_finished.emit(
                        False,
                        "Das Programm-Verzeichnis ist kein gültiges Git-Repository.",
                    )
                    return
            except Exception as e:
                self.update_finished.emit(
                    False, f"Fehler bei der Git-Repository-Überprüfung: {str(e)}"
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

                # Wechsle temporär zum Zielcommit
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

                    # Sicherheitscheck: Stelle sicher, dass kritische Dateien noch existieren
                    if not self.verify_critical_files():
                        # Rollback zum ursprünglichen Commit
                        self.update_progress.emit(
                            "Kritische Dateien fehlen. Führe Rollback durch..."
                        )
                        subprocess.run(
                            ["git", "checkout", current_commit],
                            cwd=self.repo_path,
                            check=True,
                        )
                        self.update_finished.emit(
                            False,
                            f"Der Ziel-Commit '{self.target_commit}' scheint wichtige Programmdateien zu entfernen. "
                            f"Die Aktualisierung wurde abgebrochen und zum vorherigen Stand zurückgesetzt.",
                        )
                        return

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

                # Pull der neuesten Änderungen
                self.update_progress.emit("Installiere Updates...")

                # Backup des aktuellen Zustands
                try:
                    # Pull der neuesten Änderungen
                    subprocess.run(["git", "pull"], cwd=self.repo_path, check=True)

                    # Sicherheitscheck: Stelle sicher, dass kritische Dateien noch existieren
                    if not self.verify_critical_files():
                        # Rollback zum ursprünglichen Commit
                        self.update_progress.emit(
                            "Kritische Dateien fehlen. Führe Rollback durch..."
                        )
                        subprocess.run(
                            ["git", "reset", "--hard", current_commit],
                            cwd=self.repo_path,
                            check=True,
                        )
                        self.update_finished.emit(
                            False,
                            "Das Update entfernt wichtige Programmdateien. Die Aktualisierung wurde abgebrochen.",
                        )
                        return

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

    def verify_critical_files(self):
        """
        Überprüft, ob wichtige Programmdateien noch existieren
        nach dem Update oder Checkout.
        """
        # Liste der kritischen Dateien relativ zum Repository-Root
        critical_files = [
            os.path.join("alima", "ui", "mainwindow.py"),  # Hauptfenster
            os.path.join("alima", "__init__.py"),  # Hauptmodul
            os.path.join("alima", "__main__.py"),  # Einstiegspunkt
            # Fügen Sie hier weitere kritische Dateien hinzu
        ]

        for file in critical_files:
            file_path = os.path.join(self.repo_path, file)
            if not os.path.exists(file_path):
                return False

        return True
