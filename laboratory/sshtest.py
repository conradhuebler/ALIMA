import sys
import os
import json
import requests
import socket
import threading
import time
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QTextEdit, QPushButton, QComboBox, QLabel,
                           QDialog, QLineEdit, QFormLayout, QMessageBox, QCheckBox,
                           QFileDialog, QGroupBox, QSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSettings

# Logger einrichten
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('llm_client.log', mode='a')
                    ])
logger = logging.getLogger('LLM_Client')

# Stellen Sie sicher, dass paramiko installiert ist
try:
    import paramiko
except ImportError:
    logger.info("Paramiko nicht gefunden, wird installiert...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko
    logger.info("Paramiko wurde erfolgreich installiert")

class SSHTunnelThread(QThread):
    tunnel_status = pyqtSignal(bool, str)
    debug_message = pyqtSignal(str)

    def __init__(self, ssh_config, parent=None):
        super().__init__(parent)
        self.ssh_config = ssh_config
        self.running = False
        self.client = None
        self.local_server = None

    def log_debug(self, message):
        logger.debug(message)
        self.debug_message.emit(message)

    def run(self):
        try:
            self.log_debug(f"Starte SSH-Verbindung zu {self.ssh_config['host']}:{self.ssh_config['port']}")
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connection_args = {
                'hostname': self.ssh_config['host'],
                'username': self.ssh_config['username'],
                'port': self.ssh_config['port']
            }

            # Authentifizierung
            if self.ssh_config['use_key']:
                key_path = self.ssh_config['key_path']
                self.log_debug(f"Verwende SSH-Key: {key_path}")
                if not os.path.exists(key_path):
                    raise FileNotFoundError(f"SSH-Key nicht gefunden: {key_path}")
                connection_args['key_filename'] = key_path
            else:
                self.log_debug("Verwende Passwort-Authentifizierung")
                connection_args['password'] = self.ssh_config['password']

            self.log_debug("Verbinde mit SSH-Server...")
            self.client.connect(**connection_args, timeout=10)
            self.log_debug("SSH-Verbindung erfolgreich hergestellt")

            # Tunnel einrichten
            local_port = self.ssh_config['local_port']
            remote_host = self.ssh_config['remote_host']
            remote_port = self.ssh_config['remote_port']

            self.log_debug(f"Richte Port-Forwarding ein: localhost:{local_port} -> {remote_host}:{remote_port}")

            # Teste, ob der lokale Port bereits belegt ist
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('127.0.0.1', local_port))
                sock.close()
            except socket.error as e:
                self.log_debug(f"Port {local_port} ist bereits belegt: {e}")
                self.tunnel_status.emit(False, f"Port {local_port} bereits belegt")
                return

            # Führe einen Port-Forward durch
            transport = self.client.get_transport()

            # Öffne direkten Kanal zwischen lokalem und Remote-Port
            self.log_debug("Initiiere Portforwarding...")
            try:
                transport.request_port_forward('127.0.0.1', local_port)
                # Starte Thread zum Weiterleit von lokalem Port zu Remote-Server
                forward_thread = threading.Thread(
                    target=self._port_forward_thread,
                    args=(transport, '127.0.0.1', local_port, remote_host, remote_port)
                )
                forward_thread.daemon = True
                forward_thread.start()
                self.log_debug("Portforwarding-Thread gestartet")
            except Exception as e:
                self.log_debug(f"Fehler beim Port-Forwarding: {e}")
                self.tunnel_status.emit(False, f"Fehler beim Port-Forwarding: {e}")
                return

            self.running = True
            self.tunnel_status.emit(True, f"SSH-Tunnel aktiv: localhost:{local_port} -> {remote_host}:{remote_port}")

            # Tunnel aktiv halten und überwachen
            self.log_debug("Starte Tunnel-Monitoring...")
            while self.running:
                if not transport.is_active():
                    self.log_debug("Transport ist nicht mehr aktiv")
                    self.tunnel_status.emit(False, "SSH-Verbindung unterbrochen")
                    break

                # Prüfe ob der Remote-Server noch erreichbar ist
                try:
                    transport.send_ignore()
                    self.log_debug("Keep-alive Pakete gesendet")
                except Exception as e:
                    self.log_debug(f"Fehler beim Keep-alive: {e}")
                    self.tunnel_status.emit(False, f"SSH-Verbindung unterbrochen: {e}")
                    break

                time.sleep(5)

        except paramiko.AuthenticationException:
            self.log_debug("Authentifizierung fehlgeschlagen")
            self.tunnel_status.emit(False, "Authentifizierung fehlgeschlagen")
        except paramiko.SSHException as e:
            self.log_debug(f"SSH-Fehler: {e}")
            self.tunnel_status.emit(False, f"SSH-Fehler: {e}")
        except Exception as e:
            self.log_debug(f"Unerwarteter Fehler: {e}")
            self.tunnel_status.emit(False, f"Fehler: {e}")
        finally:
            self.log_debug("SSH-Thread wird beendet")
            self.stop_tunnel()

    def _port_forward_thread(self, transport, local_host, local_port, remote_host, remote_port):
        """Eigener Thread für das Port-Forwarding"""
        while self.running:
            try:
                transport.open_channel(
                    'direct-tcpip',
                    (remote_host, remote_port),
                    (local_host, local_port)
                )
                time.sleep(1)
            except:
                if self.running:
                    self.log_debug("Fehler im Port-Forwarding-Thread")
                else:
                    break

    def stop_tunnel(self):
        self.running = False
        if self.client:
            try:
                self.log_debug("Schließe SSH-Client...")
                self.client.close()
                self.log_debug("SSH-Client geschlossen")
            except Exception as e:
                self.log_debug(f"Fehler beim Schließen des SSH-Clients: {e}")
            finally:
                self.client = None

        self.tunnel_status.emit(False, "SSH-Tunnel beendet")

class OllamaManager:
    def __init__(self):
        self.logger = logging.getLogger('LLM_Client.OllamaManager')

    def check_ollama(self, host="localhost", port=11434):
        """Prüft, ob Ollama auf dem angegebenen Host:Port verfügbar ist"""
        self.logger.debug(f"Prüfe Ollama auf {host}:{port}")
        try:
            url = f"http://{host}:{port}/api/tags"
            self.logger.debug(f"Sende Anfrage an: {url}")
            response = requests.get(url, timeout=5)
            self.logger.debug(f"Antwort: Status {response.status_code}")

            if response.status_code == 200:
                models = response.json().get('models', [])
                self.logger.debug(f"Gefundene Modelle: {len(models)}")
                return True, models

            self.logger.debug(f"Unerwarteter Statuscode: {response.status_code}")
            return False, None
        except requests.exceptions.ConnectionError as e:
            self.logger.debug(f"Verbindungsfehler: {e}")
            return False, None
        except Exception as e:
            self.logger.debug(f"Allgemeiner Fehler: {e}")
            return False, None

    def query_ollama(self, model, prompt, host="localhost", port=11434):
        """Sendet eine Anfrage an Ollama"""
        self.logger.debug(f"Sende Anfrage an Modell {model} auf {host}:{port}")
        try:
            url = f"http://{host}:{port}/api/generate"
            self.logger.debug(f"API-URL: {url}")
            data = {"model": model, "prompt": prompt}
            response = requests.post(url, json=data)
            self.logger.debug(f"Antwort-Status: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            self.logger.debug(f"Fehlerhafte Antwort: {response.text}")
            return {"error": f"Fehlercode: {response.status_code}"}
        except Exception as e:
            self.logger.debug(f"Ausnahme bei Ollama-Anfrage: {e}")
            return {"error": str(e)}

class SSHConfigDialog(QDialog):
    def __init__(self, ssh_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSH-Verbindung konfigurieren")
        self.resize(450, 400)

        # Verwende vorhandene Konfiguration oder erstelle eine neue
        self.ssh_config = ssh_config or {}

        layout = QVBoxLayout()

        # SSH-Server Konfiguration
        ssh_group = QGroupBox("SSH Server")
        ssh_form = QFormLayout()

        self.host_input = QLineEdit()
        self.host_input.setText(self.ssh_config.get('host', "192.168.1.100"))
        ssh_form.addRow("SSH Host:", self.host_input)

        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.ssh_config.get('port', 22))
        ssh_form.addRow("SSH Port:", self.port_input)

        self.username_input = QLineEdit()
        self.username_input.setText(self.ssh_config.get('username', ""))
        ssh_form.addRow("Benutzername:", self.username_input)

        self.auth_group = QGroupBox("Authentifizierung")
        auth_layout = QVBoxLayout()

        self.use_key_checkbox = QCheckBox("Mit SSH-Key verbinden")
        self.use_key_checkbox.setChecked(self.ssh_config.get('use_key', False))
        self.use_key_checkbox.toggled.connect(self.toggle_auth_method)
        auth_layout.addWidget(self.use_key_checkbox)

        key_layout = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.setText(self.ssh_config.get('key_path', ""))
        self.key_input.setEnabled(self.ssh_config.get('use_key', False))
        key_layout.addWidget(self.key_input)

        self.browse_button = QPushButton("Durchsuchen...")
        self.browse_button.setEnabled(self.ssh_config.get('use_key', False))
        self.browse_button.clicked.connect(self.browse_key_file)
        key_layout.addWidget(self.browse_button)
        auth_layout.addLayout(key_layout)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setText(self.ssh_config.get('password', ""))
        self.password_input.setEnabled(not self.ssh_config.get('use_key', False))
        auth_layout.addWidget(QLabel("Passwort:"))
        auth_layout.addWidget(self.password_input)

        self.auth_group.setLayout(auth_layout)
        ssh_form.addRow(self.auth_group)

        ssh_group.setLayout(ssh_form)
        layout.addWidget(ssh_group)

        # Remote Ollama Server Konfiguration
        ollama_group = QGroupBox("Ollama Server")
        ollama_form = QFormLayout()

        self.ollama_host_input = QLineEdit()
        self.ollama_host_input.setText(self.ssh_config.get('remote_host', "localhost"))
        ollama_form.addRow("Ollama Host:", self.ollama_host_input)

        self.ollama_port_input = QSpinBox()
        self.ollama_port_input.setRange(1, 65535)
        self.ollama_port_input.setValue(self.ssh_config.get('remote_port', 11434))
        ollama_form.addRow("Ollama Port:", self.ollama_port_input)

        self.local_port_input = QSpinBox()
        self.local_port_input.setRange(1024, 65535)
        self.local_port_input.setValue(self.ssh_config.get('local_port', 11435))
        ollama_form.addRow("Lokaler Port:", self.local_port_input)

        ollama_group.setLayout(ollama_form)
        layout.addWidget(ollama_group)

        # Test-Button
        self.test_button = QPushButton("Test-Verbindung")
        self.test_button.clicked.connect(self.test_connection)
        layout.addWidget(self.test_button)

        # Buttons
        button_layout = QHBoxLayout()
        self.connect_button = QPushButton("Speichern")
        self.connect_button.clicked.connect(self.accept)
        button_layout.addWidget(self.connect_button)

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def toggle_auth_method(self, checked):
        self.key_input.setEnabled(checked)
        self.browse_button.setEnabled(checked)
        self.password_input.setEnabled(not checked)

    def browse_key_file(self):
        initial_dir = os.path.expanduser("~/.ssh")
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")

        file_name, _ = QFileDialog.getOpenFileName(
            self, "SSH Private Key auswählen",
            initial_dir,
            "All Files (*)"
        )
        if file_name:
            self.key_input.setText(file_name)

    def test_connection(self):
        # Temporäre Konfiguration erstellen
        temp_config = self.get_ssh_config()

        try:
            # Teste SSH-Verbindung
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connection_args = {
                'hostname': temp_config['host'],
                'username': temp_config['username'],
                'port': temp_config['port'],
                'timeout': 10
            }

            if temp_config['use_key']:
                if not os.path.exists(temp_config['key_path']):
                    QMessageBox.warning(self, "Fehler", f"SSH-Key nicht gefunden: {temp_config['key_path']}")
                    return
                connection_args['key_filename'] = temp_config['key_path']
            else:
                connection_args['password'] = temp_config['password']

            # Verbindung testen
            client.connect(**connection_args)

            # Prüfen, ob Remote Ollama erreichbar ist
            transport = client.get_transport()
            try:
                # Temporärer Channel für Remote-Host-Test
                channel = transport.open_channel(
                    'direct-tcpip',
                    (temp_config['remote_host'], temp_config['remote_port']),
                    ('localhost', 0)  # Nutze einen beliebigen lokalen Port
                )

                if channel.recv_exit_status() == 0:
                    # Remote Ollama scheint erreichbar zu sein
                    QMessageBox.information(self, "Test erfolgreich",
                                     f"SSH-Verbindung zu {temp_config['host']} erfolgreich. Remote Ollama scheint erreichbar.")
                else:
                    QMessageBox.warning(self, "Teilweise erfolgreich",
                               f"SSH-Verbindung zu {temp_config['host']} erfolgreich, aber Remote Ollama unter {temp_config['remote_host']}:{temp_config['remote_port']} nicht erreichbar.")
            except Exception:
                QMessageBox.warning(self, "Teilweise erfolgreich",
                           f"SSH-Verbindung zu {temp_config['host']} erfolgreich, aber Remote Ollama unter {temp_config['remote_host']}:{temp_config['remote_port']} nicht erreichbar.")
            finally:
                client.close()

        except paramiko.AuthenticationException:
            QMessageBox.critical(self, "Authentifizierungsfehler",
                         "Benutzername oder Passwort/Key ist falsch.")
        except paramiko.SSHException as e:
            QMessageBox.critical(self, "SSH-Verbindungsfehler",
                         f"Fehler bei der SSH-Verbindung: {str(e)}")
        except socket.error as e:
            QMessageBox.critical(self, "Verbindungsfehler",
                         f"Konnte keine Verbindung zu {temp_config['host']}:{temp_config['port']} herstellen: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Fehler",
                         f"Unerwarteter Fehler: {str(e)}")

    def get_ssh_config(self):
        return {
            'host': self.host_input.text(),
            'port': self.port_input.value(),
            'username': self.username_input.text(),
            'use_key': self.use_key_checkbox.isChecked(),
            'key_path': self.key_input.text() if self.use_key_checkbox.isChecked() else "",
            'password': self.password_input.text() if not self.use_key_checkbox.isChecked() else "",
            'remote_host': self.ollama_host_input.text(),
            'remote_port': self.ollama_port_input.value(),
            'local_port': self.local_port_input.value()
        }

class QueryThread(QThread):
    response_received = pyqtSignal(dict)
    debug_message = pyqtSignal(str)

    def __init__(self, manager, model, prompt, host, port, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.model = model
        self.prompt = prompt
        self.host = host
        self.port = port
        self.logger = logging.getLogger('LLM_Client.QueryThread')

    def run(self):
        self.logger.debug(f"Starte Anfrage an {self.model} auf {self.host}:{self.port}")
        self.debug_message.emit(f"Sende Anfrage an {self.model} auf {self.host}:{self.port}...")
        response = self.manager.query_ollama(self.model, self.prompt, self.host, self.port)
        self.logger.debug(f"Antwort erhalten: {response is not None}")
        self.response_received.emit(response)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Client mit Ollama und SSH-Tunnel (Windows)")
        self.setGeometry(100, 100, 1000, 800)

        self.logger = logging.getLogger('LLM_Client.MainWindow')
        self.logger.debug("Initialisiere MainWindow")

        # Einstellungen laden
        self.settings = QSettings("OllamaClient", "LLMClient")

        self.ollama_manager = OllamaManager()
        self.ssh_thread = None
        self.ssh_config = self.load_ssh_config()
        self.is_connected_remote = False
        self.query_thread = None

        self.init_ui()

        # Lade gespeicherte Fensterposition und -größe
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Lade gespeichertes UI-Verhalten
        mode_index = self.settings.value("mode_index", 0, int)
        if mode_index in (0, 1):
            self.mode_combo.setCurrentIndex(mode_index)

        self.check_local_ollama()

    def init_ui(self):
        # Hauptwidget und Layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Server-Konfigurationsbereich
        server_group = QGroupBox("Server-Konfiguration")
        server_layout = QVBoxLayout()

        # Servermodus
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Lokal")
        self.mode_combo.addItem("Remote über SSH")
        self.mode_combo.currentIndexChanged.connect(self.toggle_remote_mode)
        mode_layout.addWidget(QLabel("Modus:"))
        mode_layout.addWidget(self.mode_combo)
        server_layout.addLayout(mode_layout)

        # Remote-Verbindungssteuerung
        self.remote_controls = QWidget()
        remote_layout = QHBoxLayout()
        self.remote_controls.setLayout(remote_layout)

        self.config_button = QPushButton("SSH konfigurieren")
        self.config_button.clicked.connect(self.show_ssh_config)
        remote_layout.addWidget(self.config_button)

        self.connect_button = QPushButton("Verbinden")
        self.connect_button.clicked.connect(self.toggle_ssh_connection)
        remote_layout.addWidget(self.connect_button)

        self.status_label = QLabel("Nicht verbunden")
        remote_layout.addWidget(self.status_label)

        server_layout.addWidget(self.remote_controls)
        self.remote_controls.setVisible(False)

        server_group.setLayout(server_layout)
        main_layout.addWidget(server_group)

        # Modell-Auswahl
        model_group = QGroupBox("LLM-Konfiguration")
        model_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Modelle aktualisieren")
        self.refresh_button.clicked.connect(self.refresh_models)
        model_layout.addWidget(self.refresh_button)

        model_layout.addWidget(QLabel("Modell:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Eingabe
        input_group = QGroupBox("Eingabe")
        input_layout = QVBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Geben Sie Ihre Anfrage ein...")
        self.input_text.setMinimumHeight(100)
        input_layout.addWidget(self.input_text)

        self.send_button = QPushButton("Anfrage senden")
        self.send_button.clicked.connect(self.send_request)
        input_layout.addWidget(self.send_button)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Debug-Bereich
        debug_group = QGroupBox("Debug-Ausgaben")
        debug_layout = QVBoxLayout()

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMaximumHeight(150)
        debug_layout.addWidget(self.debug_text)

        debug_control = QHBoxLayout()
        self.clear_debug_button = QPushButton("Debug leeren")
        self.clear_debug_button.clicked.connect(self.debug_text.clear)
        debug_control.addWidget(self.clear_debug_button)

        self.toggle_debug_button = QPushButton("Debug ein/aus")
        self.toggle_debug_button.setCheckable(True)
        self.toggle_debug_button.setChecked(True)
        debug_control.addWidget(self.toggle_debug_button)
        debug_layout.addLayout(debug_control)

        debug_group.setLayout(debug_layout)
        main_layout.addWidget(debug_group)

        # Ausgabe
        output_group = QGroupBox("Antwort")
        output_layout = QVBoxLayout()

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Status
        status_layout = QHBoxLayout()
        self.ollama_status = QLabel("Ollama-Status: Wird geprüft...")
        status_layout.addWidget(self.ollama_status)
        main_layout.addLayout(status_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Starte log capture
        self.log_debug("Anwendung gestartet")

    def log_debug(self, message):
        """Fügt eine Debug-Nachricht zum Debug-Textfeld hinzu"""
        self.logger.debug(message)
        if self.toggle_debug_button.isChecked():
            self.debug_text.append(message)

    def toggle_remote_mode(self, index):
        is_remote = index == 1
        self.remote_controls.setVisible(is_remote)

        # Bei Wechsel zurück zum lokalen Modus
        if not is_remote and self.is_connected_remote:
            self.disconnect_ssh()

        # Aktualisiere die Modelliste basierend auf dem ausgewählten Modus
        self.refresh_models()

        # Speichere Moduseinstellung
        self.settings.setValue("mode_index", index)

    def show_ssh_config(self):
        self.log_debug("Öffne SSH-Konfigurationsdialog")
        dialog = SSHConfigDialog(self.ssh_config, self)
        if dialog.exec():
            self.ssh_config = dialog.get_ssh_config()
            self.save_ssh_config()
            self.log_debug(f"SSH-Konfiguration für {self.ssh_config['username']}@{self.ssh_config['host']} gespeichert")
            QMessageBox.information(self, "SSH-Konfiguration",
                              f"SSH-Konfiguration für {self.ssh_config['username']}@{self.ssh_config['host']} gespeichert.")

    def toggle_ssh_connection(self):
        if not self.is_connected_remote:
            self.connect_ssh()
        else:
            self.disconnect_ssh()

    def connect_ssh(self):
        if not self.ssh_config:
            self.log_debug("Keine SSH-Konfiguration vorhanden")
            QMessageBox.warning(self, "Konfiguration fehlt",
                          "Bitte konfigurieren Sie zuerst die SSH-Verbindung.")
            return

        self.connect_button.setEnabled(False)
        self.status_label.setText("Verbindungsaufbau...")
        self.log_debug("Starte SSH-Verbindung")

        # SSH-Verbindung starten
        self.ssh_thread = SSHTunnelThread(self.ssh_config)
        self.ssh_thread.tunnel_status.connect(self.update_ssh_status)
        self.ssh_thread.debug_message.connect(self.log_debug)
        self.ssh_thread.start()

    def disconnect_ssh(self):
        self.log_debug("Trenne SSH-Verbindung")
        if self.ssh_thread and self.ssh_thread.isRunning():
            self.ssh_thread.stop_tunnel()
            self.connect_button.setText("Verbinden")
            self.is_connected_remote = False
            self.config_button.setEnabled(True)

    def update_ssh_status(self, is_connected, message):
        self.status_label.setText(message)
        self.connect_button.setEnabled(True)
        self.log_debug(f"SSH-Status: {message}")

        if is_connected:
            self.connect_button.setText("Trennen")
            self.is_connected_remote = True
            self.config_button.setEnabled(False)
            # Modelle vom Remote-Server abrufen
            self.refresh_models()
        else:
            self.connect_button.setText("Verbinden")
            self.is_connected_remote = False
            self.config_button.setEnabled(True)

    def check_local_ollama(self):
        self.log_debug("Prüfe lokalen Ollama-Server")
        success, models = self.ollama_manager.check_ollama()
        if success and models:
            self.log_debug(f"Lokaler Ollama-Server gefunden mit {len(models)} Modellen")
            self.update_model_list(models)
            self.ollama_status.setText("Ollama-Status: Lokal verfügbar")
        else:
            self.log_debug("Kein lokaler Ollama-Server gefunden")
            self.ollama_status.setText("Ollama-Status: Lokal nicht verfügbar")

    def refresh_models(self):
        if self.mode_combo.currentText() == "Lokal":
            self.log_debug("Aktualisiere lokale Modelle")
            success, models = self.ollama_manager.check_ollama()
            if success and models:
                self.log_debug(f"Lokaler Ollama-Server hat {len(models)} Modelle")
                self.update_model_list(models)
                self.ollama_status.setText("Ollama-Status: Lokal verfügbar")
            else:
                self.log_debug("Lokaler Ollama-Server nicht verfügbar")
                self.model_combo.clear()
                self.ollama_status.setText("Ollama-Status: Lokal nicht verfügbar")
        elif self.is_connected_remote and self.ssh_config:
            self.log_debug("Aktualisiere Remote-Modelle über SSH-Tunnel")
            host = "localhost"
            port = self.ssh_config["local_port"]
            success, models = self.ollama_manager.check_ollama(host, port)
            if success and models:
                self.log_debug(f"Remote Ollama-Server hat {len(models)} Modelle")
                self.update_model_list(models)
                self.ollama_status.setText(f"Ollama-Status: Remote verfügbar über SSH-Tunnel")
            else:
                self.log_debug("Remote Ollama-Server nicht erreichbar")
                self.model_combo.clear()
                self.ollama_status.setText(f"Ollama-Status: Remote nicht verfügbar")

    def update_model_list(self, models):
        current_model = self.model_combo.currentText()
        self.model_combo.clear()

        if not models:
            return

        model_names = []
        for model in models:
            if isinstance(model, dict):
                model_name = model.get('name', '')
                if model_name:
                    model_names.append(model_name)
            elif isinstance(model, str):
                model_names.append(model)

        self.log_debug(f"Gefundene Modelle: {', '.join(model_names)}")
        self.model_combo.addItems(model_names)

        # Versuche, das vorherige Modell wiederherzustellen
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def send_request(self):
        if self.query_thread and self.query_thread.isRunning():
            self.log_debug("Eine Anfrage wird bereits verarbeitet")
            QMessageBox.information(self, "In Bearbeitung",
                              "Eine Anfrage wird bereits bearbeitet. Bitte warten Sie.")
            return

        prompt = self.input_text.toPlainText().strip()
        if not prompt:
            self.log_debug("Keine Eingabe für Anfrage")
            QMessageBox.warning(self, "Keine Eingabe",
                          "Bitte geben Sie Text in das Eingabefeld ein.")
            return

        if self.model_combo.count() == 0:
            self.log_debug("Kein Modell ausgewählt oder verfügbar")
            QMessageBox.warning(self, "Kein Modell",
                          "Kein Modell verfügbar. Bitte prüfen Sie die Serververbindung.")
            return

        model = self.model_combo.currentText()
        self.log_debug(f"Starte Anfrage an Modell: {model}")

        # UI aktualisieren
        self.send_button.setEnabled(False)
        self.send_button.setText("Verarbeitung läuft...")
        self.output_text.clear()
        self.output_text.append("Anfrage wird verarbeitet...")

        # Zielhost und -port basierend auf Modus bestimmen
        host = "localhost"
        port = 11434  # Standard-Ollama-Port

        if self.mode_combo.currentText() == "Remote über SSH" and self.is_connected_remote:
            port = self.ssh_config["local_port"]
            self.log_debug(f"Verwende SSH-Tunnel mit lokalem Port {port}")
        else:
            self.log_debug("Verwende lokalen Ollama-Server")

        # Anfrage in einem separaten Thread ausführen
        self.query_thread = QueryThread(
            self.ollama_manager, model, prompt, host, port
        )
        self.query_thread.response_received.connect(self.handle_response)
        self.query_thread.debug_message.connect(self.log_debug)
        self.query_thread.start()

    def handle_response(self, response):
        self.send_button.setEnabled(True)
        self.send_button.setText("Anfrage senden")

        self.output_text.clear()

        if "error" in response:
            self.log_debug(f"Fehler bei der Anfrage: {response['error']}")
            self.output_text.append(f"Fehler bei der Anfrage: {response['error']}")
        else:
            self.log_debug("Antwort erfolgreich erhalten")
            self.output_text.append(response.get("response", "Keine Antwort erhalten"))

        self.query_thread = None

    def load_ssh_config(self):
        """Lädt die SSH-Konfiguration aus den Programmeinstellungen"""
        self.logger.debug("Lade SSH-Konfiguration")
        config = {}

        # Lade verschlüsselte Konfiguration
        if self.settings.contains("ssh_host"):
            config['host'] = self.settings.value("ssh_host", "")
            config['port'] = self.settings.value("ssh_port", 22, int)
            config['username'] = self.settings.value("ssh_username", "")
            config['use_key'] = self.settings.value("ssh_use_key", False, bool)
            config['key_path'] = self.settings.value("ssh_key_path", "")

            # Passwort aus Umgebungsvariable
            password_var = self.settings.value("ssh_password_var", "")
            if password_var:
                config['password'] = os.environ.get(password_var, "")
            else:
                config['password'] = ""

            config['remote_host'] = self.settings.value("ssh_remote_host", "localhost")
            config['remote_port'] = self.settings.value("ssh_remote_port", 11434, int)
            config['local_port'] = self.settings.value("ssh_local_port", 11435, int)

            self.logger.debug(f"SSH-Konfiguration geladen für {config.get('username')}@{config.get('host')}")
            return config

        return None

    def save_ssh_config(self):
        """Speichert die SSH-Konfiguration in den Programmeinstellungen"""
        if not self.ssh_config:
            return

        self.logger.debug(f"Speichere SSH-Konfiguration für {self.ssh_config.get('username')}@{self.ssh_config.get('host')}")

        self.settings.setValue("ssh_host", self.ssh_config.get('host', ""))
        self.settings.setValue("ssh_port", self.ssh_config.get('port', 22))
        self.settings.setValue("ssh_username", self.ssh_config.get('username', ""))
        self.settings.setValue("ssh_use_key", self.ssh_config.get('use_key', False))
        self.settings.setValue("ssh_key_path", self.ssh_config.get('key_path', ""))

        # Passwort wird nicht direkt gespeichert - nur für diese Session im Speicher halten
        # Hier könnte man eine Umgebungsvariable verwenden statt direkt zu speichern
        # self.settings.setValue("ssh_password_var", "OLLAMA_SSH_PASSWORD")

        self.settings.setValue("ssh_remote_host", self.ssh_config.get('remote_host', "localhost"))
        self.settings.setValue("ssh_remote_port", self.ssh_config.get('remote_port', 11434))
        self.settings.setValue("ssh_local_port", self.ssh_config.get('local_port', 11435))

        self.settings.sync()

    def closeEvent(self, event):
        """Wird aufgerufen, wenn das Fenster geschlossen wird"""
        self.logger.debug("Schließe Anwendung")

        # Speichere Fensterposition und -größe
        self.settings.setValue("geometry", self.saveGeometry())

        # Beende SSH-Verbindung wenn aktiv
        if self.is_connected_remote:
            self.disconnect_ssh()

        # Warte auf Thread-Beendigung
        if self.ssh_thread and self.ssh_thread.isRunning():
            self.ssh_thread.wait(3000)  # max. 3 Sekunden warten

        if self.query_thread and self.query_thread.isRunning():
            self.query_thread.wait(1000)  # max. 1 Sekunde warten

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Einheitlicher Look unter verschiedenen Plattformen
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
