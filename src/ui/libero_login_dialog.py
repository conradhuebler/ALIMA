#!/usr/bin/env python3
"""
Libero Login Dialog - Fetches session token via Libero Authenticate webservice
Claude Generated
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QDialogButtonBox, QApplication
)
from PyQt6.QtCore import Qt
from urllib.parse import urlparse, quote
import logging

logger = logging.getLogger(__name__)


class LiberoLoginDialog(QDialog):
    """Login dialog to fetch a Libero session token - Claude Generated"""

    def __init__(self, soap_url: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Libero-Anmeldung")
        self.setMinimumWidth(500)
        self._token = ""
        self._soap_url = soap_url

        layout = QVBoxLayout()

        # Explanation label
        hint = QLabel("Libero-Zugangsdaten eingeben um automatisch einen Token zu erstellen.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Credentials form
        form = QFormLayout()

        self._username_input = QLineEdit()
        self._username_input.setPlaceholderText("Benutzername")
        self._username_input.textChanged.connect(self._update_url_preview)
        form.addRow("Benutzername:", self._username_input)

        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_input.setPlaceholderText("Passwort (darf leer sein)")
        self._password_input.textChanged.connect(self._update_url_preview)
        form.addRow("Passwort:", self._password_input)

        # Auth URL preview — editable so user can adjust for non-standard installations
        self._url_preview = QLineEdit()
        self._url_preview.setPlaceholderText("Auth-URL wird automatisch zusammengesetzt")
        self._url_preview.setToolTip(
            "Vollständige Auth-URL — wird automatisch aus SOAP-URL + Zugangsdaten aufgebaut.\n"
            "Kann bei abweichenden Installationen manuell angepasst werden."
        )
        form.addRow("Auth-URL:", self._url_preview)

        layout.addLayout(form)

        # Fetch button
        fetch_btn_layout = QVBoxLayout()
        self._fetch_btn = QPushButton("🔑 Token erstellen")
        self._fetch_btn.clicked.connect(self._fetch)
        fetch_btn_layout.addWidget(self._fetch_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(fetch_btn_layout)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Dialog buttons — OK initially disabled
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        self.setLayout(layout)

        # Populate URL preview with initial (empty credentials) value
        self._update_url_preview()

    def _build_auth_url(self, username: str, password: str) -> str:
        """Construct the Libero authenticate URL - Claude Generated"""
        parsed = urlparse(self._soap_url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return (
            f"{parsed.scheme}://{parsed.netloc}"
            f"/libero/LiberoWebServices.Authenticate.cls"
            f"?soap_method=Login&Username={quote(username, safe='')}"
            f"&Password={quote(password, safe='')}"
        )

    def _update_url_preview(self):
        """Rebuild auth URL preview when credentials change - Claude Generated"""
        # Show masked password in preview for security
        password = self._password_input.text()
        masked_password = "****" if password else ""
        url = self._build_auth_url(
            self._username_input.text().strip(),
            masked_password
        )
        self._url_preview.setText(url)
        # Reset token/OK when credentials change
        self._token = ""
        self._ok_btn.setEnabled(False)
        self._status_label.setText("")

    def _fetch(self):
        """Call LiberoTokenFetcher using real credentials - Claude Generated"""
        from ..utils.setup_utils import LiberoTokenFetcher

        # Build URL with actual password (not masked preview)
        auth_url = self._build_auth_url(
            self._username_input.text().strip(),
            self._password_input.text()
        )
        if not auth_url:
            self._status_label.setText("❌ Keine Auth-URL — bitte SOAP Search URL in der Konfiguration eintragen.")
            self._status_label.setStyleSheet("color: red;")
            return

        self._fetch_btn.setEnabled(False)
        self._fetch_btn.setText("🔄 Verbinde...")
        self._status_label.setText("")
        QApplication.processEvents()

        try:
            result = LiberoTokenFetcher.fetch_from_url(auth_url)
        finally:
            self._fetch_btn.setEnabled(True)
            self._fetch_btn.setText("🔑 Token erstellen")

        if result.success:
            self._token = result.data
            self._status_label.setText(f"✅ {result.message}")
            self._status_label.setStyleSheet("color: green;")
            self._ok_btn.setEnabled(True)
            logger.info("Libero token fetched successfully")
        else:
            self._status_label.setText(f"❌ {result.message}")
            self._status_label.setStyleSheet("color: red;")
            self._ok_btn.setEnabled(False)
            logger.warning(f"Libero token fetch failed: {result.message}")

    @property
    def token(self) -> str:
        return self._token
