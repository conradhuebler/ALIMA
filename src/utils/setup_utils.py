#!/usr/bin/env python3
"""
First-Start Setup Utilities - Shared logic for GUI and CLI wizards
Claude Generated
"""

import os
import gzip
import tempfile
import requests
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from urllib.parse import urlparse  # BUGFIX: For parsing Ollama base_url - Claude Generated

from .config_models import (
    AlimaConfig, SystemConfig, DatabaseConfig, UIConfig,
    UnifiedProviderConfig,
    UnifiedProvider,
)


logger = logging.getLogger(__name__)

# GND Data Source URL
GND_DOWNLOAD_URL = "https://data.dnb.de/GND/authorities-gnd-sachbegriff_dnbmarc.mrc.xml.gz"


@dataclass
class SetupResult:
    """Result of a setup validation - Claude Generated"""
    success: bool
    message: str
    data: any = None


class OllamaConnectionValidator:
    """Validates Ollama connections (native or OpenAI-compatible) - Claude Generated"""

    @staticmethod
    def test_native(host: str, port: int, api_key: Optional[str] = None,
                   timeout: int = 10, use_ssl: bool = False) -> SetupResult:
        """Test native Ollama connection - Claude Generated

        Builds the URL exactly like the running client does (``ollama_endpoint``),
        so the test and the real call cannot disagree about where they connect.
        """
        from .ollama_endpoint import base_url, clean_api_key

        url = base_url(host, port, use_ssl)
        try:
            import ollama

            client_params = {"host": url, "timeout": timeout}
            key = clean_api_key(api_key or "")
            if key:
                client_params["headers"] = {"Authorization": f"Bearer {key}"}

            client = ollama.Client(**client_params)

            # Try to get models
            logger.info(f"Testing native Ollama connection to {url}")
            models_response = client.list()

            # Extract model names from response
            models = []
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        models.append(model.model)
                    elif hasattr(model, 'name'):
                        models.append(model.name)

            if not models:
                return SetupResult(
                    False,
                    "Verbindung erfolgreich, aber keine Modelle verfügbar. Bitte zuerst ein Modell herunterladen (z.B. ollama pull mistral)",
                    []
                )

            return SetupResult(True, f"Verbindung erfolgreich ({url}). {len(models)} Modelle gefunden.", models)

        except ImportError:
            return SetupResult(
                False,
                "ollama-Paket nicht installiert. Installation mit: pip install ollama",
                []
            )
        except Exception as e:
            error_msg = str(e)
            low = error_msg.lower()
            if "401" in error_msg or "unauthorized" in low:
                return SetupResult(
                    False,
                    f"{url} lehnt den Schlüssel ab (401). Schlüssel prüfen oder neu erzeugen.",
                    [])
            if "403" in error_msg or "forbidden" in low:
                return SetupResult(False, f"{url} verweigert den Zugriff (403).", [])
            if "refused" in low or "connect" in low or "timed out" in low or "name or service" in low:
                return SetupResult(
                    False,
                    f"Keine Verbindung zu {url} möglich. Läuft der Dienst, stimmen Host, Port und SSL?",
                    []
                )
            return SetupResult(False, f"Verbindung zu {url} fehlgeschlagen: {error_msg}", [])

    @staticmethod
    def test_openai_compatible(base_url: str, api_key: str,
                              timeout: int = 10) -> SetupResult:
        """Test OpenAI-compatible connection - Claude Generated"""
        try:
            import openai

            logger.info(f"Testing OpenAI-compatible connection to {base_url}")

            # Create client
            client_params = {
                "base_url": base_url,
                "api_key": api_key if api_key else "no-key-required",
                "timeout": timeout
            }

            client = openai.OpenAI(**client_params)

            # Try to get models
            models_response = client.models.list()
            models = [model.id for model in models_response.data]

            if not models:
                return SetupResult(
                    False,
                    "Verbindung erfolgreich, aber keine Modelle verfügbar.",
                    []
                )

            return SetupResult(True, f"Verbindung erfolgreich. {len(models)} Modelle gefunden.", models)

        except ImportError:
            return SetupResult(
                False,
                "openai-Paket nicht installiert. Installation mit: pip install openai",
                []
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                return SetupResult(False, f"Authentifizierung fehlgeschlagen. Bitte überprüfen Sie Ihren API-Schlüssel.", [])
            elif "404" in error_msg or "refused" in error_msg.lower():
                return SetupResult(False, f"Keine Verbindung zu {base_url} möglich. Bitte überprüfen Sie die URL.", [])
            return SetupResult(False, f"Verbindung fehlgeschlagen: {error_msg}", [])


class APIKeyValidator:
    """Validates API keys for various providers - Claude Generated"""

    @staticmethod
    def validate_gemini(api_key: str) -> SetupResult:
        """Validate Google Gemini API key - Claude Generated"""
        if not api_key or len(api_key) < 10:
            return SetupResult(False, "Ungültiges API-Schlüssel-Format")

        try:
            # google.generativeai does not exist under that name - the
            # pinned/installed SDK is google-genai (module google.genai,
            # Client() + client.models.*). Every validation attempt raised
            # ModuleNotFoundError and was reported as an invalid key, even
            # for a perfectly valid one. - Claude Generated
            from google import genai
            client = genai.Client(api_key=api_key)
            client.models.generate_content(
                model="gemini-2.5-flash", contents="test",
            )
            return SetupResult(True, "API-Schlüssel ist gültig")
        except Exception as e:
            return SetupResult(False, f"API-Schlüssel-Validierung fehlgeschlagen: {str(e)}")

    @staticmethod
    def validate_anthropic(api_key: str) -> SetupResult:
        """Validate Anthropic Claude API key - Claude Generated"""
        if not api_key or not api_key.startswith("sk-"):
            return SetupResult(False, "Ungültiges API-Schlüssel-Format (sollte mit 'sk-' beginnen)")

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            # Lightweight test - just create message to verify key
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return SetupResult(True, "API-Schlüssel ist gültig")
        except Exception as e:
            return SetupResult(False, f"API-Schlüssel-Validierung fehlgeschlagen: {str(e)}")


class GNDDatabaseDownloader:
    """Handles GND database download and import - Claude Generated"""

    @staticmethod
    def download(progress_callback: Optional[Callable[[int], None]] = None) -> SetupResult:
        """Download and extract GND database - Claude Generated

        Args:
            progress_callback: Optional callback that receives percentage (0-100)

        Returns:
            SetupResult with path to extracted XML file
        """
        try:
            logger.info(f"Downloading GND database from {GND_DOWNLOAD_URL}")

            # Send request with stream
            response = requests.get(GND_DOWNLOAD_URL, stream=True)
            response.raise_for_status()

            # Get total file size
            total_size = int(response.headers.get("content-length", 0))
            logger.info(f"File size: {total_size / (1024*1024):.1f} MB")

            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and progress_callback:
                            percent = min(100, int(downloaded * 100 / total_size))
                            progress_callback(percent)

            logger.info("Download complete, extracting...")

            # Extract gzip file
            with gzip.open(temp_gz_path, 'rb') as gz_f:
                with open(temp_xml_path, 'wb') as xml_f:
                    xml_f.write(gz_f.read())

            # Clean up gzip file
            os.remove(temp_gz_path)

            logger.info(f"GND database extracted to {temp_xml_path}")
            return SetupResult(True, "GND-Datenbank erfolgreich heruntergeladen", temp_xml_path)

        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            return SetupResult(False, f"Download fehlgeschlagen: {str(e)}")
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return SetupResult(False, f"Entpacken fehlgeschlagen: {str(e)}")


class ConfigurationBuilder:
    """Builds initial ALIMA configuration - Claude Generated"""

    @staticmethod
    def create_initial_config(
        provider_type: str,
        provider_name: str,
        base_url: str = None,
        api_key: str = None,
        models: List[str] = None,
        pipeline_model: str = "",
        vision_model: str = "",
    ) -> AlimaConfig:
        """Create initial ALIMA configuration with LLM provider - Claude Generated

        Args:
            provider_type: 'ollama', 'gemini', 'anthropic', or 'openai_compatible'
            provider_name: Human-readable name (e.g., 'My Ollama Server')
            base_url: URL for Ollama/OpenAI-compatible providers
            api_key: API key for cloud providers
            models: List of available models
            pipeline_model: Model for every classic-pipeline step (falls back
                to the first available model)
            vision_model: Model for image/OCR analysis (falls back the same way)

        Returns:
            AlimaConfig with provider configured
        """

        # Create database config with OS-specific path
        database_config = DatabaseConfig()

        # Create system config
        system_config = SystemConfig(
            first_run_completed=False,
            log_level='normal'
        )

        # Create UI config
        ui_config = UIConfig()

        # Ollama wird nativ gespeichert (Host/Port/SSL, optionaler Schlüssel für
        # Ollama Cloud), denn der Wizard testet vorher ebenfalls nativ: Test und
        # echter Aufruf gehen so durch denselben Pfad. Früher wurde daraus ein
        # openai_compatible-Eintrag mit "/v1" und dem Platzhalter-Schlüssel
        # "no-key-required"; ein Cloud-Schlüssel ging dabei verloren.
        # - Claude Generated
        if provider_type == "ollama":
            from .ollama_endpoint import (
                base_url as ollama_base_url, clean_api_key, is_ollama_cloud_host,
                parse_host,
            )

            host, port, use_ssl = parse_host(
                base_url or "http://localhost:11434", 11434, False
            )
            cloud = is_ollama_cloud_host(host)
            if cloud:
                port, use_ssl = 443, True
            provider = UnifiedProvider(
                name="ollama_cloud" if cloud else "ollama",
                provider_type="ollama",
                host=host or "localhost",
                port=port,
                use_ssl=use_ssl,
                base_url=ollama_base_url(host or "localhost", port, use_ssl),
                api_key=clean_api_key(api_key or ""),
                connection_type="native_client",
                available_models=models or [],
                enabled=True,
                preferred_model=models[0] if models else "",
            )
        else:
            provider = UnifiedProvider(
                name=provider_type,
                provider_type=provider_type,
                base_url=base_url or "",
                api_key=api_key or "",
                connection_type="native_client",
                available_models=models or [],
                enabled=True,
                preferred_model=models[0] if models else "",
            )

        # Pipeline + vision defaults from the wizard's two model questions.
        # Used to be six per-task TaskPreference entries (one per classic-pipeline
        # task); that choice only ever acted as a dead first entry of a fallback
        # chain nothing else walked, and three of six tasks had no step to land
        # on at all (see docs/settings_cleanup_phase1.md). Retired Oct 2026 to
        # the two scopes that actually vary. - Claude Generated
        default_model = models[0] if models else ""
        resolved_pipeline_model = pipeline_model or default_model
        resolved_vision_model = vision_model or default_model

        # CRITICAL VALIDATION: Reject "default", "auto", or "auto-select" strings
        # These should NEVER appear in wizard-created configs (only explicit model names)
        for label, selected_model in (
            ("Pipeline-Modell", resolved_pipeline_model),
            ("Vision-Modell", resolved_vision_model),
        ):
            if selected_model and selected_model.lower() in ["default", "auto", "auto-select", "(auto-select)"]:
                raise ValueError(
                    f"Invalid model name '{selected_model}' for {label}. "
                    f"Wizard must use explicit model names, not auto-select placeholders."
                )

        # Create unified provider config
        # "Allgemein" (zugleich Agentik und Chat) wird ausdrücklich gesetzt, auf
        # dasselbe Modell wie die Pipeline. Leer fiel es auf "erster nutzbarer
        # Anbieter" zurück (dessen erstes Modell), und ein erneuter Lauf auf
        # einer bestehenden Konfiguration leerte einen vorhandenen Wert.
        unified_config = UnifiedProviderConfig(
            providers=[provider],
            preferred_provider=provider.name,
            preferred_model=resolved_pipeline_model,
            pipeline_default_provider=provider.name,
            pipeline_default_model=resolved_pipeline_model,
            vision_default_provider=provider.name,
            vision_default_model=resolved_vision_model,
        )

        # Assemble final config
        config = AlimaConfig(
            database_config=database_config,
            system_config=system_config,
            ui_config=ui_config,
            unified_config=unified_config
        )

        return config


class LiberoTokenFetcher:
    """Fetches authentication token from Libero SOAP service - Claude Generated"""

    @staticmethod
    def fetch(soap_url: str, username: str, password: str) -> 'SetupResult':
        """Authenticate with Libero and retrieve session token.

        soap_url: any Libero SOAP URL (scheme+host extracted automatically)
        Returns: SetupResult(success=True, data=token) or SetupResult(success=False, message=error)
        """
        try:
            from urllib.parse import quote
            import xml.etree.ElementTree as ET

            parsed = urlparse(soap_url)
            if not parsed.scheme or not parsed.netloc:
                return SetupResult(False, "Ungültige SOAP-URL — zuerst eine Search URL eintragen")

            auth_url = (
                f"{parsed.scheme}://{parsed.netloc}"
                f"/libero/LiberoWebServices.Authenticate.cls"
                f"?soap_method=Login&Username={quote(username, safe='')}"
                f"&Password={quote(password, safe='')}"
            )
            response = requests.get(auth_url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            token_el = root.find('.//{http://libero.com.au}Token')
            if token_el is None:
                token_el = root.find('.//Token')

            if token_el is None or not (token_el.text or '').strip():
                msg_el = (root.find('.//{http://libero.com.au}Message')
                          or root.find('.//Message'))
                msg = msg_el.text if msg_el is not None else "Kein Token in der Antwort"
                return SetupResult(False, msg)

            return SetupResult(True, "Token erfolgreich erstellt", data=token_el.text.strip())

        except requests.RequestException as e:
            return SetupResult(False, f"Verbindungsfehler: {e}")
        except ET.ParseError as e:
            return SetupResult(False, f"Ungültige XML-Antwort: {e}")
        except Exception as e:
            return SetupResult(False, f"Fehler: {e}")

    @staticmethod
    def fetch_from_url(auth_url: str) -> 'SetupResult':
        """Fetch token using a pre-built auth URL (e.g. manually edited by user) - Claude Generated"""
        try:
            import xml.etree.ElementTree as ET

            response = requests.get(auth_url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            token_el = root.find('.//{http://libero.com.au}Token')
            if token_el is None:
                token_el = root.find('.//Token')

            if token_el is None or not (token_el.text or '').strip():
                msg_el = (root.find('.//{http://libero.com.au}Message')
                          or root.find('.//Message'))
                msg = msg_el.text if msg_el is not None else "Kein Token in der Antwort"
                return SetupResult(False, msg)

            return SetupResult(True, "Token erfolgreich erstellt", data=token_el.text.strip())

        except requests.RequestException as e:
            return SetupResult(False, f"Verbindungsfehler: {e}")
        except ET.ParseError as e:
            return SetupResult(False, f"Ungültige XML-Antwort: {e}")
        except Exception as e:
            return SetupResult(False, f"Fehler: {e}")


class PromptValidator:
    """Validates prompts.json availability - Claude Generated"""

    @staticmethod
    def check_prompts_file(prompts_path: str = "prompts.json") -> SetupResult:
        """Check if prompts.json exists and is valid - Claude Generated"""

        try:
            if not Path(prompts_path).exists():
                return SetupResult(False, f"prompts.json nicht gefunden unter {prompts_path}")

            # Try to parse as JSON
            import json
            with open(prompts_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return SetupResult(False, "prompts.json ist kein gültiges JSON")

            # Check for required prompt keys
            required_keys = ['initialisation', 'keywords', 'classification']
            missing = [k for k in required_keys if k not in data]

            if missing:
                return SetupResult(
                    False,
                    f"prompts.json fehlen erforderliche Schlüssel: {', '.join(missing)}"
                )

            return SetupResult(True, "prompts.json ist gültig")

        except Exception as e:
            return SetupResult(False, f"Fehler beim Lesen von prompts.json: {str(e)}")
