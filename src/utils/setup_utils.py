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
    CatalogConfig, PromptConfig, UnifiedProviderConfig,
    UnifiedProvider, TaskPreference, TaskType
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
                   timeout: int = 10) -> SetupResult:
        """Test native Ollama connection - Claude Generated"""
        try:
            import ollama

            # Build connection URL
            url = f"http://{host}:{port}"

            # Create client with timeout
            client_params = {
                "host": url,
                "timeout": timeout
            }
            if api_key:
                client_params["headers"] = {"Authorization": api_key}

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

            return SetupResult(True, f"Verbindung erfolgreich. {len(models)} Modelle gefunden.", models)

        except ImportError:
            return SetupResult(
                False,
                "ollama-Paket nicht installiert. Installation mit: pip install ollama",
                []
            )
        except Exception as e:
            error_msg = str(e)
            if "refused" in error_msg.lower() or "connection" in error_msg.lower():
                return SetupResult(
                    False,
                    f"Keine Verbindung zu Ollama unter {host}:{port} möglich. Stellen Sie sicher, dass Ollama läuft.",
                    []
                )
            return SetupResult(False, f"Verbindung fehlgeschlagen: {error_msg}", [])

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
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("test")
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
        task_model_selections: dict = None
    ) -> AlimaConfig:
        """Create initial ALIMA configuration with LLM provider - Claude Generated

        Args:
            provider_type: 'ollama', 'gemini', 'anthropic', or 'openai_compatible'
            provider_name: Human-readable name (e.g., 'My Ollama Server')
            base_url: URL for Ollama/OpenAI-compatible providers
            api_key: API key for cloud providers
            models: List of available models
            task_model_selections: Dict mapping task_type to selected model name

        Returns:
            AlimaConfig with provider configured
        """

        # Create database config with OS-specific path
        database_config = DatabaseConfig()

        # Create system config
        system_config = SystemConfig(
            first_run_completed=False,
            debug=False,
            log_level='INFO'
        )

        # Create UI config
        ui_config = UIConfig()

        # Create catalog config
        catalog_config = CatalogConfig()

        # Create prompt config
        prompt_config = PromptConfig()

        # ARCHITECTURAL FIX: Use OpenAI-compatible API for Ollama (simpler, more reliable)
        # Ollama supports OpenAI-compatible API as primary interface
        actual_provider_type = provider_type
        actual_base_url = base_url or ""

        if provider_type == "ollama":
            # Convert Ollama to openai_compatible provider type
            actual_provider_type = "openai_compatible"
            # Ensure base_url includes /v1 endpoint for OpenAI compatibility
            if actual_base_url and not actual_base_url.endswith("/v1"):
                actual_base_url = f"{actual_base_url}/v1"
            # Ollama doesn't require API key
            actual_api_key = "no-key-required"
        else:
            # For all other providers (OpenAI, Gemini, etc.), use provided API key
            actual_api_key = api_key or ""

        # Create provider configuration - Claude Generated (fixed parameters)
        provider = UnifiedProvider(
            name=provider_type,  # Keep original type as display name (e.g., "ollama")
            provider_type=actual_provider_type,  # Convert to openai_compatible if needed
            base_url=actual_base_url,
            api_key=actual_api_key,
            connection_type="native_client" if provider_type == "ollama" else "native_client",
            available_models=models or [],
            enabled=True,
            preferred_model=models[0] if models else ""
        )

        # Create task preferences with user-selected models - Claude Generated (enhanced)
        # Only create preferences for LLM tasks, not for non-LLM tasks (INPUT, SEARCH, DK_SEARCH)
        task_preferences = {}
        if models:
            default_model = models[0]
            llm_tasks = {TaskType.INITIALISATION, TaskType.KEYWORDS, TaskType.CLASSIFICATION,
                        TaskType.DK_CLASSIFICATION, TaskType.VISION, TaskType.CHUNKED_PROCESSING}

            for task_type in llm_tasks:
                # Use user's model selection or fall back to default
                selected_model = default_model
                if task_model_selections and task_type.name in task_model_selections:
                    selected_model = task_model_selections[task_type.name]

                task_preferences[task_type.name] = TaskPreference(
                    task_type=task_type,
                    model_priority=[
                        {
                            "provider_name": provider.name,
                            "model_name": selected_model
                        }
                    ],
                    allow_fallback=True
                )

        # Create unified provider config
        unified_config = UnifiedProviderConfig(
            providers=[provider],
            task_preferences=task_preferences,
            provider_priority=[provider.name]
        )

        # Assemble final config
        config = AlimaConfig(
            database_config=database_config,
            system_config=system_config,
            ui_config=ui_config,
            catalog_config=catalog_config,
            prompt_config=prompt_config,
            unified_config=unified_config
        )

        return config


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
