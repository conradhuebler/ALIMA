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
                    "Connection successful but no models available. Please pull a model first (e.g., ollama pull mistral)",
                    []
                )

            return SetupResult(True, f"Connection successful. Found {len(models)} models.", models)

        except ImportError:
            return SetupResult(
                False,
                "ollama package not installed. Install with: pip install ollama",
                []
            )
        except Exception as e:
            error_msg = str(e)
            if "refused" in error_msg.lower() or "connection" in error_msg.lower():
                return SetupResult(
                    False,
                    f"Cannot connect to Ollama at {host}:{port}. Make sure Ollama is running.",
                    []
                )
            return SetupResult(False, f"Connection failed: {error_msg}", [])

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
                    "Connection successful but no models available.",
                    []
                )

            return SetupResult(True, f"Connection successful. Found {len(models)} models.", models)

        except ImportError:
            return SetupResult(
                False,
                "openai package not installed. Install with: pip install openai",
                []
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                return SetupResult(False, f"Authentication failed. Please check your API key.", [])
            elif "404" in error_msg or "refused" in error_msg.lower():
                return SetupResult(False, f"Cannot connect to {base_url}. Please check the URL.", [])
            return SetupResult(False, f"Connection failed: {error_msg}", [])


class APIKeyValidator:
    """Validates API keys for various providers - Claude Generated"""

    @staticmethod
    def validate_gemini(api_key: str) -> SetupResult:
        """Validate Google Gemini API key - Claude Generated"""
        if not api_key or len(api_key) < 10:
            return SetupResult(False, "Invalid API key format")

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("test")
            return SetupResult(True, "API key is valid")
        except Exception as e:
            return SetupResult(False, f"API key validation failed: {str(e)}")

    @staticmethod
    def validate_anthropic(api_key: str) -> SetupResult:
        """Validate Anthropic Claude API key - Claude Generated"""
        if not api_key or not api_key.startswith("sk-"):
            return SetupResult(False, "Invalid API key format (should start with 'sk-')")

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            # Lightweight test - just create message to verify key
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return SetupResult(True, "API key is valid")
        except Exception as e:
            return SetupResult(False, f"API key validation failed: {str(e)}")


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
            return SetupResult(True, "GND database downloaded successfully", temp_xml_path)

        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            return SetupResult(False, f"Download failed: {str(e)}")
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return SetupResult(False, f"Extraction failed: {str(e)}")


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

        # Create provider configuration - Claude Generated (fixed parameters)
        provider = UnifiedProvider(
            name=provider_name,
            provider_type=provider_type,
            base_url=base_url or "",
            api_key=api_key or "",
            connection_type="native_client" if provider_type == "ollama" else "native_client",
            available_models=models or [],
            enabled=True,
            preferred_model=models[0] if models else ""
        )

        # Create task preferences with user-selected models - Claude Generated (enhanced)
        task_preferences = {}
        if models:
            default_model = models[0]
            for task_type in TaskType:
                # Use user's model selection or fall back to default - Claude Generated
                selected_model = default_model
                if task_model_selections and task_type.value in task_model_selections:
                    selected_model = task_model_selections[task_type.value]

                task_preferences[task_type.value] = TaskPreference(
                    task_type=task_type,
                    model_priority=[
                        {
                            "provider_name": provider_type.lower(),
                            "model_name": selected_model
                        }
                    ],
                    allow_fallback=True
                )

        # Create unified provider config
        unified_config = UnifiedProviderConfig(
            providers=[provider],
            task_preferences=task_preferences,
            provider_priority=[provider_type.lower()]
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
                return SetupResult(False, f"prompts.json not found at {prompts_path}")

            # Try to parse as JSON
            import json
            with open(prompts_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return SetupResult(False, "prompts.json is not valid JSON")

            # Check for required prompt keys
            required_keys = ['initialisation', 'keywords', 'classification']
            missing = [k for k in required_keys if k not in data]

            if missing:
                return SetupResult(
                    False,
                    f"prompts.json missing required keys: {', '.join(missing)}"
                )

            return SetupResult(True, "prompts.json is valid")

        except Exception as e:
            return SetupResult(False, f"Error reading prompts.json: {str(e)}")
