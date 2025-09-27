import requests
from typing import Optional, Union, List, Dict, Any, Callable
import os
import threading
import time
from pathlib import Path
import importlib
import logging
import base64
import json
import sys
import traceback
import socket
import subprocess
import platform
from urllib.parse import urlparse
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtNetwork import QTcpSocket, QHostInfo

# Native Ollama client import - Claude Generated
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LlmService(QObject):
    """
    A unified interface for interacting with various Large Language Models.

    This class provides a consistent API for different LLM providers like OpenAI,
    Anthropic, Google Gemini, and others. It handles initialization, configuration,
    and generation requests across all supported providers.
    """

    # Define PyQt signals for text streaming
    text_received = pyqtSignal(str, str)  # request_id, text_chunk
    generation_finished = pyqtSignal(str, str)  # request_id, message
    generation_error = pyqtSignal(str, str)  # request_id, error_message

    # Neues Signal zur Anzeige von Abbrüchen
    generation_cancelled = pyqtSignal(str)  # request_id
    
    # Signals für Provider-Status - Claude Generated
    provider_status_changed = pyqtSignal(str, bool)  # provider_name, is_reachable

    # Update Ollama URL and Port
    ollama_url_updated = pyqtSignal()
    ollama_port_updated = pyqtSignal()

    def __init__(
        self,
        providers: List[str] = None,
        config_manager = None,
        api_keys: Dict[str, str] = None,
        ollama_url: str = "http://localhost",
        ollama_port: int = 11434,
        lazy_initialization: bool = False,  # Use direct initialization with ping tests - Claude Generated
    ):
        """
        Initialize LLM interface with specified providers and API keys.

        Args:
            providers: List of provider names to initialize. If None, tries to initialize all supported providers.
            config_file: Path to configuration file for storing API keys and provider settings.
            api_keys: Dictionary of provider API keys {provider_name: api_key}.
        """
        super().__init__()  # Initialize QObject base class

        # Erweiterte Variablen für das Abbrechen von Anfragen
        self.cancel_requested = False
        self.stream_running = False
        self.current_provider = None
        self.current_request_id = None
        self.current_thread_id = None
        
        # Store lazy initialization flag - Claude Generated
        self.lazy_initialization = lazy_initialization
        
        # Provider reachability cache system - Claude Generated
        self.provider_status_cache = {}  # provider_name -> {'reachable': bool, 'last_check': timestamp, 'latency_ms': float}
        self.status_cache_timeout = 300  # 5 minutes cache timeout
        self.reachability_timer = QTimer()
        self.reachability_timer.timeout.connect(self._refresh_provider_status)
        self.reachability_timer.setSingleShot(False)
        self.reachability_timer.setInterval(60000)  # Check every minute

        # Ensure ollama_url has a scheme (legacy support)
        if not ollama_url.startswith(("http://", "https://")):
            ollama_url = "http://" + ollama_url

        self.ollama_url = ollama_url
        self.ollama_port = ollama_port
        
        # Initialize native Ollama clients holder - Claude Generated
        self.ollama_native_clients = {}  # provider_name -> ollama.Client()

        # Timeout für hängengebliebene Anfragen (in Sekunden)
        self.request_timeout = 300  # 5 Minuten
        self.watchdog_thread = None
        self.last_chunk_time = 0

        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration manager
        if config_manager is None:
            from ..utils.config_manager import get_config_manager
            config_manager = get_config_manager()
        self.config_manager = config_manager
        
        # Load current configuration
        self.alima_config = self.config_manager.load_config()

        # Dictionary to store provider clients
        self.clients = {}

        # Initialize unified provider configurations
        self._init_unified_provider_configs()

        # Initialize legacy dynamic OpenAI-compatible providers (deprecated)
        self._legacy_init_dynamic_provider_configs()

        # Initialize all providers directly (ping tests prevent blocking) - Claude Generated
        self.initialize_providers(providers)

    def set_ollama_url(self, url: str):
        self.logger.info(f"Setting Ollama URL to {url}")
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        self.ollama_url = url
        self.ollama_url_updated.emit()

    def set_ollama_port(self, port: int):
        self.logger.info(f"Setting Ollama Port to {port}")
        self.ollama_port = port
        self.ollama_port_updated.emit()

    def _init_unified_provider_configs(self):
        """Initialize all provider configurations from unified config - Claude Generated"""
        self.all_providers = {}

        # Get unified config from existing config manager - Claude Generated
        try:
            unified_config = self.config_manager.get_unified_config()

            # Initialize all enabled providers from unified config
            for provider in unified_config.get_enabled_providers():
                provider_key = provider.name  # Use original case to match get_available_providers() output - Claude Generated

                if provider.provider_type == "gemini":
                    self.all_providers[provider_key] = {
                        "module": "google.generativeai",
                        "class": None,
                        "initializer": self._init_gemini,
                        "generator": self._generate_gemini,
                        "config": provider
                    }
                elif provider.provider_type == "anthropic":
                    self.all_providers[provider_key] = {
                        "module": "anthropic",
                        "class": "Anthropic",
                        "initializer": self._init_anthropic,
                        "generator": self._generate_anthropic,
                        "config": provider
                    }
                elif provider.provider_type == "ollama":
                    self.all_providers[provider_key] = {
                        "module": "requests",
                        "class": None,
                        "initializer": self._init_ollama_native_provider,
                        "generator": self._generate_ollama,
                        "config": provider
                    }
                elif provider.provider_type == "openai_compatible":
                    self.all_providers[provider_key] = {
                        "module": "openai",
                        "class": "OpenAI",
                        "initializer": self._init_openai_compatible,
                        "generator": self._generate_openai_compatible,
                        "config": provider
                    }

        except Exception as e:
            self.logger.error(f"Failed to initialize unified provider configs: {e}")
            # Fallback to empty configuration
            self.all_providers = {}

        # Create supported_providers from all_providers for backwards compatibility
        self.supported_providers = self.all_providers.copy()

        # Also maintain legacy attributes for compatibility - using .get() for safety
        self.static_providers = {k: v for k, v in self.all_providers.items()
                               if v.get("config", {}) and hasattr(v.get("config"), 'provider_type') and
                               v["config"].provider_type in ["gemini", "anthropic", "ollama"]}
        self.openai_providers = {k: v for k, v in self.all_providers.items()
                               if v.get("config", {}) and hasattr(v.get("config"), 'provider_type') and
                               v["config"].provider_type == "openai_compatible"}

    def _legacy_init_dynamic_provider_configs(self):
        """DEPRECATED - Legacy method for OpenAI providers - Claude Generated"""
        # This method is now handled by _init_unified_provider_configs
        self.openai_providers = {}  # Keep for backward compatibility
        self.logger.info("Legacy dynamic provider config method called - now handled by unified config")

    def reload_providers(self):
        """Reload providers from current configuration - Claude Generated"""
        self.logger.info("Reloading providers from configuration")
        
        # Reload configuration
        self.alima_config = self.config_manager.load_config()
        
        # Reinitialize provider configurations
        self._legacy_init_dynamic_provider_configs()
        
        # Reinitialize all providers
        self.clients.clear()
        self.initialize_providers()
        
        self.logger.info("Provider reload completed")

    def set_api_key(self, provider: str, api_key: str):
        """
        Set API key for provider and update configuration - Claude Generated

        Args:
            provider: Provider name
            api_key: API key
        """
        if provider in self.static_providers:
            # Update static provider API key in configuration
            if provider == "gemini":
                self.alima_config.unified_config.gemini_api_key = api_key
            elif provider == "anthropic":
                self.alima_config.unified_config.anthropic_api_key = api_key
            
            # Save configuration
            self.config_manager.save_config(self.alima_config)
            
            # Reinitialize provider with new key
            self._initialize_single_provider(provider)
            
        elif provider in self.openai_providers:
            # Update OpenAI-compatible provider API key
            provider_obj = self.alima_config.unified_config.get_provider_by_name(provider)
            if provider_obj:
                provider_obj.api_key = api_key
                
                # Save configuration
                self.config_manager.save_config(self.alima_config)
                
                # Reload providers to pick up changes
                self.reload_providers()
            else:
                self.logger.warning(f"Provider {provider} not found in configuration")
        else:
            self.logger.warning(f"Unsupported provider: {provider}")

    # Neue Methode zum Abbrechen von Anfragen
    def _init_watchdog(self):
        """Starte einen Watchdog-Thread zum Überwachen von hängengebliebenen Anfragen."""
        if self.watchdog_thread is not None and self.watchdog_thread.is_alive():
            return  # Watchdog läuft bereits

        def watchdog_func():
            while self.stream_running:
                # Prüfe, ob wir länger als timeout auf Chunks gewartet haben
                if (
                    self.last_chunk_time > 0
                    and time.time() - self.last_chunk_time > self.request_timeout
                ):
                    self.logger.warning(
                        f"Request timed out after {self.request_timeout} seconds without response"
                    )
                    self.cancel_generation(reason="timeout")
                time.sleep(1)  # Überprüfe jede Sekunde

        self.watchdog_thread = threading.Thread(target=watchdog_func, daemon=True)
        self.watchdog_thread.start()

    def cancel_generation(self, reason="user_requested"):
        """
        Cancel the currently running generation request both client-side and server-side if possible.

        Args:
            reason: Reason for cancellation (user_requested, timeout, error)

        Returns:
            bool: True if cancellation was requested, False if no active request
        """
        if not self.stream_running:
            self.logger.info("No active generation to cancel")
            return False

        self.cancel_requested = True
        self.logger.info(f"Cancellation requested (reason: {reason})")

        # Versuche serverseitigen Abbruch basierend auf dem Provider
        try:
            if self.current_provider == "openai" and self.current_request_id:
                self._cancel_openai_request()
            elif self.current_provider == "chatai" and self.current_request_id:
                self._cancel_openai_compatible_request("chatai")
            elif self.current_provider == "comet" and self.current_request_id:
                self._cancel_openai_compatible_request("comet")
            elif self.current_provider == "ollama":
                self._cancel_ollama_request()
            elif self.current_provider == "anthropic" and self.current_request_id:
                self._cancel_anthropic_request()
            else:
                self.logger.info(
                    f"No server-side cancellation available for {self.current_provider}"
                )
        except Exception as e:
            self.logger.error(f"Error during server-side cancellation: {str(e)}")

        self.generation_cancelled.emit(reason)
        return True

    def _cancel_openai_request(self):
        """Abbrechen einer OpenAI-Anfrage."""
        if not self.current_request_id:
            return

        try:
            self.logger.info(f"Cancelling OpenAI request {self.current_request_id}")
            # Der spezifische OpenAI-Abbruch-Endpunkt
            self.clients["openai"].cancel(self.current_request_id)
        except (AttributeError, Exception) as e:
            # Die neuere OpenAI API hat möglicherweise keine direkte cancel-Methode
            self.logger.warning(f"OpenAI server-side cancellation failed: {str(e)}")

    def _cancel_openai_compatible_request(self, provider):
        """Abbrechen einer OpenAI-kompatiblen Anfrage (ChatAI, Comet)."""
        if not self.current_request_id:
            return

        try:
            self.logger.info(f"Cancelling {provider} request {self.current_request_id}")
            # Für OpenAI-kompatible APIs
            self.clients[provider].cancel(self.current_request_id)
        except (AttributeError, Exception) as e:
            self.logger.warning(f"{provider} server-side cancellation failed: {str(e)}")

    def _cancel_ollama_request(self):
        """Abbrechen einer Ollama-Anfrage."""
        try:
            self.logger.info("Cancelling Ollama request")
            # Ollama hat einen speziellen Endpunkt zum Abbrechen
            self.clients["ollama"].post(
                f"{self.ollama_url}:{self.ollama_port}/api/cancel",
                json={},  # Neuere Ollama-Versionen benötigen keine Modellangabe
                timeout=5,  # 5 second timeout to prevent UI blocking
            )
        except Exception as e:
            self.logger.warning(f"Ollama cancellation failed: {str(e)}")

    def _cancel_anthropic_request(self):
        """Abbrechen einer Anthropic-Anfrage."""
        if not self.current_request_id:
            return

        try:
            self.logger.info(f"Cancelling Anthropic request {self.current_request_id}")
            # Neuere Anthropic-Versionen unterstützen möglicherweise Abbrüche
            self.clients["anthropic"].cancel(self.current_request_id)
        except (AttributeError, Exception) as e:
            self.logger.warning(f"Anthropic server-side cancellation failed: {str(e)}")

    def _initialize_single_provider(self, provider: str):
        """
        Initialize a single provider - Claude Generated

        Args:
            provider: The provider name to initialize.
        """
        # Find provider info with case-insensitive fallback and enhanced debugging - Claude Generated
        provider_info = None

        self.logger.info(f"🔍 PROVIDER_INIT_LOOKUP: Searching for provider '{provider}' in supported_providers")
        self.logger.info(f"🔍 AVAILABLE_KEYS: {list(self.supported_providers.keys())}")

        if provider in self.supported_providers:
            provider_info = self.supported_providers[provider]
            self.logger.info(f"✅ PROVIDER_FOUND: '{provider}' found with exact match")
        elif provider.lower() in self.supported_providers:
            provider_info = self.supported_providers[provider.lower()]
            self.logger.info(f"✅ PROVIDER_FOUND: '{provider}' found with lowercase match")
        else:
            # Try finding with case-insensitive search - Claude Generated
            for key in self.supported_providers.keys():
                if key.lower() == provider.lower():
                    provider_info = self.supported_providers[key]
                    self.logger.info(f"✅ PROVIDER_FOUND: '{provider}' found with case-insensitive match: '{key}'")
                    break

            if not provider_info:
                self.logger.warning(f"❌ UNSUPPORTED_PROVIDER: '{provider}' not found in supported providers")
                return

        try:
            # Handle different provider types - Claude Generated (Updated for unified config)
            provider_config = provider_info.get("config")
            provider_type = provider_config.provider_type if provider_config else provider_info.get("type", "")

            if provider_type == "ollama":
                # Special handling for Ollama providers - don't import module
                module = None
                api_key = provider_config.api_key if provider_config else provider_info.get("api_key")
            else:
                # Try to import the required module for other providers
                module = importlib.import_module(provider_info["module"])
                
                # Handle API key from unified config structure - Claude Generated
                api_key = provider_config.api_key if provider_config else provider_info.get("api_key")
            
            # Check if API key is provided - warn if missing but continue for some providers - Claude Generated
            if not api_key:
                if provider_type in ["ollama"]:
                    # Ollama providers may or may not need API keys depending on setup
                    self.logger.info(f"No API key configured for {provider} - continuing without authentication")
                elif provider_type in ["gemini", "anthropic"]:
                    # API-only providers require API keys
                    self.logger.warning(f"No API key found for {provider} - initialization skipped")
                    return
                else:
                    # Other providers: warn but continue (might not need API key)
                    self.logger.info(f"No API key configured for {provider} - continuing without authentication")

            # Call the specific initializer for this provider
            if provider_info["initializer"]:
                provider_info["initializer"](provider, module, api_key, provider_info)
            else:
                self.logger.warning(f"No initializer defined for {provider}")

            self.logger.info(f"Successfully initialized {provider}")

        except ImportError as ie:
            self.logger.warning(
                f"Could not import {provider_info['module']} for {provider}: {str(ie)}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing {provider}: {str(e)}")
            self.logger.debug(traceback.format_exc())

    def _register_providers_lazy(self, providers: List[str] = None):
        """
        Register providers without testing connections (lazy initialization) - Claude Generated
        
        Args:
            providers: List of providers to register. If None, registers all providers.
        """
        if providers is None:
            providers = list(self.supported_providers.keys())
        
        # Apply Ollama routing logic for backward compatibility
        filtered_providers = []
        for provider in providers:
            if provider == "ollama":
                # Check if we have any enabled Ollama providers
                if self.alima_config.unified_config.get_enabled_ollama_providers():
                    filtered_providers.append(provider)
                    continue
                else:
                    self.logger.info("No enabled Ollama providers, skipping legacy ollama registration")
                    continue
            filtered_providers.append(provider)

        # Register providers in clients dict but don't initialize them
        for provider in filtered_providers:
            # Check with case-insensitive lookup for supported providers - Claude Generated
            if provider not in self.supported_providers:
                # Try case-insensitive search
                found = False
                for key in self.supported_providers.keys():
                    if key.lower() == provider.lower():
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Unsupported provider: {provider}")
                    continue

            # Mark provider as registered but not initialized - use original case for consistency - Claude Generated
            self.clients[provider] = "lazy_uninitialized"
            self.logger.debug(f"Registered provider for lazy initialization: {provider}")

    def _ensure_provider_initialized(self, provider: str) -> bool:
        """
        Ensure a provider is actually initialized before use (lazy loading) - Claude Generated

        Args:
            provider: Provider name to initialize

        Returns:
            True if provider is initialized, False if initialization failed
        """
        # Map legacy provider names to actual configured providers - Claude Generated
        mapped_provider = self._map_provider_name(provider)

        self.logger.info(f"🔧 ENSURE_INIT: Checking initialization for provider '{mapped_provider}' (original: '{provider}')")
        self.logger.info(f"🔧 CLIENT_KEYS_AVAILABLE: {list(self.clients.keys())}")

        if mapped_provider not in self.clients:
            self.logger.warning(f"🔧 PROVIDER_NOT_IN_CLIENTS: '{mapped_provider}' not found in self.clients")
            return False

        # If provider is already initialized (not a string), return True
        if self.clients[mapped_provider] != "lazy_uninitialized":
            self.logger.info(f"🔧 PROVIDER_ALREADY_INIT: '{mapped_provider}' is already initialized")
            return True

        # Initialize the provider now
        self.logger.info(f"🔧 LAZY_INITIALIZING: Starting lazy initialization for provider '{mapped_provider}'")
        try:
            self._initialize_single_provider(mapped_provider)
            success = mapped_provider in self.clients and self.clients[mapped_provider] != "lazy_uninitialized"
            self.logger.info(f"🔧 LAZY_INIT_RESULT: '{mapped_provider}' initialization success: {success}")
            return success
        except Exception as e:
            self.logger.error(f"🔧 LAZY_INIT_FAILED: Failed to lazy-initialize provider {mapped_provider}: {e}")
            return False

    def initialize_providers(self, providers: List[str] = None):
        """
        Initialize specified providers or all supported ones with Ollama routing logic - Claude Generated

        Args:
            providers: List of providers to initialize. If None, tries to initialize all providers.
        """
        if providers is None:
            providers = list(self.supported_providers.keys())
            
        # Apply Ollama routing logic for backward compatibility
        filtered_providers = []
        for provider in providers:
            # For backward compatibility, allow legacy "ollama" provider
            if provider == "ollama":
                # Check if we have any enabled Ollama providers
                if self.alima_config.unified_config.get_enabled_ollama_providers():
                    filtered_providers.append(provider)
                    continue
                else:
                    self.logger.info("No enabled Ollama providers, skipping legacy ollama initialization")
                    continue
            filtered_providers.append(provider)

        for provider in filtered_providers:
            # Check both original case and lowercase for backward compatibility - Claude Generated
            if provider not in self.supported_providers and provider.lower() not in self.supported_providers:
                self.logger.warning(f"Unsupported provider: {provider}")
                continue

            # Use original case for provider names - Claude Generated
            self._initialize_single_provider(provider)

    def get_available_providers(self) -> List[str]:
        """
        Get a clean list of user-facing, available providers. - Gemini Refactor

        Returns:
            List of provider names.
        """
        provider_list = []

        # 1. Handle static providers with API key check only (separate from model loading) - Claude Generated
        if self.alima_config.unified_config.gemini_api_key:
            provider_list.append("gemini")

        if self.alima_config.unified_config.anthropic_api_key:
            provider_list.append("anthropic")

        # 2. Add all enabled providers from unified config - Claude Generated
        if hasattr(self.alima_config.unified_config, 'get_enabled_providers'):
            for provider in self.alima_config.unified_config.get_enabled_providers():
                # Add all enabled providers regardless of type
                if provider.name not in provider_list:
                    provider_list.append(provider.name)
            
        # Return a unique list while preserving order
        unique_providers = []
        for provider in provider_list:
            if provider not in unique_providers:
                unique_providers.append(provider)
                
        return unique_providers

    def _map_provider_name(self, provider: str) -> str:
        """
        Map legacy provider names to actual configured provider names - Claude Generated

        Args:
            provider: Original provider name (may be legacy "ollama")

        Returns:
            Actual configured provider name
        """
        if provider == "ollama":
            # Map "ollama" to first available configured ollama provider
            for ollama_provider in self.alima_config.unified_config.get_enabled_ollama_providers():
                if ollama_provider.name in self.clients:
                    self.logger.info(f"🔄 PROVIDER_MAPPING: 'ollama' → '{ollama_provider.name}'")
                    return ollama_provider.name

            # Fallback to localhost if no configured providers found
            if "localhost" in self.clients:
                self.logger.info(f"🔄 PROVIDER_MAPPING: 'ollama' → 'localhost' (fallback)")
                return "localhost"

            self.logger.warning(f"🔄 PROVIDER_MAPPING: No available ollama providers found for 'ollama'")

        return provider  # Return original if no mapping needed

    def get_preferred_ollama_provider(self) -> Optional[str]:
        """
        Get the preferred Ollama provider based on configuration - Claude Generated
        
        Returns first available enabled Ollama provider, or legacy 'ollama' if available
        
        Returns:
            Provider name or None if no Ollama provider is available
        """
        available = self.get_available_providers()
        
        # Check for legacy Ollama provider first (backward compatibility)
        if "ollama" in available:
            return "ollama"
        
        # Check for configured Ollama providers
        for ollama_provider in self.alima_config.unified_config.get_enabled_ollama_providers():
            provider_key = ollama_provider.name  # Use the actual provider name directly - Claude Generated
            if provider_key in available:
                return provider_key
        
        return None

    def get_available_models(self, provider: str) -> List[str]:
        """
        Get available models for specified provider with direct approach - Claude Generated

        Args:
            provider: The provider name.

        Returns:
            List of model names.
        """
        # Simple approach: ping check → direct model loading
        if not self.is_provider_reachable(provider):
            return []

        # Direct access - no lazy loading complications with case-insensitive lookup - Claude Generated
        client_key = None
        if provider in self.clients:
            client_key = provider
        else:
            # Case-insensitive fallback
            for key in self.clients.keys():
                if key.lower() == provider.lower():
                    client_key = key
                    break

        if not client_key:
            self.logger.warning(f"No client found for provider '{provider}'. Available clients: {list(self.clients.keys())}")
            return []

        try:
            # Get provider config from unified configuration - Claude Generated
            config = self.config_manager.load_config() if self.config_manager else None
            if not config or not hasattr(config, 'unified_config'):
                return []

            provider_config = None
            for p in config.unified_config.providers:
                if p.name.lower() == provider.lower():
                    provider_config = p
                    break

            if not provider_config:
                return []

            # Simple provider type based model loading - Claude Generated
            self.logger.info(f"Loading models for {provider} (type: {provider_config.provider_type})")
            if provider_config.provider_type == "gemini":
                return [
                    model.name.split("/")[-1]
                    for model in self.clients[client_key].list_models()
                    if hasattr(model, "name")
                ]

            elif provider_config.provider_type == "ollama":
                # Use native Ollama client
                models_response = self.clients[client_key].list()
                if 'models' in models_response:
                    models = []
                    for model in models_response['models']:
                        # Try different possible keys for model name
                        if 'name' in model:
                            models.append(model['name'])
                        elif 'model' in model:
                            models.append(model['model'])
                        elif isinstance(model, str):
                            models.append(model)
                    return models
                return []

            elif provider_config.provider_type == "openai_compatible":
                # OpenAI-compatible provider - Claude Generated
                models = []
                try:
                    for model_obj in self.clients[client_key].models.list():
                        if hasattr(model_obj, "id"):
                            models.append(model_obj.id)
                    return models
                except Exception as openai_error:
                    self.logger.warning(f"OpenAI client models.list() failed for {provider}: {openai_error}")
                    # Fallback to direct HTTP request - Claude Generated
                    try:
                        import requests
                        base_url = provider_config.base_url.rstrip('/')
                        models_url = f"{base_url}/models"

                        headers = {}
                        if provider_config.api_key and provider_config.api_key not in ["None", "no-key-required"]:
                            headers["Authorization"] = f"Bearer {provider_config.api_key}"

                        self.logger.info(f"Fallback HTTP request to {models_url} with headers: {bool(headers)}")
                        response = requests.get(models_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        data = response.json()

                        if "data" in data:
                            models_list = [model["id"] for model in data["data"] if "id" in model]
                            self.logger.info(f"Fallback HTTP success: {len(models_list)} models found")
                            return models_list
                        return []
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback HTTP request failed for {provider}: {fallback_error}")
                        return []

            elif provider_config.provider_type == "anthropic":
                models = []
                for model_obj in self.clients[client_key].models.list():
                    if hasattr(model_obj, "id"):
                        models.append(model_obj.id)
                return models

            else:
                return []  # Unknown provider type

        except Exception as e:
            self.logger.error(f"Error getting models for {provider}: {str(e)}")
            return []

        return []

    def process_image(self, image_input: Union[str, bytes]) -> bytes:
        """
        Convert image input to bytes.

        Args:
            image_input: Path to image file or image bytes.

        Returns:
            Image content as bytes.
        """
        if isinstance(image_input, str):
            # If it's a path string, read the file
            with open(image_input, "rb") as img_file:
                return img_file.read()
        # If it's already bytes, return as is
        return image_input

    def generate_response(
        self,
        provider: str,
        model: str,
        prompt: str,
        request_id: str,
        temperature: float = 0.7,
        p_value: float = 0.1,
        seed: Optional[int] = None,
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> Union[str, Any]:  # Return type can be str or a generator
        """
        Generate a response from the specified provider using the given parameters.

        Args:
            provider: Provider name
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            image: Optional image data
            system: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generated text response (str) or a generator if streaming.
        """
        # Map legacy provider names to actual configured providers - Claude Generated
        provider = self._map_provider_name(provider)

        # Reset cancellation state
        self.cancel_requested = False
        self.stream_running = True
        self.current_provider = provider
        self.current_request_id = request_id
        self.last_chunk_time = time.time()  # Setze den Timer für den ersten Chunk

        # Starte den Watchdog
        self._init_watchdog()

        # Ensure provider is initialized if using lazy loading - Claude Generated
        if not self._ensure_provider_initialized(provider):
            error_msg = f"Provider {provider} not available or failed to initialize"
            self.generation_error.emit(request_id, error_msg)
            self.stream_running = False
            raise ValueError(error_msg)

        try:
            # Log the request
            self.logger.info(f"Generating with {provider}/{model} (stream={stream})")

            # --- Start Gemini Refactor: Dynamic Provider Dispatch ---
            provider_info = self.supported_providers.get(provider)

            if provider_info and 'generator' in provider_info:
                generator_func = provider_info['generator']
                
                # Handle the special case for native ollama generator which has a different signature
                if generator_func == self._generate_ollama_native:
                    response = generator_func(
                        provider, model, prompt, temperature, p_value, seed, image, system, stream
                    )
                else:
                    # All other generators have a consistent signature
                    response = generator_func(
                        model, prompt, temperature, p_value, seed, image, system, stream
                    )
            else:
                error_msg = f"Generation not implemented for provider: {provider}"
                self.logger.error(error_msg)
                self.generation_error.emit(self.current_request_id, error_msg)
                self.stream_running = False
                raise ValueError(error_msg)
            # --- End Gemini Refactor ---

            if stream:
                return response  # Return the generator directly
            else:
                # Nur Finish-Signal emittieren wenn keine Abbruch angefordert wurde
                if not self.cancel_requested:
                    self.generation_finished.emit(
                        self.current_request_id, "Generation finished"
                    )
                self.stream_running = False
                self.current_provider = None
                self.current_request_id = None
                return response

        except Exception as e:
            error_msg = f"Error in generate_response: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            self.generation_error.emit(self.current_request_id, error_msg)
            self.stream_running = False
            self.current_provider = None
            self.current_request_id = None
            raise e

    # Provider-specific initialization methods

    def _init_gemini(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize Gemini provider."""
        module.configure(api_key=api_key)
        self.clients[provider] = module

    def _init_openai_compatible(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize OpenAI-compatible providers (OpenAI, ChatAI, Comet) - Claude Generated"""
        params = {}
        
        # Always provide an API key for OpenAI-compatible clients (required by OpenAI library)
        # Use provided key or placeholder for providers that don't need authentication - Claude Generated
        if api_key and api_key != "None":
            params["api_key"] = api_key
        else:
            params["api_key"] = "no-key-required"  # Placeholder for providers without authentication
            self.logger.info(f"Initializing {provider} with placeholder API key (no authentication required)")

        # Add base_url if specified - Claude Generated
        provider_config = provider_info.get("config")
        if provider_config and hasattr(provider_config, 'base_url') and provider_config.base_url:
            params["base_url"] = provider_config.base_url
        elif "base_url" in provider_info:
            params["base_url"] = provider_info["base_url"]
        elif provider_info.get("params", {}).get("base_url"):
            params["base_url"] = provider_info["params"]["base_url"]

        # Create client with debug logging - Claude Generated
        client_class = getattr(module, provider_info["class"])
        self.clients[provider] = client_class(**params)

        self.logger.info(f"🔧 CLIENT_STORED: Provider '{provider}' client stored in self.clients")
        self.logger.info(f"🔧 CLIENT_KEYS: Current client keys: {list(self.clients.keys())}")

        if "base_url" in params:
            self.logger.info(
                f"{provider} initialized with base URL: {params['base_url']}"
            )

    def _init_anthropic(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize Anthropic provider."""
        client_class = getattr(module, provider_info["class"])
        self.clients[provider] = client_class(api_key=api_key)

    def _is_server_reachable(
        self, url: str, port: Optional[int] = None, timeout: int = 2
    ) -> bool:
        """Check if a server at the given URL and port is reachable.

        Args:
            url: The URL to check (without protocol if port is specified)
            port: Optional port number
            timeout: Connection timeout in seconds

        Returns:
            bool: True if server is reachable, False otherwise
        """
        import socket
        import requests
        from urllib.parse import urlparse

        # Clean up URL format
        if port is not None:
            # Strip any protocol from URL
            clean_url = url.split("://")[-1] if "://" in url else url
            # Remove any path or query params
            clean_url = clean_url.split("/")[0]
            # Handle IPv6 addresses
            if clean_url.startswith("[") and "]" in clean_url:
                clean_url = clean_url.split("]")[0] + "]"

            try:
                # Try simple socket connection first
                # Get address info to handle both IPv4 and IPv6 properly
                addr_info = socket.getaddrinfo(
                    clean_url, int(port), socket.AF_UNSPEC, socket.SOCK_STREAM
                )

                for family, socktype, proto, _, sockaddr in addr_info:
                    try:
                        sock = socket.socket(family, socktype, proto)
                        sock.settimeout(timeout)
                        result = sock.connect_ex(sockaddr)
                        sock.close()
                        if result == 0:
                            self.logger.debug(
                                f"Socket connection to {clean_url}:{port} successful"
                            )
                            return True
                    except socket.error:
                        continue

                # If socket fails, try HTTP request with full URL
                full_url = f"http://{clean_url}:{port}"
                self.logger.debug(f"Trying HTTP request to {full_url}")

                # Set shorter timeout for request to avoid long hangs
                response = requests.get(
                    full_url, timeout=timeout, headers={"Connection": "close"}
                )
                return response.status_code < 500

            except socket.gaierror:
                self.logger.debug(f"Could not resolve hostname: {clean_url}")
                return False
            except requests.exceptions.RequestException as e:
                self.logger.debug(
                    f"HTTP request failed for {clean_url}:{port} - {str(e)}"
                )
                return False
            except Exception as e:
                self.logger.debug(
                    f"Server check failed for {clean_url}:{port} - {str(e)}"
                )
                return False
        else:
            # Handle full URL case
            try:
                # Try GET instead of HEAD
                if not url.startswith(("http://", "https://")):
                    url = f"http://{url}"

                self.logger.debug(f"Trying HTTP request to {url}")
                response = requests.get(
                    url, timeout=timeout, headers={"Connection": "close"}
                )
                return response.status_code < 500
            except requests.exceptions.RequestException as e:
                self.logger.debug(f"HTTP request failed for {url} - {str(e)}")
                return False
            except Exception as e:
                self.logger.debug(f"Server check failed for {url} - {str(e)}")
                return False

    def _init_ollama(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize Ollama provider."""
        full_ollama_url = f"{self.ollama_url}:{self.ollama_port}"
        self.logger.info(f"Attempting to initialize Ollama at {full_ollama_url}")
        try:
            response = module.get(
                f"{full_ollama_url}/api/tags", timeout=5
            )  # 5 second timeout to prevent UI blocking
            response.raise_for_status()  # Raise an exception for HTTP errors
            self.clients[provider] = module
            self.logger.info(
                f"Ollama client initialized successfully. Models: {[m['name'] for m in response.json()['models']]}"
            )
        except requests.exceptions.ConnectionError as ce:
            self.logger.error(f"Ollama connection error: {ce}")
            self.logger.warning(f"Ollama server not accessible at {full_ollama_url}")
        except Exception as e:
            self.logger.error(f"Error initializing Ollama: {e}")

    def _init_ollama_native_provider(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize specific Ollama native client provider - Claude Generated"""
        try:
            # Get provider configuration from unified config (provider data is in provider_info['config'])
            provider_config = provider_info.get('config', {})
            base_url = provider_config.base_url if hasattr(provider_config, 'base_url') else "http://localhost:11434"
            provider_api_key = provider_config.api_key if hasattr(provider_config, 'api_key') else ""
            provider_enabled = provider_config.enabled if hasattr(provider_config, 'enabled') else True
            provider_name = provider


            # Check if provider is enabled
            if not provider_enabled:
                self.logger.info(f"Ollama provider {provider_name} disabled in configuration")
                return

            # Set up client parameters
            client_params = {"host": base_url}

            # Add authorization header if API key is provided
            if provider_api_key:
                client_params["headers"] = {
                    'Authorization': provider_api_key
                }
                self.logger.info(f"Initializing native Ollama client {provider_name} with authentication at {base_url}")
            else:
                self.logger.info(f"Initializing native Ollama client {provider_name} without authentication at {base_url}")
            
            # Create native Ollama client
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama library not available. Please install it: pip install ollama")
            
            client_instance = ollama.Client(**client_params)
            self.clients[provider] = client_instance
            
            self.logger.info(f"Native Ollama client {provider_name} initialized successfully at {base_url}")
            
        except Exception as e:
            self.logger.error(f"Error initializing native Ollama client: {e}")
            self.logger.debug(traceback.format_exc())

    def _init_azure_inference(
        self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]
    ):
        """Initialize Azure Inference-based providers (Azure OpenAI, GitHub Copilot)."""
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        # Get endpoint from config or environment
        endpoint_key = provider_info["params"]["endpoint"]
        endpoint = self.config.get(endpoint_key) or os.getenv(endpoint_key.upper())

        if not endpoint and provider == "github":
            # Default endpoint for GitHub if not specified
            endpoint = "https://models.inference.ai.azure.com"

        if not endpoint:
            self.logger.warning(f"Missing endpoint configuration for {provider}")
            return

        # Create client
        self.clients[provider] = ChatCompletionsClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key)
        )

        # Store additional modules for message creation
        self.clients[f"{provider}_modules"] = importlib.import_module(
            "azure.ai.inference.models"
        )

        # Default model
        default_model = provider_info["params"]["default_model"]
        self.config[f"{provider}_default_model"] = self.config.get(
            f"{provider}_default_model", default_model
        )

    # Provider-specific generation methods

    def _generate_gemini(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> str:
        """Generate response using Google Gemini."""
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": p_value,
            }

            # Fix für model_version Problem
            if not model.startswith("models/"):
                model_name = model
                if not model_name.startswith("gemini-"):
                    model_name = f"gemini-1.5-flash"  # Current valid Gemini model - Claude Generated

                model = f"models/{model_name}"
                self.logger.info(f"Verwende vollqualifiziertes Gemini Modell: {model}")

            # Create model instance with system instruction if provided
            system_instruction = system if system else None
            model_instance = self.clients["gemini"].GenerativeModel(
                model, system_instruction=system_instruction
            )

            # Vorbereiten des Inhalts
            if image:
                img_bytes = self.process_image(image)
                content = [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
            else:
                content = prompt

            # Handle streaming option
            if stream:
                try:
                    # Versuche die aktuelle Streaming-API zu verwenden
                    response = model_instance.generate_content(
                        content, generation_config=generation_config, stream=True
                    )

                    full_response = ""
                    for chunk in response:
                        # Prüfen auf Abbruchsignal
                        if self.cancel_requested:
                            self.logger.info("Gemini generation cancelled")
                            break

                        # In neueren Versionen ist chunk.text direkt verfügbar
                        if hasattr(chunk, "text"):
                            chunk_text = chunk.text
                        # In älteren Versionen müssen wir es aus den parts extrahieren
                        elif hasattr(chunk, "parts"):
                            chunk_text = "".join(
                                [
                                    part.text
                                    for part in chunk.parts
                                    if hasattr(part, "text")
                                ]
                            )
                        else:
                            # Fallback für andere Formate
                            try:
                                chunk_text = chunk.candidates[0].content.parts[0].text
                            except (AttributeError, IndexError):
                                chunk_text = str(chunk)

                        if chunk_text:
                            full_response += chunk_text
                            self.text_received.emit(self.current_request_id, chunk_text)

                    return full_response

                except (AttributeError, TypeError) as e:
                    # Fallback für den Fall, dass das Streaming nicht funktioniert
                    self.logger.warning(
                        f"Streaming fehler: {e}, verwende nicht-streaming Methode"
                    )
                    response = model_instance.generate_content(
                        content, generation_config=generation_config, stream=False
                    )
                    response_text = (
                        response.text
                        if hasattr(response, "text")
                        else response.parts[0].text
                    )
                    self.text_received.emit(self.current_request_id, response_text)
                    return response_text
            else:
                # Nicht-Streaming-Variante
                response = model_instance.generate_content(
                    content, generation_config=generation_config
                )

                # Je nach API-Version kann die Antwort unterschiedlich strukturiert sein
                if hasattr(response, "text"):
                    return response.text
                elif hasattr(response, "parts") and response.parts:
                    return response.parts[0].text
                else:
                    response.resolve()  # Für ältere API-Versionen
                    return response.text

        except Exception as e:
            self.logger.error(f"Gemini error: {str(e)}")
            self.logger.debug(traceback.format_exc())
            error_msg = f"Error with Gemini: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            raise e

    def _generate_github(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> str:
        """Generate response using Github Copilot."""
        try:
            # Prepare messages
            messages = []

            # Add system message if provided
            if system:
                messages.append({"role": "system", "content": system})

            # Add user message
            if image:
                img_bytes = self.process_image(image)
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # Create multimodal content
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ]
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})

            # Handle streaming
            if stream:
                try:
                    # Versuche zuerst die neue Methode
                    response_stream = self.clients["github"].chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=p_value,
                        stream=True,
                    )

                    full_response = ""
                    for chunk in response_stream:
                        if hasattr(chunk.choices[0], "delta") and hasattr(
                            chunk.choices[0].delta, "content"
                        ):
                            chunk_text = chunk.choices[0].delta.content
                            if chunk_text:
                                full_response += chunk_text
                                self.text_received.emit(
                                    self.current_request_id, chunk_text
                                )

                    return full_response

                except (AttributeError, TypeError, ValueError) as e:
                    self.logger.warning(
                        f"Erste Streaming-Methode fehlgeschlagen: {e}, versuche Alternative..."
                    )

                    try:
                        # Versuche es mit der alternativen Methode für ältere OpenAI-kompatible Clients
                        response_stream = self.clients[
                            "github"
                        ].chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            stream=True,
                        )

                        full_response = ""
                        for chunk in response_stream:
                            if (
                                hasattr(chunk, "choices")
                                and chunk.choices
                                and hasattr(chunk.choices[0], "delta")
                            ):
                                delta = chunk.choices[0].delta
                                if hasattr(delta, "content") and delta.content:
                                    chunk_text = delta.content
                                    full_response += chunk_text
                                    self.text_received.emit(
                                        self.current_request_id, chunk_text
                                    )

                        return full_response

                    except Exception as stream_error:
                        self.logger.warning(
                            f"Alle Streaming-Methoden fehlgeschlagen: {stream_error}, verwende non-streaming"
                        )

                        # Fallback auf non-streaming
                        response = self.clients["github"].chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            stream=False,
                        )

                        if (
                            hasattr(response, "choices")
                            and response.choices
                            and hasattr(response.choices[0], "message")
                        ):
                            response_text = response.choices[0].message.content
                            self.text_received.emit(
                                self.current_request_id, response_text
                            )
                            return response_text
                        else:
                            raise ValueError("Unerwartetes Antwortformat")

            else:
                # Non-streaming direct API call
                response = self.clients["github"].chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )

                return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Github error: {str(e)}")
            self.logger.debug(traceback.format_exc())
            error_msg = f"Error with Github: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            return error_msg

    def _generate_openai_compatible(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> str:
        """Generate response using OpenAI-compatible APIs - Claude Generated"""
        # Use the current provider being processed (set in generate_response)
        provider = self.current_provider
        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not initialized in clients")

        try:
            # Create messages array
            messages = []

            # Add system message if provided
            if system:
                messages.append({"role": "system", "content": system})

            # Add user message with optional image
            if image:
                img_bytes = self.process_image(image)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
                                },
                            },
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": prompt})

            # Set up parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": p_value,
                "stream": stream,
            }

            # Add seed if provided
            if seed is not None:
                params["seed"] = seed

            # Handle streaming option
            if stream:
                response_stream = self.clients[provider].chat.completions.create(
                    **params
                )

                # Speichere request_id für möglichen Abbruch
                if hasattr(response_stream, "id"):
                    self.current_request_id = response_stream.id
                    self.logger.info(
                        f"OpenAI compatible request ID: {self.current_request_id}"
                    )

                full_response = ""
                for chunk in response_stream:
                    # Aktualisiere den Zeitpunkt des letzten empfangenen Chunks
                    self.last_chunk_time = time.time()

                    # Prüfen auf Abbruchsignal
                    if self.cancel_requested:
                        self.logger.info(f"{provider} generation cancelled")
                        break

                    if (
                        chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        chunk_text = chunk.choices[0].delta.content
                        full_response += chunk_text
                        self.text_received.emit(self.current_request_id, chunk_text)

                return full_response
            else:
                # Make API call without streaming
                response = self.clients[provider].chat.completions.create(**params)
                return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"{provider.capitalize()} error: {str(e)}")
            error_msg = f"Error with {provider.capitalize()}: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            return error_msg

    def _generate_ollama(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> Union[str, Any]:  # Changed return type to Union[str, Generator]
        """Generate response using Ollama."""
        full_ollama_url = f"{self.ollama_url}:{self.ollama_port}"
        try:
            # Set up request data
            data = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "num_ctx": 32768,
                    "temperature": temperature,
                    "top_p": p_value,
                },
                "stream": stream,
                "think" : False
            }

            # Add seed if provided
            if seed is not None:
                data["options"]["seed"] = seed

            # Add image if provided
            if image:
                img_bytes = self.process_image(image)
                data["images"] = [base64.b64encode(img_bytes).decode()]

            # Add system prompt if provided
            if system:
                data["system"] = system

            self.logger.info(f"Sending Ollama request with model: {model}")
            self.logger.info(f"Ollama request data: {data}")
            # Make API call with streaming
            if stream:
                response = self.clients["ollama"].post(
                    f"{full_ollama_url}/api/generate",
                    json=data,
                    stream=True,
                    timeout=120,  # 120 second timeout for initial connection
                )

                # Process streaming response
                try:
                    for line in response.iter_lines():
                        # Aktualisiere den Zeitpunkt des letzten empfangenen Chunks
                        self.last_chunk_time = time.time()

                        # Prüfen auf Abbruchsignal
                        if self.cancel_requested:
                            # Bei Ollama können wir die Anfrage serverseitig abbrechen
                            try:
                                self._cancel_ollama_request()
                            except Exception as cancel_error:
                                self.logger.warning(
                                    f"Could not cancel Ollama request: {cancel_error}"
                                )

                            self.logger.info("Ollama generation cancelled")
                            break

                        if line:
                            json_response = json.loads(line)
                            if "response" in json_response:
                                chunk = json_response["response"]
                                yield chunk  # Yield the chunk
                except Exception as stream_e:
                    self.logger.error(f"Error during Ollama streaming: {stream_e}")
                    self.logger.debug(traceback.format_exc())  # Log full traceback
                    raise stream_e  # Re-raise to be caught by outer try-except
            else:
                # Non-streaming option
                data["stream"] = False
                response = self.clients["ollama"].post(
                    f"{full_ollama_url}/api/generate",
                    json=data,
                    timeout=120,  # 120 second timeout for non-streaming requests
                )
                return response.json()["response"]

        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            error_msg = f"Error with Ollama: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            raise e

    def _generate_ollama_native(
        self,
        provider: str,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> Union[str, Any]:
        """Generate response using native Ollama client - Claude Generated"""
        try:
            # Prepare messages for chat format
            messages = []
            
            # Add system message if provided
            if system:
                messages.append({
                    'role': 'system',
                    'content': system,
                })
            
            # Add user message
            if image:
                img_bytes = self.process_image(image)
                # Ollama native client expects base64 encoded images
                import base64
                img_b64 = base64.b64encode(img_bytes).decode()
                
                messages.append({
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64]  # Native client format for images
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': prompt,
                })
            
            # Set up options
            options = {
                'temperature': temperature,
                'top_p': p_value,
            }
            
            if seed is not None:
                options['seed'] = seed
            
            self.logger.info(f"Sending native Ollama request with model: {model}")
            
            # Make API call with streaming support
            if stream:
                # Use provider-specific native client streaming
                stream_response = self.clients[provider].chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream_response:
                    # Update last chunk time
                    self.last_chunk_time = time.time()
                    
                    # Check for cancellation
                    if self.cancel_requested:
                        self.logger.info("Native Ollama generation cancelled")
                        break
                    
                    # Extract content from chunk
                    if 'message' in chunk and 'content' in chunk['message']:
                        chunk_text = chunk['message']['content']
                        if chunk_text:
                            full_response += chunk_text
                            # Emit token for streaming display
                            self.text_received.emit(self.current_request_id, chunk_text)
                
                return full_response
            else:
                # Non-streaming call
                response = self.clients[provider].chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=False
                )
                
                # Extract response content
                if 'message' in response and 'content' in response['message']:
                    return response['message']['content']
                else:
                    return str(response)  # Fallback
        
        except Exception as e:
            self.logger.error(f"Native Ollama error: {str(e)}")
            error_msg = f"Error with Native Ollama: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            raise e

    def _generate_anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> str:
        """Generate response using Anthropic."""
        try:
            # Set up parameters
            params = {
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "top_p": p_value,
                "messages": [],
                "stream": stream,
            }

            # Add system message if provided
            if system:
                params["system"] = system

            # Add user message
            if image:
                img_bytes = self.process_image(image)

                # Check if Anthropic supports image in the current version
                try:
                    from anthropic import ImageContent, ContentBlock, TextContent

                    # Create content blocks
                    content_blocks = [
                        TextContent(text=prompt, type="text"),
                        ImageContent(
                            source={
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(img_bytes).decode(),
                            },
                            type="image",
                        ),
                    ]

                    params["messages"].append(
                        {"role": "user", "content": content_blocks}
                    )
                except (ImportError, AttributeError):
                    # Fallback if the current Anthropic version doesn't support images
                    self.logger.warning(
                        "This version of Anthropic Python SDK might not support images. Sending text only."
                    )
                    params["messages"].append({"role": "user", "content": prompt})
            else:
                params["messages"].append({"role": "user", "content": prompt})

            # Add seed if provided
            if seed is not None:
                params["seed"] = seed

            # Handle streaming option
            if stream:
                stream_response = self.clients["anthropic"].messages.create(**params)

                # Speichere request_id für möglichen Abbruch
                if hasattr(stream_response, "id"):
                    self.current_request_id = stream_response.id
                    self.logger.info(f"Anthropic request ID: {self.current_request_id}")

                full_response = ""
                for chunk in stream_response:
                    # Aktualisiere den Zeitpunkt des letzten empfangenen Chunks
                    self.last_chunk_time = time.time()

                    # Prüfen auf Abbruchsignal
                    if self.cancel_requested:
                        self.logger.info("Anthropic generation cancelled")
                        break

                    if hasattr(chunk, "delta") and chunk.delta.text:
                        chunk_text = chunk.delta.text
                        full_response += chunk_text
                        self.text_received.emit(self.current_request_id, chunk_text)

                return full_response
            else:
                # Non-streaming API call
                message = self.clients["anthropic"].messages.create(**params)
                return message.content[0].text

        except Exception as e:
            self.logger.error(f"Anthropic error: {str(e)}")
            error_msg = f"Error with Anthropic: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            return error_msg

    # Auch die anderen _generate_* Methoden müssten ähnlich angepasst werden,
    # um self.last_chunk_time zu aktualisieren

    def set_timeout(self, timeout_seconds: int):
        """
        Set the timeout for request watchdog in seconds.

        Args:
            timeout_seconds: Number of seconds to wait before considering a request stuck
        """
        self.request_timeout = max(10, timeout_seconds)  # Mindestens 10 Sekunden
        self.logger.info(f"Request timeout set to {self.request_timeout} seconds")

    def _generate_azure_inference(
        self,
        model: str,
        prompt: str,
        temperature: float,
        p_value: float,
        seed: Optional[int],
        image: Optional[Union[str, bytes]] = None,
        system: Optional[str] = "",
        stream: bool = True,
    ) -> str:
        """Generate response using Azure Inference-based providers (Azure OpenAI, GitHub Copilot)."""
        # Determine provider (azure or github)
        provider = [
            p
            for p in ["azure", "github"]
            if p in self.clients and f"{p}_modules" in self.clients
        ][0]

        try:
            # Get modules for this provider
            modules = self.clients[f"{provider}_modules"]
            UserMessage = modules.UserMessage
            SystemMessage = modules.SystemMessage

            # Create messages array
            messages = []

            # Add system message if provided
            if system:
                messages.append(SystemMessage(system))

            # Vision model and image handling
            vision_models = [
                "gpt-4-vision",
                "phi-3-vision",
                "phi-4-multimodal-instruct",
                "llama-3.2-90b-vision-instruct",
            ]
            supports_vision = any(vm.lower() in model.lower() for vm in vision_models)

            # Add user message with optional image
            if image and supports_vision:
                # For vision models with image
                img_bytes = self.process_image(image)
                encoded_image = base64.b64encode(img_bytes).decode("ascii")

                # Import necessary classes
                ImageContent = modules.ImageContent
                MultiModalContent = modules.MultiModalContent

                image_content = ImageContent(data=encoded_image, mime_type="image/jpeg")

                multi_modal_content = MultiModalContent(
                    text=prompt, images=[image_content]
                )

                messages.append(UserMessage(content=multi_modal_content))
            else:
                if image and not supports_vision:
                    self.logger.warning(
                        f"Model {model} does not support image processing."
                    )
                messages.append(UserMessage(prompt))

            # Set up parameters
            params = {
                "messages": messages,
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "top_p": p_value,
                "stream": stream,
            }

            # Add seed if API supports it
            if seed is not None and hasattr(modules, "CompletionParams"):
                CompletionParams = modules.CompletionParams
                params["params"] = CompletionParams(seed=seed)

            # Handle streaming option
            if stream:
                response_stream = self.clients[provider].complete_streaming(**params)

                full_response = ""
                for chunk in response_stream:
                    # Prüfen auf Abbruchsignal
                    if self.cancel_requested:
                        self.logger.info(f"{provider} generation cancelled")
                        break

                    if (
                        chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        chunk_text = chunk.choices[0].delta.content
                        full_response += chunk_text
                        self.text_received.emit(self.current_request_id, chunk_text)

                return full_response
            else:
                # Non-streaming API call
                response = self.clients[provider].complete(**params)
                return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"{provider.capitalize()} error: {str(e)}")
            error_msg = f"Error with {provider.capitalize()}: {str(e)}"
            self.generation_error.emit(self.current_request_id, error_msg)
            return error_msg

    # Zusätzlich brauchen wir eine Methode, um Provider-spezifische System-Prompts zu konfigurieren
    def set_system_prompt(self, provider: str, system_prompt: str):
        """
        Set default system prompt for a specific provider.

        Args:
            provider: Provider name
            system_prompt: Default system prompt to use when none is provided
        """
        if provider in self.supported_providers:
            self.config[f"{provider}_system_message"] = system_prompt
            self._save_config()
            self.logger.info(f"Set default system prompt for {provider}")
        else:
            self.logger.warning(f"Unsupported provider: {provider}")
    
    def ping_test_provider(self, provider_name: str, timeout: float = 3.0) -> Dict[str, Any]:
        """
        Simple ping test to check if provider server is reachable - Claude Generated
        
        Args:
            provider_name: Name of the provider to test
            timeout: Connection timeout in seconds
            
        Returns:
            Dict with test results: {'reachable': bool, 'latency_ms': float, 'error': str}
        """
        # Skip ping test for API-based services - Claude Generated
        api_services = ['gemini', 'anthropic', 'openai', 'comet', 'chatai']
        if provider_name.lower() in api_services:
            return {
                'reachable': True,
                'latency_ms': 0.0,
                'error': None,
                'method': 'api_service'
            }
        result = {
            'reachable': False,
            'latency_ms': 0.0,
            'error': None,
            'method': 'unknown'
        }
        
        try:
            # Get provider configuration
            provider_config = None
            
            # Check if it's an Ollama provider
            if hasattr(self.config_manager.config.unified_config, 'ollama_providers'):
                for provider in self.config_manager.config.unified_config.ollama_providers:
                    if provider.name == provider_name:
                        provider_config = provider
                        break
            
            # Check if it's an OpenAI-compatible provider
            if not provider_config and hasattr(self.config_manager.config.unified_config, 'openai_compatible_providers'):
                for provider in self.config_manager.config.unified_config.openai_compatible_providers:
                    if provider.name == provider_name:
                        provider_config = provider
                        break
            
            if not provider_config:
                result['error'] = f"Provider '{provider_name}' not found in configuration"
                return result
            
            # Parse URL to get host and port
            try:
                if hasattr(provider_config, 'base_url'):
                    url = provider_config.base_url
                else:
                    # For Ollama providers, construct URL
                    protocol = "https" if provider_config.host.startswith("https://") else "http"
                    if "://" in provider_config.host:
                        url = provider_config.host
                    else:
                        url = f"{protocol}://{provider_config.host}:{provider_config.port}"
                
                parsed_url = urlparse(url)
                host = parsed_url.hostname or parsed_url.netloc.split(':')[0]
                port = parsed_url.port
                
                if not port:
                    port = 443 if parsed_url.scheme == 'https' else 80
                    
            except Exception as e:
                result['error'] = f"Failed to parse provider URL: {str(e)}"
                return result
            
            # Try socket connection first (fastest)
            start_time = time.time()
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                sock.close()
                result['reachable'] = True
                result['latency_ms'] = (time.time() - start_time) * 1000
                result['method'] = 'socket'
                return result
            except (socket.timeout, socket.error) as e:
                result['error'] = f"Socket connection failed: {str(e)}"
            
            # Fallback to ping if socket fails (for some firewalls)
            try:
                start_time = time.time()
                
                # Determine ping command based on OS
                if platform.system().lower() == "windows":
                    cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), host]
                else:
                    cmd = ["ping", "-c", "1", "-W", str(int(timeout)), host]
                
                proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 1)
                
                if proc.returncode == 0:
                    result['reachable'] = True
                    result['latency_ms'] = (time.time() - start_time) * 1000
                    result['method'] = 'ping'
                    result['error'] = None
                else:
                    result['error'] = f"Ping failed: {proc.stderr.decode()}"
                    
            except subprocess.TimeoutExpired:
                result['error'] = f"Ping timeout after {timeout} seconds"
            except Exception as e:
                result['error'] = f"Ping command failed: {str(e)}"
                
        except Exception as e:
            result['error'] = f"Unexpected error during ping test: {str(e)}"
            self.logger.error(f"Ping test error for {provider_name}: {e}")
        
        return result
    
    def _qt_ping_test_provider(self, provider_name: str, timeout_ms: int = 3000) -> Dict[str, Any]:
        """
        Qt-based ping test for provider reachability - Claude Generated
        
        Args:
            provider_name: Name of the provider to test
            timeout_ms: Connection timeout in milliseconds
            
        Returns:
            Dict with test results: {'reachable': bool, 'latency_ms': float, 'error': str}
        """
        result = {
            'reachable': False,
            'latency_ms': 0.0,
            'error': None,
            'method': 'qt_tcp'
        }
        
        try:
            # Get provider configuration
            provider_config = None
            
            # Load config if not already loaded
            if not hasattr(self.config_manager, 'config') or self.config_manager.config is None:
                config = self.config_manager.load_config()
            else:
                config = self.config_manager.config
            
            # Find provider in unified configuration - Claude Generated
            if hasattr(config, 'unified_config') and config.unified_config:
                for provider in config.unified_config.providers:
                    if provider.name == provider_name:
                        provider_config = provider
                        break
            
            if not provider_config:
                result['error'] = f"Provider '{provider_name}' not found in configuration"
                return result
            
            # Parse URL to get host and port
            try:
                if hasattr(provider_config, 'base_url'):
                    url = provider_config.base_url
                else:
                    # For Ollama providers, construct URL
                    protocol = "https" if provider_config.host.startswith("https://") else "http"
                    if "://" in provider_config.host:
                        url = provider_config.host
                    else:
                        url = f"{protocol}://{provider_config.host}:{provider_config.port}"
                
                parsed_url = urlparse(url)
                host = parsed_url.hostname or parsed_url.netloc.split(':')[0]
                port = parsed_url.port
                
                if not port:
                    port = 443 if parsed_url.scheme == 'https' else 80
                    
            except Exception as e:
                result['error'] = f"Failed to parse provider URL: {str(e)}"
                return result
            
            # Use Qt TCP socket for connection test
            tcp_socket = QTcpSocket()
            start_time = time.time()
            
            # Connect to host
            tcp_socket.connectToHost(host, port)
            
            # Wait for connection with timeout
            if tcp_socket.waitForConnected(timeout_ms):
                result['reachable'] = True
                result['latency_ms'] = (time.time() - start_time) * 1000
                tcp_socket.disconnectFromHost()
                tcp_socket.waitForDisconnected(1000)
            else:
                result['error'] = f"Qt TCP connection failed: {tcp_socket.errorString()}"
            
            tcp_socket.deleteLater()
            
        except Exception as e:
            result['error'] = f"Qt ping test error: {str(e)}"
            self.logger.error(f"Qt ping test error for {provider_name}: {e}")
        
        return result
    
    def has_provider_api_key(self, provider: str) -> bool:
        """
        Check if provider has API key configured (separate from reachability) - Claude Generated
        Args:
            provider: Provider name to check
        Returns:
            True if API key is configured, False otherwise
        """
        if provider == "gemini":
            return bool(self.alima_config.unified_config.gemini_api_key)
        elif provider == "anthropic":
            return bool(self.alima_config.unified_config.anthropic_api_key)
        elif hasattr(self.alima_config.unified_config, 'get_enabled_openai_providers'):
            for openai_provider in self.alima_config.unified_config.get_enabled_openai_providers():
                if openai_provider.name == provider:
                    return bool(openai_provider.api_key)
        elif hasattr(self.alima_config.unified_config, 'get_enabled_ollama_providers'):
            for ollama_provider in self.alima_config.unified_config.get_enabled_ollama_providers():
                if ollama_provider.name == provider:
                    return ollama_provider.enabled
        return False
    
    def is_provider_reachable(self, provider_name: str, force_check: bool = False) -> bool:
        """
        Check if provider is reachable using cached status - Claude Generated
        
        Args:
            provider_name: Name of provider to check
            force_check: Force fresh reachability check, ignore cache
            
        Returns:
            True if provider is reachable, False otherwise
        """
        # Skip reachability check for API-based services - Claude Generated
        # Check if provider has API key (indicates API-based service)
        if self._is_api_based_provider(provider_name):
            return True  # Assume API services are always reachable
        current_time = time.time()
        
        # Check cache first if not forcing check
        if not force_check and provider_name in self.provider_status_cache:
            cache_entry = self.provider_status_cache[provider_name]
            cache_age = current_time - cache_entry.get('last_check', 0)
            
            # Use cached result if not expired
            if cache_age < self.status_cache_timeout:
                return cache_entry.get('reachable', False)
        
        # Perform fresh reachability check
        ping_result = self._qt_ping_test_provider(provider_name, timeout_ms=3000)
        
        # Update cache
        self.provider_status_cache[provider_name] = {
            'reachable': ping_result['reachable'],
            'last_check': current_time,
            'latency_ms': ping_result.get('latency_ms', 0.0),
            'error': ping_result.get('error')
        }
        
        # Emit status change signal
        self.provider_status_changed.emit(provider_name, ping_result['reachable'])
        
        return ping_result['reachable']

    def _is_api_based_provider(self, provider_name: str) -> bool:
        """
        Check if provider is API-based (has API key or is known API service) - Claude Generated

        Args:
            provider_name: Name of provider to check

        Returns:
            True if provider is API-based, False otherwise
        """
        # Known API-only services (no local hosting)
        known_api_services = ['gemini', 'anthropic', 'openai']
        if provider_name.lower() in known_api_services:
            return True

        # Check if provider has API key in unified config
        try:
            config = self.config_manager.load_config() if hasattr(self, 'config_manager') else None
            if config and hasattr(config, 'unified_config') and config.unified_config:
                for provider in config.unified_config.providers:
                    if provider.name == provider_name and provider.api_key:
                        return True
        except Exception as e:
            self.logger.debug(f"Error checking API key for {provider_name}: {e}")

        return False
    
    def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """
        Get detailed provider status information - Claude Generated
        
        Args:
            provider_name: Name of provider
            
        Returns:
            Dict with status info: {'reachable': bool, 'latency_ms': float, 'last_check': timestamp}
        """
        if provider_name in self.provider_status_cache:
            return self.provider_status_cache[provider_name].copy()
        else:
            return {'reachable': False, 'latency_ms': 0.0, 'last_check': 0, 'error': 'Not checked yet'}
    
    def refresh_all_provider_status(self) -> Dict[str, bool]:
        """
        Refresh reachability status for all configured providers - Claude Generated
        
        Returns:
            Dict mapping provider names to reachability status
        """
        results = {}
        
        # Load current configuration
        config = self.config_manager.load_config()
        
        # Check Ollama providers
        if hasattr(config.unified_config, 'ollama_providers'):
            for provider in config.unified_config.ollama_providers:
                if provider.enabled:
                    results[provider.name] = self.is_provider_reachable(provider.name, force_check=True)
        
        # Check OpenAI-compatible providers
        if hasattr(config.unified_config, 'openai_compatible_providers'):
            for provider in config.unified_config.openai_compatible_providers:
                if provider.enabled:
                    results[provider.name] = self.is_provider_reachable(provider.name, force_check=True)
        
        self.logger.info(f"Provider status refresh completed: {results}")
        return results
    
    def _refresh_provider_status(self):
        """
        Timer callback to refresh provider status periodically - Claude Generated
        """
        self.refresh_all_provider_status()
