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

        # Define static provider configurations (non-OpenAI compatible)
        self._init_static_provider_configs()
        
        # Initialize dynamic OpenAI-compatible providers
        self._init_dynamic_provider_configs()

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

    def _init_static_provider_configs(self):
        """Initialize static provider configurations (non-OpenAI compatible) - Claude Generated"""
        self.static_providers = {
            "gemini": {
                "module": "google.generativeai",
                "class": None,
                "initializer": self._init_gemini,
                "generator": self._generate_gemini,
            },
            "anthropic": {
                "module": "anthropic", 
                "class": "Anthropic",
                "initializer": self._init_anthropic,
                "generator": self._generate_anthropic,
            },
            "ollama": {
                "module": "requests",
                "class": None,
                "initializer": self._init_ollama,
                "generator": self._generate_ollama,
            },
        }

    def _init_dynamic_provider_configs(self):
        """Initialize dynamic OpenAI-compatible provider configurations - Claude Generated"""
        self.openai_providers = {}
        
        # Load all enabled OpenAI-compatible providers from configuration
        for provider in self.alima_config.llm.get_enabled_providers():
            provider_key = provider.name.lower().replace(" ", "_")
            self.openai_providers[provider_key] = {
                "name": provider.name,
                "module": "openai",
                "class": "OpenAI",
                "base_url": provider.base_url,
                "api_key": provider.api_key,
                "description": provider.description,
                "initializer": self._init_openai_compatible,
                "generator": self._generate_openai_compatible,
            }
            self.logger.debug(f"Configured OpenAI-compatible provider: {provider.name} -> {provider.base_url}")
        
        # Add Ollama providers as OpenAI-compatible or native clients
        for ollama_provider in self.alima_config.llm.get_enabled_ollama_providers():
            provider_key = ollama_provider.name  # Use the actual provider name directly - Claude Generated
            
            if ollama_provider.connection_type == "openai_compatible":
                # Add as OpenAI-compatible provider
                self.openai_providers[provider_key] = {
                    "name": f"Ollama {ollama_provider.name}",
                    "module": "openai",
                    "class": "OpenAI", 
                    "base_url": ollama_provider.base_url,
                    "api_key": ollama_provider.api_key,
                    "description": f"Ollama {ollama_provider.name} ({ollama_provider.host}:{ollama_provider.port})",
                    "initializer": self._init_openai_compatible,
                    "generator": self._generate_openai_compatible,
                }
                self.logger.info(f"Configured Ollama OpenAI-compatible: {ollama_provider.name} -> {ollama_provider.base_url} (API Key: {'Set' if ollama_provider.api_key else 'Not Set'})")
                
            elif ollama_provider.connection_type == "native_client":
                # Add as native Ollama client
                self.static_providers[provider_key] = {
                    "name": f"Ollama Native {ollama_provider.name}",
                    "module": "ollama",
                    "class": "Client",
                    "base_url": ollama_provider.base_url,
                    "api_key": ollama_provider.api_key,
                    "description": f"Ollama Native {ollama_provider.name} ({ollama_provider.host}:{ollama_provider.port})",
                    "initializer": self._init_ollama_native_provider,
                    "generator": self._generate_ollama_native,
                }
                self.logger.info(f"Configured Ollama Native: {ollama_provider.name} -> {ollama_provider.base_url} (API Key: {'Set' if ollama_provider.api_key else 'Not Set'})")
        
        # Keep legacy Ollama support for backward compatibility
        primary_ollama = self.alima_config.llm.get_primary_ollama_provider()
        if primary_ollama and primary_ollama.connection_type == "openai_compatible":
            self.ollama_url = primary_ollama.base_url.replace('/v1', '')  # Remove /v1 for legacy compatibility
            self.ollama_port = primary_ollama.port
        
        # Add dynamic Ollama providers to supported providers - Claude Generated
        ollama_providers = {}
        for ollama_provider in self.alima_config.llm.get_enabled_ollama_providers():
            provider_key = ollama_provider.name  # Use the actual provider name directly - Claude Generated
            
            if ollama_provider.connection_type == "native_client":
                ollama_providers[provider_key] = {
                    "name": f"Ollama Native {ollama_provider.name}",
                    "type": "ollama_native_provider",
                    "params": {},
                    "base_url": ollama_provider.base_url,
                    "api_key": ollama_provider.api_key,
                    "description": f"Ollama Native {ollama_provider.name} ({ollama_provider.host}:{ollama_provider.port})",
                    "initializer": self._init_ollama_native_provider,
                    "generator": self._generate_ollama_native,
                }
        
        # Merge static and dynamic providers for unified access
        self.supported_providers = {**self.static_providers, **self.openai_providers, **ollama_providers}

    def reload_providers(self):
        """Reload providers from current configuration - Claude Generated"""
        self.logger.info("Reloading providers from configuration")
        
        # Reload configuration
        self.alima_config = self.config_manager.load_config()
        
        # Reinitialize dynamic provider configurations
        self._init_dynamic_provider_configs()
        
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
                self.alima_config.llm.gemini = api_key
            elif provider == "anthropic":
                self.alima_config.llm.anthropic = api_key
            
            # Save configuration
            self.config_manager.save_config(self.alima_config)
            
            # Reinitialize provider with new key
            self._initialize_single_provider(provider)
            
        elif provider in self.openai_providers:
            # Update OpenAI-compatible provider API key
            provider_obj = self.alima_config.llm.get_provider_by_name(provider)
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
        # Find provider info with case-insensitive fallback - Claude Generated
        provider_info = None
        if provider in self.supported_providers:
            provider_info = self.supported_providers[provider]
        elif provider.lower() in self.supported_providers:
            provider_info = self.supported_providers[provider.lower()]
        else:
            self.logger.warning(f"Unsupported provider: {provider}")
            return

        try:
            # Handle different provider types - Claude Generated
            provider_type = provider_info.get("type", "")
            
            if provider_type == "ollama_native_provider":
                # Special handling for native Ollama providers - don't import module
                module = None
                api_key = provider_info.get("api_key")
            else:
                # Try to import the required module for other providers
                module = importlib.import_module(provider_info["module"])
                
                # Handle API key based on provider type
                api_key = None
                if provider in self.static_providers:
                    # Static providers: get API key from configuration
                    if provider == "gemini":
                        api_key = self.alima_config.llm.gemini
                    elif provider == "anthropic":
                        api_key = self.alima_config.llm.anthropic
                    # Ollama (local) doesn't need API key
                    elif provider == "ollama":
                        api_key = None
                    # Ollama native client uses API key from config
                    elif provider == "ollama_native":
                        api_key = self.alima_config.llm.ollama.native_api_key
                else:
                    # Dynamic OpenAI-compatible providers: get API key from provider config
                    api_key = provider_info.get("api_key")
            
            # Check if API key is provided - warn if missing but continue for some providers
            if not api_key:
                if provider in ["ollama"]:
                    # Local Ollama doesn't require API key
                    pass
                elif provider == "ollama_native" and not self.alima_config.llm.ollama.native_api_key:
                    # Native Ollama without API key - might be local setup
                    self.logger.info(f"No API key configured for {provider} - assuming local setup")
                elif provider in self.static_providers and provider in ["gemini", "anthropic"]:
                    # Static providers like Gemini and Anthropic require API keys
                    self.logger.warning(f"No API key found for {provider} - initialization skipped")
                    return
                else:
                    # OpenAI-compatible providers: warn but continue (might not need API key)
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
                if self.alima_config.llm.get_enabled_ollama_providers():
                    filtered_providers.append(provider)
                    continue
                else:
                    self.logger.info("No enabled Ollama providers, skipping legacy ollama registration")
                    continue
            filtered_providers.append(provider)

        # Register providers in clients dict but don't initialize them
        for provider in filtered_providers:
            if provider.lower() not in self.supported_providers:
                self.logger.warning(f"Unsupported provider: {provider}")
                continue
            
            # Mark provider as registered but not initialized
            self.clients[provider.lower()] = "lazy_uninitialized"
            self.logger.debug(f"Registered provider for lazy initialization: {provider.lower()}")

    def _ensure_provider_initialized(self, provider: str) -> bool:
        """
        Ensure a provider is actually initialized before use (lazy loading) - Claude Generated
        
        Args:
            provider: Provider name to initialize
            
        Returns:
            True if provider is initialized, False if initialization failed
        """
        if provider not in self.clients:
            return False
            
        # If provider is already initialized (not a string), return True
        if self.clients[provider] != "lazy_uninitialized":
            return True
        
        # Initialize the provider now
        self.logger.info(f"Lazy-initializing provider: {provider}")
        try:
            self._initialize_single_provider(provider)
            return provider in self.clients and self.clients[provider] != "lazy_uninitialized"
        except Exception as e:
            self.logger.error(f"Failed to lazy-initialize provider {provider}: {e}")
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
                if self.alima_config.llm.get_enabled_ollama_providers():
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
        Get list of all available providers (both initialized and lazy-loadable) - Claude Generated

        Returns:
            List of provider names.
        """
        # Get all configured providers, not just initialized ones
        return list(self.supported_providers.keys())

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
        for ollama_provider in self.alima_config.llm.get_enabled_ollama_providers():
            provider_key = ollama_provider.name  # Use the actual provider name directly - Claude Generated
            if provider_key in available:
                return provider_key
        
        return None

    def get_available_models(self, provider: str) -> List[str]:
        """
        Get available models for specified provider with reachability check and lazy initialization - Claude Generated

        Args:
            provider: The provider name.

        Returns:
            List of model names.
        """
        # First check if provider is reachable - Claude Generated
        if not self.is_provider_reachable(provider):
            self.logger.warning(f"Provider {provider} is not reachable. Skipping model loading.")
            return []
            
        # Ensure provider is initialized if using lazy loading
        if not self._ensure_provider_initialized(provider):
            self.logger.warning(f"Provider {provider} could not be initialized.")
            return []

        try:
            if provider == "gemini":
                # Ensure model.name is handled correctly, it's usually "models/model-name"
                return [
                    model.name.split("/")[-1]
                    for model in self.clients[provider].list_models()
                    if hasattr(model, "name")  # Ensure 'name' attribute exists
                ]

            elif provider == "ollama":
                if not self._is_server_reachable(self.ollama_url, self.ollama_port):
                    self.logger.warning(
                        f"Ollama server not accessible at {self.ollama_url}:{self.ollama_port}"
                    )
                    return []
                response = self.clients[provider].get(
                    f"{self.ollama_url}:{self.ollama_port}/api/tags"
                )
                response.raise_for_status()  # Raise an exception for HTTP errors
                return [model["name"] for model in response.json()["models"]]

            elif provider == "ollama_native":
                # Use native Ollama client to list models
                try:
                    models_response = self.clients[provider].list()
                    if 'models' in models_response:
                        return [model['name'] for model in models_response['models']]
                    else:
                        return []
                except Exception as e:
                    self.logger.error(f"Error getting native Ollama models: {e}")
                    return []

            elif provider in self.openai_providers:
                # Use a more robust way to get model IDs, handling potential missing 'id'
                models = []
                for model_obj in self.clients[provider].models.list():
                    if hasattr(model_obj, "id"):
                        models.append(model_obj.id)
                    else:
                        self.logger.warning(
                            f"Model object from {provider} has no 'id' attribute: {model_obj}"
                        )
                return models

            elif provider == "anthropic":
                models = []
                for model_obj in self.clients[provider].models.list():
                    if hasattr(model_obj, "id"):
                        models.append(model_obj.id)
                    else:
                        self.logger.warning(
                            f"Model object from {provider} has no 'id' attribute: {model_obj}"
                        )
                return models

            elif provider == "azure":
                return self.supported_providers[provider]["params"]["supported_models"]

            elif provider == "github":
                return self.supported_providers[provider]["params"]["supported_models"]

            # Check for flexible Ollama providers (new multi-instance system) - Claude Generated
            config = self.config_manager.load_config()
            
            # Check if it's a configured Ollama provider - Claude Generated
            for ollama_provider in config.llm.ollama_providers:
                if ollama_provider.name == provider:
                    if ollama_provider.connection_type == "native_client":
                        # Use native Ollama client
                        try:
                            if provider not in self.clients:
                                self.logger.warning(f"Native Ollama client {provider} not initialized")
                                return []
                                
                            client = self.clients[provider]
                            models_response = client.list()
                            
                            # Handle both dict format and ollama.ListResponse object format - Claude Generated
                            if isinstance(models_response, dict) and 'models' in models_response:
                                return [model['name'] for model in models_response['models']]
                            elif hasattr(models_response, 'models'):
                                # Handle ollama.ListResponse object format
                                models = []
                                for model in models_response.models:
                                    if hasattr(model, 'model'):
                                        models.append(model.model)  # Use .model attribute for model name
                                    elif hasattr(model, 'name'):
                                        models.append(model.name)   # Fallback to .name attribute
                                    else:
                                        models.append(str(model))   # Fallback to string representation
                                return models
                            else:
                                self.logger.warning(f"Unexpected response format from {provider}: {type(models_response)}")
                                self.logger.debug(f"Response content: {models_response}")
                                return []
                                
                        except Exception as e:
                            self.logger.error(f"Error getting native Ollama models from {provider}: {e}")
                            self.logger.debug(traceback.format_exc())
                            return []
                    
                    elif ollama_provider.connection_type == "openai_compatible":
                        # Use OpenAI-compatible client
                        try:
                            if provider not in self.clients:
                                self.logger.warning(f"OpenAI-compatible client {provider} not initialized")
                                return []
                            
                            models = []
                            for model_obj in self.clients[provider].models.list():
                                if hasattr(model_obj, "id"):
                                    models.append(model_obj.id)
                            return models
                            
                        except Exception as e:
                            self.logger.error(f"Error getting OpenAI-compatible models from {provider}: {e}")
                            self.logger.debug(traceback.format_exc())
                            return []
                    
                    break
            
            # Check for OpenAI-compatible providers
            for openai_provider in config.llm.openai_compatible_providers:
                if openai_provider.name == provider:
                    try:
                        if provider not in self.clients:
                            self.logger.warning(f"OpenAI-compatible client {provider} not initialized")
                            return []
                        
                        models = []
                        for model_obj in self.clients[provider].models.list():
                            if hasattr(model_obj, "id"):
                                models.append(model_obj.id)
                        return models
                        
                    except Exception as e:
                        self.logger.error(f"Error getting OpenAI-compatible models from {provider}: {e}")
                        self.logger.debug(traceback.format_exc())
                        return []

        except Exception as e:
            self.logger.error(f"Error getting models for {provider}: {str(e)}")
            self.logger.debug(traceback.format_exc())

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

            # Generate based on provider with dynamic support - Claude Generated
            if provider == "gemini":
                response = self._generate_gemini(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider == "anthropic":
                response = self._generate_anthropic(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider == "ollama":
                response = self._generate_ollama(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider == "ollama_native":
                response = self._generate_ollama_native(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider == "github":
                response = self._generate_github(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider == "azure":
                response = self._generate_azure_inference(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            elif provider in self.openai_providers:
                # Dynamic OpenAI-compatible provider support
                response = self._generate_openai_compatible(
                    model, prompt, temperature, p_value, seed, image, system, stream
                )
            else:
                error_msg = f"Generation not implemented for provider: {provider}"
                self.logger.error(error_msg)
                self.generation_error.emit(self.current_request_id, error_msg)
                self.stream_running = False
                raise ValueError(error_msg)

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
        # Use provided key or placeholder for providers that don't need authentication
        if api_key:
            params["api_key"] = api_key
        else:
            params["api_key"] = "no-key-required"  # Placeholder for providers without authentication
            self.logger.info(f"Initializing {provider} with placeholder API key (no authentication required)")

        # Add base_url if specified
        if "base_url" in provider_info:
            params["base_url"] = provider_info["base_url"]
        elif provider_info.get("params", {}).get("base_url"):
            params["base_url"] = provider_info["params"]["base_url"]

        # Create client
        client_class = getattr(module, provider_info["class"])
        self.clients[provider] = client_class(**params)

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
            # Get base_url from provider info
            base_url = provider_info.get("base_url", "http://localhost:11434")
            
            # Use provider name directly (no ollama_ prefix anymore) - Claude Generated
            provider_name = provider
            
            # Find matching Ollama provider in config
            ollama_provider = self.alima_config.llm.get_ollama_provider_by_name(provider_name)
            if not ollama_provider:
                self.logger.warning(f"No Ollama provider configuration found for: {provider_name}")
                return
                
            if not ollama_provider.enabled:
                self.logger.info(f"Ollama provider {provider_name} disabled in configuration")
                return
            
            # Set up client parameters
            client_params = {"host": base_url}
            
            # Add authorization header if API key is provided
            if ollama_provider.api_key:
                client_params["headers"] = {
                    'Authorization': ollama_provider.api_key
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
                    model_name = f"gemini-pro"  # Standardmodell

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
            if hasattr(self.config_manager.config.llm, 'ollama_providers'):
                for provider in self.config_manager.config.llm.ollama_providers:
                    if provider.name == provider_name:
                        provider_config = provider
                        break
            
            # Check if it's an OpenAI-compatible provider
            if not provider_config and hasattr(self.config_manager.config.llm, 'openai_compatible_providers'):
                for provider in self.config_manager.config.llm.openai_compatible_providers:
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
            
            # Check if it's an Ollama provider - Claude Generated
            if hasattr(config.llm, 'ollama_providers'):
                for provider in config.llm.ollama_providers:
                    if provider.name == provider_name:
                        provider_config = provider
                        break
            
            # Check if it's an OpenAI-compatible provider
            if not provider_config and hasattr(config.llm, 'openai_compatible_providers'):
                for provider in config.llm.openai_compatible_providers:
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
        api_services = ['gemini', 'anthropic', 'openai', 'comet', 'chatai']
        if provider_name.lower() in api_services:
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
        if hasattr(config.llm, 'ollama_providers'):
            for provider in config.llm.ollama_providers:
                if provider.enabled:
                    results[provider.name] = self.is_provider_reachable(provider.name, force_check=True)
        
        # Check OpenAI-compatible providers
        if hasattr(config.llm, 'openai_compatible_providers'):
            for provider in config.llm.openai_compatible_providers:
                if provider.enabled:
                    results[provider.name] = self.is_provider_reachable(provider.name, force_check=True)
        
        self.logger.info(f"Provider status refresh completed: {results}")
        return results
    
    def _refresh_provider_status(self):
        """
        Timer callback to refresh provider status periodically - Claude Generated
        """
        self.refresh_all_provider_status()
