#!/usr/bin/env python3
"""
ALIMA Configuration Manager
Unified configuration management with centralized data models.
Claude Generated - Refactored for unified provider configuration
"""

import json
import os
import sys
import platform
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict, field
import logging

# Import centralized data models
from .config_models import (
    AlimaConfig, DatabaseConfig, CatalogConfig, PromptConfig,
    UnifiedProviderConfig, UnifiedProvider, TaskPreference, PipelineStepConfig,
    OllamaProvider, OpenAICompatibleProvider, GeminiProvider, AnthropicProvider,
    TaskType, PipelineMode
)

# Re-export all config_models classes for backward compatibility
# This allows "from config_manager import AlimaConfig" to still work
from .config_models import *  # Re-export everything

# ============================================================================
# TEMPORARY BRIDGE CLASSES - For import compatibility during migration
# ============================================================================

class ProviderPreferences:
    """
    TEMPORARY BRIDGE CLASS - ProviderPreferences
    This bridges the old ProviderPreferences interface to the new unified_config system.
    Will be removed once all code is updated to use unified_config directly.
    """
    def __init__(self, unified_config: UnifiedProviderConfig):
        self._unified_config = unified_config

    @property
    def preferred_provider(self) -> str:
        return self._unified_config.provider_priority[0] if self._unified_config.provider_priority else "ollama"

    @property
    def provider_priority(self) -> List[str]:
        return self._unified_config.provider_priority

    @property
    def disabled_providers(self) -> List[str]:
        return self._unified_config.disabled_providers

    def get_provider_for_task(self, task_type: str = "general") -> str:
        return self.preferred_provider

    def get_provider_priority_for_task(self, task_type: str = "general") -> List[str]:
        return [p for p in self.provider_priority if p not in self.disabled_providers]

    def validate_preferences(self, provider_detection_service) -> Dict[str, Any]:
        """Validate provider preferences - Claude Generated"""
        return {}  # No validation issues for bridge compatibility

    def auto_cleanup(self, provider_detection_service) -> Dict[str, Any]:
        """Auto-cleanup provider preferences - Claude Generated"""
        return {}  # No cleanup needed for bridge compatibility

    def ensure_valid_configuration(self, provider_detection_service):
        """Ensure valid configuration - Claude Generated"""
        pass  # Nothing to ensure for bridge compatibility

    @property
    def prefer_faster_models(self) -> bool:
        """Get preference for faster models - Claude Generated"""
        return self._unified_config.prefer_faster_models if hasattr(self._unified_config, 'prefer_faster_models') else False

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if provider is enabled - Claude Generated"""
        return provider not in self.disabled_providers


class LLMConfig:
    """
    COMPREHENSIVE BRIDGE CLASS - LLMConfig
    This bridges ALL old LLMConfig interface patterns to unified_config system.
    Supports 50+ access patterns from 22 files - Claude Generated
    """
    def __init__(self, unified_config: UnifiedProviderConfig):
        self._unified_config = unified_config

    # ============================================================================
    # API KEY PROPERTIES
    # ============================================================================

    @property
    def gemini(self) -> str:
        return self._unified_config.gemini_api_key

    @property
    def anthropic(self) -> str:
        return self._unified_config.anthropic_api_key

    # ============================================================================
    # PREFERRED MODEL PROPERTIES
    # ============================================================================

    @property
    def gemini_preferred_model(self) -> str:
        """Get preferred Gemini model from unified config providers"""
        provider = self._unified_config.get_provider_by_name("gemini")
        return provider.preferred_model if provider else ""

    @property
    def anthropic_preferred_model(self) -> str:
        """Get preferred Anthropic model from unified config providers"""
        provider = self._unified_config.get_provider_by_name("anthropic")
        return provider.preferred_model if provider else ""

    # ============================================================================
    # DEFAULT PROVIDER PROPERTY
    # ============================================================================

    @property
    def default_provider(self) -> str:
        """Get default provider from unified config priority list"""
        return self._unified_config.provider_priority[0] if self._unified_config.provider_priority else "ollama"

    # ============================================================================
    # PROVIDER COLLECTION PROPERTIES (Legacy compatibility - return empty lists)
    # ============================================================================

    @property
    def openai_compatible_providers(self) -> List:
        """Legacy OpenAI providers list - now handled through unified_config.providers"""
        return []  # Empty list for legacy compatibility

    @property
    def ollama_providers(self) -> List:
        """Legacy Ollama providers list - now handled through unified_config.providers"""
        return []  # Empty list for legacy compatibility

    # ============================================================================
    # PROVIDER LOOKUP METHODS
    # ============================================================================

    def get_provider_by_name(self, name: str) -> Optional[Any]:
        """Bridge to unified config provider lookup"""
        return self._unified_config.get_provider_by_name(name)

    def get_ollama_provider_by_name(self, name: str) -> Optional[Any]:
        """Legacy Ollama provider lookup - bridges to unified config"""
        provider = self._unified_config.get_provider_by_name(name)
        return provider if provider and provider.provider_type == "ollama" else None

    def get_primary_ollama_provider(self) -> Optional[Any]:
        """Legacy primary Ollama provider - returns first Ollama provider"""
        for provider in self._unified_config.providers:
            if provider.provider_type == "ollama" and provider.enabled:
                return provider
        return None

    # ============================================================================
    # ENABLED PROVIDER LIST METHODS
    # ============================================================================

    def get_enabled_providers(self) -> List[Any]:
        """Get all enabled providers from unified config"""
        return self._unified_config.get_enabled_providers()

    def get_enabled_openai_providers(self) -> List[Any]:
        """Legacy enabled OpenAI providers - filters from unified config"""
        return [p for p in self._unified_config.get_enabled_providers()
                if p.provider_type == "openai_compatible"]

    def get_enabled_ollama_providers(self) -> List[Any]:
        """Legacy enabled Ollama providers - filters from unified config"""
        return [p for p in self._unified_config.get_enabled_providers()
                if p.provider_type == "ollama"]

    # ============================================================================
    # PROVIDER MANAGEMENT METHODS (Legacy compatibility)
    # ============================================================================

    def add_provider(self, provider) -> bool:
        """Legacy add provider method - not implemented in unified config"""
        return False

    def remove_provider(self, name: str) -> bool:
        """Legacy remove provider method - not implemented in unified config"""
        return False

    def add_ollama_provider(self, provider) -> bool:
        """Legacy add Ollama provider method - not implemented"""
        return False

    def remove_ollama_provider(self, name: str) -> bool:
        """Legacy remove Ollama provider method - not implemented"""
        return False


class ProviderDetectionService:
    """
    Service for detecting available LLM providers using internal ALIMA logic - Claude Generated
    Wraps LlmService to provide clean API for provider detection, capabilities, and testing
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self._llm_service = None  # Lazy initialization

    def _get_llm_service(self):
        """Lazy initialize LLM service to avoid startup delays - Claude Generated"""
        if self._llm_service is None:
            try:
                # Import here to avoid circular imports
                from ..llm.llm_service import LlmService
                self._llm_service = LlmService(lazy_initialization=True)
            except Exception as e:
                self.logger.error(f"Failed to initialize LlmService: {e}")
                raise
        return self._llm_service

    def get_available_providers(self) -> List[str]:
        """Get list of all available providers from internal LlmService - Claude Generated"""
        try:
            llm_service = self._get_llm_service()
            return llm_service.get_available_providers()
        except Exception as e:
            self.logger.error(f"Error getting available providers: {e}")
            return []

    def is_provider_reachable(self, provider: str) -> bool:
        """Test if a provider is currently reachable - Claude Generated"""
        try:
            llm_service = self._get_llm_service()
            return llm_service.is_provider_reachable(provider)
        except Exception as e:
            self.logger.warning(f"Error testing reachability for {provider}: {e}")
            return False

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a specific provider - Claude Generated"""
        try:
            llm_service = self._get_llm_service()
            models = llm_service.get_available_models(provider)
            return models if models else []
        except Exception as e:
            self.logger.warning(f"Error getting models for {provider}: {e}")
            return []

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get comprehensive information about a provider - Claude Generated"""
        info = {
            'name': provider,
            'available': provider in self.get_available_providers(),
            'reachable': False,
            'models': [],
            'model_count': 0,
            'status': 'unknown'
        }

        if not info['available']:
            info['status'] = 'not_configured'
            return info

        # Test reachability
        info['reachable'] = self.is_provider_reachable(provider)

        if info['reachable']:
            info['status'] = 'ready'
            info['models'] = self.get_available_models(provider)
            info['model_count'] = len(info['models'])
        else:
            info['status'] = 'unreachable'

        return info

    def detect_provider_capabilities(self, provider: str) -> List[str]:
        """Detect provider capabilities - Claude Generated"""
        capabilities = []

        try:
            # Basic capability detection based on provider type and configuration
            if provider == "gemini":
                capabilities.extend(["vision", "large_context", "reasoning"])
            elif provider == "anthropic":
                capabilities.extend(["large_context", "reasoning", "analysis"])
            elif "ollama" in provider.lower() or provider in ["localhost", "llmachine/ollama"]:
                capabilities.extend(["local", "privacy"])

                # Check available models for additional capabilities
                models = self.get_available_models(provider)
                if any("vision" in model.lower() or "llava" in model.lower() for model in models):
                    capabilities.append("vision")
                if any("fast" in model.lower() or "flash" in model.lower() for model in models):
                    capabilities.append("fast")
            else:
                # OpenAI-compatible or other providers
                capabilities.extend(["api_compatible"])

                # Check for fast models
                models = self.get_available_models(provider)
                if any("fast" in model.lower() or "turbo" in model.lower() for model in models):
                    capabilities.append("fast")

        except Exception as e:
            self.logger.warning(f"Error detecting capabilities for {provider}: {e}")

        return capabilities


class ConfigManager:
    """Unified ALIMA configuration manager with centralized provider management - Claude Generated"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Get OS-specific configuration paths
        self._setup_config_paths()

        self._config: Optional[AlimaConfig] = None
        self._provider_detection_service: Optional[ProviderDetectionService] = None

    def _setup_config_paths(self):
        """Setup OS-specific configuration file paths - Claude Generated"""
        system_name = platform.system().lower()

        if system_name == "windows":
            config_base = Path(os.environ.get("APPDATA", "")) / "ALIMA"
        elif system_name == "darwin":  # macOS
            config_base = Path("~/Library/Application Support/ALIMA").expanduser()
        else:  # Linux and others
            config_base = Path("~/.config/alima").expanduser()

        # Ensure directory exists
        config_base.mkdir(parents=True, exist_ok=True)

        # Primary config file path
        self.config_file = config_base / "config.json"
        self.legacy_config_file = config_base / "config.yaml"

        self.logger.info(f"Config path: {self.config_file}")

    def load_config(self, force_reload: bool = False) -> AlimaConfig:
        """Load configuration with unified provider system - Claude Generated"""
        if self._config is None or force_reload:
            self._config = self._load_config_from_file()
        return self._config

    def _load_config_from_file(self) -> AlimaConfig:
        """Load configuration from JSON file - Claude Generated"""
        config_data = {}

        # Try to load existing config
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading config from {self.config_file}: {e}")

        # Parse configuration
        return self._parse_config(config_data)

    def _parse_config(self, config_data: Dict[str, Any]) -> AlimaConfig:
        """Parse configuration data with legacy migration support - Claude Generated"""
        try:
            # DETECT: Is this a legacy configuration?
            is_legacy_config = self._is_legacy_config(config_data)

            if is_legacy_config:
                self.logger.info("🔄 Legacy configuration detected - performing migration")
                config_data = self._migrate_legacy_config(config_data)

            # Create main config sections
            database_config = DatabaseConfig(**config_data.get("database_config", config_data.get("database", {})))
            catalog_config = CatalogConfig(**config_data.get("catalog_config", config_data.get("catalog", {})))
            prompt_config = PromptConfig(**config_data.get("prompt_config", config_data.get("prompt", {})))
            system_config = SystemConfig(**config_data.get("system_config", config_data.get("system", {})))

            # Create unified provider config
            if "unified_config" in config_data:
                # Already in unified format
                unified_config_data = config_data["unified_config"]
                unified_config = self._parse_unified_config(unified_config_data)
            else:
                # Parse modern configuration format
                self.logger.info("📋 Modern configuration detected - parsing provider data")
                unified_config = self._parse_modern_config(config_data)

            # Create main config
            config = AlimaConfig(
                database_config=database_config,
                catalog_config=catalog_config,
                prompt_config=prompt_config,
                system_config=system_config,
                unified_config=unified_config,
                config_version=config_data.get("config_version", "2.0")
            )

            return config

        except Exception as e:
            self.logger.error(f"Error parsing configuration: {e}")
            self.logger.error(f"Config data keys: {list(config_data.keys()) if config_data else 'empty'}")
            # Return default configuration
            return AlimaConfig()

    def _is_legacy_config(self, config_data: Dict[str, Any]) -> bool:
        """Detect if this is a legacy configuration format - Claude Generated"""
        if not config_data:
            return False

        # Modern format indicators (current standard)
        modern_keys = ['provider_preferences', 'task_preferences', 'llm']
        has_modern = any(key in config_data for key in modern_keys)

        # Unified format indicators (our new target format)
        unified_keys = ['unified_config']
        has_unified = any(key in config_data for key in unified_keys)

        # If it has unified_config, it's already in new format
        if has_unified:
            return False

        # If it has modern keys, it's NOT legacy - it's current standard
        if has_modern:
            return False

        # Only truly old formats (YAML, old JSON schemas) are legacy
        return True

    def _migrate_legacy_config(self, legacy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy configuration to new format - Claude Generated"""
        migrated = {}

        # Create backup
        backup_file = self.config_file.parent / f"alima_config_backup_{int(time.time())}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(legacy_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📁 Backup created: {backup_file}")
        except Exception as e:
            self.logger.warning(f"Could not create backup: {e}")

        # Migrate database config
        if 'database' in legacy_data:
            migrated['database_config'] = legacy_data['database']

        # Migrate catalog config
        if 'catalog' in legacy_data:
            migrated['catalog_config'] = legacy_data['catalog']

        # Migrate system config
        if 'system' in legacy_data:
            migrated['system_config'] = legacy_data['system']

        # Create unified_config from legacy provider/task preferences
        unified_config = {}

        # Migrate provider_preferences
        if 'provider_preferences' in legacy_data:
            prefs = legacy_data['provider_preferences']
            unified_config['provider_priority'] = prefs.get('provider_priority', ["ollama", "gemini", "anthropic", "openai"])
            unified_config['disabled_providers'] = prefs.get('disabled_providers', [])

        # Migrate task_preferences
        if 'task_preferences' in legacy_data:
            task_prefs = {}
            for task_name, task_data in legacy_data['task_preferences'].items():
                if isinstance(task_data, dict) and 'model_priority' in task_data:
                    task_prefs[task_name] = {
                        'task_type': task_data.get('task_type', 'general'),
                        'model_priority': task_data.get('model_priority', []),
                        'chunked_model_priority': task_data.get('chunked_model_priority'),
                        'allow_fallback': task_data.get('allow_fallback', True)
                    }
            unified_config['task_preferences'] = task_prefs

        # Migrate LLM config
        if 'llm' in legacy_data:
            llm_data = legacy_data['llm']
            unified_config['gemini_api_key'] = llm_data.get('gemini', '')
            unified_config['anthropic_api_key'] = llm_data.get('anthropic', '')

            # Migrate Ollama providers
            if 'ollama_providers' in llm_data:
                # Convert to unified providers - simplified for now
                pass

            # Migrate OpenAI compatible providers
            if 'openai_compatible_providers' in llm_data:
                # Convert to unified providers - simplified for now
                pass

        migrated['unified_config'] = unified_config
        migrated['config_version'] = '2.0'

        self.logger.info("✅ Legacy configuration migration completed")
        return migrated

    def _parse_unified_config(self, data: Dict[str, Any]) -> UnifiedProviderConfig:
        """Parse unified provider configuration - Claude Generated"""
        unified_config = UnifiedProviderConfig()

        # Parse global settings
        unified_config.provider_priority = data.get("provider_priority", ["ollama", "gemini", "anthropic", "openai"])
        unified_config.disabled_providers = data.get("disabled_providers", [])

        # Parse individual provider configs (legacy support)
        unified_config.gemini_api_key = data.get("gemini_api_key", "")
        unified_config.anthropic_api_key = data.get("anthropic_api_key", "")

        # Auto-create UnifiedProvider objects from API keys - Claude Generated
        self._create_providers_from_api_keys(unified_config)

        # Parse task preferences
        task_prefs_data = data.get("task_preferences", {})
        for task_name, task_data in task_prefs_data.items():
            try:
                task_type = TaskType(task_data.get("task_type", "general"))
                unified_config.task_preferences[task_name] = TaskPreference(
                    task_type=task_type,
                    model_priority=task_data.get("model_priority", []),
                    chunked_model_priority=task_data.get("chunked_model_priority"),
                    allow_fallback=task_data.get("allow_fallback", True)
                )
            except Exception as e:
                self.logger.warning(f"Error parsing task preference '{task_name}': {e}")

        return unified_config

    def _parse_modern_config(self, config_data: Dict[str, Any]) -> UnifiedProviderConfig:
        """Parse modern configuration format - Claude Generated"""
        unified_config = UnifiedProviderConfig()

        # Parse provider preferences
        if 'provider_preferences' in config_data:
            prefs = config_data['provider_preferences']
            unified_config.provider_priority = prefs.get('provider_priority', ['ollama', 'gemini', 'anthropic', 'openai'])
            unified_config.disabled_providers = prefs.get('disabled_providers', [])
            unified_config.auto_fallback = prefs.get('auto_fallback', True)
            unified_config.prefer_faster_models = prefs.get('prefer_faster_models', False)

        # Parse LLM section and create providers
        if 'llm' in config_data:
            llm_data = config_data['llm']

            # Store legacy API keys
            unified_config.gemini_api_key = llm_data.get('gemini', '')
            unified_config.anthropic_api_key = llm_data.get('anthropic', '')

            # Parse OpenAI-compatible providers
            if 'openai_compatible_providers' in llm_data:
                for provider_data in llm_data['openai_compatible_providers']:
                    try:
                        # Create OpenAICompatibleProvider and convert to UnifiedProvider
                        from .config_models import OpenAICompatibleProvider, UnifiedProvider
                        provider = OpenAICompatibleProvider(**provider_data)
                        unified_provider = UnifiedProvider.from_openai_compatible_provider(provider)
                        unified_config.providers.append(unified_provider)
                    except Exception as e:
                        self.logger.warning(f"Error parsing OpenAI provider {provider_data.get('name', 'unknown')}: {e}")

            # Parse Ollama providers
            if 'ollama_providers' in llm_data:
                for provider_data in llm_data['ollama_providers']:
                    try:
                        # Create OllamaProvider and convert to UnifiedProvider
                        from .config_models import OllamaProvider, UnifiedProvider
                        provider = OllamaProvider(**provider_data)
                        unified_provider = UnifiedProvider.from_ollama_provider(provider)
                        unified_config.providers.append(unified_provider)
                    except Exception as e:
                        self.logger.warning(f"Error parsing Ollama provider {provider_data.get('name', 'unknown')}: {e}")

        # Auto-create providers from API keys
        self._create_providers_from_api_keys(unified_config)

        # Parse task preferences
        if 'task_preferences' in config_data:
            for task_name, task_data in config_data['task_preferences'].items():
                try:
                    from .config_models import TaskType, TaskPreference
                    # Try to map task_name to TaskType enum
                    try:
                        task_type = TaskType(task_data.get('task_type', task_name))
                    except ValueError:
                        # Fallback to GENERAL if task_type not recognized
                        task_type = TaskType.GENERAL
                        self.logger.warning(f"Unknown task type '{task_data.get('task_type', task_name)}', using GENERAL")

                    unified_config.task_preferences[task_name] = TaskPreference(
                        task_type=task_type,
                        model_priority=task_data.get('model_priority', []),
                        chunked_model_priority=task_data.get('chunked_model_priority'),
                        allow_fallback=task_data.get('allow_fallback', True)
                    )
                except Exception as e:
                    self.logger.warning(f"Error parsing task preference '{task_name}': {e}")

        self.logger.info(f"✅ Parsed modern config: {len(unified_config.providers)} providers, {len(unified_config.task_preferences)} task preferences")
        return unified_config

    def _create_providers_from_api_keys(self, unified_config: UnifiedProviderConfig) -> None:
        """Auto-create UnifiedProvider objects from API keys - Claude Generated"""
        # Create Gemini provider if API key exists
        if unified_config.gemini_api_key and not unified_config.get_provider_by_name("gemini"):
            from .config_models import GeminiProvider, UnifiedProvider
            gemini_provider = GeminiProvider(
                api_key=unified_config.gemini_api_key,
                enabled=True,
                description="Google Gemini API"
            )
            unified_provider = UnifiedProvider.from_gemini_provider(gemini_provider)
            unified_config.providers.append(unified_provider)
            self.logger.info("✅ Created Gemini UnifiedProvider from API key")

        # Create Anthropic provider if API key exists
        if unified_config.anthropic_api_key and not unified_config.get_provider_by_name("anthropic"):
            from .config_models import AnthropicProvider, UnifiedProvider
            anthropic_provider = AnthropicProvider(
                api_key=unified_config.anthropic_api_key,
                enabled=True,
                description="Anthropic Claude API"
            )
            unified_provider = UnifiedProvider.from_anthropic_provider(anthropic_provider)
            unified_config.providers.append(unified_provider)
            self.logger.info("✅ Created Anthropic UnifiedProvider from API key")

    def save_config(self, config: Optional[AlimaConfig] = None) -> bool:
        """Save configuration to JSON file - Claude Generated"""
        if config is None:
            config = self._config

        if config is None:
            self.logger.warning("No configuration to save")
            return False

        try:
            # Convert to dictionary
            config_dict = asdict(config)

            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def get_unified_config(self) -> UnifiedProviderConfig:
        """Get unified provider configuration - Claude Generated"""
        config = self.load_config()
        return config.unified_config

    def save_unified_config(self, unified_config: UnifiedProviderConfig) -> bool:
        """Save unified provider configuration - Claude Generated"""
        config = self.load_config()
        config.unified_config = unified_config
        return self.save_config(config)

    def get_provider_detection_service(self) -> ProviderDetectionService:
        """Get provider detection service instance - Claude Generated"""
        if self._provider_detection_service is None:
            self._provider_detection_service = ProviderDetectionService(self)
        return self._provider_detection_service

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration - Claude Generated"""
        return self.load_config().database_config

    def get_catalog_config(self) -> CatalogConfig:
        """Get catalog configuration - Claude Generated"""
        return self.load_config().catalog_config

    def get_prompt_config(self) -> PromptConfig:
        """Get prompt configuration - Claude Generated"""
        return self.load_config().prompt_config

    def update_database_config(self, database_config: DatabaseConfig) -> bool:
        """Update database configuration - Claude Generated"""
        config = self.load_config()
        config.database_config = database_config
        return self.save_config(config)

    def update_catalog_config(self, catalog_config: CatalogConfig) -> bool:
        """Update catalog configuration - Claude Generated"""
        config = self.load_config()
        config.catalog_config = catalog_config
        return self.save_config(config)

    # ============================================================================
    # UNIFIED PROVIDER CONFIG MANAGEMENT - Integrated from UnifiedProviderConfigManager
    # ============================================================================

    def validate_unified_config(self, unified_config: Optional[UnifiedProviderConfig] = None) -> List[str]:
        """Validate unified provider configuration - Claude Generated"""
        if unified_config is None:
            unified_config = self.get_unified_config()

        issues = []

        # Check for enabled providers
        enabled_providers = unified_config.get_enabled_providers()
        if not enabled_providers:
            issues.append("No enabled providers configured")

        # Check task preferences
        for task_key, task_pref in unified_config.task_preferences.items():
            for provider_info in task_pref.model_priority:
                if isinstance(provider_info, dict):
                    provider_name = provider_info.get("provider_name")
                    if provider_name and not unified_config.get_provider_by_name(provider_name):
                        issues.append(f"Task '{task_key}' references unavailable provider '{provider_name}'")

        return issues

    def add_ollama_provider(self, provider: OllamaProvider) -> bool:
        """Add Ollama provider to unified config - Claude Generated"""
        unified_config = self.get_unified_config()

        # Convert to UnifiedProvider and add
        unified_provider = UnifiedProvider.from_ollama_provider(provider)

        # Check if provider already exists
        if unified_config.get_provider_by_name(provider.name):
            return False

        unified_config.providers.append(unified_provider)
        return self.save_unified_config(unified_config)

    def add_openai_compatible_provider(self, provider: OpenAICompatibleProvider) -> bool:
        """Add OpenAI-compatible provider to unified config - Claude Generated"""
        unified_config = self.get_unified_config()

        # Convert to UnifiedProvider and add
        unified_provider = UnifiedProvider.from_openai_compatible_provider(provider)

        # Check if provider already exists
        if unified_config.get_provider_by_name(provider.name):
            return False

        unified_config.providers.append(unified_provider)
        return self.save_unified_config(unified_config)

    def update_gemini_provider(self, api_key: str, enabled: bool = True, preferred_model: str = "") -> bool:
        """Update Gemini provider configuration - Claude Generated"""
        unified_config = self.get_unified_config()

        # Find existing Gemini provider or create new one
        gemini_provider = unified_config.get_provider_by_name("gemini")
        if gemini_provider:
            gemini_provider.api_key = api_key
            gemini_provider.enabled = enabled
            gemini_provider.preferred_model = preferred_model
        else:
            # Create new Gemini provider
            gemini_unified = UnifiedProvider.from_gemini_provider(GeminiProvider(
                api_key=api_key,
                enabled=enabled,
                preferred_model=preferred_model
            ))
            unified_config.providers.append(gemini_unified)

        return self.save_unified_config(unified_config)

    def update_anthropic_provider(self, api_key: str, enabled: bool = True, preferred_model: str = "") -> bool:
        """Update Anthropic provider configuration - Claude Generated"""
        unified_config = self.get_unified_config()

        # Find existing Anthropic provider or create new one
        anthropic_provider = unified_config.get_provider_by_name("anthropic")
        if anthropic_provider:
            anthropic_provider.api_key = api_key
            anthropic_provider.enabled = enabled
            anthropic_provider.preferred_model = preferred_model
        else:
            # Create new Anthropic provider
            anthropic_unified = UnifiedProvider.from_anthropic_provider(AnthropicProvider(
                api_key=api_key,
                enabled=enabled,
                preferred_model=preferred_model
            ))
            unified_config.providers.append(anthropic_unified)

        return self.save_unified_config(unified_config)

    def remove_provider(self, provider_name: str) -> bool:
        """Remove provider from unified config - Claude Generated"""
        unified_config = self.get_unified_config()

        # Find and remove provider
        for i, provider in enumerate(unified_config.providers):
            if provider.name.lower() == provider_name.lower():
                del unified_config.providers[i]
                return self.save_unified_config(unified_config)

        return False

    def get_enabled_providers(self) -> List[UnifiedProvider]:
        """Get list of enabled providers - Claude Generated"""
        unified_config = self.get_unified_config()
        return unified_config.get_enabled_providers()

    def update_task_preference(self, task_name: str, task_preference: TaskPreference) -> bool:
        """Update task preference in unified config - Claude Generated"""
        unified_config = self.get_unified_config()
        unified_config.task_preferences[task_name] = task_preference
        return self.save_unified_config(unified_config)

    def get_task_preference(self, task_name: str) -> TaskPreference:
        """Get task preference from unified config - Claude Generated"""
        unified_config = self.get_unified_config()
        if task_name in unified_config.task_preferences:
            return unified_config.task_preferences[task_name]

        # Return default preference
        return TaskPreference(
            task_type=TaskType.GENERAL,
            model_priority=[],
            allow_fallback=True
        )

    # ============================================================================
    # BRIDGE METHODS - For import compatibility during migration
    # ============================================================================

    def get_provider_preferences(self) -> ProviderPreferences:
        """TEMPORARY BRIDGE METHOD - Return ProviderPreferences wrapper"""
        unified_config = self.get_unified_config()
        return ProviderPreferences(unified_config)

    def get_llm_config(self) -> LLMConfig:
        """TEMPORARY BRIDGE METHOD - Return LLMConfig wrapper"""
        unified_config = self.get_unified_config()
        return LLMConfig(unified_config)

    def get_config_info(self) -> dict:
        """Return configuration paths information - Claude Generated"""
        import platform
        return {
            'os': platform.system(),
            'project_config': str(self.config_file),
            'user_config': str(self.config_file),
            'system_config': 'Not used'
        }


# ============================================================================
# GLOBAL FUNCTIONS - For import compatibility
# ============================================================================

def get_config_manager() -> ConfigManager:
    """Get global ConfigManager instance - Claude Generated"""
    return ConfigManager()