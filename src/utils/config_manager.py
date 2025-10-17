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
from enum import Enum
# Import centralized data models
from .config_models import (
    AlimaConfig, DatabaseConfig, CatalogConfig, PromptConfig, UIConfig,
    UnifiedProviderConfig, UnifiedProvider, TaskPreference, PipelineStepConfig,
    OllamaProvider, OpenAICompatibleProvider, GeminiProvider, AnthropicProvider,
    TaskType, PipelineMode
)
# Import path resolution utilities - Claude Generated
from .path_utils import resolve_path

# Re-export all config_models classes for backward compatibility
# This allows "from config_manager import AlimaConfig" to still work
from .config_models import *  # Re-export everything

# ============================================================================
# TEMPORARY BRIDGE CLASSES - For import compatibility during migration
# ============================================================================



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


class AlimaConfigEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle specific types like Enum."""
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)

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

        self.logger.debug(f"Config path: {self.config_file}")

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
                self.logger.debug(f"Loaded configuration from {self.config_file}")
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
                self.logger.debug("🔄 Legacy configuration detected - performing migration")
                config_data = self._migrate_legacy_config(config_data)

            # Create main config sections
            database_config = DatabaseConfig(**config_data.get("database_config", config_data.get("database", {})))
            catalog_config = CatalogConfig(**config_data.get("catalog_config", config_data.get("catalog", {})))
            prompt_config = PromptConfig(**config_data.get("prompt_config", config_data.get("prompt", {})))
            system_config = SystemConfig(**config_data.get("system_config", config_data.get("system", {})))

            # Resolve file paths from system_config - Claude Generated
            try:
                # Resolve database path (supports absolute, relative, and ~)
                system_config.database_path = resolve_path(system_config.database_path)

                # Resolve prompts path
                system_config.prompts_path = resolve_path(system_config.prompts_path)

                # Update database_config with resolved path (for SQLite only)
                if database_config.db_type.lower() in ['sqlite', 'sqlite3']:
                    database_config.sqlite_path = system_config.database_path
                    self.logger.debug(f"✅ Resolved database path: {database_config.sqlite_path}")

                self.logger.debug(f"✅ Resolved prompts path: {system_config.prompts_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Path resolution error: {e}. Using default paths.")

            # Parse UI config - Claude Generated (Webcam Feature Fix)
            ui_config_data = config_data.get("ui_config", {})
            ui_config = UIConfig(**ui_config_data) if ui_config_data else UIConfig()

            # Create unified provider config
            if "unified_config" in config_data:
                # Already in unified format
                unified_config_data = config_data["unified_config"]
                unified_config = self._parse_unified_config(unified_config_data)
            else:
                # Parse modern configuration format
                self.logger.debug("📋 Modern configuration detected - parsing provider data")
                unified_config = self._parse_modern_config(config_data)

            # Create main config
            config = AlimaConfig(
                database_config=database_config,
                catalog_config=catalog_config,
                prompt_config=prompt_config,
                system_config=system_config,
                ui_config=ui_config,  # Claude Generated (Webcam Feature Fix)
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
            self.logger.debug(f"📁 Backup created: {backup_file}")
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

            # Initialize providers list
            if 'providers' not in unified_config:
                unified_config['providers'] = []

            # Migrate Ollama providers
            if 'ollama_providers' in llm_data:
                for provider_data in llm_data['ollama_providers']:
                    try:
                        from .config_models import OllamaProvider, UnifiedProvider
                        provider = OllamaProvider(**provider_data)
                        unified_provider = UnifiedProvider.from_ollama_provider(provider)
                        unified_config['providers'].append(asdict(unified_provider))
                        self.logger.debug(f"✅ Migrated Ollama provider: {provider.name}")
                    except Exception as e:
                        self.logger.warning(f"Error migrating Ollama provider {provider_data.get('name', 'unknown')}: {e}")

            # Migrate OpenAI compatible providers
            if 'openai_compatible_providers' in llm_data:
                for provider_data in llm_data['openai_compatible_providers']:
                    try:
                        from .config_models import OpenAICompatibleProvider, UnifiedProvider
                        provider = OpenAICompatibleProvider(**provider_data)
                        unified_provider = UnifiedProvider.from_openai_compatible_provider(provider)
                        unified_config['providers'].append(asdict(unified_provider))
                        self.logger.debug(f"✅ Migrated OpenAI provider: {provider.name}")
                    except Exception as e:
                        self.logger.warning(f"Error migrating OpenAI provider {provider_data.get('name', 'unknown')}: {e}")

            # Create built-in providers from API keys
            if unified_config['gemini_api_key']:
                try:
                    from .config_models import GeminiProvider, UnifiedProvider
                    gemini_provider = GeminiProvider(
                        api_key=unified_config['gemini_api_key'],
                        enabled=True,
                        description="Google Gemini API"
                    )
                    unified_provider = UnifiedProvider.from_gemini_provider(gemini_provider)
                    unified_config['providers'].append(asdict(unified_provider))
                    self.logger.debug("✅ Created Gemini provider from API key")
                except Exception as e:
                    self.logger.warning(f"Error creating Gemini provider: {e}")

            if unified_config['anthropic_api_key']:
                try:
                    from .config_models import AnthropicProvider, UnifiedProvider
                    anthropic_provider = AnthropicProvider(
                        api_key=unified_config['anthropic_api_key'],
                        enabled=True,
                        description="Anthropic Claude API"
                    )
                    unified_provider = UnifiedProvider.from_anthropic_provider(anthropic_provider)
                    unified_config['providers'].append(asdict(unified_provider))
                    self.logger.debug("✅ Created Anthropic provider from API key")
                except Exception as e:
                    self.logger.warning(f"Error creating Anthropic provider: {e}")

        migrated['unified_config'] = unified_config
        migrated['config_version'] = '2.0'

        self.logger.debug("✅ Legacy configuration migration completed")
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

        # Parse saved providers from config - Claude Generated
        providers_data = data.get("providers", [])
        for provider_dict in providers_data:
            try:
                provider = UnifiedProvider(**provider_dict)
                unified_config.providers.append(provider)
                self.logger.debug(f"✅ Loaded provider: {provider.name}")
            except Exception as e:
                self.logger.warning(f"Error loading provider {provider_dict.get('name', 'unknown')}: {e}")

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

        self.logger.debug(f"✅ Parsed modern config: {len(unified_config.providers)} providers, {len(unified_config.task_preferences)} task preferences")
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
            self.logger.debug("✅ Created Gemini UnifiedProvider from API key")

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
            self.logger.debug("✅ Created Anthropic UnifiedProvider from API key")

    def save_config(self, config: AlimaConfig, scope: str = 'user', preserve_unified: bool = True) -> bool:
        """Save configuration to specified scope - Claude Generated"""
        try:
            # 🔍 DEBUG: Track save_config entry point
            self.logger.critical(f"🔍 SAVE_CONFIG_ENTRY: preserve_unified={preserve_unified}")
            self.logger.critical(f"🔍 SAVE_CONFIG_UNIFIED_PROVIDERS: {len(config.unified_config.providers)} providers in input config")
            for i, p in enumerate(config.unified_config.providers):
                self.logger.critical(f"🔍 SAVE_CONFIG_INPUT_PROVIDER_{i}: {p.name} ({p.provider_type})")

            #config_path = self._get_config_path(scope)
            #config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert AlimaConfig to dictionary for serialization
            config_dict = asdict(config)
            self.logger.critical(f"🔍 SAVE_CONFIG_POST_ASDICT: {len(config_dict.get('unified_config', {}).get('providers', []))} providers after asdict")

            # DEFAULT: Always preserve unified_config unless explicitly disabled
            if preserve_unified:
                try:
                    self.logger.critical(f"🔍 SAVE_CONFIG_PRESERVING: Reading current config from {self.config_file}")
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        current_config = json.load(f)

                    # 🔍 DEBUG: Log what keys are in the current config
                    self.logger.critical(f"🔍 SAVE_CONFIG_CURRENT_KEYS: {list(current_config.keys())}")

                    # Keep existing unified_config if it exists
                    if 'unified_config' in current_config:
                        preserved_providers = current_config['unified_config'].get('providers', [])
                        self.logger.critical(f"🔍 SAVE_CONFIG_DISK_PROVIDERS: Found {len(preserved_providers)} providers on disk")

                        # CRITICAL FIX: Use INCOMING providers from input config, not preserved from disk - Claude Generated
                        incoming_providers = config_dict.get('unified_config', {}).get('providers', [])
                        self.logger.critical(f"🔍 SAVE_CONFIG_INPUT_PROVIDERS: Found {len(incoming_providers)} providers in input config")
                        for i, p in enumerate(incoming_providers):
                            self.logger.critical(f"🔍 SAVE_CONFIG_INPUT_PROVIDER_{i}: {p.get('name', 'NO_NAME')} ({p.get('provider_type', 'NO_TYPE')})")

                        incoming_task_prefs = config_dict.get('unified_config', {}).get('task_preferences', {})
                        preserved_unified_config = current_config['unified_config'].copy()

                        # Update with NEW providers and task preferences from input config - Claude Generated
                        preserved_unified_config['providers'] = incoming_providers  # ✅ USE NEW PROVIDERS FROM INPUT
                        if incoming_task_prefs:
                            preserved_unified_config['task_preferences'] = incoming_task_prefs
                            self.logger.critical(f"✅ Updated unified_config: {len(incoming_providers)} providers (from input), {len(incoming_task_prefs)} task prefs")
                        else:
                            self.logger.critical(f"✅ Updated unified_config: {len(incoming_providers)} providers (from input), no task prefs")

                        config_dict['unified_config'] = preserved_unified_config
                    else:
                        self.logger.critical("🔍 SAVE_CONFIG_NO_UNIFIED: No unified_config found in current file")
                        # CRITICAL FIX: If the config being saved has providers but file doesn't have unified_config,
                        # this means we're migrating or the input config should be preserved
                        input_providers = config_dict.get('unified_config', {}).get('providers', [])
                        if input_providers:
                            self.logger.critical(f"🔍 SAVE_CONFIG_LEGACY_OVERRIDE: Input config has {len(input_providers)} providers - preserving them despite legacy file format")
                        else:
                            # Load the current unified config from memory to avoid losing providers
                            try:
                                current_unified = self.get_unified_config()
                                if current_unified.providers:
                                    self.logger.critical(f"🔍 SAVE_CONFIG_MEMORY_PRESERVATION: Using current unified config from memory with {len(current_unified.providers)} providers")
                                    config_dict['unified_config'] = asdict(current_unified)
                                else:
                                    self.logger.critical("🔍 SAVE_CONFIG_NO_PROVIDERS: No providers found in memory either")
                            except Exception as memory_e:
                                self.logger.critical(f"🔍 SAVE_CONFIG_MEMORY_ERROR: Could not load unified config from memory: {memory_e}")
                except Exception as e:
                    self.logger.critical(f"🔍 SAVE_CONFIG_PRESERVE_ERROR: Could not preserve unified_config: {e}")
            else:
                self.logger.critical("⚠️  SAVE_CONFIG_PRESERVATION_DISABLED - providers may be lost!")

            # Save with preserved unified_config
            final_providers = config_dict.get('unified_config', {}).get('providers', [])
            self.logger.critical(f"🔍 SAVE_CONFIG_FINAL_WRITE: Writing {len(final_providers)} providers to disk")

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, cls=AlimaConfigEncoder)

            # Verify what was actually written
            with open(self.config_file, 'r', encoding='utf-8') as f:
                verification_config = json.load(f)
            written_providers = verification_config.get('unified_config', {}).get('providers', [])
            self.logger.critical(f"🔍 SAVE_CONFIG_VERIFICATION: {len(written_providers)} providers actually written to disk")

            self.logger.info(f"Configuration saved to {self.config_file}")

            # Update internal cache with saved config - Claude Generated
            self._config = config
            self.logger.debug("✅ Internal config cache updated with saved configuration")

            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def get_unified_config(self) -> UnifiedProviderConfig:
        """Get unified provider configuration - Claude Generated"""
        config = self.load_config()
        return config.unified_config

    def save_unified_config(self, unified_config: UnifiedProviderConfig) -> bool:
        """Save unified provider configuration - Claude Generated (Deprecated)"""
        import warnings
        warnings.warn(
            "save_unified_config is deprecated. Use save_config with a full AlimaConfig object instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Load the current configuration, update the unified_config part and save everything
        config = self.load_config()
        config.unified_config = unified_config
        return self.save_config(config, preserve_unified=False)  # Preservation off since we're explicitly updating this part

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

    def get_ui_config(self) -> UIConfig:
        """Get UI configuration - Claude Generated"""
        return self.load_config().ui_config

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

    def update_ui_config(self, ui_config: UIConfig) -> bool:
        """Update UI configuration - Claude Generated"""
        config = self.load_config()
        config.ui_config = ui_config
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



    def test_database_connection(self) -> tuple[bool, str]:
        """Test database connection using current configuration - Claude Generated"""
        try:
            # Get database configuration
            config = self.load_config()
            database_config = config.database_config

            # Import DatabaseManager
            from ..core.database_manager import DatabaseManager

            # Create temporary DatabaseManager for testing
            db_manager = DatabaseManager(database_config, "test_connection")

            # Test the connection
            success, message = db_manager.test_connection()

            # Clean up
            db_manager.close_connection()

            return success, message

        except Exception as e:
            return False, f"❌ Configuration error: {str(e)}"

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
    # CONFIGURATION IMPORT/EXPORT - Deploy and migrate configurations
    # ============================================================================

    def import_configuration(self, source_dir: str, create_backup: bool = True) -> Tuple[bool, str]:
        """
        Import ALIMA configuration from a source directory - Claude Generated

        Copies config.json, prompts.json, and database file from source directory
        to the OS-specific configuration directory, with automatic path adjustment.

        Args:
            source_dir: Source directory containing config files to import
            create_backup: Create backup of current configuration before importing

        Returns:
            Tuple of (success: bool, message: str)

        Example:
            ```python
            success, msg = config_manager.import_configuration("/tmp/alima_config")
            if success:
                print("Configuration imported successfully")
            ```
        """
        try:
            from .config_validator import ConfigValidator
            from shutil import copy2
            import shutil

            source_path = Path(source_dir)
            target_dir = self.config_file.parent

            # === STEP 1: VALIDATE SOURCE DIRECTORY ===
            self.logger.info(f"🔍 Validating source directory: {source_dir}")
            validator = ConfigValidator(self.logger)
            is_valid, errors = validator.validate_config_directory(source_dir)

            if not is_valid:
                error_msg = f"❌ Source directory validation failed:\n" + "\n".join(errors)
                self.logger.error(error_msg)
                return False, error_msg

            self.logger.debug(f"✅ Source directory valid")

            # === STEP 2: CREATE BACKUP (OPTIONAL) ===
            if create_backup:
                backup_time = int(time.time())

                self.logger.info("📁 Creating backup of current configuration...")

                # Backup config.json
                if self.config_file.exists():
                    backup_config = target_dir / f"config_backup_{backup_time}.json"
                    try:
                        copy2(self.config_file, backup_config)
                        self.logger.debug(f"✅ Backed up: {backup_config}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not backup config.json: {e}")

                # Backup prompts.json
                prompts_target = target_dir / "prompts.json"
                if prompts_target.exists():
                    backup_prompts = target_dir / f"prompts_backup_{backup_time}.json"
                    try:
                        copy2(prompts_target, backup_prompts)
                        self.logger.debug(f"✅ Backed up: {backup_prompts}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not backup prompts.json: {e}")

                # Backup database
                db_target = target_dir / "alima_knowledge.db"
                if db_target.exists():
                    backup_db = target_dir / f"alima_knowledge_backup_{backup_time}.db"
                    try:
                        copy2(db_target, backup_db)
                        self.logger.debug(f"✅ Backed up: {backup_db}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not backup database: {e}")

            # === STEP 3: COPY FILES ===
            self.logger.info("📋 Copying configuration files...")

            # Copy config.json (required)
            source_config = source_path / "config.json"
            if source_config.exists():
                copy2(source_config, self.config_file)
                self.logger.debug(f"✅ Copied config.json")
            else:
                return False, "❌ Source config.json not found"

            # Copy prompts.json (optional)
            source_prompts = source_path / "prompts.json"
            if source_prompts.exists():
                target_prompts = target_dir / "prompts.json"
                copy2(source_prompts, target_prompts)
                self.logger.debug(f"✅ Copied prompts.json")

            # Copy database file (optional)
            db_files = list(source_path.glob("*.db"))
            if db_files:
                source_db = db_files[0]
                target_db = target_dir / "alima_knowledge.db"
                copy2(source_db, target_db)
                self.logger.debug(f"✅ Copied database: {source_db.name}")

            # === STEP 4: ADJUST PATHS IN CONFIG ===
            self.logger.info("🔧 Adjusting paths in configuration...")

            # Load the copied config
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Ensure system_config exists
            if 'system_config' not in config_data:
                config_data['system_config'] = {}

            system_config_data = config_data['system_config']

            # Set absolute paths for database and prompts
            abs_db_path = str(target_dir / "alima_knowledge.db")
            abs_prompts_path = str(target_dir / "prompts.json")

            system_config_data['database_path'] = abs_db_path
            system_config_data['prompts_path'] = abs_prompts_path

            # Save adjusted config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False, cls=AlimaConfigEncoder)

            self.logger.debug(f"✅ Updated database_path → {abs_db_path}")
            self.logger.debug(f"✅ Updated prompts_path → {abs_prompts_path}")

            # === STEP 5: RELOAD CONFIGURATION ===
            self.logger.info("🔄 Reloading configuration...")
            self._config = None  # Clear cache
            self.load_config()

            success_msg = f"✅ Configuration successfully imported to {target_dir}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"❌ Configuration import failed: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Stack trace: {e}", exc_info=True)
            return False, error_msg

    def export_configuration(self, target_dir: str) -> Tuple[bool, str]:
        """
        Export current ALIMA configuration to a directory - Claude Generated

        Exports config.json, prompts.json, and database file with relative paths
        suitable for deployment to other systems.

        Args:
            target_dir: Target directory where configuration files will be exported

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            from shutil import copy2

            target_path = Path(target_dir)

            # Create target directory if it doesn't exist
            target_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"📦 Exporting configuration to: {target_dir}")

            # Copy config.json
            if self.config_file.exists():
                copy2(self.config_file, target_path / "config.json")
                self.logger.debug("✅ Exported config.json")

            # Copy prompts.json
            prompts_file = self.config_file.parent / "prompts.json"
            if prompts_file.exists():
                copy2(prompts_file, target_path / "prompts.json")
                self.logger.debug("✅ Exported prompts.json")

            # Copy database
            db_file = self.config_file.parent / "alima_knowledge.db"
            if db_file.exists():
                copy2(db_file, target_path / "alima_knowledge.db")
                size_mb = db_file.stat().st_size / (1024 * 1024)
                self.logger.debug(f"✅ Exported database ({size_mb:.1f} MB)")

            success_msg = f"✅ Configuration successfully exported to {target_dir}"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"❌ Configuration export failed: {e}"
            self.logger.error(error_msg)
            return False, error_msg


# ============================================================================
# GLOBAL FUNCTIONS - For import compatibility
# ============================================================================

def get_config_manager() -> ConfigManager:
    """Get global ConfigManager instance - Claude Generated"""
    return ConfigManager()