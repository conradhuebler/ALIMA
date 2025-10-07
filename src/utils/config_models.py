#!/usr/bin/env python3
"""
ALIMA Configuration Data Models
Centralized data models for all configuration structures.
Consolidates models from config_manager.py and unified_provider_config.py
Claude Generated
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging


# ============================================================================
# ENUMS
# ============================================================================

class PipelineMode(Enum):
    """Pipeline execution modes - Claude Generated"""
    SMART = "smart"        # Automatic provider/model selection
    ADVANCED = "advanced"  # Manual provider/model selection
    EXPERT = "expert"      # Full parameter control


class TaskType(Enum):
    """ALIMA pipeline step types for intelligent provider selection - Claude Generated"""
    INPUT = "input"                      # File/text input (no LLM)
    INITIALISATION = "initialisation"    # LLM keyword extraction from text
    SEARCH = "search"                    # Database search (no LLM)
    KEYWORDS = "keywords"                # LLM keyword verification with GND context
    CLASSIFICATION = "classification"    # LLM DDC/DK/RVK classification
    DK_SEARCH = "dk_search"             # Catalog search (no LLM)
    DK_CLASSIFICATION = "dk_classification" # LLM DK classification analysis
    VISION = "vision"                    # Image analysis, OCR (for image_text_extraction)
    CHUNKED_PROCESSING = "chunked"       # Large text processing
    GENERAL = "general"                  # Default fallback


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration - Claude Generated"""
    db_type: str = 'sqlite'  # 'sqlite' or 'mysql'/'mariadb'

    # SQLite specific
    sqlite_path: str = 'alima_knowledge.db'

    # MySQL/MariaDB specific
    host: str = 'localhost'
    port: int = 3306
    database: str = 'alima_knowledge'
    username: str = 'alima'
    password: str = ''

    # Connection settings
    connection_timeout: int = 30
    auto_create_tables: bool = True
    charset: str = 'utf8mb4'
    ssl_disabled: bool = False


# ============================================================================
# PROVIDER CONFIGURATIONS
# ============================================================================

@dataclass
class OpenAICompatibleProvider:
    """Configuration for OpenAI-compatible API providers - Claude Generated"""
    name: str = ''                    # Provider name (e.g. "ChatAI", "DeepSeek")
    base_url: str = ''               # API base URL
    api_key: str = ''                # API key for authentication
    enabled: bool = True             # Whether provider is active
    models: List[str] = field(default_factory=list)  # Available models (optional)
    preferred_model: str = ''        # Preferred model for this provider - Claude Generated
    description: str = ''            # Description for UI display

    def __post_init__(self):
        """Validation after initialization - Claude Generated"""
        if not self.name:
            raise ValueError("Provider name cannot be empty")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")


@dataclass
class OllamaProvider:
    """Flexible Ollama provider configuration similar to OpenAI providers - Claude Generated"""
    name: str  # Alias name (e.g., "local_home", "work_server", "cloud_instance")
    host: str  # Server host (e.g., "localhost", "192.168.1.100", "ollama.example.com")
    port: int = 11434  # Server port
    api_key: str = ''  # Optional API key for authenticated access
    enabled: bool = True  # Provider enabled/disabled state
    preferred_model: str = ''  # Preferred model for this provider - Claude Generated
    description: str = ''  # Human-readable description
    use_ssl: bool = False  # Use HTTPS instead of HTTP
    connection_type: str = 'native_client'  # 'native_client' (native ollama library) or 'openai_compatible' (OpenAI API format)

    def __post_init__(self):
        """Validation after initialization - Claude Generated"""
        if not self.name:
            raise ValueError("Ollama provider name cannot be empty")
        if not self.host:
            raise ValueError("Ollama host cannot be empty")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")

    @property
    def base_url(self) -> str:
        """Get the complete base URL for this Ollama provider - Claude Generated"""
        # Handle case where host already contains protocol
        if self.host.startswith(('http://', 'https://')):
            host_without_protocol = self.host.split('://', 1)[1]
            protocol = 'https' if self.use_ssl else 'http'
        else:
            host_without_protocol = self.host
            protocol = 'https' if self.use_ssl else 'http'

        # Only add port for local/IP addresses, not for domain names with standard ports - Claude Generated
        if self._needs_explicit_port():
            port_part = f":{self.port}"
        else:
            port_part = ""

        if self.connection_type == 'openai_compatible':
            return f"{protocol}://{host_without_protocol}{port_part}/v1"
        else:
            return f"{protocol}://{host_without_protocol}{port_part}"

    def _needs_explicit_port(self) -> bool:
        """Check if explicit port is needed - Claude Generated"""
        # Extract hostname without protocol
        if '://' in self.host:
            host_part = self.host.split('://', 1)[1]
        else:
            host_part = self.host

        # Remove port if already in host
        host_part = host_part.split(':')[0]

        # Standard HTTPS/HTTP ports don't need explicit port
        if self.use_ssl and self.port == 443:
            return False
        elif not self.use_ssl and self.port == 80:
            return False

        # localhost and IP addresses typically need explicit ports
        if host_part in ['localhost', '127.0.0.1'] or host_part.count('.') == 3:
            return True

        # Domain names with non-standard ports need explicit port
        return True

    @property
    def display_name(self) -> str:
        """Get display name with connection info - Claude Generated"""
        status = "🔐" if self.api_key else "🔓"
        ssl_indicator = "🔒" if self.use_ssl else ""
        return f"{status}{ssl_indicator} {self.name} ({self.host}:{self.port})"


@dataclass
class GeminiProvider:
    """Gemini provider configuration - Claude Generated"""
    api_key: str = ''
    enabled: bool = True
    preferred_model: str = ''
    description: str = 'Google Gemini API'

    @property
    def name(self) -> str:
        return "gemini"


@dataclass
class AnthropicProvider:
    """Anthropic provider configuration - Claude Generated"""
    api_key: str = ''
    enabled: bool = True
    preferred_model: str = ''
    description: str = 'Anthropic Claude API'

    @property
    def name(self) -> str:
        return "anthropic"


# ============================================================================
# TASK PREFERENCES & PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class TaskPreference:
    """Task-specific provider preferences with chunked-task support - Claude Generated"""
    task_type: TaskType
    model_priority: List[Dict[str, str]] = field(default_factory=list)  # [{"provider_name": "p1", "model_name": "m1"}, ...]
    chunked_model_priority: Optional[List[Dict[str, str]]] = None  # For chunked subtasks like keywords_chunked
    allow_fallback: bool = True

    # Legacy compatibility - will be removed in cleanup
    preferred_providers: List[str] = field(default_factory=list)  # Will be migrated to model_priority

    def __post_init__(self):
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        # Migrate legacy preferred_providers to model_priority if needed
        if self.preferred_providers and not self.model_priority:
            self.model_priority = [
                {"provider_name": provider, "model_name": "default"}
                for provider in self.preferred_providers
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization - Claude Generated"""
        return {
            "task_type": self.task_type.value,  # Convert enum to string
            "model_priority": self.model_priority,
            "chunked_model_priority": self.chunked_model_priority,
            "allow_fallback": self.allow_fallback,
            "preferred_providers": self.preferred_providers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPreference':
        """Create from dictionary for JSON deserialization - Claude Generated"""
        # Convert string back to enum
        task_type = TaskType(data.get("task_type", "general"))
        return cls(
            task_type=task_type,
            model_priority=data.get("model_priority", []),
            chunked_model_priority=data.get("chunked_model_priority"),
            allow_fallback=data.get("allow_fallback", True),
            preferred_providers=data.get("preferred_providers", [])
        )


@dataclass
class PipelineStepConfig:
    """Configuration for a single pipeline step - Claude Generated"""
    step_id: str

    # Task type for context (auto-derived from step_id)
    task_type: Optional[TaskType] = None

    # Configuration settings
    provider: Optional[str] = None
    model: Optional[str] = None
    task: Optional[str] = None  # Prompt task name

    # LLM parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None  # For reproducible results
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Meta settings
    enabled: bool = True
    timeout: Optional[int] = None

    def __post_init__(self):
        # Convert string enums to proper enums
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)


# ============================================================================
# UNIFIED PROVIDER CONFIGURATION
# ============================================================================

@dataclass
class UnifiedProvider:
    """Unified provider representation for all provider types - Claude Generated"""
    name: str
    provider_type: str  # 'ollama', 'openai_compatible', 'gemini', 'anthropic'
    enabled: bool = True
    api_key: str = ''
    base_url: str = ''
    preferred_model: str = ''
    description: str = ''

    # Ollama specific
    host: str = ''
    port: int = 11434
    use_ssl: bool = False
    connection_type: str = 'native_client'

    # Runtime attributes
    available_models: List[str] = field(default_factory=list)  # Models available from this provider

    @property
    def type(self) -> str:
        """Alias for provider_type for backward compatibility - Claude Generated"""
        return self.provider_type

    @type.setter
    def type(self, value: str):
        """Setter for type alias - Claude Generated"""
        self.provider_type = value

    @classmethod
    def from_ollama_provider(cls, provider: OllamaProvider) -> 'UnifiedProvider':
        """Create UnifiedProvider from OllamaProvider - Claude Generated"""
        return cls(
            name=provider.name,
            provider_type='ollama',
            enabled=provider.enabled,
            api_key=provider.api_key,
            base_url=provider.base_url,
            preferred_model=provider.preferred_model,
            description=provider.description,
            host=provider.host,
            port=provider.port,
            use_ssl=provider.use_ssl,
            connection_type=provider.connection_type
        )

    @classmethod
    def from_openai_compatible_provider(cls, provider: OpenAICompatibleProvider) -> 'UnifiedProvider':
        """Create UnifiedProvider from OpenAICompatibleProvider - Claude Generated"""
        return cls(
            name=provider.name,
            provider_type='openai_compatible',
            enabled=provider.enabled,
            api_key=provider.api_key,
            base_url=provider.base_url,
            preferred_model=provider.preferred_model,
            description=provider.description
        )

    @classmethod
    def from_gemini_provider(cls, provider: GeminiProvider) -> 'UnifiedProvider':
        """Create UnifiedProvider from GeminiProvider - Claude Generated"""
        return cls(
            name='gemini',
            provider_type='gemini',
            enabled=provider.enabled,
            api_key=provider.api_key,
            preferred_model=provider.preferred_model,
            description=provider.description
        )

    @classmethod
    def from_anthropic_provider(cls, provider: AnthropicProvider) -> 'UnifiedProvider':
        """Create UnifiedProvider from AnthropicProvider - Claude Generated"""
        return cls(
            name='anthropic',
            provider_type='anthropic',
            enabled=provider.enabled,
            api_key=provider.api_key,
            preferred_model=provider.preferred_model,
            description=provider.description
        )


@dataclass
class UnifiedProviderConfig:
    """Unified configuration for all provider types - Claude Generated"""
    # All providers as unified objects
    providers: List[UnifiedProvider] = field(default_factory=list)

    # Global settings
    provider_priority: List[str] = field(default_factory=lambda: ["ollama", "gemini", "anthropic", "openai"])
    disabled_providers: List[str] = field(default_factory=list)

    # Task-specific preferences
    task_preferences: Dict[str, TaskPreference] = field(default_factory=dict)

    # Individual provider configs (LEGACY - will be migrated to providers list)
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_preferred_model: str = ""
    anthropic_preferred_model: str = ""
    auto_fallback: bool = True
    prefer_faster_models: bool = False  # Legacy compatibility for smart_provider_selector

    # Direct preferred provider attribute (config.md Phase 1) - Claude Generated
    preferred_provider: str = "localhost"  # Explicit user choice, independent of provider_priority

    def get_enabled_providers(self) -> List[UnifiedProvider]:
        """Get list of enabled providers - Claude Generated"""
        return [p for p in self.providers if p.enabled and p.name not in self.disabled_providers]

    def get_provider_by_name(self, name: str) -> Optional[UnifiedProvider]:
        """Get provider by name - Claude Generated"""
        for provider in self.providers:
            if provider.name.lower() == name.lower():
                return provider
        return None

    def get_task_preference(self, task_type: TaskType) -> TaskPreference:
        """Get task preference, with fallback to defaults - Claude Generated"""
        task_key = task_type.value
        if task_key in self.task_preferences:
            return self.task_preferences[task_key]

        # Return default task preference
        return TaskPreference(
            task_type=task_type,
            model_priority=[],
            allow_fallback=True
        )

    def get_model_priority_for_task(self, task_name: str, is_chunked: bool = False) -> List[Dict[str, str]]:
        """Get model priority for a task, with chunked variant support - Claude Generated"""
        task_pref = self.task_preferences.get(task_name)

        if not task_pref:
            return []

        # Use chunked priority if available and requested
        if is_chunked and task_pref.chunked_model_priority:
            return task_pref.chunked_model_priority

        return task_pref.model_priority

    @classmethod
    def from_legacy_config(cls, legacy_data: Dict[str, Any]) -> 'UnifiedProviderConfig':
        """Create UnifiedProviderConfig from legacy configuration - Claude Generated"""
        config = cls()

        # Migrate basic settings if present
        if 'auto_fallback' in legacy_data:
            config.auto_fallback = legacy_data['auto_fallback']
        if 'prefer_faster_models' in legacy_data:
            config.prefer_faster_models = legacy_data['prefer_faster_models']
        if 'provider_priority' in legacy_data:
            config.provider_priority = legacy_data['provider_priority']
        if 'disabled_providers' in legacy_data:
            config.disabled_providers = legacy_data['disabled_providers']

        return config


# ============================================================================
# OTHER CONFIGURATION CLASSES
# ============================================================================

@dataclass
class CatalogConfig:
    """Catalog API configuration - Claude Generated"""
    catalog_token: str = ''
    catalog_search_url: str = 'https://katalog.ub.uni-leipzig.de/Search/Results'
    catalog_details_url: str = 'https://katalog.ub.uni-leipzig.de/Record'


@dataclass
class PromptConfig:
    """Prompt configuration settings - Claude Generated"""
    prompts_file: str = 'prompts.json'
    custom_prompts_dir: str = 'custom_prompts'
    prompt_cache_enabled: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration - Claude Generated"""
    debug: bool = False
    log_level: str = 'INFO'
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    temp_dir: str = '/tmp'


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class AlimaConfig:
    """Main ALIMA configuration with unified provider system - Claude Generated"""
    # Core configuration sections
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    catalog_config: CatalogConfig = field(default_factory=CatalogConfig)
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    system_config: SystemConfig = field(default_factory=SystemConfig)

    # UNIFIED PROVIDER CONFIGURATION - single source of truth
    unified_config: UnifiedProviderConfig = field(default_factory=UnifiedProviderConfig)

    # Legacy compatibility attributes - will be removed
    @property
    def database(self) -> DatabaseConfig:
        return self.database_config

    @property
    def catalog(self) -> CatalogConfig:
        return self.catalog_config

    @property
    def system(self) -> SystemConfig:
        return self.system_config


    # Version and metadata
    config_version: str = '2.0'  # Incremented for unified config
    created_at: str = field(default_factory=lambda: "")
    updated_at: str = field(default_factory=lambda: "")