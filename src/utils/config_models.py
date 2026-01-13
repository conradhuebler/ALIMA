#!/usr/bin/env python3
"""
ALIMA Configuration Data Models
Centralized data models for all configuration structures.
Consolidates models from config_manager.py and unified_provider_config.py
Claude Generated
"""

import json
import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import uuid  # P2.11: For ProviderId system - Claude Generated


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
# LLM TASK DISPLAY INFORMATION - Shared across all UI components
# ============================================================================
# Single source of truth for configurable LLM tasks shown in wizards and settings - Claude Generated
LLM_TASK_DISPLAY_INFO = [
    # (TaskType enum, icon+label, german description)
    (TaskType.INITIALISATION, '🔤 Initialisation', 'Erste Keyword-Generierung'),
    (TaskType.KEYWORDS, '🔑 Keywords', 'Finale Keyword-Verifikation'),
    (TaskType.CLASSIFICATION, '📚 Classification', 'DDC/DK/RVK Klassifizierung'),
    (TaskType.DK_CLASSIFICATION, '📖 DK Classification', 'DK-spezifische Klassifizierung'),
    (TaskType.VISION, '👁️ Vision', 'Bild-/OCR-Analyse'),
    (TaskType.CHUNKED_PROCESSING, '📄 Chunked', 'Große Texte in Chunks'),
]

# ============================================================================
# HELPER FUNCTIONS FOR DATABASE CONFIGURATION
# ============================================================================

def get_default_db_path() -> str:
    r"""Get OS-specific default database path - Claude Generated

    Returns:
        str: Default database path for current OS
        - Windows: C:\Users\{user}\AppData\Local\ALIMA\alima_knowledge.db
        - macOS: ~/Library/Application Support/ALIMA/alima_knowledge.db
        - Linux/Other: ~/.config/alima/alima_knowledge.db
    """
    system = platform.system()

    if system == "Windows":
        app_data = os.getenv("APPDATA")
        if app_data:
            base_path = os.path.join(app_data, "ALIMA")
        else:
            base_path = os.path.expanduser("~/AppData/Local/ALIMA")
    elif system == "Darwin":
        base_path = os.path.expanduser("~/Library/Application Support/ALIMA")
    else:
        # Linux and others
        base_path = os.path.expanduser("~/.config/alima")

    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, "alima_knowledge.db")


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration - Claude Generated

    UNIFIED SINGLE SOURCE OF TRUTH for database configuration.
    The sqlite_path is now the only database path definition.
    """
    db_type: str = 'sqlite'  # 'sqlite' or 'mysql'/'mariadb'

    # SQLite specific - THIS IS THE SINGLE SOURCE OF TRUTH FOR DB PATH
    sqlite_path: str = field(default_factory=get_default_db_path)

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
    # Allowed LLM tasks only
    LLM_TASKS = {TaskType.INITIALISATION, TaskType.KEYWORDS, TaskType.CLASSIFICATION,
                 TaskType.DK_CLASSIFICATION, TaskType.VISION, TaskType.CHUNKED_PROCESSING}

    task_type: TaskType
    model_priority: List[Dict[str, str]] = field(default_factory=list)  # [{"provider_name": "p1", "model_name": "m1"}, ...]
    chunked_model_priority: Optional[List[Dict[str, str]]] = None  # For chunked subtasks like keywords_chunked
    allow_fallback: bool = True


    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)

        # Validate: only allow LLM tasks in task_preferences
        if self.task_type not in self.LLM_TASKS:
            raise ValueError(
                f"TaskPreference only supports LLM tasks. Got {self.task_type.value}. "
                f"Allowed tasks: {', '.join(t.value for t in self.LLM_TASKS)}"
            )

        # DEPRECATED: Legacy preferred_providers field is no longer auto-migrated
        # Reason: Auto-migration with "default" causes unwanted auto-select behavior
        # Users should explicitly configure model_priority with real model names
        # (No automatic fallback - will use empty model_priority instead of "default")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization - Claude Generated"""
        return {
            "task_type": self.task_type.value,  # Convert enum to string
            "model_priority": self.model_priority,
            "chunked_model_priority": self.chunked_model_priority,
            "allow_fallback": self.allow_fallback
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
            allow_fallback=data.get("allow_fallback", True)
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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # P2.11: Unique provider ID - Claude Generated
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

    # P1.8 REVERT: Field is actually used in CLI, comprehensive_settings_dialog, pipeline_config_dialog
    preferred_provider: str = ""  # Explicit user choice, independent of provider_priority

    def get_enabled_providers(self) -> List[UnifiedProvider]:
        """Get list of enabled providers - Claude Generated"""
        return [p for p in self.providers if p.enabled and p.name not in self.disabled_providers]

    def get_enabled_ollama_providers(self) -> List[UnifiedProvider]:
        """Get enabled Ollama providers - Claude Generated (BUGFIX: Restored from bridge layer)"""
        return [p for p in self.get_enabled_providers() if p.provider_type == "ollama"]

    def get_enabled_openai_providers(self) -> List[UnifiedProvider]:
        """Get enabled OpenAI-compatible providers - Claude Generated (BUGFIX: Restored from bridge layer)"""
        return [p for p in self.get_enabled_providers() if p.provider_type == "openai_compatible"]

    def get_provider_by_name(self, name: str) -> Optional[UnifiedProvider]:
        """Get provider by name - Claude Generated"""
        for provider in self.providers:
            if provider.name.lower() == name.lower():
                return provider
        return None

    def get_provider_by_type(self, provider_type: str) -> Optional[UnifiedProvider]:
        """
        Get first enabled provider matching the given type - Claude Generated

        Args:
            provider_type: Provider type to search for (e.g., 'ollama', 'openai_compatible')

        Returns:
            First enabled provider of that type, or None if not found
        """
        for provider in self.providers:
            if provider.provider_type == provider_type and provider.enabled:
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

class CatalogType(Enum):
    """Catalog API type for classification search - Claude Generated"""
    LIBERO_SOAP = "libero_soap"  # Libero SOAP API (original)
    MARCXML_SRU = "marcxml_sru"  # MARC XML via SRU protocol (DNB, GBV, etc.)
    AUTO = "auto"  # Auto-detect based on configuration


@dataclass
class CatalogConfig:
    """Catalog API configuration - Claude Generated
    
    Supports two catalog types:
    1. LIBERO_SOAP: Original Libero SOAP API (requires token)
    2. MARCXML_SRU: Standard MARC XML via SRU protocol (DNB, Library of Congress, etc.)
    
    For MARCXML_SRU, you can use presets: "dnb", "loc", "gbv", "swb", "k10plus"
    Or configure a custom SRU endpoint with sru_base_url.
    """
    # Catalog type selection
    catalog_type: str = 'libero_soap'  # 'libero_soap', 'marcxml_sru', or 'auto'
    
    # Libero SOAP configuration (original)
    catalog_token: str = ''
    catalog_search_url: str = 'https://katalog.ub.uni-leipzig.de/Search/Results'
    catalog_details_url: str = 'https://katalog.ub.uni-leipzig.de/Record'
    
    # MARC XML / SRU configuration - Claude Generated
    sru_base_url: str = ''  # SRU endpoint URL (e.g., "https://services.dnb.de/sru/dnb")
    sru_database: str = ''  # SRU database name (optional, depends on endpoint)
    sru_schema: str = 'marcxml'  # Record schema: 'marcxml' or 'MARC21-xml'
    sru_preset: str = ''  # Use preset: "dnb", "loc", "gbv", "swb", "k10plus"
    sru_max_records: int = 50  # Maximum records per search
    
    # EXPERT OPTION: Set to False to allow non-GND-validated keywords in DK search
    # Default True: Only GND-validated keywords are used (recommended for quality control)
    # Set to False: Include plain text keywords if GND validation fails
    strict_gnd_validation_for_dk_search: bool = True
    
    def get_catalog_type(self) -> str:
        """Get the effective catalog type - Claude Generated"""
        if self.catalog_type == 'auto':
            # Auto-detect: prefer SRU if configured, otherwise SOAP
            if self.sru_preset or self.sru_base_url:
                return 'marcxml_sru'
            return 'libero_soap'
        return self.catalog_type


@dataclass
class PromptConfig:
    """Prompt configuration settings - Claude Generated"""
    prompts_file: str = 'prompts.json'
    custom_prompts_dir: str = 'custom_prompts'
    prompt_cache_enabled: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration - Claude Generated

    NOTE: database_path has been moved to DatabaseConfig.sqlite_path
    This is now the single source of truth for database configuration.
    """
    debug: bool = False
    log_level: str = 'INFO'
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    temp_dir: str = '/tmp'

    # Configurable file paths (can be absolute or relative to project root) - Claude Generated
    prompts_path: str = 'prompts.json'          # Path to prompts configuration file

    # First-run wizard tracking - Claude Generated
    first_run_completed: bool = False  # Set to true after wizard completion
    skip_first_run_check: bool = False  # Set to true to disable first-run dialog on empty config


@dataclass
class UIConfig:
    """UI-specific configuration - Claude Generated"""
    enable_webcam_input: bool = False  # Enable webcam capture in Pipeline tab
    # Future UI options can be added here


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
    ui_config: UIConfig = field(default_factory=UIConfig)  # Claude Generated - Webcam Feature

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