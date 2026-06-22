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


class ProviderScope(str, Enum):
    """Scope for default provider/model resolution - Claude Generated.

    A ``str`` subclass so existing string callers keep working: comparisons like
    ``ProviderScope.AGENTIC == "agentic"`` are True, and the resolver accepts
    either the enum member or the bare string.
    """
    GENERAL = "general"    # preferred -> first enabled
    PIPELINE = "pipeline"  # pipeline_default -> preferred -> first enabled
    AGENTIC = "agentic"    # agentic_default -> pipeline_default -> preferred -> first enabled


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
    chunking_threshold: Optional[int] = None  # 0 = auto-detect, >0 = explicit threshold (keywords task only)


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
            "allow_fallback": self.allow_fallback,
            "chunking_threshold": self.chunking_threshold,
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
            chunking_threshold=data.get("chunking_threshold"),
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
    repetition_penalty: Optional[float] = None  # Penalise repeated tokens (Ollama repeat_penalty / OpenAI repetition_penalty)
    think: Optional[bool] = None  # Enable/disable thinking mode (Ollama/OpenAI-compat extended thinking)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Meta settings
    enabled: bool = True
    timeout: Optional[int] = None

    # Iterative refinement - Claude Generated
    enable_iterative_refinement: bool = False  # Opt-in by default
    max_refinement_iterations: int = 2  # Maximum iterations for missing concept search

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

    @property
    def is_usable(self) -> bool:
        """Whether this provider can plausibly serve requests (config-level check).

        gemini/anthropic are cloud-only and cannot do anything without an API
        key, so a keyless stub left enabled in the list must not be picked as the
        implicit "first enabled" default fallback. ollama (local) and
        openai_compatible (may be a keyless local endpoint) are assumed usable —
        reachability/SDK is verified at runtime by llm_service. Claude Generated.
        """
        if self.provider_type in ("gemini", "anthropic") and not self.api_key:
            return False
        return True

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
    # No hardcoded provider list (per CLAUDE.md: read providers from config, not
    # a baked-in cloud list). Empty = derive ordering from the configured
    # providers. Claude Generated.
    provider_priority: List[str] = field(default_factory=list)
    disabled_providers: List[str] = field(default_factory=list)

    # Pipeline default provider/model - single source of truth for pipeline execution
    # Claude Generated - Provider Strategy Simplification
    pipeline_default_provider: str = ""  # e.g. "ollama", "gemini"
    pipeline_default_model: str = ""     # e.g. "cogito:32b", "gemini-1.5-flash"

    # Agentic / workflow default provider/model (WP-ι cleanup)
    # Falls back to pipeline_default, then preferred_provider if unset.
    agentic_default_provider: str = ""
    agentic_default_model: str = ""

    # Task-specific preferences (for step overrides)
    task_preferences: Dict[str, TaskPreference] = field(default_factory=dict)

    # Individual provider configs (LEGACY - will be migrated to providers list)
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_preferred_model: str = ""
    anthropic_preferred_model: str = ""

    # P1.8 REVERT: Field is actually used in CLI, comprehensive_settings_dialog, pipeline_config_dialog
    preferred_provider: str = ""  # Explicit user choice, independent of provider_priority
    preferred_model: str = ""     # Model for preferred_provider — central general default
                                  # (baseline for pipeline + chat when no pipeline_default set)

    # Per-model chunking thresholds - Claude Generated
    # Format: {"provider_name": {"model_name": threshold_int}}
    # e.g., {"ollama_local": {"cogito:32b": 1000, "cogito:14b": 500}}
    model_chunking_thresholds: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_chunking_threshold(self, provider_name: str, model_name: str) -> Optional[int]:
        """Get configured chunking threshold for provider/model - Claude Generated

        Args:
            provider_name: Provider name (e.g., "ollama_local")
            model_name: Model name (e.g., "cogito:32b")

        Returns:
            Configured threshold if set, None for auto-detect
        """
        provider_thresholds = self.model_chunking_thresholds.get(provider_name, {})
        return provider_thresholds.get(model_name)

    def set_chunking_threshold(self, provider_name: str, model_name: str, threshold: int):
        """Set chunking threshold for provider/model - Claude Generated

        Args:
            provider_name: Provider name
            model_name: Model name
            threshold: Chunking threshold (keywords before splitting)
        """
        if provider_name not in self.model_chunking_thresholds:
            self.model_chunking_thresholds[provider_name] = {}
        self.model_chunking_thresholds[provider_name][model_name] = threshold

    def remove_chunking_threshold(self, provider_name: str, model_name: str):
        """Remove chunking threshold (revert to auto-detect) - Claude Generated

        Args:
            provider_name: Provider name
            model_name: Model name
        """
        if provider_name in self.model_chunking_thresholds:
            self.model_chunking_thresholds[provider_name].pop(model_name, None)
            # Clean up empty provider dict
            if not self.model_chunking_thresholds[provider_name]:
                del self.model_chunking_thresholds[provider_name]

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

    def resolve_provider_model(self, provider_name: str) -> tuple[str, str]:
        """Return a usable (provider, model) for a known provider name.

        If the provider is unknown, model is empty.  This is the single place
        that knows how to turn a bare provider name into a provider/model pair.
        Claude Generated (default-model cleanup).
        """
        provider = self.get_provider_by_name(provider_name)
        if provider is None:
            return provider_name, ""
        model = (
            getattr(provider, "preferred_model", "") or
            (list(getattr(provider, "available_models", None) or []) or [""])[0]
        )
        return provider.name, model

    def resolve_default_provider_model(
        self,
        *,
        scope: "ProviderScope | str" = ProviderScope.GENERAL,
        fallback_to_first_enabled: bool = True,
    ) -> tuple[str, str]:
        """Resolve the default (provider, model) for a given scope.

        Hierarchy per scope (first match wins):
          - pipeline:  pipeline_default -> preferred -> first enabled provider
          - agentic:   agentic_default -> pipeline_default -> preferred -> first enabled provider
          - general:   preferred -> first enabled provider

        Empty strings are treated as unset.  When ``fallback_to_first_enabled`` is
        True and no configured default is available, the first enabled provider's
        preferred/first model is used.  Claude Generated (default-model cleanup).

        Args:
            scope: Which default to resolve ("general", "pipeline", or "agentic").
            fallback_to_first_enabled: Whether to fall back to the first enabled
                provider when no explicit default is configured.

        Returns:
            Tuple of (provider, model). May be ("", "") if no provider is configured.
        """
        # Determine the ordered list of candidate (provider, model) settings.
        candidate_fields: list[tuple[str, str]] = []
        if scope == "agentic":
            candidate_fields.append(
                (self.agentic_default_provider, self.agentic_default_model)
            )
        if scope in ("agentic", "pipeline"):
            candidate_fields.append(
                (self.pipeline_default_provider, self.pipeline_default_model)
            )
        # General fallback is always last.
        candidate_fields.append((self.preferred_provider, self.preferred_model))

        for provider, model in candidate_fields:
            if provider:
                if not model:
                    _, model = self.resolve_provider_model(provider)
                return provider, model

        if fallback_to_first_enabled:
            # Pick the first enabled *usable* provider, not just the first enabled
            # one: an enabled-but-keyless cloud stub (e.g. a leftover gemini entry)
            # would otherwise be chosen and then fail with "No client found".
            for candidate in self.get_enabled_providers():
                if not candidate.is_usable:
                    continue
                provider, model = self.resolve_provider_model(candidate.name)
                if provider:
                    return provider, model

        return "", ""

    def fill_empty_default_models(self) -> None:
        """Persist resolved models for any default-provider with an empty model.

        When a ``*_default_provider`` is set but its ``*_default_model`` is empty,
        resolve the provider's preferred/first model and store it.  This makes the
        choice durable instead of re-deriving it on every read (and re-saving it as
        empty).  A no-op when the provider has no resolvable model yet (e.g. models
        not detected).  Claude Generated (default-model cleanup).
        """
        pairs = (
            ("pipeline_default_provider", "pipeline_default_model"),
            ("agentic_default_provider", "agentic_default_model"),
            ("preferred_provider", "preferred_model"),
        )
        for provider_attr, model_attr in pairs:
            provider = getattr(self, provider_attr, "") or ""
            model = getattr(self, model_attr, "") or ""
            if provider and not model:
                _, resolved = self.resolve_provider_model(provider)
                if resolved:
                    setattr(self, model_attr, resolved)

    def sync_legacy_from_providers(self) -> None:
        """Refresh the legacy gemini/anthropic mirror fields from their provider objects.

        Keys and preferred models are actually edited on the ``UnifiedProvider``
        objects (the UI's provider editor and ``set_api_key`` both target them), so
        the provider object is authoritative.  These legacy top-level fields are a
        read-only mirror still consulted by some UI/status code; this keeps them
        from going stale.  Only overwrites from a non-empty provider value so a
        first-time legacy seed (before the provider is materialized) isn't clobbered.
        Claude Generated (key cleanup).
        """
        for name, key_attr, model_attr in (
            ("gemini", "gemini_api_key", "gemini_preferred_model"),
            ("anthropic", "anthropic_api_key", "anthropic_preferred_model"),
        ):
            provider = self.get_provider_by_name(name)
            if provider is None:
                continue
            if getattr(provider, "api_key", ""):
                setattr(self, key_attr, provider.api_key)
            if getattr(provider, "preferred_model", ""):
                setattr(self, model_attr, provider.preferred_model)

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
    catalog_search_url: str = ''
    catalog_details_url: str = ''
    catalog_web_search_url: str = ''   # Web frontend search URL (BiblioClient web fallback)
    catalog_web_record_url: str = ''   # Web frontend record base URL (BiblioClient web fallback)
    
    # MARC XML / SRU configuration - Claude Generated
    sru_base_url: str = ''  # SRU endpoint URL (e.g., "https://services.dnb.de/sru/dnb")
    sru_database: str = ''  # SRU database name (optional, depends on endpoint)
    sru_schema: str = 'marcxml'  # Record schema: 'marcxml' or 'MARC21-xml'
    sru_preset: str = ''  # Use preset: "dnb", "loc", "gbv", "swb", "k10plus"
    sru_max_records: int = 50  # Maximum records per search

    # finc / VuFind-JSON catalog configuration - Claude Generated
    # Used by FincClient (src/utils/clients/finc_client.py) to talk to a local
    # finc/VuFind instance (e.g. TU Freiberg finc solrproxy). Sits alongside
    # Libero/SRU; priority is set in tool_registry._init_suggesters and
    # pipeline_utils.execute_dk_search (finc is preferred when configured).
    finc_base_url: str = ''  # e.g. "https://finc.example.org/fincsolrproxy/proxy.php"
    finc_web_record_url: str = ''  # e.g. "https://katalog.example.org/Record/" (used to build web_url)
    finc_default_limit: int = 20  # Max records per search (1..100)
    finc_timeout: int = 30  # HTTP timeout in seconds
    finc_institution_filter: str = ''  # Optional default filter[]=facet:"value" (VuFind facet syntax)
    # Opt-in: use finc (per-title udk_raw) for the DK classification search
    # instead of Libero/SRU. Requires finc_base_url. Default off so finc stays
    # reachable via the search_finc tool without changing the DK backend. - Claude Generated
    finc_dk_enabled: bool = False
    # Opt-in: in the keyword search step, harvest finc catalog titles and
    # reconcile their subjects against the local GND cache to enrich the
    # selection pool (agentic finc_subject_harvest step). Requires finc_base_url. - Claude Generated
    finc_harvest_enabled: bool = False

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

    # Output paths - Claude Generated
    autosave_dir: str = str(Path.home() / "Documents" / "ALIMA_Results")

    # Contact email for API polite pools (Crossref, OpenAlex) - Claude Generated
    contact_email: str = ''

    # DOI resolver: which sources to query - Claude Generated
    doi_use_crossref: bool = True
    doi_use_openalex: bool = True
    doi_use_datacite: bool = True

    # First-run wizard tracking - Claude Generated
    first_run_completed: bool = False  # Set to true after wizard completion
    skip_first_run_check: bool = False  # Set to true to disable first-run dialog on empty config


@dataclass
class UIConfig:
    """UI-specific configuration - Claude Generated"""
    enable_webcam_input: bool = False  # Enable webcam capture in Pipeline tab
    font_size: int = 10  # Global base font size in pt (8–16) — Claude Generated
    ptheta_banner_seen: bool = False  # WP10 P-θ.4 one-time banner flag


@dataclass
class RepetitionDetectionConfig:
    """Configuration for LLM repetition detection - Claude Generated

    Detects when LLMs fall into repetitive output loops and enables
    automatic abort with parameter variation suggestions.

    Tuned to avoid false positives - requires substantial repetition before triggering.
    """
    # Master switch
    enabled: bool = True              # Enable repetition detection
    auto_abort: bool = True           # Automatically abort on detection
    show_suggestions: bool = True     # Show parameter variation suggestions
    grace_period_seconds: float = 2.0 # Wait before aborting (0 = immediate, Claude Generated)

    # N-gram detection
    ngram_size: int = 6               # Words per N-gram (longer = more specific)
    ngram_threshold: int = 8          # Repetitions before flagging (higher = more lenient)

    # Window similarity detection
    window_size: int = 300            # Characters per comparison window
    window_similarity_threshold: float = 0.90  # Jaccard threshold (0.0-1.0)
    min_windows: int = 4              # Minimum windows before comparison

    # Character pattern detection
    char_repeat_threshold: int = 80   # Consecutive identical chars ("!!!", "aaa")

    # Processing control
    min_text_length: int = 1000       # Start checking after N characters
    check_interval: int = 200         # Check every N characters


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class ChatConfig:
    """Chat-widget configuration (WP10 P-δ.1). Claude Generated.

    Independent of the pipeline-LLM defaults — chat may use a smaller /
    cheaper model. ``no_cache_writes=True`` blocks chat tool calls from
    persisting into the MCP cache (Audit-Finding S5).
    """
    default_provider: str = ""
    default_model: str = ""
    max_iterations: int = 30
    # Max tokens per LLM call in chat. Reasoning models can spend the whole
    # budget on the thinking channel and return empty content on long inputs —
    # a larger default leaves headroom for an actual answer. - Claude Generated
    max_tokens: int = 8192
    no_cache_writes: bool = True
    temperature: float = 0.5
    system_prompt_override: str = ""
    # Chat system-prompt tier: 'auto' (compact for small/code models, full
    # otherwise), 'compact' (always short), 'full' (always long). The full
    # ruleset can overwhelm small/code models into empty/wrong answers. - Claude Generated
    system_prompt_tier: str = "auto"
    # P-ε: when True, mutation tools skip the inline confirm bubble and
    # apply the change immediately. Destructive ops (cache writes, catalog
    # calls with cost) still require confirmation regardless of this flag.
    autonomous_pipeline: bool = False
    # Webapp-only, config-file setting (no UI): path to a SQLite DB that logs
    # every chat turn (prompt, response, tool calls + results) for later
    # analysis. Empty = disabled. Relative paths resolve against the config
    # directory. - Claude Generated
    session_log_db: str = ""


@dataclass
class AlimaConfig:
    """Main ALIMA configuration with unified provider system - Claude Generated"""
    # Core configuration sections
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    catalog_config: CatalogConfig = field(default_factory=CatalogConfig)
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    system_config: SystemConfig = field(default_factory=SystemConfig)
    ui_config: UIConfig = field(default_factory=UIConfig)  # Claude Generated - Webcam Feature
    repetition_config: RepetitionDetectionConfig = field(default_factory=RepetitionDetectionConfig)  # Claude Generated
    chat_config: ChatConfig = field(default_factory=ChatConfig)  # WP10 P-δ.1

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