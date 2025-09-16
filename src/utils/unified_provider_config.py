#!/usr/bin/env python3
"""
Unified Provider Configuration System
Consolidates LLMConfig and ProviderPreferences into a single, coherent system.
Supports Smart/Advanced/Expert modes for flexible pipeline control.
Claude Generated
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from .config_manager import (
    OpenAICompatibleProvider, 
    OllamaProvider, 
    LLMConfig, 
    ProviderPreferences
)


class PipelineMode(Enum):
    """Pipeline execution modes - Claude Generated"""
    SMART = "smart"        # Automatic provider/model selection
    ADVANCED = "advanced"  # Manual provider/model selection
    EXPERT = "expert"      # Full parameter control


class TaskType(Enum):
    """LLM task types for intelligent provider selection - Claude Generated"""
    TEXT_ANALYSIS = "text_analysis"      # Keywords, concept extraction
    VISION = "vision"                    # Image analysis, OCR
    CLASSIFICATION = "classification"    # DDC, DK classification  
    CHUNKED_PROCESSING = "chunked"       # Large text processing
    GENERAL = "general"                  # Default fallback


@dataclass
class TaskPreference:
    """Task-specific provider preferences with chunked-task support - Claude Generated"""
    task_type: TaskType
    model_priority: List[Dict[str, str]] = field(default_factory=list)  # [{"provider_name": "p1", "model_name": "m1"}, ...]
    chunked_model_priority: Optional[List[Dict[str, str]]] = None  # For chunked subtasks like keywords_chunked
    performance_preference: str = "balanced"  # "quality", "speed", "balanced"
    allow_fallback: bool = True
    
    # Legacy compatibility
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


@dataclass
class PipelineStepConfig:
    """Configuration for a single pipeline step - Claude Generated"""
    step_id: str
    mode: PipelineMode = PipelineMode.SMART
    
    # Smart Mode settings
    task_type: Optional[TaskType] = None
    quality_preference: str = "balanced"  # "high", "balanced", "fast"
    
    # Advanced/Expert Mode settings  
    provider: Optional[str] = None
    model: Optional[str] = None
    task: Optional[str] = None  # Prompt task name
    
    # Expert Mode parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Meta settings
    enabled: bool = True
    timeout: Optional[int] = None
    
    def __post_init__(self):
        # Convert string enums to proper enums
        if isinstance(self.mode, str):
            self.mode = PipelineMode(self.mode)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
    
    def is_manual_override(self) -> bool:
        """Check if this step uses manual provider/model selection - Claude Generated"""
        return self.mode in [PipelineMode.ADVANCED, PipelineMode.EXPERT]
    
    def get_manual_config(self) -> Dict[str, Any]:
        """Get manual configuration for Advanced/Expert modes - Claude Generated"""
        if not self.is_manual_override():
            return {}
            
        config = {}
        if self.provider:
            config["provider"] = self.provider
        if self.model:
            config["model"] = self.model
        if self.task:
            config["task"] = self.task
            
        # Expert mode parameters
        if self.mode == PipelineMode.EXPERT:
            if self.temperature is not None:
                config["temperature"] = self.temperature
            if self.top_p is not None:
                config["top_p"] = self.top_p
            if self.max_tokens is not None:
                config["max_tokens"] = self.max_tokens
            config.update(self.custom_params)
                
        return config


@dataclass
class UnifiedProvider:
    """Unified provider definition combining connection and capability info - Claude Generated"""
    name: str                           # Display name (e.g., "Ollama-Local", "Gemini-API")
    type: str                          # "ollama", "gemini", "openai_compatible", "anthropic"
    connection_config: Dict[str, Any]   # Connection parameters (host, API keys, etc.)
    capabilities: List[str] = field(default_factory=list)  # ["vision", "fast", "large_context"]
    available_models: List[str] = field(default_factory=list)  # Detected or configured models
    enabled: bool = True
    description: str = ""
    
    @classmethod
    def from_ollama_provider(cls, ollama_provider: OllamaProvider) -> 'UnifiedProvider':
        """Convert OllamaProvider to UnifiedProvider - Claude Generated"""
        return cls(
            name=ollama_provider.name,
            type="ollama",
            connection_config={
                "host": ollama_provider.host,
                "port": ollama_provider.port,
                "api_key": ollama_provider.api_key,
                "use_ssl": ollama_provider.use_ssl,
                "connection_type": ollama_provider.connection_type
            },
            capabilities=["local", "privacy", "custom_models"],
            enabled=ollama_provider.enabled,
            description=ollama_provider.description
        )
    
    @classmethod  
    def from_openai_provider(cls, openai_provider: OpenAICompatibleProvider) -> 'UnifiedProvider':
        """Convert OpenAICompatibleProvider to UnifiedProvider - Claude Generated"""
        capabilities = []
        name_lower = openai_provider.name.lower()
        if 'openai' in name_lower:
            capabilities = ["function_calling", "structured_output"]
        elif 'chatai' in name_lower:
            capabilities = ["academic", "german"]
        elif 'anthropic' in name_lower:
            capabilities = ["reasoning", "analysis"]
            
        return cls(
            name=openai_provider.name,
            type="openai_compatible",
            connection_config={
                "base_url": openai_provider.base_url,
                "api_key": openai_provider.api_key
            },
            capabilities=capabilities,
            available_models=openai_provider.models,
            enabled=openai_provider.enabled,
            description=openai_provider.description
        )
    
    def get_base_url(self) -> str:
        """Get connection URL for this provider - Claude Generated"""
        if self.type == "ollama":
            host = self.connection_config.get("host", "localhost")
            port = self.connection_config.get("port", 11434)
            ssl = self.connection_config.get("use_ssl", False)
            protocol = "https" if ssl else "http"
            return f"{protocol}://{host}:{port}"
        elif self.type == "openai_compatible":
            return self.connection_config.get("base_url", "")
        elif self.type == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/models"
        elif self.type == "anthropic":
            return "https://api.anthropic.com/v1"
        else:
            return ""


@dataclass
class UnifiedProviderConfig:
    """
    Unified configuration system replacing LLMConfig + ProviderPreferences
    Supports Smart/Advanced/Expert pipeline modes - Claude Generated
    """
    
    # Provider definitions
    providers: List[UnifiedProvider] = field(default_factory=list)
    
    # Global preferences (from ProviderPreferences)
    preferred_provider: str = "ollama"
    provider_priority: List[str] = field(default_factory=lambda: ["ollama", "gemini", "anthropic", "openai"])
    disabled_providers: List[str] = field(default_factory=list)
    
    # Task-specific preferences
    task_preferences: Dict[str, TaskPreference] = field(default_factory=dict)
    
    # Static provider configs (for non-OpenAI-compatible)
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Performance settings
    auto_fallback: bool = True
    fallback_timeout: int = 30
    prefer_faster_models: bool = False
    
    # Version and metadata
    config_version: str = "2.0"  # New unified version
    
    def __post_init__(self):
        """Initialize default task preferences if not provided - Claude Generated"""
        if not self.task_preferences:
            self._setup_default_task_preferences()
    
    def _setup_default_task_preferences(self):
        """Setup empty default task preferences - User must configure via UI - Claude Generated"""
        self.task_preferences = {
            # Legacy compatibility entries only
            "text_analysis": TaskPreference(
                task_type=TaskType.TEXT_ANALYSIS,
                preferred_providers=[],  # Empty by default
                performance_preference="balanced"
            ),
            "vision": TaskPreference(
                task_type=TaskType.VISION,
                preferred_providers=[],  # Empty by default  
                performance_preference="quality"
            ),
            "chunked": TaskPreference(
                task_type=TaskType.CHUNKED_PROCESSING,
                preferred_providers=[],  # Empty by default
                performance_preference="balanced"
            )
        }
    
    @classmethod
    def from_legacy_config(cls, llm_config: LLMConfig, provider_preferences: ProviderPreferences) -> 'UnifiedProviderConfig':
        """Convert legacy LLMConfig + ProviderPreferences to UnifiedProviderConfig - Claude Generated"""
        
        # Convert providers
        unified_providers = []
        
        # Convert Ollama providers
        for ollama_provider in llm_config.ollama_providers:
            unified_providers.append(UnifiedProvider.from_ollama_provider(ollama_provider))
        
        # Convert OpenAI-compatible providers
        for openai_provider in llm_config.openai_compatible_providers:
            unified_providers.append(UnifiedProvider.from_openai_provider(openai_provider))
        
        # Create unified config
        unified_config = cls(
            providers=unified_providers,
            preferred_provider=provider_preferences.preferred_provider,
            provider_priority=provider_preferences.provider_priority.copy(),
            disabled_providers=provider_preferences.disabled_providers.copy(),
            gemini_api_key=llm_config.gemini,
            anthropic_api_key=llm_config.anthropic,
            auto_fallback=provider_preferences.auto_fallback,
            fallback_timeout=provider_preferences.fallback_timeout,
            prefer_faster_models=provider_preferences.prefer_faster_models
        )
        
        # Convert task-specific overrides to task preferences
        if provider_preferences.vision_provider:
            unified_config.task_preferences["vision"] = TaskPreference(
                task_type=TaskType.VISION,
                preferred_providers=[provider_preferences.vision_provider],
                performance_preference="quality"
            )
        
        if provider_preferences.text_provider:
            unified_config.task_preferences["text_analysis"] = TaskPreference(
                task_type=TaskType.TEXT_ANALYSIS,
                preferred_providers=[provider_preferences.text_provider],
                performance_preference="balanced"
            )
            
        if provider_preferences.classification_provider:
            unified_config.task_preferences["classification"] = TaskPreference(
                task_type=TaskType.CLASSIFICATION,
                preferred_providers=[provider_preferences.classification_provider],
                performance_preference="quality"
            )
        
        return unified_config
    
    def get_provider_by_name(self, name: str) -> Optional[UnifiedProvider]:
        """Get provider by name - Claude Generated"""
        for provider in self.providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def get_enabled_providers(self) -> List[UnifiedProvider]:
        """Get all enabled providers - Claude Generated"""
        return [p for p in self.providers if p.enabled and p.name not in self.disabled_providers]
    
    def get_providers_for_task(self, task_type: TaskType) -> List[str]:
        """Get ordered provider list for specific task type - Claude Generated"""
        task_key = task_type.value
        
        if task_key in self.task_preferences:
            task_pref = self.task_preferences[task_key]
            return [p for p in task_pref.preferred_providers if p not in self.disabled_providers]
        
        # Fallback to global priority
        return [p for p in self.provider_priority if p not in self.disabled_providers]
    
    def add_provider(self, provider: UnifiedProvider) -> bool:
        """Add new provider - Claude Generated"""
        if self.get_provider_by_name(provider.name):
            return False  # Provider already exists
        self.providers.append(provider)
        return True
    
    def remove_provider(self, name: str) -> bool:
        """Remove provider by name - Claude Generated"""
        for i, provider in enumerate(self.providers):
            if provider.name.lower() == name.lower():
                del self.providers[i]
                # Clean up references
                if name in self.provider_priority:
                    self.provider_priority.remove(name)
                if name in self.disabled_providers:
                    self.disabled_providers.remove(name)
                return True
        return False
    
    def get_task_preference(self, task_type: TaskType) -> TaskPreference:
        """Get task preference, with fallback to defaults - Claude Generated"""
        task_key = task_type.value
        if task_key in self.task_preferences:
            return self.task_preferences[task_key]
        
        # Return default task preference
        return TaskPreference(
            task_type=task_type,
            preferred_providers=self.provider_priority.copy(),
            performance_preference="balanced"
        )
    
    def get_model_priority_for_task(self, task_name: str, is_chunked: bool = False) -> List[Dict[str, str]]:
        """Get model priority for specific task, with chunked task support - Claude Generated"""
        # Handle chunked tasks: keywords_chunked -> keywords base task
        base_task_name = task_name
        if task_name.endswith('_chunked'):
            base_task_name = task_name[:-8]  # Remove '_chunked' suffix
            is_chunked = True
        
        # Get task preference
        if base_task_name in self.task_preferences:
            task_pref = self.task_preferences[base_task_name]
            
            # Use chunked model priority if available and requested
            if is_chunked and task_pref.chunked_model_priority:
                return task_pref.chunked_model_priority
            
            # Use standard model priority
            if task_pref.model_priority:
                return task_pref.model_priority
            
            # Legacy fallback: convert preferred_providers to model_priority
            if task_pref.preferred_providers:
                return [
                    {"provider_name": provider, "model_name": "default"}
                    for provider in task_pref.preferred_providers
                ]
        
        # Fallback to global provider priority
        return [
            {"provider_name": provider, "model_name": "default"}
            for provider in self.provider_priority
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization - Claude Generated"""
        result = asdict(self)
        
        # Convert enums in task_preferences
        converted_task_preferences = {}
        for key, task_pref in self.task_preferences.items():
            converted_task_preferences[key] = {
                "task_type": task_pref.task_type.value,
                "model_priority": task_pref.model_priority,
                "chunked_model_priority": task_pref.chunked_model_priority,
                "preferred_providers": task_pref.preferred_providers,  # Legacy compatibility
                "performance_preference": task_pref.performance_preference,
                "allow_fallback": task_pref.allow_fallback
            }
        result["task_preferences"] = converted_task_preferences
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedProviderConfig':
        """Create from dictionary (JSON deserialization) - Claude Generated"""
        # Convert providers
        providers = []
        for provider_data in data.get("providers", []):
            providers.append(UnifiedProvider(**provider_data))
        
        # Convert task preferences
        task_preferences = {}
        for key, task_data in data.get("task_preferences", {}).items():
            task_preferences[key] = TaskPreference(
                task_type=TaskType(task_data["task_type"]),
                model_priority=task_data.get("model_priority", []),
                chunked_model_priority=task_data.get("chunked_model_priority"),
                preferred_providers=task_data.get("preferred_providers", []),  # Legacy support
                performance_preference=task_data.get("performance_preference", "balanced"),
                allow_fallback=task_data.get("allow_fallback", True)
            )
        
        return cls(
            providers=providers,
            preferred_provider=data.get("preferred_provider", "ollama"),
            provider_priority=data.get("provider_priority", ["ollama", "gemini", "anthropic", "openai"]),
            disabled_providers=data.get("disabled_providers", []),
            task_preferences=task_preferences,
            gemini_api_key=data.get("gemini_api_key", ""),
            anthropic_api_key=data.get("anthropic_api_key", ""),
            auto_fallback=data.get("auto_fallback", True),
            fallback_timeout=data.get("fallback_timeout", 30),
            prefer_faster_models=data.get("prefer_faster_models", False),
            config_version=data.get("config_version", "2.0")
        )


class UnifiedProviderConfigManager:
    """Manager for UnifiedProviderConfig with migration and validation - Claude Generated"""
    
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self._unified_config: Optional[UnifiedProviderConfig] = None
    
    def get_unified_config(self) -> UnifiedProviderConfig:
        """Get unified configuration, migrating from legacy if needed - Claude Generated"""
        if self._unified_config is None:
            self._unified_config = self._load_or_migrate_config()
        return self._unified_config
    
    def _load_or_migrate_config(self) -> UnifiedProviderConfig:
        """Load unified config or migrate from legacy system - Claude Generated"""
        if not self.config_manager:
            # Return default config if no config manager
            return UnifiedProviderConfig()
        
        try:
            # Try to load existing unified config first
            alima_config = self.config_manager.load_config()
            
            # Check if we already have unified config
            if hasattr(alima_config, 'unified_provider_config'):
                self.logger.info("Loading existing unified provider config")
                return alima_config.unified_provider_config
            
            # Migrate from legacy system
            self.logger.info("Migrating from legacy LLMConfig + ProviderPreferences")
            legacy_llm_config = alima_config.llm
            legacy_preferences = alima_config.provider_preferences
            
            unified_config = UnifiedProviderConfig.from_legacy_config(
                legacy_llm_config, 
                legacy_preferences
            )
            
            self.logger.info(f"Migration completed: {len(unified_config.providers)} providers, "
                           f"{len(unified_config.task_preferences)} task preferences")
            
            return unified_config
            
        except Exception as e:
            self.logger.error(f"Error loading/migrating provider config: {e}")
            self.logger.info("Falling back to default unified config")
            return UnifiedProviderConfig()
    
    def save_unified_config(self, unified_config: UnifiedProviderConfig) -> bool:
        """Save unified configuration - Claude Generated"""
        try:
            self._unified_config = unified_config
            
            if self.config_manager:
                # Update main config with unified config
                alima_config = self.config_manager.load_config()
                alima_config.unified_provider_config = unified_config
                return self.config_manager.save_config(alima_config)
            else:
                self.logger.warning("No config manager available for saving")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving unified provider config: {e}")
            return False
    
    def validate_config(self, unified_config: UnifiedProviderConfig) -> List[str]:
        """Validate unified configuration - Claude Generated"""
        issues = []
        
        # Check for enabled providers
        enabled_providers = unified_config.get_enabled_providers()
        if not enabled_providers:
            issues.append("No enabled providers configured")
        
        # Check preferred provider exists
        if unified_config.preferred_provider not in [p.name for p in enabled_providers]:
            issues.append(f"Preferred provider '{unified_config.preferred_provider}' not available")
        
        # Check provider priority list
        available_names = [p.name for p in enabled_providers]
        for provider_name in unified_config.provider_priority:
            if provider_name not in available_names:
                issues.append(f"Priority provider '{provider_name}' not available")
        
        # Check task preferences
        for task_key, task_pref in unified_config.task_preferences.items():
            for provider_name in task_pref.preferred_providers:
                if provider_name not in available_names:
                    issues.append(f"Task '{task_key}' prefers unavailable provider '{provider_name}'")
        
        return issues


# Global instance for easy access
_unified_config_manager = None

def get_unified_config_manager(config_manager=None) -> UnifiedProviderConfigManager:
    """Get global unified configuration manager - Claude Generated"""
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedProviderConfigManager(config_manager)
    return _unified_config_manager