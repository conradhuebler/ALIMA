#!/usr/bin/env python3
"""
ALIMA Configuration Manager
Handles all application configuration with system-wide and fallback support.
Includes LLM providers, database settings, catalog tokens, and all other configs.
Claude Generated
"""

import json
import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict, field
import logging


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
class LLMConfig:
    """LLM provider configuration - Claude Generated"""
    # Individual provider API Keys (non-OpenAI compatible)
    gemini: str = ''
    gemini_preferred_model: str = ''  # Preferred model for Gemini provider - Claude Generated
    anthropic: str = ''
    anthropic_preferred_model: str = ''  # Preferred model for Anthropic provider - Claude Generated
    
    # OpenAI-compatible providers (flexible list)
    openai_compatible_providers: List[OpenAICompatibleProvider] = field(default_factory=list)
    
    # Ollama providers (flexible multi-instance list)
    ollama_providers: List[OllamaProvider] = field(default_factory=list)

    
    def get_provider_by_name(self, name: str) -> Optional[OpenAICompatibleProvider]:
        """Get OpenAI-compatible provider by name - Claude Generated"""
        for provider in self.openai_compatible_providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def add_provider(self, provider: OpenAICompatibleProvider) -> bool:
        """Add new OpenAI-compatible provider - Claude Generated"""
        if self.get_provider_by_name(provider.name):
            return False  # Provider already exists
        self.openai_compatible_providers.append(provider)
        return True
    
    def remove_provider(self, name: str) -> bool:
        """Remove OpenAI-compatible provider by name - Claude Generated"""
        for i, provider in enumerate(self.openai_compatible_providers):
            if provider.name.lower() == name.lower():
                del self.openai_compatible_providers[i]
                return True
        return False
    
    def get_enabled_providers(self) -> List[OpenAICompatibleProvider]:
        """Get list of enabled OpenAI-compatible providers - Claude Generated"""
        return [p for p in self.openai_compatible_providers if p.enabled]
    
    # Ollama provider management methods - Claude Generated
    def get_ollama_provider_by_name(self, name: str) -> Optional[OllamaProvider]:
        """Get Ollama provider by name - Claude Generated"""
        for provider in self.ollama_providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def add_ollama_provider(self, provider: OllamaProvider) -> bool:
        """Add new Ollama provider - Claude Generated"""
        if self.get_ollama_provider_by_name(provider.name):
            return False  # Provider already exists
        self.ollama_providers.append(provider)
        return True
    
    def remove_ollama_provider(self, name: str) -> bool:
        """Remove Ollama provider by name - Claude Generated"""
        for i, provider in enumerate(self.ollama_providers):
            if provider.name.lower() == name.lower():
                del self.ollama_providers[i]
                return True
        return False
    
    def get_enabled_ollama_providers(self) -> List[OllamaProvider]:
        """Get list of enabled Ollama providers - Claude Generated"""
        return [p for p in self.ollama_providers if p.enabled]
    
    def get_enabled_openai_providers(self) -> List[OpenAICompatibleProvider]:
        """Get list of enabled OpenAI-compatible providers - Claude Generated"""
        return [p for p in self.openai_compatible_providers if p.enabled]
    
    def get_primary_ollama_provider(self) -> Optional[OllamaProvider]:
        """Get first enabled Ollama provider (primary) - Claude Generated"""
        enabled = self.get_enabled_ollama_providers()
        return enabled[0] if enabled else None
    
    def resolve_provider_type(self, config_name: str) -> tuple[str, dict]:
        """
        Resolve configuration name to provider type and config - Claude Generated
        
        Args:
            config_name: Configuration name (e.g., "LLMachine", "ChatAI", "Gemini")
            
        Returns:
            Tuple of (provider_type, provider_config)
            
        Examples:
            "LLMachine" → ("ollama", {"host": "139.20.140.163", "port": 11434})
            "ChatAI" → ("openai", {"base_url": "...", "api_key": "..."})
            "Gemini" → ("gemini", {"api_key": "..."})
        """
        # Check Ollama providers first
        for ollama_provider in self.get_enabled_ollama_providers():
            if ollama_provider.name == config_name:
                config = {
                    "host": ollama_provider.host,
                    "port": ollama_provider.port,
                    "api_key": ollama_provider.api_key,
                    "use_ssl": ollama_provider.use_ssl,
                    "connection_type": ollama_provider.connection_type
                }
                return ("ollama", config)
                
        # Check OpenAI-compatible providers
        for openai_provider in self.get_enabled_openai_providers():
            if openai_provider.name == config_name:
                config = {
                    "base_url": openai_provider.base_url,
                    "api_key": openai_provider.api_key
                }
                # Return the specific provider name, not generic "openai"
                return (openai_provider.name.lower(), config)
                
        # Check static providers
        if config_name.lower() == "gemini":
            config = {"api_key": self.gemini}
            return ("gemini", config)
        elif config_name.lower() == "anthropic":
            config = {"api_key": self.anthropic}
            return ("anthropic", config)
            
        # Unknown provider - return as-is for backward compatibility
        return (config_name, {})
    
    @classmethod
    def create_default(cls) -> 'LLMConfig':
        """Create LLMConfig with default OpenAI-compatible and Ollama providers - Claude Generated"""
        default_openai_providers = [
            OpenAICompatibleProvider(
                name="OpenAI",
                base_url="https://api.openai.com/v1", 
                api_key="",
                enabled=False,
                description="Official OpenAI API"
            )
        ]
        
        default_ollama_providers = [
            OllamaProvider(
                name="localhost",
                host="localhost",
                port=11434,
                api_key="",
                enabled=True,
                description="Local Ollama instance",
                use_ssl=False,
                connection_type="native_client"
            )
        ]
        
        return cls(
            gemini="",
            gemini_preferred_model="",
            anthropic="",
            anthropic_preferred_model="",
            openai_compatible_providers=default_openai_providers,
            ollama_providers=default_ollama_providers
        )


@dataclass
class CatalogConfig:
    """Library catalog configuration - Claude Generated"""
    catalog_token: str = ''
    catalog_search_url: str = ''
    catalog_details_url: str = ''


@dataclass
class SystemConfig:
    """System-wide configuration - Claude Generated"""
    debug: bool = False
    log_level: str = 'INFO'
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    temp_dir: str = '/tmp'


@dataclass
class ProviderPreferences:
    """Universal LLM provider preferences for all tasks - Claude Generated"""
    
    # Global provider preferences
    preferred_provider: str = "ollama"  # First choice: ollama, gemini, anthropic, openai, chatai, etc.
    provider_priority: List[str] = field(default_factory=lambda: ["ollama", "gemini", "anthropic", "openai"])  # Fallback order
    disabled_providers: List[str] = field(default_factory=list)  # Providers to never use (e.g. ["gemini"])
    
    # Task-specific provider overrides
    vision_provider: Optional[str] = None  # Override for image/vision tasks (gemini-2.0-flash, gpt-4o, claude-3-5-sonnet)
    text_provider: Optional[str] = None    # Override for text-only tasks 
    classification_provider: Optional[str] = None  # Override for classification tasks
    
    # Preferred models per provider
    preferred_models: Dict[str, str] = field(default_factory=lambda: {
        "ollama": "cogito:32b",
        "gemini": "gemini-2.0-flash", 
        "anthropic": "claude-3-5-sonnet",
        "openai": "gpt-4o"
    })
    
    # REMOVED: task_preferences moved to root-level config.task_preferences - Claude Generated
    
    # Fallback behavior
    auto_fallback: bool = True  # Automatically try next provider if preferred fails
    fallback_timeout: int = 30  # Seconds to wait before trying fallback
    
    # Performance preferences
    prefer_faster_models: bool = False  # Prioritize speed over quality
    max_cost_per_request: Optional[float] = None  # Maximum cost threshold (future use)
    
    def get_provider_for_task(self, task_type: str = "general") -> str:
        """Get the preferred provider for a specific task type - Claude Generated"""
        # Check task-specific overrides first
        if task_type == "vision" and self.vision_provider:
            return self.vision_provider
        elif task_type == "text" and self.text_provider:
            return self.text_provider
        elif task_type == "classification" and self.classification_provider:
            return self.classification_provider
        
        # Return global preferred provider
        return self.preferred_provider
    
    def get_provider_priority_for_task(self, task_type: str = "general") -> List[str]:
        """Get the full provider priority list for a task, excluding disabled providers - Claude Generated"""
        task_provider = self.get_provider_for_task(task_type)
        
        # Start with task-specific provider if different from preferred
        if task_provider != self.preferred_provider:
            priority_list = [task_provider] + self.provider_priority
        else:
            priority_list = self.provider_priority[:]
        
        # Remove duplicates and disabled providers
        filtered_list = []
        for provider in priority_list:
            if provider not in self.disabled_providers and provider not in filtered_list:
                filtered_list.append(provider)
        
        return filtered_list
    
    def get_preferred_model(self, provider: str) -> Optional[str]:
        """Get the preferred model for a specific provider - Claude Generated"""
        return self.preferred_models.get(provider)
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled (not in disabled list) - Claude Generated"""
        return provider not in self.disabled_providers
    
    def disable_provider(self, provider: str):
        """Disable a specific provider - Claude Generated"""
        if provider not in self.disabled_providers:
            self.disabled_providers.append(provider)
    
    def enable_provider(self, provider: str):
        """Enable a previously disabled provider - Claude Generated"""
        if provider in self.disabled_providers:
            self.disabled_providers.remove(provider)
    
    # REMOVED: get_model_priority_for_task() - task preferences moved to root-level config - Claude Generated
    
    # REMOVED: set_task_preference() - task preferences moved to root-level config - Claude Generated
    
    # REMOVED: get_task_preference() - task preferences moved to root-level config - Claude Generated
    
    def validate_preferences(self, detection_service: 'ProviderDetectionService') -> Dict[str, List[str]]:
        """Validate all provider preferences against available providers - Claude Generated"""
        validation_issues = {
            'missing_providers': [],
            'invalid_task_overrides': [],
            'invalid_models': [],
            'warnings': []
        }
        
        available_providers = detection_service.get_available_providers()
        
        # Validate preferred provider
        if self.preferred_provider not in available_providers:
            validation_issues['missing_providers'].append(f"Preferred provider '{self.preferred_provider}' not available")
        
        # Validate provider priority list
        for provider in self.provider_priority:
            if provider not in available_providers:
                validation_issues['missing_providers'].append(f"Priority provider '{provider}' not available")
        
        # Validate task-specific overrides
        task_providers = {
            'vision': self.vision_provider,
            'text': self.text_provider,
            'classification': self.classification_provider
        }
        
        for task, provider in task_providers.items():
            if provider and provider not in available_providers:
                validation_issues['invalid_task_overrides'].append(f"Task '{task}' provider '{provider}' not available")
        
        # Validate preferred models
        for provider, model in self.preferred_models.items():
            if provider in available_providers:
                available_models = detection_service.get_available_models(provider)
                if available_models and model not in available_models:
                    validation_issues['invalid_models'].append(f"Model '{model}' not available for provider '{provider}'")
            else:
                validation_issues['warnings'].append(f"Cannot validate model '{model}' - provider '{provider}' not available")
        
        return validation_issues
    
    def auto_cleanup(self, detection_service: 'ProviderDetectionService') -> Dict[str, Any]:
        """Auto-cleanup invalid provider references - Claude Generated"""
        cleanup_report = {
            'cleaned_providers': [],
            'updated_preferred': None,
            'cleaned_models': [],
            'cleaned_task_overrides': []
        }
        
        available_providers = detection_service.get_available_providers()
        
        # Clean provider priority list - remove unavailable providers
        original_priority = self.provider_priority[:]
        self.provider_priority = [p for p in self.provider_priority if p in available_providers]
        
        for removed in set(original_priority) - set(self.provider_priority):
            cleanup_report['cleaned_providers'].append(f"Removed unavailable provider '{removed}' from priority list")
        
        # Update preferred provider if not available
        if self.preferred_provider not in available_providers:
            if self.provider_priority:
                old_preferred = self.preferred_provider
                self.preferred_provider = self.provider_priority[0]
                cleanup_report['updated_preferred'] = f"Changed preferred provider from '{old_preferred}' to '{self.preferred_provider}'"
            else:
                # Fallback to first available provider
                if available_providers:
                    old_preferred = self.preferred_provider
                    self.preferred_provider = available_providers[0]
                    cleanup_report['updated_preferred'] = f"Changed preferred provider from '{old_preferred}' to '{self.preferred_provider}' (fallback)"
        
        # Clean task-specific overrides
        task_overrides = [
            ('vision', 'vision_provider'),
            ('text', 'text_provider'), 
            ('classification', 'classification_provider')
        ]
        
        for task_name, attr_name in task_overrides:
            provider = getattr(self, attr_name)
            if provider and provider not in available_providers:
                setattr(self, attr_name, None)
                cleanup_report['cleaned_task_overrides'].append(f"Cleared unavailable {task_name} provider '{provider}'")
        
        # Clean preferred models for unavailable providers
        models_to_remove = []
        for provider in list(self.preferred_models.keys()):
            if provider not in available_providers:
                models_to_remove.append(provider)
        
        for provider in models_to_remove:
            model = self.preferred_models.pop(provider)
            cleanup_report['cleaned_models'].append(f"Removed model preference '{model}' for unavailable provider '{provider}'")
        
        # Update disabled providers list - remove providers that are not available anymore
        original_disabled = self.disabled_providers[:]
        self.disabled_providers = [p for p in self.disabled_providers if p in available_providers]
        
        for removed in set(original_disabled) - set(self.disabled_providers):
            cleanup_report['cleaned_providers'].append(f"Removed unavailable provider '{removed}' from disabled list")
        
        return cleanup_report
    
    def ensure_valid_configuration(self, detection_service: 'ProviderDetectionService') -> bool:
        """Ensure preferences have at least one valid provider configured - Claude Generated"""
        available_providers = detection_service.get_available_providers()
        
        if not available_providers:
            return False  # No providers available at all
        
        # Make sure we have at least one provider in our priority list
        if not any(p in available_providers for p in self.provider_priority):
            # Add all available providers to priority list
            self.provider_priority = available_providers[:]
        
        # Make sure preferred provider is valid
        if self.preferred_provider not in available_providers:
            if self.provider_priority:
                self.preferred_provider = self.provider_priority[0]
            else:
                self.preferred_provider = available_providers[0]
        
        return True


@dataclass
class AlimaConfig:
    """Complete ALIMA configuration - Claude Generated"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=lambda: LLMConfig.create_default())
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    provider_preferences: ProviderPreferences = field(default_factory=ProviderPreferences)  # Claude Generated
    
    # Task-specific model preferences for Smart Mode - Claude Generated
    task_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Version and metadata
    config_version: str = '1.0'
    last_updated: str = ''


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
    
    def has_provider_api_key(self, provider: str) -> bool:
        """Check if provider has API key configured - Claude Generated"""
        try:
            llm_service = self._get_llm_service()
            return llm_service.has_provider_api_key(provider)
        except Exception as e:
            self.logger.warning(f"Error checking API key for {provider}: {e}")
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
    
    def detect_provider_capabilities(self, provider: str) -> List[str]:
        """Detect capabilities of a provider based on available models and configuration - Claude Generated"""
        capabilities = []
        
        try:
            models = self.get_available_models(provider)
            models_str = ' '.join(models).lower()
            
            # Vision capability detection
            vision_indicators = ['vision', 'gpt-4o', 'claude-3', 'gemini-2.0', 'llava', 'minicpm-v']
            if any(indicator in models_str for indicator in vision_indicators):
                capabilities.append('vision')
            
            # Speed capability (based on model names)
            speed_indicators = ['flash', 'mini', 'haiku', '14b', 'turbo']
            if any(indicator in models_str for indicator in speed_indicators):
                capabilities.append('fast')
            
            # Large context capability
            large_context_indicators = ['gpt-4', 'claude-3', 'gemini-1.5', 'cogito:32b']
            if any(indicator in models_str for indicator in large_context_indicators):
                capabilities.append('large_context')
            
            # Provider-specific capabilities
            if provider.startswith('ollama') or 'ollama' in provider.lower():
                capabilities.extend(['local', 'privacy', 'custom_models'])
            
            if 'gemini' in provider.lower():
                capabilities.extend(['multimodal', 'google'])
            
            if 'anthropic' in provider.lower():
                capabilities.extend(['reasoning', 'analysis'])
            
            if 'openai' in provider.lower():
                capabilities.extend(['function_calling', 'structured_output'])
            
            if 'chatai' in provider.lower():
                capabilities.extend(['academic', 'german'])
            
        except Exception as e:
            self.logger.warning(f"Error detecting capabilities for {provider}: {e}")
        
        return list(set(capabilities))  # Remove duplicates
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get comprehensive information about a provider - Claude Generated"""
        info = {
            'name': provider,
            'available': provider in self.get_available_providers(),
            'has_api_key': False,
            'reachable': False,
            'models': [],
            'model_count': 0,
            'capabilities': [],
            'description': '',
            'status': 'unknown'
        }
        
        # Check API key availability first
        info['has_api_key'] = self.has_provider_api_key(provider)
        
        if not info['available']:
            info['status'] = 'not_configured'
            return info
        
        # Test reachability
        info['reachable'] = self.is_provider_reachable(provider)
        
        if info['reachable']:
            info['status'] = 'ready'
            info['models'] = self.get_available_models(provider)
            info['model_count'] = len(info['models'])
            info['capabilities'] = self.detect_provider_capabilities(provider)
            
            # Generate description
            caps_str = ', '.join(info['capabilities'][:3])
            model_info = f"{info['model_count']} models"
            info['description'] = f"{caps_str} | {model_info}" if caps_str else model_info
        else:
            # Provider configured but unreachable
            if info['has_api_key']:
                info['status'] = 'configured_unreachable'
                info['description'] = "API key configured but unreachable"
            else:
                info['status'] = 'unreachable'
                info['description'] = "Not configured properly"
        
        return info
    
    def get_all_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers - Claude Generated"""
        providers = self.get_available_providers()
        return {provider: self.get_provider_info(provider) for provider in providers}
    
    def get_providers_with_capability(self, capability: str) -> List[str]:
        """Get all providers that have a specific capability - Claude Generated"""
        matching_providers = []
        
        for provider in self.get_available_providers():
            if capability in self.detect_provider_capabilities(provider):
                matching_providers.append(provider)
        
        return matching_providers
    
    def get_vision_providers(self) -> List[str]:
        """Get all providers with vision capabilities - Claude Generated"""
        return self.get_providers_with_capability('vision')
    
    def get_local_providers(self) -> List[str]:
        """Get all local providers (typically Ollama) - Claude Generated"""
        return self.get_providers_with_capability('local')
    
    def validate_provider_list(self, providers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of providers against available providers - Claude Generated
        
        Returns:
            Tuple of (valid_providers, invalid_providers)
        """
        available = set(self.get_available_providers())
        valid = [p for p in providers if p in available]
        invalid = [p for p in providers if p not in available]
        
        return valid, invalid
    
    def cleanup_provider_preferences(self, preferences: 'ProviderPreferences') -> 'ProviderPreferences':
        """Clean up provider preferences by removing unavailable providers - Claude Generated"""
        available = set(self.get_available_providers())
        
        # Clean up provider priority
        preferences.provider_priority = [p for p in preferences.provider_priority if p in available]
        
        # Clean up disabled providers (only keep ones that exist)
        preferences.disabled_providers = [p for p in preferences.disabled_providers if p in available]
        
        # Clean up preferred provider
        if preferences.preferred_provider not in available:
            if preferences.provider_priority:
                preferences.preferred_provider = preferences.provider_priority[0]
            elif available:
                preferences.preferred_provider = list(available)[0]
            else:
                preferences.preferred_provider = "ollama"  # fallback
        
        # Clean up task-specific overrides
        if preferences.vision_provider and preferences.vision_provider not in available:
            preferences.vision_provider = None
        
        if preferences.text_provider and preferences.text_provider not in available:
            preferences.text_provider = None
        
        if preferences.classification_provider and preferences.classification_provider not in available:
            preferences.classification_provider = None
        
        # Clean up preferred models
        for provider in list(preferences.preferred_models.keys()):
            if provider not in available:
                del preferences.preferred_models[provider]
        
        return preferences


class ConfigManager:
    """Manages all ALIMA configuration with OS-specific paths and priority fallback - Claude Generated"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Get OS-specific configuration paths
        self._setup_config_paths()
        
        self._config: Optional[AlimaConfig] = None
        self._provider_detection_service: Optional[ProviderDetectionService] = None  # Claude Generated
    
    def _setup_config_paths(self):
        """Setup OS-specific configuration file paths - Claude Generated"""
        system_name = platform.system().lower()
        
        # Project config is always in current directory
        self.project_config_path = Path("alima_config.json")
        
        if system_name == "windows":
            # Windows paths
            appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
            local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
            programdata = os.environ.get("PROGRAMDATA", "C:\\ProgramData")
            
            self.user_config_path = Path(appdata) / "ALIMA" / "config.json"
            self.system_config_path = Path(programdata) / "ALIMA" / "config.json"
            
        elif system_name == "darwin":  # macOS
            # macOS paths following Apple guidelines
            self.user_config_path = Path.home() / "Library" / "Application Support" / "ALIMA" / "config.json"
            self.system_config_path = Path("/Library") / "Application Support" / "ALIMA" / "config.json"
            
        else:  # Linux and other Unix-like systems
            # Follow XDG Base Directory Specification
            xdg_config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
            xdg_config_dirs = os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")
            
            self.user_config_path = Path(xdg_config_home) / "alima" / "config.json"
            # Use first directory from XDG_CONFIG_DIRS, fallback to /etc
            system_config_dir = Path(xdg_config_dirs[0]) if xdg_config_dirs else Path("/etc")
            self.system_config_path = system_config_dir / "alima" / "config.json"
        
        self.logger.debug(f"Config paths for {system_name}:")
        self.logger.debug(f"  Project: {self.project_config_path}")
        self.logger.debug(f"  User: {self.user_config_path}")
        self.logger.debug(f"  System: {self.system_config_path}")
    
    def get_config_info(self) -> Dict[str, str]:
        """Get information about configuration paths - Claude Generated"""
        system_name = platform.system()
        return {
            "os": system_name,
            "project_config": str(self.project_config_path),
            "user_config": str(self.user_config_path),
            "system_config": str(self.system_config_path),
        }
        
    def load_config(self, force_reload: bool = False) -> AlimaConfig:
        """Load configuration with priority fallback - Claude Generated"""
        if self._config is not None and not force_reload:
            # 🔍 DEBUG: Log cache hit - Claude Generated
            self.logger.critical(f"🔍 CONFIG_LOAD_CACHE_HIT: Using cached config")
            return self._config
        elif force_reload:
            # 🔍 DEBUG: Log forced reload - Claude Generated
            self.logger.critical(f"🔍 CONFIG_FORCE_RELOAD: Forcing config reload from file")
            self._config = None  # Clear cache
            
        # Try loading from different sources (priority order)
        config_sources = [
            ("project", self.project_config_path),
            ("user", self.user_config_path), 
            ("system", self.system_config_path),
        ]
        
        for source_name, config_path in config_sources:
            if config_path.exists():
                try:
                    config_data = self._load_config_file(config_path)
                    if config_data:
                        self._config = self._parse_config(config_data, source_name)
                        # 🔍 DEBUG: Log loaded preferred model values - Claude Generated
                        self.logger.critical(f"🔍 CONFIG_LOADED_FROM_FILE: {config_path}")
                        self.logger.critical(f"🔍 LOADED_gemini_preferred_model: '{self._config.llm.gemini_preferred_model}'")
                        self.logger.critical(f"🔍 LOADED_anthropic_preferred_model: '{self._config.llm.anthropic_preferred_model}'")
                        self.logger.info(f"Loaded config from {source_name}: {config_path}")
                        return self._config
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {source_name} ({config_path}): {e}")
                    continue
        
        # Use default configuration if nothing found
        self.logger.info("Using default configuration")
        self._config = AlimaConfig()
        return self._config
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON config file - Claude Generated"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config ({config_path}): {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading config ({config_path}): {e}")
            return None
    
    def _parse_config(self, config_data: Dict[str, Any], source_name: str) -> AlimaConfig:
        """Parse configuration data into AlimaConfig - Claude Generated"""
        

        # Parse new format
        config = AlimaConfig()
        
        # Database section
        if "database" in config_data:
            db_data = config_data["database"]
            config.database = DatabaseConfig(
                db_type=db_data.get("db_type", "sqlite"),
                sqlite_path=db_data.get("sqlite_path", "alima_knowledge.db"),
                host=db_data.get("host", "localhost"),
                port=db_data.get("port", 3306),
                database=db_data.get("database", "alima_knowledge"),
                username=db_data.get("username", "alima"),
                password=db_data.get("password", ""),
                connection_timeout=db_data.get("connection_timeout", 30),
                auto_create_tables=db_data.get("auto_create_tables", True),
                charset=db_data.get("charset", "utf8mb4"),
                ssl_disabled=db_data.get("ssl_disabled", False)
            )
        
        # LLM section
        if "llm" in config_data:
            llm_data = config_data["llm"]
            
            # Parse OpenAI-compatible providers
            openai_providers = []
            if "openai_compatible_providers" in llm_data:
                # New format: list of provider objects
                for provider_data in llm_data["openai_compatible_providers"]:
                    try:
                        provider = OpenAICompatibleProvider(
                            name=provider_data.get("name", ""),
                            base_url=provider_data.get("base_url", ""),
                            api_key=provider_data.get("api_key", ""),
                            enabled=provider_data.get("enabled", True),
                            models=provider_data.get("models", []),
                            preferred_model=provider_data.get("preferred_model", ""),  # Include preferred_model - Claude Generated
                            description=provider_data.get("description", "")
                        )
                        openai_providers.append(provider)
                    except ValueError as e:
                        self.logger.warning(f"Skipping invalid provider: {e}")
            
            # Parse Ollama providers (new multi-instance format)
            ollama_providers = []
            if "ollama_providers" in llm_data:
                # New format: list of Ollama provider objects
                for provider_data in llm_data["ollama_providers"]:
                    try:
                        provider = OllamaProvider(
                            name=provider_data.get("name", ""),
                            host=provider_data.get("host", "localhost"),
                            port=provider_data.get("port", 11434),
                            api_key=provider_data.get("api_key", ""),
                            enabled=provider_data.get("enabled", True),
                            preferred_model=provider_data.get("preferred_model", ""),  # Include preferred_model - Claude Generated
                            description=provider_data.get("description", ""),
                            use_ssl=provider_data.get("use_ssl", False),
                            connection_type=provider_data.get("connection_type", "native_client")
                        )
                        ollama_providers.append(provider)
                    except ValueError as e:
                        self.logger.warning(f"Skipping invalid Ollama provider: {e}")
            
            config.llm = LLMConfig(
                gemini=llm_data.get("gemini", ""),
                gemini_preferred_model=llm_data.get("gemini_preferred_model", ""),
                anthropic=llm_data.get("anthropic", ""),
                anthropic_preferred_model=llm_data.get("anthropic_preferred_model", ""),
                openai_compatible_providers=openai_providers,
                ollama_providers=ollama_providers
            )
        
        # Catalog section
        if "catalog" in config_data:
            cat_data = config_data["catalog"]
            config.catalog = CatalogConfig(
                catalog_token=cat_data.get("catalog_token", ""),
                catalog_search_url=cat_data.get("catalog_search_url", ""),
                catalog_details_url=cat_data.get("catalog_details_url", "")
            )
        
        # System section  
        if "system" in config_data:
            sys_data = config_data["system"]
            config.system = SystemConfig(
                debug=sys_data.get("debug", False),
                log_level=sys_data.get("log_level", "INFO"),
                cache_dir=sys_data.get("cache_dir", "cache"),
                data_dir=sys_data.get("data_dir", "data"),
                temp_dir=sys_data.get("temp_dir", "/tmp")
            )
        
        # Provider preferences section - Claude Generated
        if "provider_preferences" in config_data:
            pref_data = config_data["provider_preferences"]
            config.provider_preferences = ProviderPreferences(
                preferred_provider=pref_data.get("preferred_provider", "ollama"),
                provider_priority=pref_data.get("provider_priority", ["ollama", "gemini", "anthropic", "openai"]),
                disabled_providers=pref_data.get("disabled_providers", []),
                vision_provider=pref_data.get("vision_provider"),
                text_provider=pref_data.get("text_provider"),
                classification_provider=pref_data.get("classification_provider"),
                preferred_models=pref_data.get("preferred_models", {
                    "ollama": "cogito:32b",
                    "gemini": "gemini-2.0-flash", 
                    "anthropic": "claude-3-5-sonnet",
                    "openai": "gpt-4o"
                }),
                # REMOVED: task_preferences loading - moved to root-level config - Claude Generated
                auto_fallback=pref_data.get("auto_fallback", True),
                fallback_timeout=pref_data.get("fallback_timeout", 30),
                prefer_faster_models=pref_data.get("prefer_faster_models", False),
                max_cost_per_request=pref_data.get("max_cost_per_request")
            )
        
        # Task preferences for Smart Mode - Claude Generated
        config.task_preferences = config_data.get("task_preferences", {})
        
        # Metadata
        config.config_version = config_data.get("config_version", "1.0")
        config.last_updated = config_data.get("last_updated", "")
        
        return config
    
    def save_config(self, config: AlimaConfig, scope: str = "user") -> bool:
        """Save configuration - Claude Generated"""
        try:
            # 🔍 DEBUG: Log save operation start - Claude Generated
            self.logger.critical(f"🔍 CONFIG_SAVE_START: scope='{scope}'")
            self.logger.critical(f"🔍 SAVE_gemini_preferred_model: '{config.llm.gemini_preferred_model}'")
            self.logger.critical(f"🔍 SAVE_anthropic_preferred_model: '{config.llm.anthropic_preferred_model}'")
            
            if scope == "system":
                config_path = self.system_config_path
            elif scope == "project":
                config_path = self.project_config_path
            else:  # user (default)
                config_path = self.user_config_path
            
            # Create directory if needed for all scopes
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            from datetime import datetime
            config.last_updated = datetime.now().isoformat()
            
            # Create configuration structure
            config_dict = asdict(config)
            config_data = {
                **config_dict,
                "_metadata": {
                    "version": config.config_version,
                    "created_by": "ALIMA Configuration Manager",
                    "scope": scope
                }
            }
            
            # Write configuration file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 🔍 DEBUG: Log successful save and cache update - Claude Generated
            self.logger.critical(f"🔍 CONFIG_WRITTEN_TO_FILE: {config_path}")
            
            # 🔍 DEBUG: Verify JSON file content after writing - Claude Generated
            self._debug_verify_json_content(config_path)
            
            self.logger.info(f"Saved config to {scope}: {config_path}")
            
            # Update cached config
            old_cache = self._config
            self._config = config
            self.logger.critical(f"🔍 CONFIG_CACHE_UPDATED: old_cache={'None' if old_cache is None else 'exists'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {scope} config: {e}")
            return False
    
    def _debug_verify_json_content(self, config_path: Path):
        """Debug method to verify JSON file contains expected preferred_model fields - Claude Generated"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            # Check if llm section exists
            if 'llm' in saved_data:
                llm_data = saved_data['llm']
                
                # Check static providers
                gemini_pref = llm_data.get('gemini_preferred_model', 'MISSING')
                anthropic_pref = llm_data.get('anthropic_preferred_model', 'MISSING')
                
                self.logger.critical(f"🔍 JSON_FILE_CONTENT: gemini_preferred_model='{gemini_pref}'")
                self.logger.critical(f"🔍 JSON_FILE_CONTENT: anthropic_preferred_model='{anthropic_pref}'")
                
                # Check OpenAI-compatible providers
                if 'openai_compatible_providers' in llm_data:
                    openai_providers = llm_data['openai_compatible_providers']
                    for i, provider in enumerate(openai_providers):
                        pref_model = provider.get('preferred_model', 'MISSING')
                        provider_name = provider.get('name', f'provider_{i}')
                        self.logger.critical(f"🔍 JSON_FILE_CONTENT: openai_compatible[{provider_name}].preferred_model='{pref_model}'")
                
                # Check Ollama providers
                if 'ollama_providers' in llm_data:
                    ollama_providers = llm_data['ollama_providers']
                    for i, provider in enumerate(ollama_providers):
                        pref_model = provider.get('preferred_model', 'MISSING')
                        provider_name = provider.get('name', f'provider_{i}')
                        self.logger.critical(f"🔍 JSON_FILE_CONTENT: ollama[{provider_name}].preferred_model='{pref_model}'")
            else:
                self.logger.critical(f"🔍 JSON_FILE_CONTENT: llm section MISSING from saved file!")
                
        except Exception as e:
            self.logger.error(f"🔍 JSON_VERIFICATION_ERROR: {e}")
    
    def invalidate_cache(self):
        """Invalidate the configuration cache to force reload - Claude Generated"""
        self.logger.critical(f"🔍 CONFIG_CACHE_INVALIDATED: Forcing next load_config() to reload from file")
        self._config = None
    
    def get_database_connection_string(self) -> str:
        """Get database connection string - Claude Generated"""
        config = self.load_config()
        db = config.database
        
        if db.db_type == 'sqlite':
            return f"sqlite:///{db.sqlite_path}"
        elif db.db_type in ['mysql', 'mariadb']:
            ssl_part = "?ssl_disabled=true" if db.ssl_disabled else ""
            return f"mysql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}{ssl_part}"
        else:
            raise ValueError(f"Unsupported database type: {db.db_type}")
    
    def get_database_connection_info(self) -> Dict[str, Any]:
        """Get database connection info for direct access - Claude Generated"""
        config = self.load_config()
        db = config.database
        
        if db.db_type == 'sqlite':
            return {
                'type': 'sqlite',
                'path': db.sqlite_path,
                'connection_timeout': db.connection_timeout
            }
        elif db.db_type in ['mysql', 'mariadb']:
            return {
                'type': db.db_type,
                'host': db.host,
                'port': db.port,
                'database': db.database,
                'username': db.username,
                'password': db.password,
                'charset': db.charset,
                'connection_timeout': db.connection_timeout,
                'ssl_disabled': db.ssl_disabled
            }
        else:
            raise ValueError(f"Unsupported database type: {db.db_type}")
    
    def test_database_connection(self) -> tuple[bool, str]:
        """Test database connection - Claude Generated"""
        try:
            config = self.load_config()
            db = config.database
            
            if db.db_type == 'sqlite':
                import sqlite3
                conn = sqlite3.connect(db.sqlite_path, timeout=db.connection_timeout)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                return True, f"SQLite connection successful: {db.sqlite_path}"
                
            elif db.db_type in ['mysql', 'mariadb']:
                try:
                    import pymysql
                except ImportError:
                    return False, "PyMySQL not installed. Install with: pip install pymysql"
                
                connection = pymysql.connect(
                    host=db.host,
                    port=db.port,
                    user=db.username,
                    password=db.password,
                    database=db.database,
                    charset=db.charset,
                    connect_timeout=db.connection_timeout,
                    ssl_disabled=db.ssl_disabled
                )
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                connection.close()
                return True, f"MySQL connection successful: {db.username}@{db.host}:{db.port}/{db.database}"
                
            else:
                return False, f"Unsupported database type: {db.db_type}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_catalog_config(self) -> Dict[str, Any]:
        """Get catalog configuration in unified format - Claude Generated"""
        config = self.load_config()
        cat = config.catalog
        
        return {
            "catalog_token": cat.catalog_token,
            "catalog_search_url": cat.catalog_search_url,
            "catalog_details_url": cat.catalog_details_url  # Unified key name - Claude Generated
        }
    
    def get_provider_preferences(self) -> ProviderPreferences:
        """Get provider preferences configuration - Claude Generated"""
        config = self.load_config()
        return config.provider_preferences
    
    def update_provider_preferences(self, preferences: ProviderPreferences) -> bool:
        """Update provider preferences and save configuration - Claude Generated"""
        try:
            config = self.load_config()
            config.provider_preferences = preferences
            return self.save_config(config)
        except Exception as e:
            self.logger.error(f"Failed to update provider preferences: {e}")
            return False
    
    def get_provider_detection_service(self) -> ProviderDetectionService:
        """Get provider detection service instance - Claude Generated"""
        if self._provider_detection_service is None:
            self._provider_detection_service = ProviderDetectionService(config_manager=self)
        return self._provider_detection_service
    
    def get_available_providers(self) -> List[str]:
        """Get list of all available providers from internal ALIMA logic - Claude Generated"""
        return self.get_provider_detection_service().get_available_providers()
    
    def get_all_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all available providers - Claude Generated"""
        return self.get_provider_detection_service().get_all_provider_info()
    
    def cleanup_provider_preferences_auto(self) -> bool:
        """Automatically cleanup provider preferences by removing unavailable providers - Claude Generated"""
        try:
            preferences = self.get_provider_preferences()
            detection_service = self.get_provider_detection_service()
            cleaned_preferences = detection_service.cleanup_provider_preferences(preferences)
            return self.update_provider_preferences(cleaned_preferences)
        except Exception as e:
            self.logger.error(f"Failed to cleanup provider preferences: {e}")
            return False
    
    def create_sample_configs(self) -> Dict[str, str]:
        """Create sample configuration files - Claude Generated"""
        samples = {}
        
        # Complete sample config
        sample_config = AlimaConfig(
            database=DatabaseConfig(
                db_type='sqlite',
                sqlite_path='alima_knowledge.db'
            ),
            llm=LLMConfig(
                gemini='your_gemini_api_key_here',
                anthropic='your_anthropic_api_key_here',
                openai='your_openai_api_key_here'
            ),
            catalog=CatalogConfig(
                catalog_token='your_catalog_token_here',
                catalog_search_url='https://your-catalog-server.com/search',
                catalog_details_url='https://your-catalog-server.com/details'
            ),
            system=SystemConfig(
                debug=False,
                log_level='INFO',
                cache_dir='cache',
                data_dir='data'
            )
        )
        
        samples['complete'] = json.dumps(asdict(sample_config), indent=2)
        
        # MySQL variant
        mysql_config = AlimaConfig(
            database=DatabaseConfig(
                db_type='mysql',
                host='localhost',
                port=3306,
                database='alima_knowledge',
                username='alima',
                password='your_password_here',
                charset='utf8mb4'
            )
        )
        samples['mysql'] = json.dumps(asdict(mysql_config), indent=2)
        
        return samples


# Global instance for easy access
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager - Claude Generated"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience functions for backward compatibility
def get_config() -> AlimaConfig:
    """Get current configuration - Claude Generated"""
    return get_config_manager().load_config()

#def get_llm_config() -> Dict[str, Any]:
#    """Get LLM configuration - Claude Generated"""
#    return get_config_manager().get_llm_config()

def get_catalog_config() -> Dict[str, Any]: 
    """Get catalog configuration - Claude Generated"""
    return get_config_manager().get_catalog_config()


if __name__ == "__main__":
    # Demo/test functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="ALIMA Configuration Manager")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")
    parser.add_argument("--show-paths", action="store_true", help="Show OS-specific configuration paths")
    parser.add_argument("--test-db", action="store_true", help="Test database connection")
    parser.add_argument("--create-samples", action="store_true", help="Show sample configurations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    manager = ConfigManager()
    
    if args.show_config:
        config = manager.load_config()
        print("Current Configuration:")
        print(json.dumps(asdict(config), indent=2))
    
    if args.show_paths:
        config_info = manager.get_config_info()
        print(f"🖥️  Configuration Paths for {config_info['os']}:")
        print(f"   Project:  {config_info['project_config']}")
        print(f"   User:     {config_info['user_config']}")
        print(f"   System:   {config_info['system_config']}")
        print()
        
        # Show which files exist
        from pathlib import Path
        paths = [
            ("Project", config_info['project_config']),
            ("User", config_info['user_config']),
            ("System", config_info['system_config']),
        ]
        
        print("📁 File Status:")
        for name, path in paths:
            exists = Path(path).exists()
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"   {name:8}: {status}")
    
    if args.test_db:
        success, message = manager.test_database_connection()
        print(f"Database Test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Message: {message}")
    
    if args.create_samples:
        samples = manager.create_sample_configs()
        print("=== Complete Configuration Sample ===")
        print(samples['complete'])
        print("\n=== MySQL Configuration Sample ===")
        print(samples['mysql'])
    