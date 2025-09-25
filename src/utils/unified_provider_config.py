#!/usr/bin/env python3
"""
TEMPORARY BRIDGE FILE - unified_provider_config.py
This file is temporarily restored to prevent ImportErrors from 11 files.
All imports are redirected to the new centralized config_models.py

This file will be removed once all import references are updated.
Claude Generated - Bridge for migration
"""

import logging
from typing import Dict, List, Optional, Any

# Import all classes from the new centralized location
from .config_models import (
    # Core enums and types
    PipelineMode,
    TaskType,

    # Data classes
    TaskPreference,
    PipelineStepConfig,
    UnifiedProvider,
    UnifiedProviderConfig,

    # Provider-specific classes
    OllamaProvider,
    OpenAICompatibleProvider,
    GeminiProvider,
    AnthropicProvider
)

# Bridge for the config manager function
def get_unified_config_manager(config_manager=None):
    """
    TEMPORARY BRIDGE FUNCTION
    Returns the config manager itself since UnifiedProviderConfigManager
    functionality is now integrated into ConfigManager.
    """
    if config_manager is None:
        from .config_manager import ConfigManager
        config_manager = ConfigManager()

    return config_manager


class UnifiedProviderConfigManager:
    """
    TEMPORARY BRIDGE CLASS - UnifiedProviderConfigManager
    This class now delegates all functionality to the main ConfigManager.
    """

    def __init__(self, config_manager=None):
        """Initialize with bridge to main ConfigManager"""
        self.logger = logging.getLogger(__name__)
        if config_manager is None:
            from .config_manager import ConfigManager
            config_manager = ConfigManager()
        self.config_manager = config_manager
        self._unified_config: Optional[UnifiedProviderConfig] = None

    def get_unified_config(self) -> UnifiedProviderConfig:
        """Get unified configuration - delegates to ConfigManager"""
        return self.config_manager.get_unified_config()

    def save_unified_config(self, unified_config: UnifiedProviderConfig) -> bool:
        """Save unified configuration - delegates to ConfigManager"""
        return self.config_manager.save_unified_config(unified_config)

    def validate_config(self, unified_config: Optional[UnifiedProviderConfig] = None) -> List[str]:
        """Validate unified configuration - delegates to ConfigManager"""
        return self.config_manager.validate_unified_config(unified_config)


# Backwards compatibility exports
__all__ = [
    'PipelineMode',
    'TaskType',
    'TaskPreference',
    'PipelineStepConfig',
    'UnifiedProvider',
    'UnifiedProviderConfig',
    'UnifiedProviderConfigManager',
    'OllamaProvider',
    'OpenAICompatibleProvider',
    'GeminiProvider',
    'AnthropicProvider',
    'get_unified_config_manager'
]