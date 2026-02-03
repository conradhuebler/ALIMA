#!/usr/bin/env python3
"""
SmartProviderSelector - Simplified LLM Provider Selection Engine
Selects providers based on explicit overrides or pipeline defaults.
Claude Generated - Refactored for Provider Strategy Simplification
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .config_manager import ConfigManager, ProviderDetectionService
from .config_models import UnifiedProviderConfig, TaskType


@dataclass
class ProviderAttempt:
    """Record of a provider selection attempt - Claude Generated"""
    provider: str
    model: str
    success: bool
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SmartSelection:
    """Result of provider selection - Claude Generated"""
    provider: str
    model: str
    config: Dict[str, Any]
    attempts: List[ProviderAttempt]
    fallback_used: bool = False
    selection_time: float = 0.0

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

    @property
    def successful_attempts(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.success)


class SmartProviderSelector:
    """
    Simplified LLM provider selection engine - Claude Generated

    2-Tier Selection Strategy:
    - TIER 1: Explicit override (from UI/CLI)
    - TIER 2: Pipeline default from config

    Fallback: First available provider if no default set
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self.provider_detection_service = self.config_manager.get_provider_detection_service()

        # Load unified config
        try:
            self.unified_config = self.config_manager.get_unified_config()
            self.logger.debug("SmartProviderSelector initialized")
        except Exception as e:
            self.logger.warning(f"Failed to load unified config: {e}")
            self.unified_config = None

        # Simple availability cache
        self._availability_cache: Dict[str, Tuple[bool, float]] = {}
        self._cache_timeout = 300  # 5 minutes

    def select_provider(
        self,
        task_type: TaskType = TaskType.GENERAL,
        required_capabilities: Optional[List[str]] = None,
        prefer_fast: bool = False,
        task_name: str = "",
        step_id: str = "",
        explicit_provider: Optional[str] = None,
        explicit_model: Optional[str] = None,
    ) -> SmartSelection:
        """
        Select provider using simplified 2-tier strategy - Claude Generated

        Args:
            task_type: Type of LLM task (for logging only now)
            required_capabilities: Not used (kept for API compatibility)
            prefer_fast: Not used (kept for API compatibility)
            task_name: Task name for logging
            step_id: Pipeline step ID for logging
            explicit_provider: Explicit provider override (TIER 1)
            explicit_model: Explicit model override (TIER 1)

        Returns:
            SmartSelection with provider, model, config
        """
        start_time = time.time()
        unified_config = self.config_manager.get_unified_config()
        attempts = []

        # === TIER 1: Explicit override ===
        if explicit_provider:
            self.logger.info(f"🎯 TIER 1: Explicit override: {explicit_provider}/{explicit_model or 'auto'}")
            return self._try_provider(
                explicit_provider, explicit_model, attempts, start_time, unified_config
            )

        # === TIER 2: Pipeline default from config ===
        default_provider = unified_config.pipeline_default_provider
        default_model = unified_config.pipeline_default_model

        if default_provider:
            self.logger.info(f"⚙️ TIER 2: Pipeline default: {default_provider}/{default_model or 'auto'}")
            try:
                return self._try_provider(
                    default_provider, default_model, attempts, start_time, unified_config
                )
            except RuntimeError:
                self.logger.warning(f"Pipeline default provider '{default_provider}' failed, trying fallback")

        # === Fallback: First available provider ===
        enabled_providers = unified_config.get_enabled_providers()
        for provider_obj in enabled_providers:
            provider_name = provider_obj.name
            if self._is_provider_available(provider_name):
                self.logger.info(f"📋 Fallback: Using first available provider: {provider_name}")
                return self._try_provider(
                    provider_name,
                    provider_obj.preferred_model,
                    attempts,
                    start_time,
                    unified_config
                )

        # All providers failed
        error_msg = self._build_error_message(task_name, task_type, attempts)
        raise RuntimeError(error_msg)

    def _try_provider(
        self,
        provider: str,
        model: Optional[str],
        attempts: List[ProviderAttempt],
        start_time: float,
        unified_config: UnifiedProviderConfig
    ) -> SmartSelection:
        """Try to use a specific provider - Claude Generated"""

        if not self._is_provider_available(provider):
            attempts.append(ProviderAttempt(
                provider=provider,
                model=model or "",
                success=False,
                error_message="Provider not available"
            ))
            raise RuntimeError(f"Provider '{provider}' is not available")

        # Get model if not specified
        if not model:
            model = self._get_model_for_provider(provider, unified_config)

        if not model:
            attempts.append(ProviderAttempt(
                provider=provider,
                model="",
                success=False,
                error_message="No model available"
            ))
            raise RuntimeError(f"No model available for provider '{provider}'")

        # Get provider configuration
        config = self._get_provider_config(provider, unified_config)

        # Validate configuration
        success, error = self._validate_provider_config(provider, model, config)
        if not success:
            attempts.append(ProviderAttempt(
                provider=provider,
                model=model,
                success=False,
                error_message=error
            ))
            raise RuntimeError(f"Provider '{provider}' validation failed: {error}")

        # Success
        attempts.append(ProviderAttempt(
            provider=provider,
            model=model,
            success=True,
            response_time=time.time() - start_time
        ))

        return SmartSelection(
            provider=provider,
            model=model,
            config=config,
            attempts=attempts,
            fallback_used=len(attempts) > 1,
            selection_time=time.time() - start_time
        )

    def _get_model_for_provider(self, provider: str, unified_config: UnifiedProviderConfig) -> str:
        """Get the best model for a provider - Claude Generated"""
        # Check provider's preferred model first
        provider_obj = unified_config.get_provider_by_name(provider)
        if provider_obj and provider_obj.preferred_model:
            return provider_obj.preferred_model

        # Check static provider models
        if provider == "gemini" and unified_config.gemini_preferred_model:
            return unified_config.gemini_preferred_model
        elif provider == "anthropic" and unified_config.anthropic_preferred_model:
            return unified_config.anthropic_preferred_model

        # Fallback to first available model
        available_models = self.provider_detection_service.get_available_models(provider)
        if available_models:
            return available_models[0]

        return ""

    def _get_provider_config(self, provider: str, unified_config: UnifiedProviderConfig) -> Dict[str, Any]:
        """Get configuration for a specific provider - Claude Generated"""
        if provider == "gemini":
            return {"api_key": unified_config.gemini_api_key}
        elif provider == "anthropic":
            return {"api_key": unified_config.anthropic_api_key}

        # Check unified providers list
        provider_obj = unified_config.get_provider_by_name(provider)
        if not provider_obj:
            provider_obj = unified_config.get_provider_by_type(provider)

        if provider_obj:
            config = {
                "api_key": provider_obj.api_key,
                "base_url": provider_obj.base_url
            }
            if provider_obj.host:
                config["host"] = provider_obj.host
            if provider_obj.port:
                config["port"] = provider_obj.port
            return config

        return {}

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is currently available - Claude Generated"""
        # Check cache first
        if provider in self._availability_cache:
            is_available, timestamp = self._availability_cache[provider]
            if time.time() - timestamp < self._cache_timeout:
                return is_available

        # Check if provider is disabled
        unified_config = self.config_manager.get_unified_config()
        if provider in unified_config.disabled_providers:
            self._availability_cache[provider] = (False, time.time())
            return False

        # Check availability via detection service
        available_providers = self.provider_detection_service.get_available_providers()
        is_available = provider in available_providers

        # Verify reachability
        if is_available:
            is_available = self.provider_detection_service.is_provider_reachable(provider)

        self._availability_cache[provider] = (is_available, time.time())
        return is_available

    def _validate_provider_config(
        self, provider: str, model: str, config: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate provider configuration - Claude Generated"""
        if not model:
            return False, "No model specified"

        if not config:
            return False, "No configuration available"

        # API key validation for cloud providers
        if provider in ["gemini", "anthropic"]:
            if not config.get("api_key"):
                return False, f"No API key for {provider}"
        elif provider == "ollama":
            if not config.get("host"):
                return False, "No host configured for Ollama"

        return True, None

    def _build_error_message(
        self, task_name: str, task_type: TaskType, attempts: List[ProviderAttempt]
    ) -> str:
        """Build detailed error message - Claude Generated"""
        error_details = []
        for attempt in attempts:
            msg = attempt.error_message or "Provider unavailable"
            error_details.append(f"  • {attempt.provider}: {msg}")

        return (
            f"❌ No available providers for task '{task_name or task_type.value}'.\n\n"
            f"Attempted {len(attempts)} provider(s):\n" +
            "\n".join(error_details) + "\n\n"
            f"💡 Suggestions:\n"
            f"  1. Set a pipeline default in Settings > Pipeline\n"
            f"  2. Verify provider is running (for local providers like Ollama)\n"
            f"  3. Check API keys for cloud providers"
        )

    def get_provider_capabilities(self, provider: str) -> List[str]:
        """Get capabilities for a specific provider - Claude Generated"""
        return self.provider_detection_service.detect_provider_capabilities(provider)


# Convenience functions for easy integration
def select_provider_for_task(
    task_type: TaskType = TaskType.GENERAL,
    required_capabilities: Optional[List[str]] = None,
    prefer_fast: bool = False
) -> SmartSelection:
    """Convenience function for quick provider selection - Claude Generated"""
    selector = SmartProviderSelector()
    return selector.select_provider(task_type, required_capabilities, prefer_fast)
