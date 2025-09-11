#!/usr/bin/env python3
"""
SmartProviderSelector - Universal LLM Provider Selection Engine
Intelligently selects providers and models based on preferences, task type, and availability.
Claude Generated
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .config_manager import ConfigManager, ProviderPreferences, ProviderDetectionService
from ..llm.llm_service import LlmService


class TaskType(Enum):
    """LLM task types for provider optimization - Claude Generated"""
    GENERAL = "general"
    VISION = "vision"
    TEXT = "text" 
    CLASSIFICATION = "classification"
    CHUNKED = "chunked"


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
    """Result of smart provider selection - Claude Generated"""
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
    Universal LLM provider selection engine - Claude Generated
    
    Features:
    - Task-specific provider optimization
    - Automatic fallback with intelligent retry logic
    - Performance tracking and learning
    - Configuration-driven preferences
    - Provider availability testing
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager or ConfigManager()
        self.provider_detection_service = self.config_manager.get_provider_detection_service()  # Claude Generated
        
        # Performance tracking
        self._provider_performance: Dict[str, List[float]] = {}
        self._provider_failures: Dict[str, int] = {}
        self._last_successful: Dict[str, float] = {}
        
        # Cache for availability testing
        self._availability_cache: Dict[str, Tuple[bool, float]] = {}
        self._cache_timeout = 300  # 5 minutes
    
    def select_provider(self, 
                       task_type: TaskType = TaskType.GENERAL,
                       required_capabilities: Optional[List[str]] = None,
                       prefer_fast: bool = False) -> SmartSelection:
        """
        Select the best provider for a given task - Claude Generated
        
        Args:
            task_type: Type of LLM task
            required_capabilities: List of required capabilities (e.g., ["vision", "large_context"])
            prefer_fast: Prioritize speed over quality
            
        Returns:
            SmartSelection with provider, model, config, and attempt history
        """
        start_time = time.time()
        preferences = self.config_manager.get_provider_preferences()
        
        # Auto-validate and cleanup preferences before using them - Claude Generated
        validation_issues = preferences.validate_preferences(self.provider_detection_service)
        if any(validation_issues.values()):
            self.logger.info("Found provider preference issues, performing auto-cleanup...")
            cleanup_report = preferences.auto_cleanup(self.provider_detection_service)
            
            # Log cleanup actions
            for category, actions in cleanup_report.items():
                if actions:
                    if isinstance(actions, list):
                        for action in actions:
                            self.logger.info(f"Preference cleanup - {category}: {action}")
                    else:
                        self.logger.info(f"Preference cleanup - {category}: {actions}")
            
            # Save cleaned preferences back to config
            self.config_manager.save_config()
        
        # Ensure we have a valid configuration
        preferences.ensure_valid_configuration(self.provider_detection_service)
        
        # Get provider priority list for this task
        provider_priority = preferences.get_provider_priority_for_task(task_type.value)
        
        # Filter by capabilities if specified
        if required_capabilities:
            provider_priority = self._filter_by_capabilities(provider_priority, required_capabilities)
        
        # Apply speed preference
        if prefer_fast or preferences.prefer_faster_models:
            provider_priority = self._sort_by_speed(provider_priority)
        
        attempts = []
        
        for provider in provider_priority:
            if not self._is_provider_available(provider):
                attempt = ProviderAttempt(
                    provider=provider,
                    model="",
                    success=False,
                    error_message="Provider not available"
                )
                attempts.append(attempt)
                continue
            
            # Get preferred model for this provider
            model = self._get_model_for_provider(provider, task_type, preferences, prefer_fast)
            
            # Get provider configuration
            config = self._get_provider_config(provider)
            
            # Test the provider
            attempt_start = time.time()
            success, error = self._test_provider(provider, model, config)
            response_time = time.time() - attempt_start
            
            attempt = ProviderAttempt(
                provider=provider,
                model=model,
                success=success,
                error_message=error,
                response_time=response_time
            )
            attempts.append(attempt)
            
            if success:
                # Update performance tracking
                self._record_success(provider, response_time)
                
                selection = SmartSelection(
                    provider=provider,
                    model=model,
                    config=config,
                    attempts=attempts,
                    fallback_used=len(attempts) > 1,
                    selection_time=time.time() - start_time
                )
                
                self.logger.info(f"Selected provider: {provider} with model: {model} for task: {task_type.value} (attempts: {len(attempts)})")
                return selection
            else:
                # Record failure
                self._record_failure(provider, error)
                self.logger.warning(f"Provider {provider} failed: {error}")
        
        # If we get here, all providers failed
        raise RuntimeError(f"No available providers for task type {task_type.value}. Attempted: {[a.provider for a in attempts]}")
    
    def _filter_by_capabilities(self, providers: List[str], required_capabilities: List[str]) -> List[str]:
        """Filter providers by required capabilities using dynamic detection - Claude Generated"""
        filtered = []
        
        for provider in providers:
            provider_caps = self.provider_detection_service.detect_provider_capabilities(provider)
            if all(capability in provider_caps for capability in required_capabilities):
                filtered.append(provider)
        
        return filtered
    
    def _sort_by_speed(self, providers: List[str]) -> List[str]:
        """Sort providers by speed performance using dynamic capability detection - Claude Generated"""
        speed_ranking = {}
        
        for provider in providers:
            if provider in self._provider_performance:
                # Use historical performance data if available
                avg_time = sum(self._provider_performance[provider]) / len(self._provider_performance[provider])
                speed_ranking[provider] = avg_time
            else:
                # Estimate speed based on dynamic capability detection
                capabilities = self.provider_detection_service.detect_provider_capabilities(provider)
                
                if 'fast' in capabilities:
                    speed_ranking[provider] = 1.5  # Fast models
                elif 'local' in capabilities:
                    speed_ranking[provider] = 2.0  # Local models (hardware dependent)
                elif 'reasoning' in capabilities or 'analysis' in capabilities:
                    speed_ranking[provider] = 4.0  # More thoughtful, slower
                else:
                    speed_ranking[provider] = 3.0  # Default estimate
        
        # Sort by speed (lower is better)
        return sorted(providers, key=lambda p: speed_ranking.get(p, 999.0))
    
    def _get_model_for_provider(self, provider: str, task_type: TaskType, preferences: ProviderPreferences, prefer_fast: bool) -> str:
        """Get the appropriate model for a provider and task using dynamic model detection - Claude Generated"""
        # Check for preferred model in preferences first
        preferred = preferences.get_preferred_model(provider)
        
        if preferred and not prefer_fast:
            return preferred
        
        # Get available models for this provider
        available_models = self.provider_detection_service.get_available_models(provider)
        
        if not available_models:
            return preferred or ""
        
        models_str = ' '.join(available_models).lower()
        
        # Task-specific model selection
        if task_type == TaskType.VISION:
            # Look for vision-capable models
            vision_indicators = ['vision', 'gpt-4o', 'claude-3', 'gemini-2.0', 'llava', 'minicpm-v']
            for model in available_models:
                if any(indicator in model.lower() for indicator in vision_indicators):
                    return model
        
        # Speed-optimized model selection
        if prefer_fast:
            speed_indicators = ['flash', 'mini', 'haiku', '14b', 'turbo']
            for model in available_models:
                if any(indicator in model.lower() for indicator in speed_indicators):
                    return model
        
        # Quality-optimized model selection (default)
        quality_indicators = ['2.0', '4o', 'claude-3-5', 'cogito:32b', 'opus']
        for model in available_models:
            if any(indicator in model.lower() for indicator in quality_indicators):
                return model
        
        # Fallback to preferred or first available model
        return preferred or available_models[0]
    
    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider - Claude Generated"""
        llm_config = self.config_manager.load_config().llm
        
        if provider == "gemini":
            return {"api_key": llm_config.gemini}
        elif provider == "anthropic":
            return {"api_key": llm_config.anthropic}
        elif provider == "ollama":
            # Use primary Ollama provider or fallback to legacy config
            ollama_provider = llm_config.get_primary_ollama_provider()
            if ollama_provider:
                return {
                    "host": ollama_provider.host,
                    "port": ollama_provider.port,
                    "api_key": ollama_provider.api_key,
                    "use_ssl": ollama_provider.use_ssl,
                    "connection_type": ollama_provider.connection_type
                }
            else:
                return {
                    "host": llm_config.ollama_host,
                    "port": llm_config.ollama_port
                }
        else:
            # Try to find in OpenAI-compatible providers
            openai_provider = llm_config.get_provider_by_name(provider)
            if openai_provider:
                return {
                    "base_url": openai_provider.base_url,
                    "api_key": openai_provider.api_key
                }
        
        return {}
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is currently available using provider detection service - Claude Generated"""
        # Check cache first
        if provider in self._availability_cache:
            is_available, timestamp = self._availability_cache[provider]
            if time.time() - timestamp < self._cache_timeout:
                return is_available
        
        # Check if provider is disabled in preferences
        preferences = self.config_manager.get_provider_preferences()
        if not preferences.is_provider_enabled(provider):
            self._availability_cache[provider] = (False, time.time())
            return False
        
        # Use provider detection service for availability check
        available_providers = self.provider_detection_service.get_available_providers()
        is_available = provider in available_providers
        
        # If available in config, also check reachability
        if is_available:
            is_reachable = self.provider_detection_service.is_provider_reachable(provider)
            is_available = is_reachable
        
        self._availability_cache[provider] = (is_available, time.time())
        return is_available
    
    def _test_provider(self, provider: str, model: str, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Test if a provider/model combination is working - Claude Generated"""
        try:
            # For now, assume success if we have configuration
            # In the future, this could do a minimal API test call
            if not model:
                return False, "No model specified"
            
            if not config:
                return False, "No configuration available"
            
            # Basic config validation
            if provider in ["gemini", "anthropic"]:
                if not config.get("api_key"):
                    return False, f"No API key for {provider}"
            elif provider == "ollama":
                if not config.get("host"):
                    return False, "No host configured for Ollama"
            else:
                if not config.get("api_key") or not config.get("base_url"):
                    return False, f"Incomplete configuration for {provider}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _record_success(self, provider: str, response_time: float):
        """Record successful provider usage - Claude Generated"""
        if provider not in self._provider_performance:
            self._provider_performance[provider] = []
        
        # Keep only last 10 measurements
        self._provider_performance[provider].append(response_time)
        if len(self._provider_performance[provider]) > 10:
            self._provider_performance[provider].pop(0)
        
        self._last_successful[provider] = time.time()
        
        # Reset failure count on success
        if provider in self._provider_failures:
            self._provider_failures[provider] = 0
    
    def _record_failure(self, provider: str, error: Optional[str]):
        """Record provider failure - Claude Generated"""
        if provider not in self._provider_failures:
            self._provider_failures[provider] = 0
        
        self._provider_failures[provider] += 1
        self.logger.warning(f"Provider {provider} failure #{self._provider_failures[provider]}: {error}")
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all available providers - Claude Generated"""
        stats = {}
        
        # Get all available providers dynamically
        available_providers = self.provider_detection_service.get_available_providers()
        
        for provider in available_providers:
            perf = self._provider_performance.get(provider, [])
            failures = self._provider_failures.get(provider, 0)
            last_success = self._last_successful.get(provider, 0)
            
            stats[provider] = {
                "average_response_time": sum(perf) / len(perf) if perf else None,
                "total_requests": len(perf),
                "failure_count": failures,
                "last_successful": last_success,
                "is_available": self._is_provider_available(provider),
                "is_reachable": self.provider_detection_service.is_provider_reachable(provider),
                "model_count": len(self.provider_detection_service.get_available_models(provider)),
                "capabilities": self.provider_detection_service.detect_provider_capabilities(provider)
            }
        
        return stats
    
    def reset_performance_tracking(self):
        """Reset all performance tracking data - Claude Generated"""
        self._provider_performance.clear()
        self._provider_failures.clear()
        self._last_successful.clear()
        self._availability_cache.clear()
        self.logger.info("Provider performance tracking reset")


# Convenience functions for easy integration
def select_provider_for_task(task_type: TaskType = TaskType.GENERAL, 
                           required_capabilities: Optional[List[str]] = None,
                           prefer_fast: bool = False) -> SmartSelection:
    """Convenience function for quick provider selection - Claude Generated"""
    selector = SmartProviderSelector()
    return selector.select_provider(task_type, required_capabilities, prefer_fast)


def select_vision_provider(prefer_fast: bool = False) -> SmartSelection:
    """Select optimal provider for vision/image analysis tasks - Claude Generated"""
    return select_provider_for_task(TaskType.VISION, ["vision"], prefer_fast)


def select_text_provider(prefer_fast: bool = False) -> SmartSelection:
    """Select optimal provider for text-only tasks - Claude Generated"""
    return select_provider_for_task(TaskType.TEXT, [], prefer_fast)