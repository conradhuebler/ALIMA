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

from .config_manager import ConfigManager, ProviderDetectionService
from ..llm.llm_service import LlmService
from .config_models import (
    UnifiedProviderConfig,
    TaskPreference,
    TaskType as UnifiedTaskType
)


class TaskType(Enum):
    """LLM task types for provider optimization - Claude Generated"""
    GENERAL = "general"
    VISION = "vision"
    TEXT = "text" 
    CLASSIFICATION = "classification"
    CHUNKED = "chunked"
    
    def to_unified_task_type(self) -> 'UnifiedTaskType':
        """Map SmartProviderSelector TaskType to UnifiedTaskType - Claude Generated"""
        mapping = {
            TaskType.GENERAL: UnifiedTaskType.GENERAL,
            TaskType.VISION: UnifiedTaskType.VISION,
            TaskType.TEXT: UnifiedTaskType.KEYWORDS,
            TaskType.CLASSIFICATION: UnifiedTaskType.CLASSIFICATION,
            TaskType.CHUNKED: UnifiedTaskType.CHUNKED_PROCESSING
        }
        return mapping.get(self, UnifiedTaskType.GENERAL)
    
    @classmethod
    def from_pipeline_step(cls, step_id: str, task_name: str = "") -> 'TaskType':
        """Map Pipeline step to appropriate TaskType - Claude Generated"""
        # Handle chunked tasks first
        if task_name.endswith('_chunked') or 'chunked' in task_name.lower():
            return cls.CHUNKED
            
        # Map pipeline steps to task types - updated for real ALIMA pipeline steps
        step_mapping = {
            "input": cls.GENERAL,                    # File/text input (no LLM)
            "initialisation": cls.TEXT,              # LLM keyword extraction
            "search": cls.GENERAL,                   # Database search (no LLM)
            "keywords": cls.TEXT,                    # LLM keyword verification
            "verification": cls.TEXT,                # Legacy alias for keywords
            "classification": cls.CLASSIFICATION,    # LLM DDC/DK/RVK classification
            "dk_search": cls.GENERAL,               # Catalog search (no LLM)
            "dk_classification": cls.CLASSIFICATION, # LLM DK classification analysis
            "image_text_extraction": cls.VISION     # Image analysis, OCR
        }
        
        return step_mapping.get(step_id, cls.GENERAL)


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
        
        # Unified Configuration Integration - Claude Generated
        try:
            # Load unified config instead of legacy config structure
            self.config = self.config_manager.load_config(force_reload=True)
            self.unified_config = self.config_manager.get_unified_config()
            self.logger.info("SmartProviderSelector initialized with unified configuration support")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Unified Configuration support: {e}")
            self.config = None
            self.unified_config = None
        
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
                       prefer_fast: bool = False,
                       task_name: str = "",
                       step_id: str = "") -> SmartSelection:
        """
        Select the best provider for a given task - Claude Generated
        
        Args:
            task_type: Type of LLM task
            required_capabilities: List of required capabilities (e.g., ["vision", "large_context"])
            prefer_fast: Prioritize speed over quality
            task_name: Specific task name (e.g., "keywords", "keywords_chunked") for Task Preference lookup
            step_id: Pipeline step identifier (e.g., "keywords", "initialisation") for automatic TaskType mapping
            
        Returns:
            SmartSelection with provider, model, config, and attempt history
        """
        start_time = time.time()
        unified_config = self.config_manager.get_unified_config()
        
        # Auto-detect TaskType from step_id if not explicitly provided - Claude Generated
        if task_type == TaskType.GENERAL and step_id:
            detected_task_type = TaskType.from_pipeline_step(step_id, task_name)
            if detected_task_type != TaskType.GENERAL:
                task_type = detected_task_type
                self.logger.info(f"Auto-detected TaskType {task_type.value} from step_id '{step_id}' and task_name '{task_name}'")
        
        # TODO: Implement validation in UnifiedProviderConfig if needed
        # validation_issues = unified_config.validate_preferences(self.provider_detection_service)
        # if any(validation_issues.values()):
        #     self.logger.info("Found provider preference issues, performing auto-cleanup...")
        #     cleanup_report = unified_config.auto_cleanup(self.provider_detection_service)
        #\n        #     # Log cleanup actions\n        #     for category, actions in cleanup_report.items():\n        #         if actions:\n        #             if isinstance(actions, list):
        #                 for action in actions:
        #                     self.logger.info(f"Preference cleanup - {category}: {action}")
        #             else:
        #                 self.logger.info(f"Preference cleanup - {category}: {actions}")
        #
        #     # Save cleaned preferences back to config - Claude Generated
        #     try:
        #         current_config = self.config_manager.load_config()
        #         self.config_manager.save_config(current_config, "user")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to save cleaned preferences: {e}")


        # Ensure we have a valid configuration
        # TODO: Implement ensure_valid_configuration in UnifiedProviderConfig if needed\n        # unified_config.ensure_valid_configuration(self.provider_detection_service)
        
        # Get provider priority list for this task - enhanced with 3-tier hierarchy - Claude Generated
        provider_priority = []
        
        # === TIER 1: Task-specific preferences from unified_config.task_preferences (highest priority) ===
        if self.unified_config and task_name and task_name in self.unified_config.task_preferences:
            task_preference = self.unified_config.task_preferences[task_name]
            model_priorities = task_preference.model_priority if task_preference else []

            # CRITICAL DEBUG: Log task preference loading attempt - Claude Generated
            self.logger.info(f"🔍 TIER1_TASK_PREFS: task_name='{task_name}' found in config, model_priorities={model_priorities}")

            if model_priorities and model_priorities[0].get("provider_name") != "auto":
                # Extract unique providers from model_priority, maintaining order
                for entry in model_priorities:
                    provider = entry.get("provider_name")
                    if provider and provider not in provider_priority:
                        provider_priority.append(provider)
                self.logger.info(f"TIER 1: Using task-specific provider priority for {task_name}: {provider_priority}")
            else:
                self.logger.info(f"🔍 TIER1_SKIP: task_name='{task_name}' has no usable model_priorities or is set to 'auto'")
        elif self.unified_config and task_name:
            self.logger.info(f"🔍 TIER1_MISS: task_name='{task_name}' not found in task_preferences. Available: {list(self.unified_config.task_preferences.keys())}")
        else:
            self.logger.info(f"🔍 TIER1_NO_CONFIG: unified_config={self.unified_config is not None}, task_name='{task_name}'")
        
        # === TIER 2: Provider defaults from unified config (medium priority) ===
        if not provider_priority and self.unified_config:
            # Map TaskType to task names for legacy lookup
            task_lookup_name = task_name
            if not task_lookup_name:
                task_type_mapping = {
                    TaskType.TEXT: "text_analysis",
                    TaskType.VISION: "image_text_extraction", 
                    TaskType.CLASSIFICATION: "classification",
                    TaskType.CHUNKED: "text_analysis"
                }
                task_lookup_name = task_type_mapping.get(task_type, "general")
            
            # TODO: Implement task-specific provider priority in UnifiedProviderConfig\n            provider_priority = unified_config.provider_priority  # Fallback to general priority
            self.logger.info(f"TIER 2: Using provider defaults for {task_type.value}: {provider_priority}")
        
        # === TIER 3: Detection service fallback (lowest priority) ===
        if not provider_priority:
            provider_priority = list(self.provider_detection_service.get_available_providers())
            self.logger.info(f"TIER 3: Using detection service fallback: {provider_priority}")
        
        # Filter by capabilities if specified
        if required_capabilities:
            provider_priority = self._filter_by_capabilities(provider_priority, required_capabilities)
        
        # Apply speed preference
        if prefer_fast or unified_config.prefer_faster_models:
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
            model = self._get_model_for_provider(provider, task_type, unified_config, prefer_fast, task_name)
            
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
    
    def _get_model_for_provider(self, provider: str, task_type: TaskType, unified_config, prefer_fast: bool = False, task_name: str = "") -> str:
        """Get the best model for a provider with Task Preference hierarchical selection - Claude Generated"""
        available_models = self.provider_detection_service.get_available_models(provider)
        if not available_models:
            self.logger.warning(f"No models available for provider {provider}")
            return ""
        
        # HIERARCHICAL MODEL SELECTION - PRIORITY ORDER:
        # 1. Task-specific model preferences (highest priority)
        # 2. Provider config preferred model
        # 3. Speed/quality optimization
        # 4. First available model (fallback)
        
        selected_model = None
        selection_reason = ""
        
        # === PRIORITY 1: Task-specific model preferences (root-level config.unified_config.task_preferences) ===
        if self.config and task_name and task_name in self.config.unified_config.task_preferences:
            try:
                task_data = self.config.unified_config.task_preferences[task_name]

                # CRITICAL DEBUG: Log task preference model selection attempt - Claude Generated
                self.logger.info(f"🔍 PRIO1_MODEL_SEARCH: provider='{provider}', task_name='{task_name}', task_data={task_data}")

                # Determine if this is a chunked task
                is_chunked = task_type == TaskType.CHUNKED or task_name.endswith('_chunked')

                # Get appropriate model priority list
                if is_chunked and hasattr(task_data, 'chunked_model_priority') and task_data.chunked_model_priority:
                    model_priorities = task_data.chunked_model_priority
                    self.logger.info(f"Using chunked model priority for task {task_name}")
                else:
                    model_priorities = getattr(task_data, 'model_priority', [])
                    self.logger.info(f"Using standard model priority for task {task_name}")
                
                # Find first available model from task priorities that matches this provider
                for priority_entry in model_priorities:
                    if priority_entry.get("provider_name") == provider:
                        preferred_model = priority_entry.get("model_name")
                        if preferred_model == "auto" or preferred_model == "default":
                            # "auto"/"default" means use provider's preferred model
                            break
                        elif preferred_model and preferred_model in available_models:
                            selected_model = preferred_model
                            selection_reason = f"task-specific preference (rank {model_priorities.index(priority_entry) + 1})"
                            if is_chunked:
                                selection_reason += " [chunked]"
                            break
                        elif preferred_model:
                            # Try fuzzy matching for model names
                            fuzzy_match = self._find_fuzzy_model_match(preferred_model, available_models)
                            if fuzzy_match:
                                selected_model = fuzzy_match
                                selection_reason = f"task-specific preference via fuzzy match ('{preferred_model}' -> '{fuzzy_match}')"
                                if is_chunked:
                                    selection_reason += " [chunked]"
                                break
                                
                if selected_model:
                    self.logger.info(f"TIER 1: Selected model {selected_model} for {provider} via {selection_reason}")
                    # CRITICAL DEBUG: Log successful task preference model selection - Claude Generated
                    self.logger.info(f"🔍 PRIO1_SUCCESS: provider='{provider}', task='{task_name}', model='{selected_model}', reason='{selection_reason}'")
                    return selected_model
                    
            except Exception as e:
                self.logger.warning(f"Failed to use task preferences for {provider}: {e}")
        
        # === PRIORITY 2: Provider config preferred model ===
        preferred_model = self._get_preferred_model_from_config(provider)
        if preferred_model:
            # Verify the preferred model is actually available
            if preferred_model in available_models:
                selected_model = preferred_model
                selection_reason = "provider config preference"
            else:
                # Try fuzzy matching
                fuzzy_match = self._find_fuzzy_model_match(preferred_model, available_models)
                if fuzzy_match:
                    selected_model = fuzzy_match
                    selection_reason = f"provider config via fuzzy match ('{preferred_model}' -> '{fuzzy_match}')"
                else:
                    self.logger.warning(f"Preferred model {preferred_model} not available for {provider}")
        
        if selected_model:
            self.logger.info(f"TIER 2: Selected model {selected_model} for {provider} via {selection_reason}")
            return selected_model
        
        # === PRIORITY 3: Speed/quality optimization ===
        if prefer_fast or unified_config.prefer_faster_models:
            # Prioritize models with speed indicators
            fast_models = [m for m in available_models if any(indicator in m.lower() 
                          for indicator in ['flash', 'mini', 'haiku', '14b', 'turbo', 'fast'])]
            if fast_models:
                selected_model = fast_models[0]
                selection_reason = "speed optimization"
        
        if selected_model:
            self.logger.info(f"TIER 3: Selected model {selected_model} for {provider} via {selection_reason}")
            return selected_model
        
        # === PRIORITY 4: First available model (fallback) ===
        selected_model = available_models[0]
        selection_reason = "fallback (first available)"
        self.logger.info(f"TIER 4: Selected model {selected_model} for {provider} via {selection_reason}")
        return selected_model
    
    def _get_preferred_model_from_config(self, provider: str) -> Optional[str]:
        """Get preferred model directly from provider configuration - Claude Generated"""
        try:
            # 🔍 DEBUG: Log preferred model request - Claude Generated
            self.logger.critical(f"🔍 PREFERRED_MODEL_REQUEST: provider='{provider}'")
            
            # Force reload to ensure we get latest saved config - Claude Generated
            config = self.config_manager.load_config(force_reload=True)
            
            # 🔍 DEBUG: Try to find provider with name mapping - Claude Generated
            self.logger.critical(f"🔍 SEARCHING_FOR_PROVIDER: '{provider}' in all provider lists")
            
            # 🔍 DEBUG: Log what we found in config - Claude Generated
            self.logger.critical(f"🔍 CONFIG_gemini_preferred_model: '{config.unified_config.gemini_preferred_model}'")
            self.logger.critical(f"🔍 CONFIG_anthropic_preferred_model: '{config.unified_config.anthropic_preferred_model}'")
            
            # Check static providers first
            if provider == "gemini":
                preferred = config.unified_config.gemini_preferred_model or None
                self.logger.critical(f"🔍 PREFERRED_MODEL_FOUND: gemini -> '{preferred}'")
                return preferred
            elif provider == "anthropic":
                preferred = config.unified_config.anthropic_preferred_model or None
                self.logger.critical(f"🔍 PREFERRED_MODEL_FOUND: anthropic -> '{preferred}'")
                return preferred
            
            # Check providers in unified provider list - Claude Generated
            for unified_provider in config.unified_config.providers:
                if unified_provider.name == provider:
                    preferred = unified_provider.preferred_model or None
                    self.logger.critical(f"🔍 PREFERRED_MODEL_FOUND: {unified_provider.provider_type} '{provider}' -> '{preferred}'")
                    return preferred

                # Fuzzy matching for common Ollama names - Claude Generated
                if unified_provider.provider_type == "ollama":
                    self.logger.critical(f"🔍 CHECKING_OLLAMA_PROVIDER: '{unified_provider.name}' vs requested '{provider}'")

                    # Fuzzy matching for common provider name variations - Claude Generated
                    if self._provider_names_match(unified_provider.name, provider):
                        preferred = unified_provider.preferred_model or None
                        self.logger.critical(f"🔍 PREFERRED_MODEL_FOUND: ollama '{provider}' -> '{preferred}' (fuzzy match: '{unified_provider.name}')")
                        return preferred
            
            self.logger.critical(f"🔍 PREFERRED_MODEL_FOUND: '{provider}' -> None (not found)")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting preferred model for {provider}: {e}")
            return None
    
    def _find_fuzzy_model_match(self, preferred_model: str, available_models: List[str]) -> Optional[str]:
        """Find fuzzy match for model name in available models - Claude Generated"""
        if not preferred_model or not available_models:
            return None
        
        preferred_lower = preferred_model.lower()
        
        # Exact match (case insensitive)
        for model in available_models:
            if model.lower() == preferred_lower:
                return model
        
        # Partial match - preferred in available
        for model in available_models:
            if preferred_lower in model.lower():
                return model
        
        # Partial match - available in preferred
        for model in available_models:
            if model.lower() in preferred_lower:
                return model
        
        # No fuzzy match found
        return None
    
    def _provider_names_match(self, config_name: str, requested_name: str) -> bool:
        """Check if provider names match with fuzzy logic for common variations - Claude Generated"""
        # Normalize names for comparison
        config_normalized = config_name.lower().replace(' ', '').replace('-', '').replace('/', '').replace('_', '')
        requested_normalized = requested_name.lower().replace(' ', '').replace('-', '').replace('/', '').replace('_', '')
        
        # Direct match after normalization
        if config_normalized == requested_normalized:
            return True
        
        # Common ollama name variations
        ollama_variations = {
            'ollama': ['localhost', 'local', 'ollama.com', 'ollamacom'],
            'localhost': ['ollama', 'local'],
            'local': ['ollama', 'localhost'],
        }
        
        # Check if requested name is a known variation of config name
        if requested_normalized in ollama_variations.get(config_normalized, []):
            return True
        if config_normalized in ollama_variations.get(requested_normalized, []):
            return True
        
        # Check if one contains the other (e.g., "LLMachine/Ollama" contains "ollama")
        if 'ollama' in config_normalized and 'ollama' in requested_normalized:
            return True
            
        return False
    
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
    
    
    def _get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider - Claude Generated"""
        # Use unified config instead of legacy llm config
        unified_config = self.config_manager.get_unified_config()

        if provider == "gemini":
            return {"api_key": unified_config.gemini_api_key}
        elif provider == "anthropic":
            return {"api_key": unified_config.anthropic_api_key}
        else:
            # Try to find in unified providers list
            provider_obj = unified_config.get_provider_by_name(provider)
            if provider_obj:
                config = {
                    "api_key": provider_obj.api_key,
                    "base_url": provider_obj.base_url
                }
                # Add provider-specific fields
                if hasattr(provider_obj, 'host'):
                    config["host"] = provider_obj.host
                if hasattr(provider_obj, 'port'):
                    config["port"] = provider_obj.port
                return config

        return {}
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is currently available using provider detection service - Claude Generated"""
        # Check cache first
        if provider in self._availability_cache:
            is_available, timestamp = self._availability_cache[provider]
            if time.time() - timestamp < self._cache_timeout:
                return is_available
        
        # Check if provider is disabled in preferences
        unified_config = self.config_manager.get_unified_config()
        if provider in unified_config.disabled_providers:
            self._availability_cache[provider] = (False, time.time())
            return False
        
        # Use provider detection service for availability check with debug logging - Claude Generated
        available_providers = self.provider_detection_service.get_available_providers()
        is_available = provider in available_providers

        self.logger.info(f"🔍 AVAILABILITY_CHECK: provider='{provider}', available_providers={available_providers}")
        self.logger.info(f"🔍 AVAILABILITY_RESULT: provider='{provider}' in available_providers: {is_available}")

        # If available in config, also check reachability
        if is_available:
            is_reachable = self.provider_detection_service.is_provider_reachable(provider)
            self.logger.info(f"🔍 REACHABILITY_CHECK: provider='{provider}' reachable: {is_reachable}")
            is_available = is_reachable
        else:
            self.logger.warning(f"🔍 PROVIDER_NOT_AVAILABLE: '{provider}' not found in available providers")
        
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
    
    def select_with_manual_override(self,
                                   manual_provider: Optional[str] = None,
                                   manual_model: Optional[str] = None,
                                   task_type: TaskType = TaskType.GENERAL,
                                   prefer_fast: bool = False,
                                   validate_manual: bool = True) -> SmartSelection:
        """
        Select provider with optional manual override - Claude Generated
        
        Args:
            manual_provider: Force specific provider (None = use smart selection)
            manual_model: Force specific model (None = use smart selection for provider)
            task_type: Type of task for smart selection fallback
            prefer_fast: Speed vs quality preference for smart selection  
            validate_manual: Whether to validate manual choices
            
        Returns:
            SmartSelection with provider/model and fallback information
        """
        start_time = time.time()
        attempts = []
        
        # If manual override is specified
        if manual_provider:
            self.logger.info(f"Manual override requested: {manual_provider}/{manual_model or 'auto'}")
            
            # Validate manual provider if requested
            if validate_manual and not self._is_provider_available(manual_provider):
                self.logger.warning(f"Manual provider '{manual_provider}' not available, falling back to smart selection")
                return self._fallback_to_smart_selection(task_type, prefer_fast, attempts, start_time)
            
            # Get provider configuration
            provider_config = self._get_provider_config(manual_provider)
            if not provider_config:
                self.logger.warning(f"No configuration found for manual provider '{manual_provider}'")
                return self._fallback_to_smart_selection(task_type, prefer_fast, attempts, start_time)
            
            # Handle model selection
            if manual_model:
                # Validate manual model if requested
                if validate_manual:
                    available_models = self.provider_detection_service.get_available_models(manual_provider)
                    if available_models and manual_model not in available_models:
                        self.logger.warning(f"Manual model '{manual_model}' not available for provider '{manual_provider}', using provider default")
                        manual_model = self._get_model_for_provider(manual_provider, task_type, self.config_manager.get_unified_config(), prefer_fast, "")
                
                selected_model = manual_model
            else:
                # Auto-select model for manual provider
                selected_model = self._get_model_for_provider(manual_provider, task_type, self.config_manager.get_unified_config(), prefer_fast, "")
            
            # Create manual selection result
            attempts.append(ProviderAttempt(
                provider=manual_provider,
                model=selected_model,
                success=True,
                response_time=time.time() - start_time
            ))
            
            return SmartSelection(
                provider=manual_provider,
                model=selected_model,
                config=provider_config,
                attempts=attempts,
                fallback_used=False,
                selection_time=time.time() - start_time
            )
        
        # No manual override - use smart selection
        return self.select_provider(task_type, prefer_fast=prefer_fast)
    
    def _fallback_to_smart_selection(self, task_type: TaskType, prefer_fast: bool, 
                                   existing_attempts: List[ProviderAttempt], start_time: float) -> SmartSelection:
        """Fallback to smart selection with existing attempt history - Claude Generated"""
        self.logger.info("Falling back to smart provider selection")
        smart_selection = self.select_provider(task_type, prefer_fast=prefer_fast)
        
        # Combine attempt histories
        combined_attempts = existing_attempts + smart_selection.attempts
        
        return SmartSelection(
            provider=smart_selection.provider,
            model=smart_selection.model,
            config=smart_selection.config,
            attempts=combined_attempts,
            fallback_used=True,  # Mark that fallback was used
            selection_time=time.time() - start_time
        )
    
    def validate_manual_choice(self, provider: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate a manual provider/model choice - Claude Generated
        
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "issues": List[str],
                "suggestions": List[str],
                "available_models": List[str]
            }
        """
        result = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "available_models": []
        }
        
        # Check provider availability
        if not self._is_provider_available(provider):
            result["valid"] = False
            result["issues"].append(f"Provider '{provider}' is not available or disabled")
            
            # Suggest alternatives
            available_providers = self.provider_detection_service.get_available_providers()
            if available_providers:
                result["suggestions"].append(f"Available providers: {', '.join(available_providers[:3])}")
        else:
            # Provider is available, get available models
            available_models = self.provider_detection_service.get_available_models(provider)
            result["available_models"] = available_models or []
            
            # Check model if specified
            if model:
                if available_models and model not in available_models:
                    result["valid"] = False
                    result["issues"].append(f"Model '{model}' not available for provider '{provider}'")
                    
                    # Suggest similar models
                    similar_models = [m for m in available_models if model.lower() in m.lower() or m.lower() in model.lower()]
                    if similar_models:
                        result["suggestions"].append(f"Similar models: {', '.join(similar_models[:3])}")
                    elif available_models:
                        result["suggestions"].append(f"Available models: {', '.join(available_models[:3])}")
        
        return result
    
    def get_provider_capabilities(self, provider: str) -> List[str]:
        """Get capabilities for a specific provider - Claude Generated"""
        return self.provider_detection_service.detect_provider_capabilities(provider)
    
    def get_optimal_model_for_task(self, provider: str, task_type: TaskType, prefer_fast: bool = False, task_name: str = "") -> Optional[str]:
        """Get the optimal model for a provider/task combination - Claude Generated"""
        unified_config = self.config_manager.get_unified_config()
        return self._get_model_for_provider(provider, task_type, unified_config, prefer_fast, task_name)


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