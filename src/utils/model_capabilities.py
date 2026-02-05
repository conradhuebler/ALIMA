"""Model capabilities registry for auto-configuring chunking thresholds - Claude Generated

This module provides a lightweight pattern-matching system to automatically detect
appropriate chunking thresholds based on model names. Different LLM models have
varying context window sizes and processing capabilities, requiring different
keyword chunking strategies.

Usage:
    from model_capabilities import get_chunking_threshold

    # Auto-detect threshold based on model
    threshold = get_chunking_threshold("ollama", "cogito:32b")  # Returns 1000

    # Explicit override takes precedence
    threshold = get_chunking_threshold("ollama", "cogito:32b", explicit_override=750)  # Returns 750
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelCapabilities:
    """Defines capabilities for a model pattern"""
    provider_pattern: str  # Regex pattern to match provider:model
    keyword_chunking_threshold: int  # Max keywords before chunking
    context_window: int  # Context window size (for documentation/future use)
    description: str = ""  # Human-readable description


# Compile-time registry mapping model patterns to capabilities
# Patterns are evaluated in order - first match wins
KNOWN_CAPABILITIES = [
    # === Large Context Models (1000+ keywords) ===
    ModelCapabilities(
        r".*cogito.*(32b|70b)",
        keyword_chunking_threshold=1000,
        context_window=128000,
        description="Cogito large models (32B, 70B parameters)"
    ),
    ModelCapabilities(
        r".*qwen.*(32b|72b)",
        keyword_chunking_threshold=1000,
        context_window=128000,
        description="Qwen large models (32B, 72B parameters)"
    ),
    ModelCapabilities(
        r"gpt-4.*",
        keyword_chunking_threshold=800,
        context_window=128000,
        description="GPT-4 family (all variants)"
    ),
    ModelCapabilities(
        r"claude-3.*",
        keyword_chunking_threshold=800,
        context_window=200000,
        description="Claude 3 family (Opus, Sonnet, Haiku)"
    ),
    ModelCapabilities(
        r".*mixtral.*8x7b",
        keyword_chunking_threshold=800,
        context_window=32000,
        description="Mixtral 8x7B MoE model"
    ),

    # === Medium Models (500 keywords - default) ===
    ModelCapabilities(
        r".*cogito.*(14b|13b)",
        keyword_chunking_threshold=500,
        context_window=32000,
        description="Cogito medium models (13B, 14B parameters)"
    ),
    ModelCapabilities(
        r"gpt-3\.5.*",
        keyword_chunking_threshold=500,
        context_window=16000,
        description="GPT-3.5 family"
    ),
    ModelCapabilities(
        r".*llama.*13b",
        keyword_chunking_threshold=500,
        context_window=4096,
        description="LLaMA 13B models"
    ),
    ModelCapabilities(
        r".*mistral.*7b",
        keyword_chunking_threshold=500,
        context_window=8192,
        description="Mistral 7B models"
    ),

    # === Small Models (200-300 keywords) ===
    ModelCapabilities(
        r".*(phi-2|phi2)",
        keyword_chunking_threshold=200,
        context_window=2048,
        description="Phi-2 small model"
    ),
    ModelCapabilities(
        r".*tinyllama",
        keyword_chunking_threshold=200,
        context_window=2048,
        description="TinyLLaMA small model"
    ),
    ModelCapabilities(
        r".*gemma.*7b",
        keyword_chunking_threshold=300,
        context_window=8192,
        description="Gemma 7B model"
    ),
    ModelCapabilities(
        r".*llama.*7b",
        keyword_chunking_threshold=300,
        context_window=4096,
        description="LLaMA 7B models"
    ),

    # === Tiny Models (150 keywords) ===
    ModelCapabilities(
        r".*(phi-1|phi1)",
        keyword_chunking_threshold=150,
        context_window=2048,
        description="Phi-1 tiny model"
    ),
    ModelCapabilities(
        r".*gemma.*2b",
        keyword_chunking_threshold=150,
        context_window=8192,
        description="Gemma 2B model"
    ),
]


def get_model_capabilities(provider: str, model: str) -> Optional[ModelCapabilities]:
    """Match provider:model against capability patterns.

    Args:
        provider: Provider name (e.g., "ollama", "openai", "anthropic")
        model: Model name (e.g., "cogito:32b", "gpt-4", "claude-3-opus")

    Returns:
        ModelCapabilities object if pattern matches, None otherwise

    Example:
        >>> cap = get_model_capabilities("ollama", "cogito:32b")
        >>> cap.keyword_chunking_threshold
        1000
    """
    if not provider or not model:
        return None

    search_string = f"{provider}:{model}".lower()

    for cap in KNOWN_CAPABILITIES:
        if re.search(cap.provider_pattern, search_string, re.IGNORECASE):
            return cap

    return None  # No match found


def get_chunking_threshold(
    provider: str,
    model: str,
    explicit_override: Optional[int] = None,
    config_manager=None
) -> int:
    """Get appropriate chunking threshold with priority handling - Claude Generated

    Priority order:
    1. Explicit override (if provided and > 0)
    2. Per-model config from UnifiedProviderConfig.model_chunking_thresholds
    3. Model pattern match from registry
    4. Default fallback (500)

    Args:
        provider: Provider name (e.g., "ollama", "openai")
        model: Model name (e.g., "cogito:32b", "gpt-4")
        explicit_override: User-specified threshold (overrides auto-detection)
        config_manager: Optional ConfigManager for per-model config lookup

    Returns:
        Keyword chunking threshold (number of keywords before splitting)

    Example:
        >>> # Auto-detect for large model
        >>> get_chunking_threshold("ollama", "cogito:32b")
        1000

        >>> # Explicit override takes precedence
        >>> get_chunking_threshold("ollama", "cogito:32b", explicit_override=750)
        750

        >>> # Per-model config lookup
        >>> get_chunking_threshold("ollama", "cogito:32b", config_manager=cm)
        800  # If configured in model_chunking_thresholds

        >>> # Unknown model uses default
        >>> get_chunking_threshold("custom", "unknown-model")
        500
    """
    # Priority 1: Explicit override
    if explicit_override is not None and explicit_override > 0:
        return explicit_override

    # Priority 2: Per-model config from UnifiedProviderConfig - Claude Generated
    if config_manager:
        try:
            unified_config = config_manager.get_unified_config()
            config_threshold = unified_config.get_chunking_threshold(provider, model)
            if config_threshold and config_threshold > 0:
                return config_threshold
        except Exception:
            pass  # Fall through to pattern match

    # Priority 3: Model pattern match
    cap = get_model_capabilities(provider, model)
    if cap:
        return cap.keyword_chunking_threshold

    # Priority 4: Default fallback
    return 500


def describe_model_capabilities(provider: str, model: str) -> str:
    """Get human-readable description of model capabilities.

    Args:
        provider: Provider name
        model: Model name

    Returns:
        Description string with capabilities, or "Unknown" if no match

    Example:
        >>> describe_model_capabilities("ollama", "cogito:32b")
        'Cogito large models (32B, 70B parameters): 1000 keywords, 128000 tokens context'
    """
    cap = get_model_capabilities(provider, model)
    if cap:
        return f"{cap.description}: {cap.keyword_chunking_threshold} keywords, {cap.context_window} tokens context"
    return "Unknown model (using default: 500 keywords)"
