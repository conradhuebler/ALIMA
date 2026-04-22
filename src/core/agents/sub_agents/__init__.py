"""Tool registry for v4 workflow agents - Claude Generated.

Only ``CachingToolRegistry`` is exported. The legacy SubAgents
(KeywordExtractionAgent, SearchAgent, KeywordSelectionAgent,
ClassificationAgent) and their ``BaseSubAgent`` base class were removed in
the Phase 5 cleanup — their responsibilities now live as deterministic
functions + ``LLMAgentStep`` in ``src/core/agents/steps/``.
"""

from .caching_tool_registry import CachingToolRegistry, create_caching_registry

__all__ = [
    "CachingToolRegistry",
    "create_caching_registry",
]
