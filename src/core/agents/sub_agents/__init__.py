"""SubAgents for MetaAgent Orchestration - Claude Generated

Specialized agents for ALIMA pipeline stages:
- KeywordExtractionAgent: Extract keywords from abstract
- SearchAgent: Search GND/SWB catalogs
- KeywordSelectionAgent: Select relevant GND entries
- ClassificationAgent: Assign DK/RVK classifications
"""

from .base_sub_agent import BaseSubAgent, SubAgentResult
from .caching_tool_registry import CachingToolRegistry, create_caching_registry
from .keyword_extraction_agent import KeywordExtractionAgent
from .search_agent import SearchAgent
from .keyword_selection_agent import KeywordSelectionAgent
from .classification_agent import ClassificationAgent

__all__ = [
    "BaseSubAgent",
    "SubAgentResult",
    "CachingToolRegistry",
    "create_caching_registry",
    "KeywordExtractionAgent",
    "SearchAgent",
    "KeywordSelectionAgent",
    "ClassificationAgent",
]