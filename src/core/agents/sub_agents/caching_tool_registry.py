"""Caching Tool Registry - Claude Generated

Wrapper around ToolRegistry that caches results to avoid redundant tool calls.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.agents.shared_context import ToolResultCache
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class CachingToolRegistry:
    """Wrapper around ToolRegistry that caches results.

    All tool calls go through the cache first. If a result is already
    cached, it returns immediately without executing the tool.

    Provides metrics for cache hit/miss rates.
    """

    def __init__(
        self,
        inner_registry: ToolRegistry,
        cache: ToolResultCache,
        cache_enabled: bool = True,
    ):
        """Initialize the caching wrapper.

        Args:
            inner_registry: The actual ToolRegistry to wrap
            cache: ToolResultCache instance to use for caching
            cache_enabled: Whether caching is enabled (default: True)
        """
        self.inner_registry = inner_registry
        self.cache = cache
        self.cache_enabled = cache_enabled
        self.logger = logging.getLogger(__name__)

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names.

        Returns:
            List of tool names from inner registry
        """
        return self.inner_registry.get_tool_names()

    def get_tool_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """Get JSON schemas for tools.

        Args:
            tool_names: Optional list of tool names. If None, returns all.

        Returns:
            List of tool schemas
        """
        return self.inner_registry.get_tool_schemas(tool_names)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with caching.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            Tool result as JSON string
        """
        if not self.cache_enabled:
            return self._execute_tool(tool_name, args)

        # Check cache first
        cached_result = self.cache.get(tool_name, args)
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for {tool_name}")
            return cached_result

        # Execute and cache
        self.logger.debug(f"Cache MISS for {tool_name}, executing...")
        result = self._execute_tool(tool_name, args)
        self.cache.set(tool_name, args, result)

        return result

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute tool through inner registry.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tool result as JSON string
        """
        try:
            result = self.inner_registry.execute(tool_name, args)
            return result
        except Exception as e:
            self.logger.error(f"Tool execution error for {tool_name}: {e}")
            # Return error as JSON
            return json.dumps({
                "error": str(e),
                "tool": tool_name,
                "args": args,
            })

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit/miss counts, hit rate, etc.
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the tool result cache."""
        self.cache.clear()
        self.logger.info("Tool result cache cleared")

    def preload_common_searches(self, keywords: List[str]) -> None:
        """Pre-populate cache with common GND searches.

        Useful for warm-starting the cache before agent execution.

        Args:
            keywords: List of keywords to search for
        """
        self.logger.info(f"Preloading cache with {len(keywords)} keywords...")

        for keyword in keywords[:20]:  # Limit to 20 to avoid overwhelming
            try:
                # Pre-search GND entries
                self.execute("search_gnd", {"term": keyword, "min_results": 5})
            except Exception as e:
                self.logger.warning(f"Preload failed for '{keyword}': {e}")

        stats = self.get_cache_stats()
        self.logger.info(f"Cache preloaded: {stats['cache_size']} entries")


def create_caching_registry(
    config_manager=None,
    cache: Optional[ToolResultCache] = None,
) -> CachingToolRegistry:
    """Factory function to create a CachingToolRegistry.

    Args:
        config_manager: Configuration manager for inner registry
        cache: Optional existing cache to use

    Returns:
        Configured CachingToolRegistry instance
    """
    inner_registry = ToolRegistry(config_manager=config_manager)
    inner_registry.register_all_tools()

    if cache is None:
        cache = ToolResultCache()

    return CachingToolRegistry(inner_registry, cache)