"""MCPToolAdapter — wraps src/mcp ToolRegistry entries as BaseChatTool.

WP10 P-δ.2. Claude Generated.

Each adapter borrows the schema (description + parameters) from the
existing ``ToolDefinition`` and dispatches via the MCP registry. The
adapter ignores the chat ``session`` argument — MCP handlers are
session-free.

``build_mcp_chat_tools`` honours ``ChatConfig.no_cache_writes`` (default
``True``) by excluding write-ish tools. Currently the only write-ish MCP
tool is ``store_search_result``; any future write tools should be added
to ``_WRITE_TOOLS`` below.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.ui.chat_tools.base import BaseChatTool

logger = logging.getLogger(__name__)

# Tools that mutate persistent state (DB / cache writes). Excluded when
# ChatConfig.no_cache_writes is True.
_WRITE_TOOLS = frozenset({"store_search_result"})


class MCPToolAdapter(BaseChatTool):
    """Wraps a single MCP tool. Ignores `session`, delegates to MCP registry."""

    def __init__(self, mcp_tool_name: str, mcp_registry: Any) -> None:
        self._mcp_tool_name = mcp_tool_name
        self._mcp_registry = mcp_registry
        tool_def = mcp_registry._tools.get(mcp_tool_name)
        if tool_def is None:
            raise ValueError(f"MCP tool not registered: {mcp_tool_name!r}")
        self.name = tool_def.name
        self.description = tool_def.description
        self.parameters_schema = tool_def.parameters

    def available_for(self, session: Any) -> bool:
        # MCP tools are global; always available regardless of session state.
        return True

    def execute(self, session: Any, **kwargs: Any) -> str:
        return self._mcp_registry.execute(self._mcp_tool_name, kwargs)


def build_mcp_chat_tools(
    *,
    no_cache_writes: bool,
    mcp_registry: Any,
) -> List[BaseChatTool]:
    """Wrap every MCP tool in an adapter, filtering writes per config."""
    if mcp_registry is None:
        return []
    adapters: List[BaseChatTool] = []
    for name in mcp_registry.get_tool_names():
        if no_cache_writes and name in _WRITE_TOOLS:
            logger.debug("Skipping MCP tool %r (no_cache_writes=True)", name)
            continue
        try:
            adapters.append(MCPToolAdapter(name, mcp_registry))
        except ValueError as e:
            logger.warning("Could not adapt MCP tool %r: %s", name, e)
    return adapters
