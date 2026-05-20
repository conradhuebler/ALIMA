"""ChatToolRegistry — AgentLoop-compatible session-scoped tool registry.

WP10 P-δ.2. Claude Generated.

AgentLoop expects an object exposing:
- ``get_tool_schemas(names: Optional[List[str]]) -> List[Dict]``
- ``execute(name: str, args: Dict) -> str``

This registry adds session-binding and BaseChatTool semantics on top of
that contract.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.ui.chat_tools.base import BaseChatTool

logger = logging.getLogger(__name__)


class ChatToolRegistry:
    """Holds session-scoped chat tools and dispatches AgentLoop calls.

    Created fresh per chat turn so the bound session never leaks across
    conversations.
    """

    def __init__(self, session: Any):
        self._session = session
        self._tools: Dict[str, BaseChatTool] = {}

    def add(self, tool: BaseChatTool) -> None:
        """Register a tool. Raises ValueError on name collision."""
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has empty name")
        if tool.name in self._tools:
            raise ValueError(f"Tool name collision: {tool.name!r}")
        self._tools[tool.name] = tool

    def available_tools(self) -> List[str]:
        """Return registered tool names, sorted."""
        return sorted(self._tools.keys())

    def get_tool_schemas(
        self, tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """AgentLoop contract: return JSON-Schema dicts for selected tools.

        ``None`` returns all; an empty list also returns all (mirrors
        `ToolRegistry.get_tool_schemas` behaviour at
        ``src/mcp/tool_registry.py:40-44``).
        """
        if tool_names:
            return [
                self._tools[n].to_tool_schema()
                for n in tool_names
                if n in self._tools
            ]
        return [t.to_tool_schema() for t in self._tools.values()]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """AgentLoop contract: execute tool, return JSON string."""
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        tool = self._tools[tool_name]
        try:
            result = tool.execute(self._session, **(arguments or {}))
        except TypeError as e:
            logger.error("Tool %r arg error: %s", tool_name, e)
            return json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})
        except Exception as e:
            logger.exception("Tool %r execution error", tool_name)
            return json.dumps({"error": str(e)})
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
