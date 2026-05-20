"""BaseChatTool ABC — WP10 P-δ.2 Chat-Tool-Layer. Claude Generated.

Chat-Tools differ from MCP-Tools: they are session-scoped (each chat turn
binds to a SharedContext snapshot) instead of global. The ABC mirrors the
MCP `ToolDefinition.to_schema()` shape so an AgentLoop-compatible registry
can expose both side-by-side.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseChatTool(ABC):
    """Session-scoped chat tool.

    Subclasses set the class-level attributes ``name``, ``description``,
    ``parameters_schema`` (JSON-Schema subset) and implement
    ``available_for`` + ``execute``.
    """

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        """Return True if this tool can run against the current session.

        Default: always available. ALIMA tools override to gate on populated
        SharedContext fields.
        """
        return True

    @abstractmethod
    def execute(self, session: Any, **kwargs: Any) -> str:
        """Execute the tool. MUST return a JSON-serializable string."""

    def to_tool_schema(self) -> Dict[str, Any]:
        """Export as JSON-Schema dict for LLM consumption.

        Shape matches `ToolDefinition.to_schema()` in `src/mcp/mcp_types.py`
        so the same downstream LlmService converters apply.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
