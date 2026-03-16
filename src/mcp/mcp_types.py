"""Shared types for ALIMA MCP tool system - Claude Generated"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


@dataclass
class ToolDefinition:
    """Schema definition for a tool that agents can call."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None  # Set by registry

    def to_schema(self) -> Dict[str, Any]:
        """Export as JSON Schema for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
