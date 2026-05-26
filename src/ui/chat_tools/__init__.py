"""Chat-Tool-Layer — WP10 P-δ.2. Claude Generated.

Public entry points:

* ``BaseChatTool``           — ABC for session-scoped chat tools.
* ``ChatToolRegistry``       — AgentLoop-compatible registry.
* ``build_chat_toolset``     — factory that assembles a session-bound
  registry from the four tool layers (generic, ALIMA, MCP-adapter).
"""
from __future__ import annotations

from typing import Any, List

from src.ui.chat_tools.base import BaseChatTool
from src.ui.chat_tools.registry import ChatToolRegistry
from src.ui.chat_tools.generic import generic_tools
from src.ui.chat_tools.alima import alima_tools
from src.ui.chat_tools.mcp_adapter import build_mcp_chat_tools


def build_chat_toolset(
    session: Any,
    *,
    chat_config: Any,
    mcp_registry: Any = None,
    pipeline_manager: Any = None,
    kb_manager: Any = None,
    proposal_gateway: Any = None,
) -> ChatToolRegistry:
    """Assemble a ChatToolRegistry for the given chat session.

    Layers in order:
      1. Generic SharedContext tools     (`generic.py`)
      2. ALIMA read-only pipeline tools  (`alima.py`)
      3. Mutation tools (P-ε)            (`mutations.py`) — only when
         ``pipeline_manager``, ``kb_manager`` and ``proposal_gateway``
         are all supplied. Without them the agent gets read-only access.
      4. MCP read-only adapter           (`mcp_adapter.py`)

    Tools whose ``available_for(session)`` returns False are skipped, so
    the agent never sees options that would no-op on empty state.

    ``chat_config`` is expected to have ``no_cache_writes: bool``
    (see ``src/utils/config_models.py:716-730``). MCP write tools are
    filtered when ``no_cache_writes`` is True.
    """
    registry = ChatToolRegistry(session=session)

    candidates: List[BaseChatTool] = []
    candidates.extend(generic_tools())
    candidates.extend(alima_tools(mcp_registry=mcp_registry))

    if pipeline_manager is not None and kb_manager is not None and proposal_gateway is not None:
        from src.ui.chat_tools.mutations import mutation_tools
        from src.ui.chat_tools.pipeline import pipeline_tools
        session_id = getattr(session, "session_id", "")
        common_kwargs = dict(
            pipeline_manager=pipeline_manager,
            kb_manager=kb_manager,
            gateway=proposal_gateway,
            chat_config=chat_config,
            session_id=str(session_id),
        )
        candidates.extend(mutation_tools(**common_kwargs))
        candidates.extend(pipeline_tools(**common_kwargs))

    for tool in candidates:
        if tool.available_for(session):
            registry.add(tool)

    no_cache_writes = bool(getattr(chat_config, "no_cache_writes", True))
    for mcp_tool in build_mcp_chat_tools(
        no_cache_writes=no_cache_writes,
        mcp_registry=mcp_registry,
    ):
        try:
            registry.add(mcp_tool)
        except ValueError:
            # Name collision with a session tool — session tools win.
            continue

    return registry


__all__ = [
    "BaseChatTool",
    "ChatToolRegistry",
    "build_chat_toolset",
    "generic_tools",
    "alima_tools",
    "build_mcp_chat_tools",
]
