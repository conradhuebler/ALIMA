"""ChatSession — session-scoped state for the chat-agent layer.

WP10 P-δ.3. Claude Generated.

Holds the conversation transcript and a reference to the current
``SharedContext`` snapshot from the last pipeline run. Passed to
``build_chat_toolset`` (see ``src/ui/chat_tools/__init__.py``) to bind
tools to the session, and to ``ChatAgentWorker.run()`` as the LLM
conversation backbone.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatSession:
    """Container for a single chat conversation.

    Attributes:
        session_id: Stable UUID for log correlation.
        messages: Chat history as a list of ``{"role", "content"}`` dicts.
        last_shared_context: Snapshot of the latest pipeline ``SharedContext``;
            None until a pipeline has run. Tools gate their availability on
            populated fields here (see ``BaseChatTool.available_for``).
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_shared_context: Optional[Any] = None  # SharedContext (avoid circular import)
    # Provider/model the agent asked for via ``switch_llm_model`` (only reachable
    # when ``ChatConfig.allow_model_switch`` is on). Applies from the next turn —
    # the running turn's loop is already bound to its model — and is cleared when
    # the operator changes the toolbar pick. Never persisted. - Claude Generated
    requested_provider: str = ""
    requested_model: str = ""

    def append(self, role: str, content: str) -> None:
        """Append a turn. ``role`` should be 'user' or 'assistant'."""
        self.messages.append({"role": role, "content": content})

    def clear_requested_model(self) -> None:
        """Drop an agent-requested model — the operator picked one. - Claude Generated"""
        self.requested_provider = ""
        self.requested_model = ""

    def reset(self) -> None:
        """Clear the transcript. Keeps session_id + last_shared_context."""
        self.messages = []
