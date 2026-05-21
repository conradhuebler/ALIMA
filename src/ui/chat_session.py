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

    def append(self, role: str, content: str) -> None:
        """Append a turn. ``role`` should be 'user' or 'assistant'."""
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        """Clear the transcript. Keeps session_id + last_shared_context."""
        self.messages = []
