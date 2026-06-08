"""MessageEntry — unified message data-structure for chat + pipeline log.

Claude Generated.

Provides a single semantic envelope for every item that can appear in the
shared PipelineChatPanel QTextBrowser.  Used by UnifiedMessageRenderer for
history tracking and by session-export code.

Streaming tokens are *not* recorded as MessageEntry rows; they are renderer
state that gets folded into the final message when streaming ends.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict


class MessageRole(Enum):
    PIPELINE_LOG = auto()       # Timestamped pipeline line (add_pipeline_message)
    USER_BUBBLE = auto()        # WhatsApp-style green bubble
    ASSISTANT_BUBBLE = auto()   # Grey markdown bubble
    TOOL_MARKER = auto()        # Monospace tool call / result
    SYSTEM_MESSAGE = auto()     # Centered italic status
    PROPOSAL_BUBBLE = auto()    # Mutation proposal with clickable anchors
    RESULT_CARD = auto()        # Pre-formatted rich HTML result block (DK/RVK card)


@dataclass
class MessageEntry:
    """One fully-rendered message in the shared log.

    Fields
    ------
    role
        Semantic category — drives export formatting and test assertions.
    content
        Plain-text content (HTML tags stripped).  For ASSISTANT_BUBBLE this
        is the *final* text after Markdown rendering.
    timestamp
        Local creation time.
    metadata
        Role-specific extras (see below).

    Metadata keys by role
    ---------------------
    PIPELINE_LOG:
        ``level`` (str), ``step_id`` (Optional[str])
    ASSISTANT_BUBBLE:
        ``model_label`` (str), ``rendered_markdown`` (bool)
    TOOL_MARKER:
        ``tool_name`` (Optional[str])
    PROPOSAL_BUBBLE:
        ``audit_id`` (int), ``tool_name`` (str)
    RESULT_CARD:
        ``kind`` (Optional[str]) — e.g. "dk_classifications", "dk_search"
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
