"""trim_history_to_turn_boundary: the chat history window used to cut mid-turn.

Both the Qt6 chat panel (``session.messages[-CHAT_HISTORY_WINDOW:]``) and the
webapp (``prior_history[-CHAT_HISTORY_WINDOW:]``) sliced the last N messages
by raw count. Once a session has had enough tool exchanges, that cut point
lands inside one: either on an orphaned "tool" message (its matching
assistant ``tool_use`` fell outside the window) or on an "assistant" message
(Anthropic requires the first message overall to be "user"). Either shape
400s on the very next turn - exactly the class of bug this session spent
most of its time fixing in message *conversion*; this is the same failure
produced by message *selection* instead. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.core.chat_prompts import trim_history_to_turn_boundary


def _user(text="u"):
    return {"role": "user", "content": text}


def _assistant(text="a", tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(name="t", call_id="id", content="r"):
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": content}


class ShortHistoryIsReturnedWholeTest(unittest.TestCase):
    def test_fewer_messages_than_the_window_come_back_unchanged(self):
        messages = [_user("1"), _assistant("2")]
        self.assertEqual(trim_history_to_turn_boundary(messages, window=5), messages)

    def test_empty_history(self):
        self.assertEqual(trim_history_to_turn_boundary([], window=5), [])


class NaiveCutWouldStartOnAnOrphanedToolMessageTest(unittest.TestCase):
    """The reported mechanism: enough tool exchanges that a plain [-N:] slice
    begins on a "tool" message whose assistant tool_use is now out of frame."""

    def test_the_window_extends_back_to_the_user_turn_boundary(self):
        messages = [
            _user("Frage 1"),
            _assistant("ruft Tool", tool_calls=[{"id": "A", "name": "x", "arguments": {}}]),
            _tool(call_id="A"),
            _assistant("Antwort 1"),
        ]
        # window=2 would naively cut to [tool, assistant] - orphaned tool first.
        trimmed = trim_history_to_turn_boundary(messages, window=2)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed, messages)  # the whole turn, not a fragment

    def test_does_not_over_extend_past_the_nearest_safe_boundary(self):
        messages = [
            _user("Frage 1"),
            _assistant("Antwort 1"),
            _user("Frage 2"),
            _assistant("ruft Tool", tool_calls=[{"id": "B", "name": "x", "arguments": {}}]),
            _tool(call_id="B"),
            _assistant("Antwort 2"),
        ]
        # Naive cut for window=3 lands on index 3 (assistant-with-tool_calls) -
        # must walk back only to index 2 ("Frage 2"), not all the way to index 0.
        trimmed = trim_history_to_turn_boundary(messages, window=3)
        self.assertEqual(trimmed, messages[2:])


class NaiveCutWouldStartOnAnAssistantMessageTest(unittest.TestCase):
    """Anthropic requires the first message to be "user" - a cut landing on
    a plain assistant reply (no pending tool) is still an invalid shape."""

    def test_the_window_extends_back_past_a_leading_assistant_message(self):
        messages = [
            _user("Frage 1"),
            _assistant("Antwort 1"),
            _assistant("zweite Antwortzeile"),  # pathological but must not crash
        ]
        trimmed = trim_history_to_turn_boundary(messages, window=2)
        self.assertEqual(trimmed[0]["role"], "user")


class ManyTurnsOfHistoryStillTrimsNormallyTest(unittest.TestCase):
    """The common case - several clean, tool-free turns - must still produce
    a tight window, not silently fall back to the whole history."""

    def test_a_long_tool_free_history_trims_to_the_requested_window(self):
        messages = []
        for i in range(10):
            messages.append(_user(f"Frage {i}"))
            messages.append(_assistant(f"Antwort {i}"))
        trimmed = trim_history_to_turn_boundary(messages, window=4)
        self.assertEqual(len(trimmed), 4)
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed, messages[-4:])


class ZeroOrNegativeWindowTest(unittest.TestCase):
    def test_a_window_of_zero_returns_nothing(self):
        self.assertEqual(trim_history_to_turn_boundary([_user(), _assistant()], window=0), [])


if __name__ == "__main__":
    unittest.main()
