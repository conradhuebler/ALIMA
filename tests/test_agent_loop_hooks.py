"""Tests for AgentLoop hooks added in WP10 P-δ.3. Claude Generated.

Two cases:
1. ``on_tool_call`` + ``on_tool_result`` fire exactly once per tool dispatch.
2. ``should_stop`` breaks the loop between iterations.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop, ThinkStreamFilter
from src.core.data_models import AgentResponse, ToolCall


def _make_registry() -> MagicMock:
    """Mock ToolRegistry: schema-less, execute returns fixed JSON."""
    reg = MagicMock()
    reg.get_tool_schemas.return_value = [
        {"name": "get_keywords", "description": "", "parameters": {}}
    ]
    reg.execute.return_value = '{"count": 3, "keywords": ["A", "B", "C"]}'
    return reg


def _make_llm_service(responses):
    """Mock LlmService whose ``generate_with_tools`` returns each
    ``AgentResponse`` in order then raises ``StopIteration``-ish behavior
    by repeating the last one."""
    svc = MagicMock()
    it = iter(responses)
    last = [None]

    def _gen(**kwargs):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    svc.generate_with_tools.side_effect = _gen
    return svc


class TestAgentLoopHooks(unittest.TestCase):

    def test_on_tool_call_and_result_invoked_per_dispatch(self):
        """One tool-dispatch ⇒ on_tool_call called once, on_tool_result once."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"})
        responses = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),  # final answer
        ]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        call_log = []
        result_log = []

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            on_tool_call=lambda tc: call_log.append(tc),
            on_tool_result=lambda name, res: result_log.append((name, res)),
        )
        result = loop.run(
            system_prompt="sys",
            user_prompt="ask",
            tools=["get_keywords"],
            provider="ollama",
            model="m",
        )

        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0].name, "get_keywords")
        self.assertEqual(call_log[0].arguments, {"kind": "initial"})

        self.assertEqual(len(result_log), 1)
        name, result_str = result_log[0]
        self.assertEqual(name, "get_keywords")
        self.assertIn('"count": 3', result_str)

        self.assertEqual(result.content, "Done.")

    def test_should_stop_breaks_loop_between_iterations(self):
        """should_stop returning True after first call halts the loop."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={})
        # Provide many tool-call responses; the loop should stop before
        # consuming most of them.
        responses = [
            AgentResponse(content="", tool_calls=[tool_call]) for _ in range(10)
        ]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        stop_counter = {"calls": 0}

        def _should_stop() -> bool:
            stop_counter["calls"] += 1
            # Halt on the second invocation, so we get at least 1 iteration.
            return stop_counter["calls"] >= 2

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            max_iterations=10,
            should_stop=_should_stop,
        )
        loop.run(system_prompt="sys", user_prompt="ask", tools=[], provider="p", model="m")

        # Only one LLM call should have happened before should_stop returned True.
        self.assertEqual(llm.generate_with_tools.call_count, 1)

    def test_should_stop_breaks_mid_tool_batch(self):
        """A multi-tool response stops after the in-flight tool, not all of them.

        should_stop returns False at the iteration boundary and for the first
        tool, then True — so only one of three queued tools executes. - Claude Generated"""
        batch = [
            ToolCall(id="t1", name="get_keywords", arguments={"i": 1}),
            ToolCall(id="t2", name="get_keywords", arguments={"i": 2}),
            ToolCall(id="t3", name="get_keywords", arguments={"i": 3}),
        ]
        responses = [AgentResponse(content="", tool_calls=batch) for _ in range(5)]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        calls = {"n": 0}

        def _should_stop() -> bool:
            calls["n"] += 1
            # boundary(False) → tc1(False) → tc2(True)
            return calls["n"] >= 3

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            max_iterations=5,
            should_stop=_should_stop,
        )
        loop.run(system_prompt="sys", user_prompt="ask", tools=[], provider="p", model="m")

        # Only the first queued tool executed before the mid-batch break.
        self.assertEqual(registry.execute.call_count, 1)

    def test_status_callback_suppressed_when_on_tool_call_set(self):
        """Phase F: when ``on_tool_call`` is wired, status lines for the
        call/result are suppressed — the bus/hook already carries the
        same info, so the chat log no longer shows duplicates."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"})

        # Case 1: ``on_tool_call`` set ⇒ status callback is silent.
        responses1 = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),
        ]
        llm1 = _make_llm_service(responses1)
        registry1 = _make_registry()
        status_log: list[str] = []
        loop = AgentLoop(
            llm_service=llm1,
            tool_registry=registry1,
            status_callback=status_log.append,
            on_tool_call=lambda tc: None,
        )
        loop.run(system_prompt="sys", user_prompt="ask", tools=["get_keywords"], provider="p", model="m")
        # The legacy "  🔧 …" / "    ✓ …" lines are gone.
        self.assertEqual([s for s in status_log if s.lstrip().startswith("🔧")], [])
        self.assertEqual([s for s in status_log if s.lstrip().startswith("✓")], [])

        # Case 2: ``on_tool_call`` NOT set ⇒ status callback still
        # receives the legacy lines (backwards-compat for headless CLI).
        # Fresh response list — case 1 already consumed the iterator.
        responses2 = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),
        ]
        llm2 = _make_llm_service(responses2)
        registry2 = _make_registry()
        status_log2: list[str] = []
        loop2 = AgentLoop(
            llm_service=llm2,
            tool_registry=registry2,
            status_callback=status_log2.append,
        )
        loop2.run(system_prompt="sys", user_prompt="ask", tools=["get_keywords"], provider="p", model="m")
        self.assertTrue(any(s.lstrip().startswith("🔧") for s in status_log2))
        self.assertTrue(any(s.lstrip().startswith("✓") for s in status_log2))


class TestThinkStreamFilter(unittest.TestCase):
    """<think> stream filter: diversion, tag splits, flush semantics. Claude Generated."""

    def _run(self, chunks):
        texts, thinks = [], []
        f = ThinkStreamFilter(texts.append, thinks.append)
        for c in chunks:
            f.feed(c)
        f.flush()
        return "".join(texts), "".join(thinks)

    def test_plain_text_passthrough(self):
        text, think = self._run(["Hello ", "world"])
        self.assertEqual(text, "Hello world")
        self.assertEqual(think, "")

    def test_think_content_diverted(self):
        text, think = self._run(["a<think>x</think>b"])
        self.assertEqual(text, "ab")
        self.assertEqual(think, "x")

    def test_tags_split_across_token_boundaries(self):
        full = "pre<think>secret</think>post"
        # Every possible single split point of the whole stream.
        for i in range(1, len(full)):
            text, think = self._run([full[:i], full[i:]])
            self.assertEqual(text, "prepost", f"split at {i}")
            self.assertEqual(think, "secret", f"split at {i}")
        # Char-by-char (worst case).
        text, think = self._run(list(full))
        self.assertEqual(text, "prepost")
        self.assertEqual(think, "secret")

    def test_unclosed_think_flushes_to_thinking(self):
        text, think = self._run(["a<think>never closed"])
        self.assertEqual(text, "a")
        self.assertEqual(think, "never closed")

    def test_lone_angle_bracket_stays_text(self):
        text, think = self._run(list("a < b and c > d, <thin fabric>"))
        self.assertEqual(text, "a < b and c > d, <thin fabric>")
        self.assertEqual(think, "")

    def test_multiple_think_segments(self):
        text, think = self._run(["<think>one</think>A<think>two</think>B"])
        self.assertEqual(text, "AB")
        self.assertEqual(think, "onetwo")


class TestAgentLoopThinking(unittest.TestCase):
    """on_thinking wiring, think-stripping, separators. Claude Generated."""

    def _streaming_llm(self, chunks, content):
        """LLM mock that streams ``chunks`` via stream_callback then returns
        a final (no-tool) response with ``content``."""
        svc = MagicMock()

        def _gen(**kwargs):
            cb = kwargs.get("stream_callback")
            if cb:
                for c in chunks:
                    cb(c)
            return AgentResponse(content=content, tool_calls=[])

        svc.generate_with_tools.side_effect = _gen
        return svc

    def test_think_stream_diverted_to_on_thinking(self):
        llm = self._streaming_llm(
            ["<think>plan", "ning</think>", "Antwort"],
            "<think>planning</think>Antwort",
        )
        tokens, thinks = [], []
        loop = AgentLoop(
            llm_service=llm,
            tool_registry=_make_registry(),
            stream_callback=tokens.append,
            status_callback=lambda s: None,
            on_thinking=thinks.append,
        )
        result = loop.run(system_prompt="s", user_prompt="u", tools=[], provider="p", model="m")
        self.assertEqual("".join(tokens), "Antwort")
        self.assertEqual("".join(thinks), "planning")
        # Persisted/final content is think-stripped.
        self.assertEqual(result.content, "Antwort")
        assistant_msgs = [m for m in result.messages if m.get("role") == "assistant"]
        self.assertTrue(all("<think>" not in (m.get("content") or "") for m in assistant_msgs))

    def test_stream_unfiltered_without_on_thinking(self):
        llm = self._streaming_llm(
            ["<think>plan</think>", "Antwort"], "<think>plan</think>Antwort"
        )
        tokens = []
        loop = AgentLoop(
            llm_service=llm,
            tool_registry=_make_registry(),
            stream_callback=tokens.append,
            status_callback=lambda s: None,
        )
        result = loop.run(system_prompt="s", user_prompt="u", tools=[], provider="p", model="m")
        # Pipeline back-compat: without on_thinking the stream is untouched.
        self.assertEqual("".join(tokens), "<think>plan</think>Antwort")
        # AgentResult content is still think-stripped.
        self.assertEqual(result.content, "Antwort")

    def test_reasoning_channel_routed_to_on_thinking(self):
        responses = [AgentResponse(content="Done.", tool_calls=[], reasoning="deep thought")]
        llm = _make_llm_service(responses)
        thinks = []
        loop = AgentLoop(
            llm_service=llm,
            tool_registry=_make_registry(),
            stream_callback=lambda t: None,
            on_thinking=thinks.append,
        )
        result = loop.run(system_prompt="s", user_prompt="u", tools=[], provider="p", model="m")
        self.assertEqual(thinks, ["deep thought"])
        self.assertEqual(result.content, "Done.")

    def test_reasoning_excerpt_suppressed_when_streaming(self):
        """The 💭 status excerpt duplicates streamed prose → only emitted
        when no stream_callback is wired."""
        tc = ToolCall(id="t1", name="get_keywords", arguments={})
        responses = [
            AgentResponse(content="Ich suche jetzt.", tool_calls=[tc]),
            AgentResponse(content="Fertig.", tool_calls=[]),
        ]
        llm = _make_llm_service(responses)
        status_log = []
        loop = AgentLoop(
            llm_service=llm,
            tool_registry=_make_registry(),
            stream_callback=lambda t: None,
            status_callback=status_log.append,
        )
        loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")
        self.assertEqual([s for s in status_log if "💭" in s], [])

        # Without streaming the excerpt still appears (CLI/status-only paths).
        responses2 = [
            AgentResponse(content="Ich suche jetzt.", tool_calls=[tc]),
            AgentResponse(content="Fertig.", tool_calls=[]),
        ]
        llm2 = _make_llm_service(responses2)
        status_log2 = []
        loop2 = AgentLoop(
            llm_service=llm2,
            tool_registry=_make_registry(),
            status_callback=status_log2.append,
        )
        loop2.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")
        self.assertTrue(any("💭" in s for s in status_log2))

    def test_tool_turn_prose_joined_with_blank_line(self):
        """Prose accumulated across tool turns gets a paragraph separator."""
        tc = ToolCall(id="t1", name="get_keywords", arguments={})
        responses = [
            AgentResponse(content="Erst suchen.", tool_calls=[tc]),
            AgentResponse(content="Dann filtern.", tool_calls=[tc]),
        ]
        llm = _make_llm_service(responses)
        loop = AgentLoop(
            llm_service=llm,
            tool_registry=_make_registry(),
            max_iterations=2,
            status_callback=lambda s: None,
        )
        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")
        self.assertEqual(result.content, "Erst suchen.\n\nDann filtern.")

    def test_max_iterations_forced_answer_is_streamed(self):
        """The forced final answer (max iterations, empty prose) must reach
        the stream so per-iteration bubble finalize doesn't lose it."""
        tc = ToolCall(id="t1", name="get_keywords", arguments={})

        svc = MagicMock()

        def _gen(**kwargs):
            if kwargs.get("tools"):
                return AgentResponse(content="", tool_calls=[tc])
            return AgentResponse(content="Erzwungene Antwort.", tool_calls=[])

        svc.generate_with_tools.side_effect = _gen
        tokens = []
        loop = AgentLoop(
            llm_service=svc,
            tool_registry=_make_registry(),
            max_iterations=2,
            stream_callback=tokens.append,
            status_callback=lambda s: None,
        )
        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")
        self.assertEqual("".join(tokens), "Erzwungene Antwort.")
        self.assertEqual(result.content, "Erzwungene Antwort.")


if __name__ == "__main__":
    unittest.main()
