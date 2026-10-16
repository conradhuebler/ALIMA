"""_generate_anthropic / _generate_anthropic_with_tools against real SDK shapes.

Found while testing Anthropic end-to-end for the first time: the package was
never installed (ImportError silently logged as ``warning``, looking like "not
configured" rather than "the dependency is missing"), and the request/response
handling had five independent defects, each confirmed against the installed
``anthropic`` SDK's own types rather than assumed:

* ``temperature``/``top_p`` were sent unconditionally. Claude Opus 4.7+ rejects
  both outright, every current model 400s on sending both at once, and Sonnet
  5.5 / Haiku 5.5 400 on any non-default value - ALIMA's fixed low-temperature
  tuning is non-default on all of them.
* ``seed`` was sent when set; the Messages API has no such field (the
  tool-calling sibling already excluded it from its signature for the same
  reason - P-eta).
* ``max_tokens`` was hardcoded to 1024 - no other provider in this text path
  caps output at all.
* The image branch imported ``ImageContent``/``ContentBlock``/``TextContent``
  from ``anthropic`` - none of these names exist in the installed SDK, so the
  ``except (ImportError, AttributeError)`` always fired and every "vision" call
  silently sent text only. The model then answered blind.
* Response parsing read ``content[0].text`` / ``chunk.delta.text``
  unconditionally. Every current model runs adaptive thinking by default and
  cannot disable it; a ``ThinkingBlock``/``ThinkingDelta`` carries ``.thinking``,
  not ``.text`` - ``AttributeError`` on the very first real call to a current
  model, both streaming and non-streaming.

Fixtures use the real ``anthropic.types`` block classes (not ad-hoc mocks) so a
future SDK rename shows up as a fixture failure here, not as a silent pass.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from anthropic.types import (
    RawContentBlockDeltaEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
)


def _make_service():
    """A bare LlmService with just enough state for _generate_anthropic*."""
    from src.llm.llm_service import LlmService

    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc.current_output_format = "json"
    svc.cancel_requested = False
    svc.current_request_id = "req-0"
    svc.last_chunk_time = 0.0
    svc.text_received = MagicMock()
    svc.generation_error = MagicMock()
    # Deliberately NOT "anthropic": self.clients is keyed by the configured
    # instance NAME, and "claude" is what exposed the hardcoded-literal bug
    # in the real config. A fixture keyed "anthropic" would pass by
    # coincidence even with that bug still in place. - Claude Generated
    svc.current_provider = "claude"
    return svc

#: The provider NAME these fixtures register the fake client under.
PROVIDER_NAME = "claude"


class _FakeNonStreamingMessages:
    """Records the request body; returns a fixed fake Message."""

    def __init__(self, response_content):
        self.captured_params = None
        self._response = MagicMock()
        self._response.content = response_content
        self._response.id = "msg_fake"

    def create(self, **params):
        self.captured_params = params
        return self._response


class _FakeStream:
    """Mimics the ``with client.messages.stream(**p) as s:`` contract.

    Supports both consumption styles: ``.text_stream`` (used by the plain
    text path) and direct iteration over raw delta events (used by the
    tool-calling path, which also needs thinking deltas - a channel
    ``.text_stream`` deliberately filters out). - Claude Generated
    """

    def __init__(self, text_chunks, final_id="msg_stream_fake", events=None):
        self._chunks = list(text_chunks)
        self._final_id = final_id
        self._events = list(events) if events is not None else [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0,
                delta=TextDelta(type="text_delta", text=chunk),
            )
            for chunk in self._chunks
        ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        """Synthesize the final Message the way the real SDK accumulates
        one: text/thinking deltas collapse into their own blocks, in the
        order they appeared. Not a mock stand-in - a real round trip needs
        the content blocks that were actually streamed, or a bug in how the
        final message is read (as opposed to the live deltas) would pass
        silently. - Claude Generated
        """
        content = []
        for event in self._events:
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                if content and content[-1].type == "text":
                    content[-1] = TextBlock(type="text", text=content[-1].text + delta.text)
                else:
                    content.append(TextBlock(type="text", text=delta.text))
            elif delta_type == "thinking_delta":
                if content and content[-1].type == "thinking":
                    content[-1] = ThinkingBlock(
                        type="thinking", thinking=content[-1].thinking + delta.thinking,
                        signature="sig",
                    )
                else:
                    content.append(ThinkingBlock(
                        type="thinking", thinking=delta.thinking, signature="sig",
                    ))
        msg = MagicMock()
        msg.id = self._final_id
        msg.content = content
        msg.stop_reason = "end_turn"
        return msg


class _FakeStreamingMessages:
    def __init__(self, text_chunks=None, events=None):
        self.captured_params = None
        self._text_chunks = text_chunks or []
        self._events = events

    def stream(self, **params):
        self.captured_params = params
        return _FakeStream(self._text_chunks, events=self._events)


class TextPathDoesNotSendRejectedFieldsTest(unittest.TestCase):
    """temperature/top_p/seed must never reach the Anthropic request body."""

    def test_non_streaming_request_carries_no_temperature_or_top_p(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertNotIn("temperature", fake_messages.captured_params)
        self.assertNotIn("top_p", fake_messages.captured_params)

    def test_non_streaming_request_carries_no_seed_even_when_set(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=42, stream=False,
        )
        self.assertNotIn("seed", fake_messages.captured_params)

    def test_streaming_request_carries_no_temperature_top_p_or_seed(self):
        svc = _make_service()
        fake_messages = _FakeStreamingMessages(["ok"])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=7, stream=True,
        )
        for forbidden in ("temperature", "top_p", "seed"):
            self.assertNotIn(forbidden, fake_messages.captured_params)

    def test_max_tokens_is_not_the_old_1024_cap(self):
        """1024 silently truncated every non-trivial ALIMA output."""
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertGreater(fake_messages.captured_params["max_tokens"], 1024)


class ToolPathDoesNotSendRejectedFieldsTest(unittest.TestCase):
    """Same bug, same fix, in the tool-calling generator."""

    def test_no_temperature_or_top_p_in_the_request(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        self.assertNotIn("temperature", fake_messages.captured_params)
        self.assertNotIn("top_p", fake_messages.captured_params)


class ResponseParsingSkipsThinkingBlocksTest(unittest.TestCase):
    """The headline bug: adaptive thinking is on by default on every current
    model and cannot be disabled - the first content block is routinely a
    ThinkingBlock, which has no ``.text``."""

    def test_a_leading_thinking_block_does_not_raise(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([
            ThinkingBlock(type="thinking", thinking="hmm", signature="sig"),
            TextBlock(type="text", text="die Antwort"),
        ])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertEqual(result, "die Antwort")

    def test_only_thinking_blocks_yields_empty_text_not_a_crash(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([
            ThinkingBlock(type="thinking", thinking="hmm", signature="sig"),
        ])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertEqual(result, "")

    def test_multiple_text_blocks_are_concatenated(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([
            TextBlock(type="text", text="erster Teil "),
            ThinkingBlock(type="thinking", thinking="hmm", signature="sig"),
            TextBlock(type="text", text="zweiter Teil"),
        ])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertEqual(result, "erster Teil zweiter Teil")


class StreamingUsesTheSdkTextStreamTest(unittest.TestCase):
    """Streaming must go through ``.stream()``/``.text_stream``, which already
    filters thinking deltas - not a hand-rolled raw-event loop."""

    def test_text_chunks_are_emitted_and_concatenated(self):
        svc = _make_service()
        fake_messages = _FakeStreamingMessages(["Hallo ", "Welt"])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=True,
        )
        self.assertEqual(result, "Hallo Welt")
        self.assertEqual(
            [c.args for c in svc.text_received.emit.call_args_list],
            [("req-0", "Hallo "), ("req-0", "Welt")],
        )

    def test_the_final_message_id_becomes_the_request_id(self):
        """So a later cancel request addresses the right generation."""
        svc = _make_service()
        fake_messages = _FakeStreamingMessages(["x"])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=True,
        )
        self.assertEqual(svc.current_request_id, "msg_stream_fake")

    def test_cancellation_stops_the_loop_early(self):
        svc = _make_service()

        class _CancellingStream(_FakeStream):
            @property
            def text_stream(self):
                for chunk in self._chunks:
                    yield chunk
                    svc.cancel_requested = True

        class _Messages:
            def stream(self, **params):
                return _CancellingStream(["a", "b", "c"])

        svc.clients = {PROVIDER_NAME: MagicMock(messages=_Messages())}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=True,
        )
        self.assertEqual(result, "a")


class VisionSendsRealContentBlocksTest(unittest.TestCase):
    """The old code imported classes that don't exist in the SDK and always
    fell back to text-only - every "vision" call answered blind."""

    @staticmethod
    def _real_image_bytes(fmt: str) -> bytes:
        """A real, tiny, decodable image - not a fixture that merely starts
        with the right magic bytes. The media-type bug (see below) only
        shows up against bytes PIL can actually open."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buf, format=fmt)
        return buf.getvalue()

    def _image_block(self, image_bytes):
        svc = _make_service()
        svc.process_image = MagicMock(return_value=image_bytes)
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        svc._generate_anthropic(
            model="claude-opus-5-5", prompt="Was zeigt das Bild?",
            temperature=0.1, p_value=0.1, seed=None,
            image="/tmp/does-not-matter", stream=False,
        )
        content = fake_messages.captured_params["messages"][0]["content"]
        self.assertIn("text", [b["type"] for b in content])
        return next(b for b in content if b["type"] == "image")

    def test_an_image_call_sends_an_image_content_block(self):
        image_block = self._image_block(self._real_image_bytes("JPEG"))
        self.assertEqual(image_block["source"]["type"], "base64")

    def test_a_png_is_labeled_png_not_hardcoded_jpeg(self):
        """The actual bug: media_type was "image/jpeg" unconditionally, and
        PNG is the first format the file picker offers. Anthropic validates
        the declared type against the bytes and rejects a mismatch."""
        image_block = self._image_block(self._real_image_bytes("PNG"))
        self.assertEqual(image_block["source"]["media_type"], "image/png")

    def test_a_gif_keeps_its_own_media_type(self):
        image_block = self._image_block(self._real_image_bytes("GIF"))
        self.assertEqual(image_block["source"]["media_type"], "image/gif")

    def test_a_webp_keeps_its_own_media_type(self):
        image_block = self._image_block(self._real_image_bytes("WEBP"))
        self.assertEqual(image_block["source"]["media_type"], "image/webp")

    def test_a_format_anthropic_does_not_accept_is_converted_to_png(self):
        """BMP and TIFF are in the same file-picker filter but Anthropic's
        vision input does not accept either - convert rather than let the
        call 400 on a file the UI itself offered."""
        for fmt in ("BMP", "TIFF"):
            with self.subTest(fmt=fmt):
                image_block = self._image_block(self._real_image_bytes(fmt))
                self.assertEqual(image_block["source"]["media_type"], "image/png")
                # And the re-encoded bytes must actually decode as PNG.
                import base64
                import io

                from PIL import Image

                decoded = base64.b64decode(image_block["source"]["data"])
                with Image.open(io.BytesIO(decoded)) as im:
                    self.assertEqual(im.format, "PNG")

    def test_undecodable_bytes_are_sent_rather_than_dropped(self):
        """Detection itself can fail (corrupt file) - the image must still be
        sent (with a best-guess media_type) rather than silently disappear,
        which is exactly the old failure mode this whole class is about."""
        image_block = self._image_block(b"this is not an image at all")
        self.assertEqual(image_block["type"], "image")
        self.assertIn(image_block["source"]["media_type"],
                      ("image/jpeg", "image/png", "image/gif", "image/webp"))

    def test_no_dead_import_of_nonexistent_sdk_classes(self):
        """Pin it structurally: these names never existed under this import."""
        import anthropic

        for name in ("ImageContent", "ContentBlock", "TextContent"):
            self.assertFalse(hasattr(anthropic, name), name)


if __name__ == "__main__":
    unittest.main()


class MissingDependencyIsVisibleTest(unittest.TestCase):
    """The actual root blocker: ``anthropic`` was never in requirements.txt and
    was never installed. The ImportError this produces was caught and logged
    at ``warning`` - indistinguishable from "provider simply not configured".
    This is the exact failure shape ``error_visibility.log_caught`` exists for
    (see its own docstring: an undeclared dependency made DNB lookups fail one
    id at a time, read as routine for months). - Claude Generated
    """

    def test_a_missing_module_is_logged_as_a_defect_not_a_warning(self):
        from types import SimpleNamespace

        from src.llm.llm_service import LlmService

        svc = LlmService.__new__(LlmService)
        svc.logger = MagicMock()
        svc.supported_providers = {
            "anthropic": {
                "module": "anthropic",
                "class": "Anthropic",
                "initializer": MagicMock(),
                "config": SimpleNamespace(provider_type="anthropic", api_key="x"),
            },
        }
        svc.clients = {}

        import sys
        import unittest.mock as mock

        with mock.patch.dict(sys.modules, {"anthropic": None}):
            svc._initialize_single_provider("anthropic")

        self.assertTrue(
            svc.logger.error.called,
            "a missing dependency must surface at ERROR, not warning",
        )


class ClientIsLookedUpByInstanceNameNotProviderTypeTest(unittest.TestCase):
    """The actual reported bug: "Error: Anthropic client not initialized".

    ``self.clients`` is keyed by the configured provider instance NAME
    (``UnifiedProvider.name``, e.g. "claude") - set in
    ``_init_unified_provider_configs``: ``provider_key = provider.name``.
    Both Anthropic generators hardcoded the literal string "anthropic" as the
    lookup key instead, so any instance not named exactly that raised "client
    not initialized" even though it *was* initialized, just under a different
    key. - Claude Generated
    """

    def test_tool_path_finds_the_client_under_its_configured_name(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        # Registered under "claude", NOT "anthropic".
        svc.clients = {"claude": MagicMock(messages=fake_messages)}

        # Must not raise, and must reach the real client.
        svc._generate_anthropic_with_tools(
            provider="claude", model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        self.assertIsNotNone(fake_messages.captured_params)

    def test_tool_path_raises_a_named_error_for_the_wrong_key(self):
        svc = _make_service()
        svc.clients = {"claude": MagicMock()}

        with self.assertRaises(ValueError) as ctx:
            svc._generate_anthropic_with_tools(
                provider="not-registered", model="claude-opus-5-5",
                messages=[{"role": "user", "content": "hi"}],
                tools=[],  temperature=0.1, top_p=0.1, max_tokens=4096,
            )
        self.assertIn("not-registered", str(ctx.exception))

    def test_text_path_finds_the_client_under_its_configured_name(self):
        svc = _make_service()
        svc.current_provider = "claude"
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {"claude": MagicMock(messages=fake_messages)}

        result = svc._generate_anthropic(
            model="claude-opus-5-5", prompt="hi",
            temperature=0.1, p_value=0.1, seed=None, stream=False,
        )
        self.assertEqual(result, "ok")

    def test_text_path_raises_a_named_error_for_the_wrong_key(self):
        svc = _make_service()
        svc.current_provider = "not-registered"
        svc.clients = {"claude": MagicMock()}

        with self.assertRaises(ValueError) as ctx:
            svc._generate_anthropic(
                model="claude-opus-5-5", prompt="hi",
                temperature=0.1, p_value=0.1, seed=None, stream=False,
            )
        self.assertIn("not-registered", str(ctx.exception))


class ParallelToolResultsMergeIntoOneTurnTest(unittest.TestCase):
    """The second real-world bug: HTTP 400 "unexpected tool_use_id found in
    tool_result blocks ... Each tool_result block must have a corresponding
    tool_use block in the previous message."

    AgentLoop appends one ``{"role": "tool", ...}`` dict per call when an
    assistant turn makes several calls in parallel (the default) - a shape
    OpenAI accepts as separate messages. Anthropic requires every tool_result
    for one assistant turn in a SINGLE user message; the second (and later)
    tool-result messages ended up each starting their own user turn, orphaned
    from the assistant's tool_use blocks. - Claude Generated
    """

    @staticmethod
    def _convert(svc_messages):
        """Drive the real conversion logic via a captured request."""
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}
        svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=svc_messages, tools=[],
            temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        return fake_messages.captured_params["messages"]

    def test_two_parallel_tool_results_land_in_one_user_message(self):
        messages = self._convert([
            {"role": "user", "content": "mach zwei Dinge"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_A", "name": "a", "input": {}},
                {"type": "tool_use", "id": "toolu_B", "name": "b", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "Ergebnis A", "name": "a"},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "Ergebnis B", "name": "b"},
        ])
        # Exactly one user turn follows the assistant turn, not two.
        self.assertEqual(len(messages), 3)
        tool_turn = messages[2]
        self.assertEqual(tool_turn["role"], "user")
        ids = [b["tool_use_id"] for b in tool_turn["content"]]
        self.assertEqual(ids, ["toolu_A", "toolu_B"])

    def test_three_parallel_tool_results_all_merge(self):
        messages = self._convert([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_A", "name": "a", "input": {}},
                {"type": "tool_use", "id": "toolu_B", "name": "b", "input": {}},
                {"type": "tool_use", "id": "toolu_C", "name": "c", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "A"},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "B"},
            {"role": "tool", "tool_call_id": "toolu_C", "content": "C"},
        ])
        self.assertEqual(len(messages), 2)
        ids = [b["tool_use_id"] for b in messages[1]["content"]]
        self.assertEqual(ids, ["toolu_A", "toolu_B", "toolu_C"])

    def test_a_single_tool_call_still_works(self):
        """No regression on the common, non-parallel case."""
        messages = self._convert([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_A", "name": "a", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "A"},
        ])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"][0]["tool_use_id"], "toolu_A")

    def test_two_separate_tool_turns_do_not_merge_into_each_other(self):
        """Two genuinely separate round-trips (assistant -> tool -> assistant
        -> tool) must stay as two user turns, not collapse into one."""
        messages = self._convert([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_A", "name": "a", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "A"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_B", "name": "b", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "B"},
        ])
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["content"][0]["tool_use_id"], "toolu_A")
        self.assertEqual(messages[3]["content"][0]["tool_use_id"], "toolu_B")

    def test_an_is_error_result_keeps_its_own_flag_after_merging(self):
        messages = self._convert([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_A", "name": "a", "input": {}},
                {"type": "tool_use", "id": "toolu_B", "name": "b", "input": {}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "ok", "is_error": False},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "boom", "is_error": True},
        ])
        blocks = {b["tool_use_id"]: b for b in messages[1]["content"]}
        self.assertFalse(blocks["toolu_A"]["is_error"])
        self.assertTrue(blocks["toolu_B"]["is_error"])

    def test_a_tool_result_does_not_merge_into_an_unrelated_user_turn(self):
        """A real user message with list-shaped content (e.g. multimodal
        image+text) directly before a tool-role entry must stay its own turn
        - the merge guard checks block *type*, not just "is a list". No
        assistant message in between: that is the only shape where the two
        candidate "previous" messages could be confused for each other."""
        messages = self._convert([
            {"role": "user", "content": [
                {"type": "text", "text": "beschreibe das Bild"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_A", "content": "A"},
        ])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertNotIn(
            "tool_result",
            [b.get("type") for b in messages[0]["content"]],
        )
        self.assertEqual(messages[1]["content"][0]["type"], "tool_result")


class GenericAssistantToolCallsConvertCorrectlyTest(unittest.TestCase):
    """The actual root cause of the user's reported 400, reproduced with the
    REAL generic shape AgentLoop stores - not a hand-built Anthropic-native
    fixture (which is exactly what let this slip through the first round of
    tests: every earlier fixture here already had ``content`` as a list of
    blocks, so none of them exercised the branch that actually fires).

    ``_build_assistant_tool_message`` stores
    ``{"content": "<text>", "tool_calls": [{"id", "name", "arguments"}]}`` -
    OpenAI's own wire shape, deliberately generic so each provider converts
    it ("Store in generic format - converters in LlmService will transform
    for each provider", its own docstring). Anthropic had no such converter:
    the assistant message's ``tool_use`` block was silently dropped, so the
    following ``tool_result`` referenced an id that appeared nowhere in the
    conversation - "unexpected tool_use_id found in tool_result blocks".
    - Claude Generated
    """

    @staticmethod
    def _convert(svc_messages):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}
        svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=svc_messages, tools=[],
            temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        return fake_messages.captured_params["messages"]

    def test_the_real_agent_loop_shape_produces_a_tool_use_block(self):
        messages = self._convert([
            {"role": "user", "content": "Welcher Fachreferent ist aktiv?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_01Rxj4977u", "name": "search_webindex",
                     "arguments": {"query": "Fachreferent", "max_results": 10}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01Rxj4977u", "content": "4 Felder"},
        ])
        self.assertEqual(len(messages), 3)
        assistant_turn = messages[1]
        self.assertEqual(assistant_turn["role"], "assistant")
        tool_use_blocks = [b for b in assistant_turn["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_use_blocks), 1)
        self.assertEqual(tool_use_blocks[0]["id"], "toolu_01Rxj4977u")
        self.assertEqual(tool_use_blocks[0]["name"], "search_webindex")
        self.assertEqual(
            tool_use_blocks[0]["input"],
            {"query": "Fachreferent", "max_results": 10},
        )
        # And the id the following tool_result references actually exists now.
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "toolu_01Rxj4977u")

    def test_text_alongside_a_tool_call_becomes_its_own_block(self):
        messages = self._convert([
            {
                "role": "assistant",
                "content": "Ich schaue das nach.",
                "tool_calls": [
                    {"id": "toolu_X", "name": "a", "arguments": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_X", "content": "A"},
        ])
        blocks = messages[0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "Ich schaue das nach."})
        self.assertEqual(blocks[1]["type"], "tool_use")

    def test_empty_text_produces_no_text_block(self):
        """The model often emits ONLY a tool_use block - a spurious empty
        text block there is itself a 400 on some validations."""
        messages = self._convert([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "toolu_X", "name": "a", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "toolu_X", "content": "A"},
        ])
        blocks = messages[0]["content"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "tool_use")

    def test_a_parallel_real_shape_turn_still_merges_its_results(self):
        """Combine both fixes: the real generic assistant shape AND two
        parallel calls in one turn."""
        messages = self._convert([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_A", "name": "a", "arguments": {}},
                    {"id": "toolu_B", "name": "b", "arguments": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_A", "content": "A"},
            {"role": "tool", "tool_call_id": "toolu_B", "content": "B"},
        ])
        self.assertEqual(len(messages), 2)
        tool_use_ids = [b["id"] for b in messages[0]["content"] if b["type"] == "tool_use"]
        self.assertEqual(tool_use_ids, ["toolu_A", "toolu_B"])
        tool_result_ids = [b["tool_use_id"] for b in messages[1]["content"]]
        self.assertEqual(tool_result_ids, ["toolu_A", "toolu_B"])

    def test_string_arguments_are_parsed_to_a_dict(self):
        """Defensive, matching the Ollama converter: arguments should already
        be a dict (ToolCall.arguments is typed as one), but a JSON string
        must not reach the API as a bare string in `input`."""
        messages = self._convert([
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "toolu_X", "name": "a", "arguments": '{"q": "x"}'}],
            },
            {"role": "tool", "tool_call_id": "toolu_X", "content": "A"},
        ])
        tool_use = next(b for b in messages[0]["content"] if b["type"] == "tool_use")
        self.assertEqual(tool_use["input"], {"q": "x"})

    def test_a_text_only_assistant_turn_is_unaffected(self):
        """No tool_calls key at all - must not be treated as a tool turn."""
        messages = self._convert([
            {"role": "user", "content": "hallo"},
            {"role": "assistant", "content": "hi!"},
        ])
        self.assertEqual(messages, [
            {"role": "user", "content": "hallo"},
            {"role": "assistant", "content": "hi!"},
        ])


class ThinkingIsStreamedInChatTest(unittest.TestCase):
    """Chat streaming for Anthropic: thinking_callback was accepted by every
    other provider's ``_with_tools`` generator but not Anthropic's, and
    ``.text_stream`` - the convenience the tool path already used for live
    text - deliberately filters out thinking deltas (confirmed from the
    installed SDK's own ``__stream_text__`` source), so even wiring the
    parameter through would have shown nothing without also switching to
    iterating the raw stream. - Claude Generated
    """

    @staticmethod
    def _run(events, thinking_callback=None, stream_callback=None):
        svc = _make_service()
        fake_messages = _FakeStreamingMessages(events=events)
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}
        response = svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.1, top_p=0.1, max_tokens=4096,
            stream_callback=stream_callback or MagicMock(),
            thinking_callback=thinking_callback,
        )
        return fake_messages, response

    def _events(self, *, text="Antwort", thinking="ich überlege kurz"):
        return [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0,
                delta=ThinkingDelta(type="thinking_delta", thinking=thinking),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=1,
                delta=TextDelta(type="text_delta", text=text),
            ),
        ]

    def test_thinking_requests_adaptive_display_summarized(self):
        fake_messages, _ = self._run(self._events(), thinking_callback=MagicMock())
        self.assertEqual(
            fake_messages.captured_params["thinking"],
            {"type": "adaptive", "display": "summarized"},
        )

    def test_no_thinking_callback_means_no_thinking_param(self):
        """An older model that may reject {"type": "adaptive"} must be
        unaffected when nobody asked to see the reasoning."""
        fake_messages, _ = self._run(self._events(), thinking_callback=None)
        self.assertNotIn("thinking", fake_messages.captured_params)

    def test_thinking_text_reaches_the_callback_live(self):
        sink = MagicMock()
        self._run(self._events(thinking="Schritt 1... Schritt 2..."), thinking_callback=sink)
        sink.assert_called_once_with("Schritt 1... Schritt 2...")

    def test_text_still_reaches_stream_callback(self):
        text_sink = MagicMock()
        self._run(
            self._events(text="die Antwort"),
            thinking_callback=MagicMock(), stream_callback=text_sink,
        )
        text_sink.assert_called_once_with("die Antwort")

    def test_reasoning_is_captured_on_the_response(self):
        _, response = self._run(self._events(thinking="mein Gedankengang"), thinking_callback=MagicMock())
        self.assertEqual(response.reasoning, "mein Gedankengang")

    def test_multiple_thinking_deltas_accumulate_in_order(self):
        events = [
            RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                      delta=ThinkingDelta(type="thinking_delta", thinking="A")),
            RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                      delta=ThinkingDelta(type="thinking_delta", thinking="B")),
            RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                      delta=ThinkingDelta(type="thinking_delta", thinking="C")),
        ]
        _, response = self._run(events, thinking_callback=MagicMock())
        self.assertEqual(response.reasoning, "ABC")

    def test_non_delta_events_are_ignored_not_crashed_on(self):
        """message_start/content_block_start/... carry no `.delta` at all."""
        events = [MagicMock(type="message_start")] + self._events()
        fake_messages, response = self._run(events, thinking_callback=MagicMock())
        self.assertIn("Antwort", response.content)


class NonStreamingReasoningIsCapturedTest(unittest.TestCase):
    """Same reasoning field, non-streaming path (chat without a live view,
    or a batch/background run)."""

    def test_a_leading_thinking_block_is_captured_as_reasoning(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([
            ThinkingBlock(type="thinking", thinking="mein Gedankengang", signature="sig"),
            TextBlock(type="text", text="die Antwort"),
        ])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        response = svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        self.assertEqual(response.content, "die Antwort")
        self.assertEqual(response.reasoning, "mein Gedankengang")

    def test_no_thinking_block_means_empty_reasoning_not_a_crash(self):
        svc = _make_service()
        fake_messages = _FakeNonStreamingMessages([TextBlock(type="text", text="ok")])
        svc.clients = {PROVIDER_NAME: MagicMock(messages=fake_messages)}

        response = svc._generate_anthropic_with_tools(
            provider=PROVIDER_NAME, model="claude-opus-5-5",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.1, top_p=0.1, max_tokens=4096,
        )
        self.assertEqual(response.reasoning, "")
