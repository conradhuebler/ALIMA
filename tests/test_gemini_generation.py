"""_generate_gemini / _generate_gemini_with_tools against real google-genai
shapes.

Gemini was completely dead before this: the code imported
``google.generativeai`` and called module-level ``configure()``/
``GenerativeModel()``, but the pinned, installed package is ``google-genai``
(module ``google.genai``), whose API is a ``Client()`` object with
``client.models.*`` methods and no module-level configure at all -
``import google.generativeai`` raises ``ModuleNotFoundError`` outright.

Ported to the real installed SDK, fixing four confirmed defects the dead
import had hidden behind "nobody could reach this code" the same way the
Anthropic path hid its own five:

* ``self.clients`` keyed by the literal ``"gemini"`` instead of the
  configured instance name (same bug as Anthropic's pre-fix generators).
* A configured model not starting with "gemini-" was silently replaced with
  "gemini-1.5-flash" - no error, just the wrong model ran.
* ``hasattr(part, "function_call")`` is always True (it is a declared
  pydantic field, present and ``None`` on every text-only part too) - the old
  code would have treated every text part as a tool call.
* Two parallel calls to the same tool in one turn produced the identical id
  (``f"gemini_{name}"``), and the IDs reaching Anthropic/OpenAI tool_result
  pairing must be unique within a turn.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from google.genai import types as genai_types


def _make_service():
    from src.llm.llm_service import LlmService

    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc.current_output_format = "json"
    svc.cancel_requested = False
    svc.current_request_id = "req-0"
    svc.current_provider = "my-gemini"
    svc.text_received = MagicMock()
    svc.generation_error = MagicMock()
    return svc


#: Deliberately not "gemini" - the real-world config names instances freely,
#: and that mismatch is exactly what made every call fail.
PROVIDER_NAME = "my-gemini"


def _response(parts):
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(
            content=genai_types.Content(role="model", parts=parts),
        )],
    )


class _FakeModels:
    """Records the request; returns a fixed fake response."""

    def __init__(self, response=None, stream_chunks=None):
        self.captured_kwargs = None
        self._response = response
        self._stream_chunks = stream_chunks or []

    def generate_content(self, **kwargs):
        self.captured_kwargs = kwargs
        return self._response

    def generate_content_stream(self, **kwargs):
        self.captured_kwargs = kwargs
        return iter(self._stream_chunks)


def _text_chunk(text):
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(
            content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=text)]),
        )],
    )


class InitUsesTheRealClientTest(unittest.TestCase):
    def test_init_constructs_a_client_not_a_bare_module(self):
        from src.llm.llm_service import LlmService
        from google.genai import Client

        svc = LlmService.__new__(LlmService)
        svc.clients = {}
        svc._init_gemini(PROVIDER_NAME, __import__("google.genai", fromlist=["Client"]),
                         "sk-test", {"class": "Client"})
        self.assertIsInstance(svc.clients[PROVIDER_NAME], Client)


class ClientIsLookedUpByInstanceNameTest(unittest.TestCase):
    """self.clients keyed by the configured name, not the literal "gemini"."""

    def test_text_path_raises_a_named_error_for_the_wrong_key(self):
        svc = _make_service()
        svc.clients = {"some-other-name": MagicMock()}
        with self.assertRaises(ValueError) as ctx:
            svc._generate_gemini(
                model="gemini-2.5-flash", prompt="hi",
                temperature=0.5, p_value=0.9, seed=None, stream=False,
            )
        self.assertIn(PROVIDER_NAME, str(ctx.exception))

    def test_tool_path_raises_a_named_error_for_the_wrong_key(self):
        svc = _make_service()
        svc.clients = {"some-other-name": MagicMock()}
        with self.assertRaises(ValueError) as ctx:
            svc._generate_gemini_with_tools(
                provider=PROVIDER_NAME, model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}], tools=[],
                temperature=0.5, top_p=0.9, max_tokens=128,
            )
        self.assertIn(PROVIDER_NAME, str(ctx.exception))

    def test_tool_path_finds_the_client_under_its_configured_name(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(response.content, "ok")


class ModelStringIsNeverSilentlySubstitutedTest(unittest.TestCase):
    """A configured model not starting with "gemini-" used to be silently
    replaced with "gemini-1.5-flash" - no error, just the wrong model ran."""

    def test_text_path_passes_the_model_through_unmodified(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini(
            model="models/my-custom-alias", prompt="hi",
            temperature=0.5, p_value=0.9, seed=None, stream=False,
        )
        self.assertEqual(fake_models.captured_kwargs["model"], "models/my-custom-alias")

    def test_tool_path_passes_the_model_through_unmodified(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-3.0-ultra-preview",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(fake_models.captured_kwargs["model"], "gemini-3.0-ultra-preview")


class TextPathStreamingAndConfigTest(unittest.TestCase):
    def test_non_streaming_returns_the_response_text(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="die Antwort")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        result = svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=None, stream=False,
        )
        self.assertEqual(result, "die Antwort")
        config = fake_models.captured_kwargs["config"]
        self.assertEqual(config.temperature, 0.3)
        self.assertEqual(config.top_p, 0.8)

    def test_streaming_emits_each_chunk_and_concatenates(self):
        svc = _make_service()
        fake_models = _FakeModels(stream_chunks=[_text_chunk("Hallo "), _text_chunk("Welt")])
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        result = svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=None, stream=True,
        )
        self.assertEqual(result, "Hallo Welt")
        self.assertEqual(
            [c.args for c in svc.text_received.emit.call_args_list],
            [("req-0", "Hallo "), ("req-0", "Welt")],
        )

    def test_json_mode_is_requested_unless_xml(self):
        svc = _make_service()
        svc.current_output_format = "json"
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="{}")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=None, stream=False,
        )
        self.assertEqual(
            fake_models.captured_kwargs["config"].response_mime_type, "application/json",
        )

    def test_xml_output_format_skips_json_mode(self):
        """OCR/vision steps request "xml" specifically to get raw text back -
        forcing JSON mode there breaks them (separate, known finding; this
        pins that the Gemini port does not newly introduce the same bug)."""
        svc = _make_service()
        svc.current_output_format = "xml"
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="plain")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=None, stream=False,
        )
        self.assertIsNone(fake_models.captured_kwargs["config"].response_mime_type)

    def test_seed_reaches_the_config_when_set(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=42, stream=False,
        )
        self.assertEqual(fake_models.captured_kwargs["config"].seed, 42)

    def test_system_prompt_reaches_the_config(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi", system="Du bist hilfreich.",
            temperature=0.3, p_value=0.8, seed=None, stream=False,
        )
        self.assertEqual(
            fake_models.captured_kwargs["config"].system_instruction, "Du bist hilfreich.",
        )


class VisionSendsARealImagePartTest(unittest.TestCase):
    @staticmethod
    def _real_png_bytes():
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(buf, format="PNG")
        return buf.getvalue()

    def test_an_image_call_sends_a_part_with_the_real_media_type(self):
        svc = _make_service()
        svc.process_image = MagicMock(return_value=self._real_png_bytes())
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}

        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="Was zeigt das Bild?",
            temperature=0.3, p_value=0.8, seed=None,
            image="/tmp/irrelevant.png", stream=False,
        )
        contents = fake_models.captured_kwargs["contents"]
        image_part = contents[1]
        self.assertEqual(image_part.inline_data.mime_type, "image/png")

    def test_a_gif_is_converted_to_png(self):
        """Gemini's inline-data support does not include GIF, unlike
        Anthropic - the shared detector must convert it, not just relabel it."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=(4, 5, 6)).save(buf, format="GIF")

        svc = _make_service()
        svc.process_image = MagicMock(return_value=buf.getvalue())
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}

        svc._generate_gemini(
            model="gemini-2.5-flash", prompt="hi",
            temperature=0.3, p_value=0.8, seed=None,
            image="/tmp/irrelevant.gif", stream=False,
        )
        image_part = fake_models.captured_kwargs["contents"][1]
        self.assertEqual(image_part.inline_data.mime_type, "image/png")


class ToolCallingConvertsMessagesAndToolsTest(unittest.TestCase):
    @staticmethod
    def _run(messages, tools=None):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="ok")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=messages, tools=tools or [],
            temperature=0.5, top_p=0.9, max_tokens=512,
        )
        return fake_models

    def test_user_and_assistant_roles_map_to_user_and_model(self):
        fake = self._run([
            {"role": "user", "content": "Frage"},
            {"role": "assistant", "content": "Antwort"},
        ])
        contents = fake.captured_kwargs["contents"]
        self.assertEqual([c.role for c in contents], ["user", "model"])

    def test_tool_role_becomes_a_function_response(self):
        fake = self._run([
            {"role": "tool", "name": "search", "content": "Ergebnis"},
        ])
        content = fake.captured_kwargs["contents"][0]
        self.assertEqual(content.role, "function")
        fr = content.parts[0].function_response
        self.assertEqual(fr.name, "search")
        self.assertEqual(fr.response, {"result": "Ergebnis"})

    def test_system_message_is_extracted_not_sent_as_content(self):
        fake = self._run([
            {"role": "system", "content": "Systemanweisung"},
            {"role": "user", "content": "Frage"},
        ])
        self.assertEqual(fake.captured_kwargs["config"].system_instruction, "Systemanweisung")
        self.assertEqual(len(fake.captured_kwargs["contents"]), 1)

    def test_tool_schema_uses_the_raw_json_schema_field(self):
        """parameters_json_schema takes the JSON Schema dict directly; the
        stricter `parameters` field requires Google's own Schema type and
        would reject it."""
        fake = self._run(
            [{"role": "user", "content": "hi"}],
            tools=[{
                "name": "search", "description": "sucht",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }],
        )
        tool = fake.captured_kwargs["config"].tools[0]
        decl = tool.function_declarations[0]
        self.assertEqual(decl.name, "search")
        self.assertIsNone(decl.parameters)
        self.assertEqual(
            decl.parameters_json_schema,
            {"type": "object", "properties": {"q": {"type": "string"}}},
        )

    def test_no_tools_means_no_tools_key_in_the_config(self):
        fake = self._run([{"role": "user", "content": "hi"}], tools=[])
        self.assertIsNone(fake.captured_kwargs["config"].tools)


class FunctionCallDetectionChecksTheValueTest(unittest.TestCase):
    """part.function_call ALWAYS exists as an attribute (declared pydantic
    field) even on a text-only part - hasattr() alone cannot distinguish
    them; only checking the value (None vs not) can."""

    def test_a_text_only_part_yields_no_tool_calls(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([genai_types.Part.from_text(text="nur Text")]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(response.content, "nur Text")
        self.assertEqual(response.tool_calls, [])

    def test_a_function_call_part_is_detected(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([
            genai_types.Part.from_function_call(name="search", args={"q": "x"}),
        ]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "search")
        self.assertEqual(response.tool_calls[0].arguments, {"q": "x"})

    def test_a_part_with_neither_text_nor_function_call_does_not_crash(self):
        """The actual divergence: hasattr(part, "function_call") is True on
        EVERY part (it is a declared field, just None when unset), so a part
        that is neither text nor a function call - e.g. only a `thought`
        flag set on an otherwise empty part, which real Gemini "thinking"
        responses can include - would hit `fc.name` on None and crash.
        Checking the value must make this a no-op instead."""
        svc = _make_service()
        empty_part = genai_types.Part()
        empty_part.thought = True
        fake_models = _FakeModels(response=_response([
            empty_part,
            genai_types.Part.from_text(text="Antwort"),
        ]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(response.content, "Antwort")
        self.assertEqual(response.tool_calls, [])

    def test_mixed_text_and_function_call_parts(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([
            genai_types.Part.from_text(text="Ich suche das. "),
            genai_types.Part.from_function_call(name="search", args={"q": "x"}),
        ]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(response.content, "Ich suche das. ")
        self.assertEqual(len(response.tool_calls), 1)


class ParallelSameToolCallsGetDistinctIdsTest(unittest.TestCase):
    """The actual confirmed bug: two calls to the same tool in one turn
    produced the identical id ("gemini_{name}"), and tool_result pairing
    downstream requires unique ids within a turn."""

    def test_two_calls_to_the_same_tool_get_different_ids(self):
        svc = _make_service()
        fake_models = _FakeModels(response=_response([
            genai_types.Part.from_function_call(name="search", args={"q": "a"}),
            genai_types.Part.from_function_call(name="search", args={"q": "b"}),
        ]))
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}
        response = svc._generate_gemini_with_tools(
            provider=PROVIDER_NAME, model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            temperature=0.5, top_p=0.9, max_tokens=128,
        )
        ids = [tc.id for tc in response.tool_calls]
        self.assertEqual(len(ids), len(set(ids)), ids)


class ModelListingUsesTheRealClientTest(unittest.TestCase):
    def test_get_available_models_calls_client_models_list(self):
        svc = _make_service()
        svc.config_manager = MagicMock()

        from types import SimpleNamespace

        provider_cfg = SimpleNamespace(provider_type="gemini")
        svc.config_manager.load_config.return_value = SimpleNamespace(
            unified_config=SimpleNamespace(providers=[
                SimpleNamespace(name=PROVIDER_NAME, provider_type="gemini"),
            ]),
        )
        svc.is_provider_reachable = MagicMock(return_value=True)

        fake_model = SimpleNamespace(name="models/gemini-2.5-flash")
        fake_models = MagicMock()
        fake_models.list.return_value = [fake_model]
        svc.clients = {PROVIDER_NAME: MagicMock(models=fake_models)}

        result = svc.get_available_models(PROVIDER_NAME)
        self.assertEqual(result, ["gemini-2.5-flash"])
        fake_models.list.assert_called_once()


if __name__ == "__main__":
    unittest.main()
