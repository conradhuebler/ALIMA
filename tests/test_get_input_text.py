"""Der Agent kommt an den vollen Eingabetext. Claude Generated.

Die Kontextzeile im Prompt ist gekürzt (GUI: 500 Zeichen, CLI/Webapp: 1500);
``get_input_text`` liefert den Text im Ganzen aus dem ``SharedContext`` der
Sitzung. Der Text muss dafür in jedem Einstiegspfad dort hinterlegt sein.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.core.headless_agent import shared_context_for_text
from src.ui.chat_tools import build_chat_toolset
from src.ui.chat_tools.generic import GetInputTextTool

LONG = "Cadmium " * 400  # 3200 Zeichen, über jeder Kürzungsgrenze


def _session(ctx):
    return SimpleNamespace(last_shared_context=ctx, session_id="s")


class ToolTest(unittest.TestCase):
    def test_returns_the_whole_text_with_title_and_source(self):
        ctx = SharedContext(abstract=LONG, working_title="Titel", input_type="doi",
                            source_value="10.1/x")
        out = json.loads(GetInputTextTool().execute(_session(ctx)))
        self.assertEqual(out["text"], LONG)
        self.assertEqual((out["title"], out["input_type"], out["source"]),
                         ("Titel", "doi", "10.1/x"))
        self.assertEqual((out["length"], out["truncated"]), (len(LONG), False))

    def test_max_chars_cuts_and_says_so(self):
        ctx = SharedContext(abstract=LONG)
        out = json.loads(GetInputTextTool().execute(_session(ctx), max_chars=100))
        self.assertEqual(len(out["text"]), 100)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["length"], len(LONG))
        out = json.loads(GetInputTextTool().execute(_session(ctx), max_chars="kaputt"))
        self.assertFalse(out["truncated"])

    def test_without_text_it_is_not_offered_and_answers_with_an_error(self):
        for ctx in (None, SharedContext(abstract="")):
            tool = GetInputTextTool()
            self.assertFalse(tool.available_for(_session(ctx)))
            self.assertIn("error", json.loads(tool.execute(_session(ctx))))

    def test_it_is_part_of_the_chat_toolset_only_with_a_text(self):
        cfg = SimpleNamespace(no_cache_writes=True, allow_model_switch=False)
        with_text = build_chat_toolset(_session(SharedContext(abstract=LONG)), chat_config=cfg)
        without = build_chat_toolset(_session(None), chat_config=cfg)
        self.assertIn("get_input_text", with_text.available_tools())
        self.assertNotIn("get_input_text", without.available_tools())


class SeedingTest(unittest.TestCase):
    def test_helper_keeps_the_full_untrimmed_text(self):
        ctx = shared_context_for_text(LONG, title="T", input_type="doi", source="10.1/x")
        self.assertEqual(ctx.abstract, LONG)
        self.assertEqual((ctx.working_title, ctx.input_type, ctx.source_value), ("T", "doi", "10.1/x"))

    def test_empty_text_means_no_context(self):
        self.assertIsNone(shared_context_for_text(""))
        self.assertIsNone(shared_context_for_text("  \n"))

    def test_the_runner_seeds_the_session_with_it(self):
        """HeadlessAgentRunner.run legt shared_context in die Sitzung; das Werkzeug
        sieht den Text also schon im ersten Zug."""
        from src.core.headless_agent import HeadlessAgentRunner

        captured = {}

        class _Loop:
            def __init__(self, *a, **k):
                pass

            def run(self, *a, **k):
                return SimpleNamespace(content="ok", tool_log=[], iterations=1, error=None)

        import src.core.headless_agent as ha

        orig_loop, orig_build = ha.AgentLoop, None
        import src.ui.chat_tools as ct

        orig_build = ct.build_chat_toolset

        def spy(session, **kw):
            captured["text"] = getattr(getattr(session, "last_shared_context", None), "abstract", None)
            return orig_build(session, **kw)

        ha.AgentLoop = _Loop
        ct.build_chat_toolset = spy
        try:
            runner = HeadlessAgentRunner(
                llm_service=MagicMock(), pipeline_manager=None,
                chat_config=SimpleNamespace(no_cache_writes=True, allow_model_switch=False,
                                            max_iterations=1, max_tokens=100, temperature=0.1),
                gateway=None,
            )
            runner.run("Frage", provider="p", model="m", context_str="kurz",
                       shared_context=shared_context_for_text(LONG))
        except Exception:
            pass  # Ziel ist nur, was build_chat_toolset gesehen hat
        finally:
            ha.AgentLoop = orig_loop
            ct.build_chat_toolset = orig_build
        self.assertEqual(captured.get("text"), LONG)


class EntryPointsSeedTheTextTest(unittest.TestCase):
    """Quelltext-Wächter: CLI und Webapp-Einzelabruf übergeben den vollen Text.

    Beide Pfade bauen ihre Kontextzeile gekürzt; ohne ``shared_context=`` am
    ``runner.run``-Aufruf hätte ``get_input_text`` dort nichts zu liefern. Das
    prüft nur, dass der Aufruf verdrahtet ist, nicht den Lauf selbst.
    """

    def _source(self, path):
        import pathlib

        return pathlib.Path(path).read_text(encoding="utf-8")

    def test_cli_agent_passes_the_full_text(self):
        src = self._source("src/cli/commands/agent_cmd.py")
        self.assertRegex(src, r"shared_context=shared_context_for_text\(\s*context_text")

    def test_webapp_one_shot_agent_passes_the_full_text(self):
        src = self._source("src/webapp/routers/agent.py")
        self.assertRegex(src, r"shared_context=shared_context_for_text\(\s*input_text")


class PromptTest(unittest.TestCase):
    def test_the_prompts_point_to_the_tool_where_the_context_is_trimmed(self):
        from src.core import chat_prompts

        self.assertIn("get_input_text", chat_prompts.USER_PROMPT_VERSCHLAGWORTUNG)
        self.assertIn("get_input_text", chat_prompts.USER_PROMPT_GENERAL)


if __name__ == "__main__":
    unittest.main()
