"""Der Chat sieht den Text im Eingabefeld. Claude Generated.

Vorher kannte der Agent nur, was ein geladenes Ergebnis trug; ein ins Feld
getippter oder eingefügter Text war für ``get_input_text`` unsichtbar. Getestet
an einem Stub, der die Methoden des Panels bindet (wie ``test_chat_llm_selection``).
"""

from __future__ import annotations

import json
import logging
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.agents.shared_context import SharedContext
from src.ui.chat_tools.generic import GetInputTextTool
from src.ui.pipeline_chat_panel import PipelineChatPanel
from src.ui.pipeline_tab import PipelineTab

LONG = "Cadmium " * 100  # 800 Zeichen, über der 500er Kürzung


def _panel(provider=None, pm_ctx=None, current_context=""):
    stub = SimpleNamespace(
        session=SimpleNamespace(last_shared_context=None),
        current_context=current_context,
        logger=logging.getLogger("test"),
        pipeline_manager=SimpleNamespace(last_shared_context=pm_ctx) if pm_ctx is not None else None,
        _input_text_provider=provider,
        _seeded_input=None,
        _context_from_input=False,
    )
    for name in ("_seed_input_text", "_apply_seeded_input", "_refresh_shared_context"):
        setattr(stub, name, getattr(PipelineChatPanel, name).__get__(stub))
    return stub


class SeedingTest(unittest.TestCase):
    def test_the_field_text_reaches_get_input_text_and_the_context_line(self):
        panel = _panel(lambda: (LONG, "text", ""))
        panel._seed_input_text()
        out = json.loads(GetInputTextTool().execute(panel.session))
        self.assertEqual(out["text"], LONG)
        self.assertTrue(panel.current_context.startswith("Abstract: Cadmium"))
        self.assertTrue(panel.current_context.endswith("..."))      # Zeile gekürzt, Text nicht
        self.assertLess(len(panel.current_context), len(LONG))

    def test_source_and_type_travel_along(self):
        panel = _panel(lambda: ("Text", "doi", "10.1/x"))
        panel._seed_input_text()
        ctx = panel.session.last_shared_context
        self.assertEqual((ctx.input_type, ctx.source_value), ("doi", "10.1/x"))

    def test_a_loaded_result_wins_over_the_field(self):
        loaded = SharedContext(abstract="Ergebnis-Abstract")
        panel = _panel(lambda: ("Feldtext", "text", ""), current_context="Titel: T")
        panel.session.last_shared_context = loaded
        panel._seed_input_text()
        self.assertIs(panel.session.last_shared_context, loaded)
        self.assertEqual(loaded.abstract, "Ergebnis-Abstract")
        self.assertEqual(panel.current_context, "Titel: T")

    def test_a_refresh_during_the_turn_keeps_the_text_and_spares_the_pipeline_context(self):
        """Bus events replace the session context with the manager's mid-turn."""
        pm_ctx = SharedContext(abstract="")
        panel = _panel(lambda: ("Feldtext", "text", ""), pm_ctx=pm_ctx)
        panel._seed_input_text()
        panel._refresh_shared_context()
        self.assertEqual(panel.session.last_shared_context.abstract, "Feldtext")
        self.assertEqual(pm_ctx.abstract, "")                        # Original unverändert
        self.assertIsNot(panel.session.last_shared_context, pm_ctx)

    def test_a_pipeline_context_with_text_takes_over_on_refresh(self):
        pm_ctx = SharedContext(abstract="Lauf-Abstract")
        panel = _panel(lambda: ("Feldtext", "text", ""), pm_ctx=pm_ctx)
        panel._seed_input_text()
        panel._refresh_shared_context()
        self.assertIs(panel.session.last_shared_context, pm_ctx)

    def test_empty_field_or_no_provider_seeds_nothing_and_drops_a_stale_seed(self):
        text = {"v": "Erst Text"}
        panel = _panel(lambda: (text["v"], "text", ""))
        panel._seed_input_text()
        self.assertIsNotNone(panel._seeded_input)
        text["v"] = "   "
        panel._seed_input_text()
        self.assertIsNone(panel._seeded_input)
        self.assertIsNone(_panel(None)._seeded_input)

    def test_an_edited_field_updates_the_context_line_but_a_real_one_is_kept(self):
        text = {"v": "Eins"}
        panel = _panel(lambda: (text["v"], "text", ""))
        panel._seed_input_text()
        self.assertEqual(panel.current_context, "Abstract: Eins")
        panel.session.last_shared_context = None
        text["v"] = "Zwei"
        panel._seed_input_text()
        self.assertEqual(panel.current_context, "Abstract: Zwei")

        real = _panel(lambda: ("Feld", "text", ""), current_context="Titel: Echt")
        real._seed_input_text()
        self.assertEqual(real.current_context, "Titel: Echt")

    def test_a_failing_provider_does_not_break_the_send(self):
        def boom():
            raise RuntimeError("widget weg")

        panel = _panel(boom)
        panel._seed_input_text()
        self.assertIsNone(panel._seeded_input)
        self.assertIsNone(panel.session.last_shared_context)


class TabProviderTest(unittest.TestCase):
    def test_reads_text_type_and_source_from_the_input_widget(self):
        widget = SimpleNamespace(get_current_text=lambda: "Text", current_source_type="doi",
                                 current_source_data="10.1/x")
        tab = SimpleNamespace(unified_input=widget)
        fn = PipelineTab._chat_input_text.__get__(tab)
        self.assertEqual(fn(), ("Text", "doi", "10.1/x"))

    def test_no_widget_yields_empty(self):
        fn = PipelineTab._chat_input_text.__get__(SimpleNamespace(unified_input=None))
        self.assertEqual(fn(), ("", "", ""))

    def test_the_tab_wires_the_provider_into_the_panel(self):
        """Quelltext-Wächter: die Verdrahtung in ``PipelineTab.__init__`` fehlt sonst
        unbemerkt, und der Chat sähe das Eingabefeld wieder nicht."""
        src = Path("src/ui/pipeline_tab.py").read_text(encoding="utf-8")
        self.assertRegex(src, r"stream_widget\.set_input_text_provider\(self\._chat_input_text\)")


if __name__ == "__main__":
    unittest.main()
