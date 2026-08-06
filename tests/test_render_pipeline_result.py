"""Tests for the shared pipeline-result emission (GUI + webapp). Claude Generated.

``render_pipeline_result`` is the one producer of the final result blocks
(completion line, GND keywords, Schlagwortketten, DK cards, workflow report).
The webapp bug it fixes: these blocks were only emitted by the GUI mixin, so
webapp runs ended with nothing but step collapsibles.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.render_events import MockTransport
from src.utils.pipeline_formatters import render_pipeline_result


class _MockCheckBox:
    def __init__(self):
        self.toggled = MagicMock()

    def isChecked(self):
        return True


def _make_renderer():
    from src.ui.unified_message_renderer import UnifiedMessageRenderer

    transport = MockTransport()
    return UnifiedMessageRenderer(transport, _MockCheckBox()), transport


class TestRenderPipelineResult(unittest.TestCase):

    def test_full_state_emits_all_sections(self):
        renderer, transport = _make_renderer()
        state = SimpleNamespace(
            final_llm_analysis=SimpleNamespace(
                extracted_gnd_keywords=["Chemie", "Informatik"],
                response_full_text=(
                    "Schlagwortketten:\nChemie → Informatik\nohne Pfeil\n"
                ),
            ),
            dk_classifications=None,
            report_markdown="| a | b |\n|---|---|\n| 1 | 2 |",
        )
        render_pipeline_result(renderer, state, "12.3s")
        blocks = transport.of_type("block")
        joined = "\n".join(b["html"] for b in blocks)
        self.assertIn("Pipeline vollständig abgeschlossen in 12.3s", joined)
        self.assertIn("2 GND-Schlagworte", joined)
        self.assertIn("Chemie", joined)
        self.assertIn("Schlagwortketten", joined)
        self.assertIn("→", joined)
        # report_markdown rendered as an actual HTML table block.
        self.assertIn("<table>", joined)

    def test_empty_state_emits_only_completion_line(self):
        renderer, transport = _make_renderer()
        render_pipeline_result(renderer, None)
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)
        self.assertIn("Pipeline vollständig abgeschlossen", blocks[0]["html"])
        # No duration suffix without a duration.
        self.assertNotIn(" in ", blocks[0]["html"])

    def test_state_without_result_data_emits_no_empty_cards(self):
        renderer, transport = _make_renderer()
        state = SimpleNamespace(
            final_llm_analysis=None,
            dk_classifications=[],
            dk_statistics={},
            rvk_provenance={},
            report_markdown="",
        )
        render_pipeline_result(renderer, state)
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)  # only the completion line


if __name__ == "__main__":
    unittest.main()
