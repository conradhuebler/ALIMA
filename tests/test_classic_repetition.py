"""Classic path: AlimaManager._generate_response with the new detector - Claude Generated

Pins the warning-bar callback contract (warn/grace/resolved, unchanged
signature), the cut on abort, and ``classic_repetition_action``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.utils.config_models import AlimaConfig, RepetitionDetectionConfig

ROOT = Path(__file__).resolve().parent.parent
LOOP = (
    "Das Werk behandelt die Auswirkungen von Mikroplastik auf marine "
    "Ökosysteme und deren Nahrungsketten. "
) * 200


def _prefix() -> str:
    data = json.loads((ROOT / "examples" / "classic_result.json").read_text(encoding="utf-8"))
    return data["results"]["initial_llm_call_details"]["response_full_text"][:2000]


def _gen(text, size=19, record=None):
    def g():
        try:
            for i in range(0, len(text), size):
                yield text[i:i + size]
        finally:
            if record is not None:
                record.append("closed")
    return g()


def _manager(texts, **cfg_kw):
    from src.core.alima_manager import AlimaManager

    cfg = AlimaConfig()
    cfg.repetition_config = RepetitionDetectionConfig(grace_period_seconds=0, **cfg_kw)
    cm = MagicMock()
    cm.load_config.return_value = cfg
    llm = MagicMock()
    closed = []
    llm.generate_response.side_effect = [_gen(t, record=closed) for t in texts]
    mgr = AlimaManager(llm, MagicMock(), cm, MagicMock())
    return mgr, llm, closed


PROMPT = SimpleNamespace(models=["m"], system="", output_format="xml")


def _run(mgr, callback=None, stream=None):
    with patch("src.core.state_bus.AlimaStateBus") as bus:
        out = mgr._generate_response(
            "req", PROMPT, "prompt", provider="fake", stream_callback=stream,
            temperature=0.3, p_value=0.1, on_repetition_detected=callback,
        )
    return out, bus


class ClassicRepetitionTest(unittest.TestCase):
    def test_retry_regenerates_with_varied_sampling(self):
        """Gate: default action retry → second call with higher temperature/top_p."""
        prefix = _prefix()
        mgr, llm, closed = _manager([prefix + LOOP, prefix])
        self.assertEqual(mgr.repetition_config.classic_repetition_action, "retry")
        out, bus = _run(mgr)
        self.assertEqual(out, prefix)
        self.assertEqual(llm.generate_response.call_count, 2)
        first = llm.generate_response.call_args_list[0].kwargs
        second = llm.generate_response.call_args_list[1].kwargs
        self.assertEqual((first["temperature"], first["p_value"]), (0.3, 0.1))
        self.assertGreater(second["temperature"], 0.3)
        self.assertGreaterEqual(second["p_value"], 0.9)
        self.assertIn("closed", closed, "looping stream must be closed")
        self.assertTrue(bus.return_value.emit_event.called)

    def test_abort_returns_the_text_before_the_loop(self):
        prefix = _prefix()
        mgr, llm, _closed = _manager([prefix + LOOP], classic_repetition_action="abort")
        out, _bus = _run(mgr)
        self.assertEqual(llm.generate_response.call_count, 1)
        self.assertTrue(out.startswith(prefix))
        self.assertLess(len(out), len(prefix) + 450)

    def test_retry_cap_then_partial_result(self):
        prefix = _prefix()
        mgr, llm, _c = _manager([prefix + LOOP, LOOP], max_retries=1)
        out, _b = _run(mgr)
        self.assertEqual(llm.generate_response.call_count, 2)
        self.assertLess(len(out), 450)

    def test_callback_contract_is_unchanged(self):
        """(result, suggestions, grace_period, resolved, grace_seconds)."""
        calls = []
        mgr, _llm, _c = _manager([_prefix() + LOOP], classic_repetition_action="abort")
        _run(mgr, callback=lambda *a: calls.append(a))
        self.assertTrue(calls)
        for result, suggestions, grace, resolved, seconds in calls:
            self.assertIsInstance(suggestions, list)
            self.assertIsInstance(grace, bool)
            self.assertIsInstance(resolved, bool)
            self.assertIsInstance(seconds, float)
        last = calls[-1][0]
        self.assertIn("abgebrochen", last.details)

    def test_auto_abort_off_streams_everything(self):
        text = _prefix() + LOOP
        mgr, llm, _c = _manager([text], auto_abort=False)
        out, _b = _run(mgr)
        self.assertEqual(out, text)
        self.assertEqual(llm.generate_response.call_count, 1)

    def test_loop_free_text_is_returned_whole(self):
        text = _prefix()
        shown = []
        mgr, _llm, _c = _manager([text])
        out, _b = _run(mgr, stream=shown.append)
        self.assertEqual(out, text)
        self.assertEqual("".join(shown), text)


if __name__ == "__main__":
    unittest.main()
