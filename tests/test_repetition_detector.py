"""Information-rate repetition detector - Claude Generated

Pins the measurement (``src/utils/repetition_detector.py``): loops of several
shapes are stopped, loop-free model answers from ``examples/`` are not, the
result does not depend on how the stream is chunked, the thinking channel has
its own thresholds, and the cut keeps the text before the loop.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.utils.config_models import RepetitionDetectionConfig
from src.utils.repetition_detector import (
    CHANNEL_ANSWER,
    CHANNEL_THINKING,
    EVENT_ABORT,
    EVENT_GRACE,
    EVENT_RESOLVED,
    EVENT_WARN,
    RepetitionDetector,
    RepetitionGuard,
    RepetitionMonitor,
    information_rate,
    retry_sampling_params,
)

ROOT = Path(__file__).resolve().parent.parent

SENTENCE = (
    "Das Werk behandelt die Auswirkungen von Mikroplastik auf marine "
    "Ökosysteme und deren Nahrungsketten. "
)
ENTRIES = [
    "- Mikroplastik (GND: 1056637645)\n", "- Meeresökologie (GND: 4074684-1)\n",
    "- Nahrungskette (GND: 4041301-8)\n", "- Umweltgift (GND: 4061636-2)\n",
    "- Plankton (GND: 4046190-6)\n",
]


def _model_answers():
    """Loop-free model answers shipped in examples/ (classic + agentic)."""
    out = []
    for name in ("classic_result.json", "agentic_result.json"):
        data = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))

        def walk(o, key=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v, k)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v, key)
            elif isinstance(o, str) and key in ("response_full_text", "response_text"):
                yield o

        out.extend(t for t in walk(data) if len(t) >= 1600)
    return list(dict.fromkeys(out))


def _prefix() -> str:
    return max(_model_answers(), key=len)[:3000]


def _play(text, cfg=None, channel=CHANNEL_ANSWER, sizes=(1, 40), seed=0):
    """Feed in random chunks (sizes 1-40), grace 0 → list of (event, pos), cut."""
    cfg = cfg or RepetitionDetectionConfig()
    mon = RepetitionMonitor(RepetitionDetector.from_config(cfg, channel), True, 0.0)
    rng = random.Random(seed)
    events, pos, cut = [], 0, None
    while pos < len(text):
        piece = text[pos:pos + rng.randint(*sizes)]
        pos += len(piece)
        ev = mon.feed(piece)
        if ev:
            events.append((ev[0], pos))
            if ev[0] == EVENT_ABORT:
                cut = ev[1].position
                return events, cut, ev[1]
    ev = mon.finish()
    if ev:
        events.append((ev[0] + "@end", pos))
        return events, ev[1].position, ev[1]
    return events, cut, None


class InformationRateTest(unittest.TestCase):
    def test_repeated_text_has_a_low_rate_fresh_text_a_high_one(self):
        fresh = _prefix()[:400]
        self.assertGreater(information_rate(fresh, b""), 0.4)
        self.assertLess(information_rate(fresh, fresh.encode("utf-8")), 0.05)

    def test_rate_is_per_character(self):
        # The same random pattern in ASCII and in box-drawing characters
        # (3 UTF-8 bytes each): per character the rates are alike, per byte
        # the box text would come out at about a third.
        rng = random.Random(1)
        pattern = [rng.randrange(3) for _ in range(400)]
        ascii_text = "".join("-|+"[i] for i in pattern)
        box_text = "".join("─│┼"[i] for i in pattern)
        ratio = information_rate(box_text) / information_rate(ascii_text)
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 2.0)


class LoopsAreStoppedTest(unittest.TestCase):
    """Gate: each loop shape ends at abort level with the default thresholds."""

    CASES = {
        "same word": "Schlagwort " * 600,
        "sentence": SENTENCE * 80,
        "cyclic list": "".join(ENTRIES) * 60,
        "character run": "!" * 6000,
        "counter loop": "".join(f"{i}. Mikroplastik in marinen Ökosystemen\n" for i in range(1, 400)),
    }

    def test_each_loop_aborts(self):
        for name, text in self.CASES.items():
            with self.subTest(loop=name):
                events, cut, res = _play(text)
                self.assertTrue(any(e.startswith(EVENT_ABORT) for e, _ in events), events)
                self.assertLess(events[-1][1], 3000, f"{name}: aborted only at {events[-1][1]}")

    def test_counter_loop_needs_the_masked_rate(self):
        text = self.CASES["counter loop"]
        _events, _cut, res = _play(text)
        self.assertEqual(res.detection_type, "counter_loop")
        det = RepetitionDetector.from_config(RepetitionDetectionConfig())
        det.add_chunk(text)
        # plain rate stays above the abort threshold: only the masked one catches it
        self.assertGreater(det.smoothed_rate, RepetitionDetectionConfig().abort_threshold)

    def test_loop_after_long_text_keeps_the_text_before(self):
        prefix = _prefix()
        for name, loop in (("sentence", SENTENCE * 80), ("list", "".join(ENTRIES) * 60)):
            with self.subTest(loop=name):
                events, cut, _res = _play(prefix + loop)
                self.assertTrue(events and events[-1][0].startswith(EVENT_ABORT), events)
                self.assertGreaterEqual(cut, len(prefix) - 20, "cut lost loop-free text")
                self.assertLessEqual(cut, len(prefix) + 400 + 50, "cut kept too much loop")
                self.assertLess(events[-1][1] - len(prefix), 2500)


class NormalTextIsNotStoppedTest(unittest.TestCase):
    """Gate: loop-free answers from examples/ never warn, prose never aborts."""

    def test_model_answers_stay_clean(self):
        answers = _model_answers()
        self.assertGreaterEqual(len(answers), 3)
        for i, text in enumerate(answers):
            with self.subTest(answer=i, chars=len(text)):
                events, _cut, _res = _play(text, seed=i)
                self.assertEqual(events, [])

    def test_repo_prose_does_not_abort(self):
        for rel in ("README.md", "docs/user_rules.md", "AIChangelog.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")[:60000]
            with self.subTest(file=rel):
                events, _cut, _res = _play(text)
                self.assertFalse(any(e.startswith(EVENT_ABORT) for e, _ in events), events)

    def test_distinct_gnd_id_list_is_not_a_counter_loop(self):
        rng = random.Random(3)
        words = ["Umwelt", "Boden", "Chemie", "Toxikologie", "Plankton", "Sediment",
                 "Ökologie", "Wasser", "Analytik", "Polymer", "Fisch", "Küste"]
        lines = []
        for i in range(300):
            w = f"{rng.choice(words)}{rng.choice(['', 'forschung', 'belastung', 'analyse', 'schutz'])}"
            lines.append(f"{i + 1}. {w} (GND: {rng.randint(4000000, 4999999)}-{rng.randint(0, 9)})\n")
        events, _cut, _res = _play("".join(lines))
        self.assertFalse(any(e.startswith(EVENT_ABORT) for e, _ in events), events)


class ChunkSizeIndependenceTest(unittest.TestCase):
    def test_same_events_and_cut_for_stream_sized_chunks(self):
        """Chunks of 1-40 characters (what providers stream): identical events."""
        text = _prefix() + SENTENCE * 80
        reference = None
        for sizes in ((1, 1), (7, 7), (40, 40), (1, 40)):
            for seed in range(3):
                events, cut, _ = _play(text, sizes=sizes, seed=seed)
                # event kinds at the same segment index, same cut
                sig = (tuple((e, p // 400) for e, p in events), cut)
                if reference is None:
                    reference = sig
                with self.subTest(sizes=sizes, seed=seed):
                    self.assertEqual(sig, reference)

    def test_same_abort_and_cut_for_large_chunks(self):
        """A chunk spanning several segments reports only the last state (a warn
        in between is folded into the abort), but abort segment and cut stay."""
        text = _prefix() + SENTENCE * 80
        ref_events, ref_cut, _ = _play(text, sizes=(1, 40))
        ref_abort = [p // 400 for e, p in ref_events if e.startswith(EVENT_ABORT)]
        for size in (400, 1000):
            events, cut, _ = _play(text, sizes=(size, size))
            with self.subTest(size=size):
                self.assertEqual(cut, ref_cut)
                aborts = [e for e, _ in events if e.startswith(EVENT_ABORT)]
                self.assertEqual(len(aborts), 1)
                self.assertLessEqual(abs(ref_abort[0] - [p for e, p in events if e.startswith(EVENT_ABORT)][0] // 400), size // 400)


class ThinkingChannelTest(unittest.TestCase):
    def test_thinking_uses_its_own_thresholds(self):
        cfg = RepetitionDetectionConfig(
            warn_threshold=0.3, abort_threshold=0.2,
            thinking_warn_threshold=0.02, thinking_abort_threshold=0.01,
        )
        a = RepetitionDetector.from_config(cfg, CHANNEL_ANSWER)
        t = RepetitionDetector.from_config(cfg, CHANNEL_THINKING)
        self.assertEqual((a.warn_threshold, a.abort_threshold), (0.3, 0.2))
        self.assertEqual((t.warn_threshold, t.abort_threshold), (0.02, 0.01))

    def test_thinking_loop_aborts_with_defaults(self):
        loop = "".join(
            f"Wait, let me reconsider the keyword {w}. Actually, I should check whether "
            "it is in the GND list. Let me look again. "
            for w in ["Mikroplastik", "Cadmium", "Plankton"] * 40
        )
        events, _cut, res = _play(loop, channel=CHANNEL_THINKING)
        self.assertTrue(events and events[-1][0].startswith(EVENT_ABORT), events)
        self.assertEqual(res.channel, CHANNEL_THINKING)

    def test_guard_measures_reasoning_and_inline_think_separately(self):
        cfg = RepetitionDetectionConfig(grace_period_seconds=0)
        # reasoning channel loops
        g = RepetitionGuard(cfg)
        sink = []
        think = g.wrap_thinking(sink.append)
        for i in range(0, 8000, 30):
            think((SENTENCE * 80)[i:i + 30])
            if g.tripped:
                break
        self.assertTrue(g.tripped)
        self.assertEqual(g.result.channel, CHANNEL_THINKING)
        self.assertTrue(sink, "the caller's thinking callback still receives the text")
        # inline <think> loop inside the answer stream; answer text before it kept
        g2 = RepetitionGuard(cfg)
        stream = g2.wrap_stream(lambda _t: None)
        raw = "Vorbemerkung. <think>" + SENTENCE * 80
        for i in range(0, len(raw), 17):
            stream(raw[i:i + 17])
            if g2.tripped:
                break
        self.assertTrue(g2.tripped)
        response = SimpleNamespace(content="".join(g2._raw_content), reasoning="", tool_calls=["x"])
        info = g2.apply(response)
        self.assertEqual(info["channel"], CHANNEL_THINKING)
        self.assertTrue(response.content.startswith("Vorbemerkung. <think>"))
        self.assertTrue(response.content.endswith("</think>"), "open think tag closed after the cut")
        self.assertEqual(response.tool_calls, [])


class MonitorTest(unittest.TestCase):
    def _monitor(self, grace, clock):
        return RepetitionMonitor(
            RepetitionDetector.from_config(RepetitionDetectionConfig()), True, grace, clock=clock
        )

    def test_grace_period_delays_the_abort(self):
        now = [0.0]
        mon = self._monitor(5.0, lambda: now[0])
        events = []
        for i in range(0, 20000, 40):
            ev = mon.feed(("Schlagwort " * 2000)[i:i + 40])
            if ev:
                events.append(ev[0])
            if ev and ev[0] == EVENT_ABORT:
                break
            now[0] += 0.05
        self.assertEqual(events[0], EVENT_GRACE)
        self.assertEqual(events[-1], EVENT_ABORT)
        self.assertGreaterEqual(now[0], 5.0)

    def test_resolution_after_warn(self):
        cfg = RepetitionDetectionConfig()
        mon = RepetitionMonitor(RepetitionDetector.from_config(cfg), True, 100.0, clock=lambda: 0.0)
        events = []
        for i in range(0, 2400, 40):
            ev = mon.feed((SENTENCE * 30)[i:i + 40])
            if ev:
                events.append(ev[0])
        text = _prefix()
        for i in range(0, len(text), 40):
            ev = mon.feed(text[i:i + 40])
            if ev:
                events.append(ev[0])
        self.assertIn(EVENT_GRACE, events)
        self.assertEqual(events[-1], EVENT_RESOLVED)

    def test_auto_abort_off_only_warns(self):
        mon = RepetitionMonitor(
            RepetitionDetector.from_config(RepetitionDetectionConfig()), False, 0.0
        )
        events = [ev[0] for ev in (mon.feed(c) for c in ["Schlagwort "] * 600) if ev]
        self.assertEqual(events, [EVENT_WARN])
        self.assertIsNone(mon.finish())

    def test_warn_level_between_the_thresholds(self):
        """A rate between abort and warn threshold warns and never aborts."""
        cfg = RepetitionDetectionConfig(warn_threshold=0.5, abort_threshold=0.0)
        events, _cut, _res = _play(SENTENCE * 80, cfg=cfg)
        self.assertEqual([e for e, _ in events], [EVENT_WARN])

    def test_disabled_detector_never_reports(self):
        cfg = RepetitionDetectionConfig(enabled=False)
        events, _cut, _res = _play("Schlagwort " * 600, cfg=cfg)
        self.assertEqual(events, [])


class RetryParamsTest(unittest.TestCase):
    def test_retry_raises_temperature_and_top_p_within_bounds(self):
        p1 = retry_sampling_params(0.3, 0.1, 1)
        self.assertEqual(p1, {"temperature": 0.8, "top_p": 0.9})
        p3 = retry_sampling_params(1.1, 0.95, 3)
        self.assertEqual(p3, {"temperature": 1.2, "top_p": 1.0})
        self.assertEqual(set(retry_sampling_params(None, None, 9)), {"temperature", "top_p"})


if __name__ == "__main__":
    unittest.main()
