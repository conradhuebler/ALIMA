"""A long line without newline must not look like a stalled step. Claude Generated"""
from src.core._pipeline_agentic import _AgenticStreamFilter


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _make():
    out, clock = [], _Clock()
    return out, clock, _AgenticStreamFilter(out.append, clock=clock)


def test_silent_long_line_gets_one_progress_line_per_interval():
    out, clock, f = _make()
    f('{"keywords": [')
    clock.t = 2.0
    f('{"keyword": "Erz"}, ')
    assert out == []                      # not yet: still within the interval
    clock.t = 5.5
    f('{"keyword": "Silber"}, ')
    assert len(out) == 1 and out[0].startswith("⏳ Antwort läuft")
    clock.t = 6.0
    f("x")
    assert len(out) == 1                  # no second line before the next interval
    clock.t = 10.6
    f("y")
    assert len(out) == 2


def test_progress_counts_characters_and_real_output_resets_it():
    out, clock, f = _make()
    f("a" * 100)
    clock.t = 6.0
    f("b" * 20)
    assert "120 Zeichen" in out[0]
    f("\n")                               # a complete non-JSON line passes through
    assert out[-1].strip() == ("a" * 100 + "b" * 20)


def test_normal_streaming_prints_no_progress_line():
    out, clock, f = _make()
    for i in range(10):
        clock.t = i * 0.5
        f("Status\n")
    assert all("Antwort läuft" not in m for m in out)


def test_no_tokens_no_progress():
    out, clock, f = _make()
    clock.t = 100.0
    f("")
    assert out == []
