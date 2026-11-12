"""Repetition detection for LLM output streams by information content - Claude Generated

A model caught in a loop keeps producing text that says nothing new. That is
measured directly: every completed segment of ``segment_chars`` characters is
Deflate-compressed (raw, level 9) with the text generated before it as the
preset dictionary (``zdict``, the last 32 KB — Deflate's window cannot reach
further). The *information rate* of the segment is compressed bytes per
character: loop-free model answers stayed above 0.15 in the calibration, text
that only repeats what came before falls to about 0.02. The rate is smoothed
over the last ``smoothing_segments`` segments and compared with two
thresholds:

* below ``warn_threshold``  → level ``"warn"``
* below ``abort_threshold`` → level ``"abort"``

A second rate is measured on the same segment with every digit replaced by
``0``: a loop that only advances a counter ("1. X", "2. X", …) carries the
counter as information Deflate cannot predict and stays near 0.10 on the
plain rate. With digits masked it is an exact loop again. The masked rate
leads to ``"abort"`` only below ``abort_threshold × COUNTER_LOOP_FACTOR`` and
only while the plain rate is already below ``warn_threshold``; loop-free
answers, GND-ID lists included, stayed above 0.11 masked in the calibration.

The prompt is deliberately **not** part of the history: a model that copies
GND terms from its input list is doing its job, not looping. Each output
channel (answer, thinking) has its own detector and therefore its own history.

Segments are fixed character ranges, so the result does not depend on how a
provider chunks its stream (pinned by ``tests/test_repetition_detector.py``).

The thresholds are calibrated with ``scripts/calibrate_repetition.py``; the
numbers behind the defaults are recorded in ``AIChangelog.md`` (2026-11-12).

Three layers:

* ``RepetitionDetector`` — one channel, pure measurement, sticky level.
* ``RepetitionMonitor`` — detector + grace period + auto-abort switch as an
  event state machine (``warn`` / ``grace`` / ``resolved`` / ``abort``); used
  by the classic path (``AlimaManager``) and by ``RepetitionGuard``.
* ``RepetitionGuard`` — all channels of one tool-calling request
  (``LlmService.generate_with_tools``): answer text, inline ``<think>`` text
  inside the answer stream, and the separate reasoning channel. Wraps the
  stream callbacks and ``should_stop`` and cuts the loop tail off the response.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config_models import RepetitionDetectionConfig

logger = logging.getLogger(__name__)

# Deflate's window: a preset dictionary longer than this has no effect.
HISTORY_BYTES = 32768
# The masked-digit rate must fall below abort_threshold × this for a counter loop.
COUNTER_LOOP_FACTOR = 0.5

_DIGIT_MASK = str.maketrans("0123456789", "0000000000")

LEVEL_WARN = "warn"
LEVEL_ABORT = "abort"

EVENT_WARN = "warn"
EVENT_GRACE = "grace"
EVENT_RESOLVED = "resolved"
EVENT_ABORT = "abort"

CHANNEL_ANSWER = "answer"
CHANNEL_THINKING = "thinking"


def information_rate(segment: str, history: bytes = b"") -> float:
    """Compressed bytes per character of ``segment`` given ``history`` - Claude Generated

    Raw Deflate (no zlib header/checksum), level 9, the last ``HISTORY_BYTES``
    of ``history`` (UTF-8) as preset dictionary. Per character rather than per
    byte, so text with many multi-byte characters (box drawing, umlauts) is not
    rated lower just for its encoding. Empty segment → 1.0.
    """
    if not segment:
        return 1.0
    if history:
        comp = zlib.compressobj(
            9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, history[-HISTORY_BYTES:]
        )
    else:
        comp = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    raw = segment.encode("utf-8")
    size = len(comp.compress(raw)) + len(comp.flush())
    return size / len(segment)


@dataclass
class RepetitionResult:
    """Result of a repetition check - Claude Generated"""

    is_repetitive: bool
    level: str = ""                  # "warn" | "abort"
    detection_type: str = "low_information"   # or "counter_loop"
    channel: str = CHANNEL_ANSWER    # "answer" | "thinking"
    details: str = ""                # human-readable description
    confidence: float = 0.0          # 0..1, how far below the warn threshold
    repeated_content: str = ""       # sample of the repeating text
    position: int = 0                # estimated loop start (char offset in the channel text)
    rate: float = 0.0                # smoothed rate that crossed ``threshold``
    threshold: float = 0.0           # the threshold that was crossed
    text_length: int = 0             # channel text length at detection

    @property
    def loop_length(self) -> int:
        return max(0, self.text_length - self.position)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable form (logging, AgentResponse.repetition) - Claude Generated"""
        return {
            "is_repetitive": self.is_repetitive,
            "level": self.level,
            "detection_type": self.detection_type,
            "channel": self.channel,
            "details": self.details,
            "confidence": round(self.confidence, 3),
            "repeated_content": self.repeated_content[:100],
            "position": self.position,
            "loop_length": self.loop_length,
            "rate": round(self.rate, 4),
            "threshold": self.threshold,
            "text_length": self.text_length,
        }


class RepetitionDetector:
    """Information-rate measurement for one output channel - Claude Generated

    ``add_chunk`` returns the current (sticky) result: the level of the last
    completed segment evaluation, or ``None`` while the text is fine. Between
    segment boundaries the previous evaluation stands, so a caller never sees
    a "clean" answer just because the next segment is not complete yet.
    """

    def __init__(
        self,
        warn_threshold: float,
        abort_threshold: float,
        segment_chars: int = 400,
        smoothing_segments: int = 4,
        channel: str = CHANNEL_ANSWER,
        enabled: bool = True,
    ):
        self.warn_threshold = float(warn_threshold)
        self.abort_threshold = float(min(abort_threshold, warn_threshold))
        self.segment_chars = max(50, int(segment_chars))
        self.smoothing_segments = max(1, int(smoothing_segments))
        self.channel = channel
        self.enabled = enabled
        self.reset()

    @classmethod
    def from_config(
        cls, config: Optional[RepetitionDetectionConfig], channel: str = CHANNEL_ANSWER
    ) -> "RepetitionDetector":
        cfg = config if isinstance(config, RepetitionDetectionConfig) else RepetitionDetectionConfig()
        if channel == CHANNEL_THINKING:
            warn, abort = cfg.thinking_warn_threshold, cfg.thinking_abort_threshold
        else:
            warn, abort = cfg.warn_threshold, cfg.abort_threshold
        return cls(
            warn_threshold=warn,
            abort_threshold=abort,
            segment_chars=cfg.segment_chars,
            smoothing_segments=cfg.smoothing_segments,
            channel=channel,
            enabled=cfg.enabled,
        )

    def reset(self) -> None:
        """Start a new generation - Claude Generated"""
        self._segments: List[str] = []
        self._rates: List[float] = []
        self._masked_rates: List[float] = []
        self._pending = ""
        self._history = bytearray()
        self._masked_history = bytearray()
        self._current: Optional[RepetitionResult] = None
        self.smoothed_rate: Optional[float] = None
        self.smoothed_masked_rate: Optional[float] = None

    # -- feeding ---------------------------------------------------------

    def add_chunk(self, chunk: str) -> Optional[RepetitionResult]:
        """Add streamed text; return the current result (None = no repetition) - Claude Generated"""
        if not self.enabled:
            return None
        if chunk:
            self._pending += chunk
            size = self.segment_chars
            while len(self._pending) >= size:
                segment, self._pending = self._pending[:size], self._pending[size:]
                self._evaluate(segment)
        return self._current

    @property
    def current(self) -> Optional[RepetitionResult]:
        return self._current

    @property
    def level(self) -> Optional[str]:
        return self._current.level if self._current else None

    @property
    def text(self) -> str:
        return "".join(self._segments) + self._pending

    @property
    def segment_rates(self) -> List[float]:
        return list(self._rates)

    @property
    def masked_segment_rates(self) -> List[float]:
        return list(self._masked_rates)

    @staticmethod
    def _append_history(buf: bytearray, raw: bytes) -> None:
        buf += raw
        if len(buf) > HISTORY_BYTES:
            del buf[: len(buf) - HISTORY_BYTES]

    def _evaluate(self, segment: str) -> None:
        masked = segment.translate(_DIGIT_MASK)
        rate = information_rate(segment, bytes(self._history))
        masked_rate = information_rate(masked, bytes(self._masked_history))
        self._segments.append(segment)
        self._rates.append(rate)
        self._masked_rates.append(masked_rate)
        self._append_history(self._history, segment.encode("utf-8"))
        self._append_history(self._masked_history, masked.encode("utf-8"))

        window = self.smoothing_segments
        if len(self._rates) < window:
            return
        smoothed = sum(self._rates[-window:]) / window
        smoothed_masked = sum(self._masked_rates[-window:]) / window
        self.smoothed_rate = smoothed
        self.smoothed_masked_rate = smoothed_masked
        counter_threshold = self.abort_threshold * COUNTER_LOOP_FACTOR
        if smoothed < self.abort_threshold:
            self._current = self._make_result(LEVEL_ABORT, smoothed, self.abort_threshold)
        elif smoothed < self.warn_threshold and smoothed_masked < counter_threshold:
            self._current = self._make_result(
                LEVEL_ABORT, smoothed_masked, counter_threshold, detection_type="counter_loop"
            )
        elif smoothed < self.warn_threshold:
            self._current = self._make_result(LEVEL_WARN, smoothed, self.warn_threshold)
        else:
            self._current = None

    def _run_start_segment(self) -> int:
        """Index of the first segment of the current low-rate run - Claude Generated"""
        k = len(self._rates) - 1
        while k > 0 and self._rates[k - 1] < self.warn_threshold:
            k -= 1
        return max(0, k)

    def _make_result(
        self, level: str, rate: float, threshold: float, detection_type: str = "low_information"
    ) -> RepetitionResult:
        start = self._run_start_segment() * self.segment_chars
        text_len = len(self._segments) * self.segment_chars + len(self._pending)
        sample = self._segments[-1][:100] if self._segments else ""
        smoothed = self.smoothed_rate if self.smoothed_rate is not None else rate
        confidence = 0.0
        if self.warn_threshold > 0:
            confidence = max(0.0, min(1.0, 1.0 - smoothed / self.warn_threshold))
        label = "Denkkanal" if self.channel == CHANNEL_THINKING else "Antwort"
        what = "Zählschleife, Rate ohne Ziffern" if detection_type == "counter_loop" else "Informationsrate"
        return RepetitionResult(
            is_repetitive=True,
            level=level,
            detection_type=detection_type,
            channel=self.channel,
            details=(
                f"{label}: {what} {rate:.3f} < {threshold:.3f} "
                f"(Mittel der letzten {self.smoothing_segments} × {self.segment_chars} Zeichen)"
            ),
            confidence=confidence,
            repeated_content=sample,
            position=start,
            rate=rate,
            threshold=threshold,
            text_length=text_len,
        )

    # -- loop start --------------------------------------------------------

    def loop_start(self) -> int:
        """Character offset where the current loop begins (refined) - Claude Generated

        The segment-level estimate (start of the low-rate run) is refined with
        short sliding windows (48 characters, step 8) over the two segments
        around it: the cut is placed where the contiguous run of windows below
        the midpoint between the fresh-text and the loop level begins. For an
        exact loop this keeps one full instance of the repeated unit — its
        first occurrence is new text, only the repeats carry no information.
        Without a clear edge the segment-level estimate stands: the start of
        the first low-rate segment, which keeps all loop-free text and at most
        one segment of loop.
        Returns ``len(text)`` when no repetition is current.
        """
        text = self.text
        if not self._current:
            return len(text)
        coarse = self._current.position
        size = self.segment_chars
        lo = max(0, coarse - size)
        hi = min(len(text), coarse + size)
        width, step = 48, 8
        positions = list(range(lo, max(lo, hi - width) + 1, step))
        if len(positions) < 4:
            return coarse
        rates = [
            information_rate(text[p:p + width], text[max(0, p - HISTORY_BYTES):p].encode("utf-8"))
            for p in positions
        ]
        ordered = sorted(rates)
        r_lo = ordered[len(ordered) // 5]
        r_hi = ordered[(len(ordered) * 4) // 5]
        if r_hi - r_lo < 0.1:
            # The whole region is either loop or fresh text: no edge to find.
            return coarse
        mid = (r_lo + r_hi) / 2
        low = [r < mid for r in rates]
        if not low[-1]:
            return coarse
        # Walk left from the loop side while the windows stay low; a single
        # high window inside the run (a varied repeat) does not end it. The
        # first instance of the repeated unit is fresh text and ends the walk,
        # so it is kept — and so is low-rate structure *before* it (repeated
        # JSON keys), which a left-to-right search would mistake for the loop.
        i = len(low) - 1
        while i > 0:
            if low[i - 1]:
                i -= 1
            elif i >= 2 and low[i - 2]:
                i -= 2
            else:
                break
        # A low window holds less than half fresh text, the high one left of
        # it at least half: the loop starts about 16 characters into window i.
        return min(len(text), positions[i] + width // 3)


class RepetitionMonitor:
    """Detector + grace period + auto-abort as an event state machine - Claude Generated

    ``feed`` returns at most one event per call:

    * ``("warn", result)``     — the rate fell below the warn threshold
    * ``("grace", result)``    — below the abort threshold, grace period starts
    * ``("resolved", None)``   — back above the warn threshold after warn/grace
    * ``("abort", result)``    — abort level held for ``grace_seconds``

    With ``auto_abort=False`` the abort level is reported as ``warn`` and
    never ends the stream. ``finish()`` is called once the stream is over: a
    text that *ends* at abort level counts as aborted even when the grace
    period had not run out (the model looped until it stopped by itself, or the
    text arrived in one piece without a stream).
    """

    def __init__(
        self,
        detector: RepetitionDetector,
        auto_abort: bool = True,
        grace_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.detector = detector
        self.auto_abort = auto_abort
        self.grace_seconds = max(0.0, float(grace_seconds or 0.0))
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self._grace_start: Optional[float] = None
        self._signalled = False      # warn or grace was reported
        self._warned = False
        self.aborted = False
        self.result: Optional[RepetitionResult] = None
        self.detections = 0
        self.resolutions = 0

    def feed(self, chunk: str) -> Optional[Tuple[str, Optional[RepetitionResult]]]:
        if self.aborted:
            return None
        res = self.detector.add_chunk(chunk)
        level = res.level if res else None

        if level is None:
            self._grace_start = None
            self._warned = False
            if self._signalled:
                self._signalled = False
                self.resolutions += 1
                return (EVENT_RESOLVED, None)
            return None

        self.result = res
        if level == LEVEL_ABORT and self.auto_abort:
            now = self._clock()
            if self._grace_start is None:
                self._grace_start = now
                self.detections += 1
                self._signalled = True
                if self.grace_seconds <= 0:
                    return self._abort(res)
                return (EVENT_GRACE, res)
            if now - self._grace_start >= self.grace_seconds:
                return self._abort(res)
            return None

        # warn level, or abort level without auto-abort
        if not self._warned and not self._signalled:
            self._warned = True
            self._signalled = True
            self.detections += 1
            return (EVENT_WARN, res)
        return None

    def finish(self) -> Optional[Tuple[str, Optional[RepetitionResult]]]:
        """End of stream: abort if the text ends at abort level - Claude Generated"""
        if self.aborted:
            return None
        res = self.detector.current
        if res and res.level == LEVEL_ABORT and self.auto_abort:
            if self._grace_start is None:
                self.detections += 1
            return self._abort(res)
        return None

    def _abort(self, res: RepetitionResult) -> Tuple[str, RepetitionResult]:
        self.aborted = True
        res.position = self.detector.loop_start()
        self.result = res
        return (EVENT_ABORT, res)


# ---------------------------------------------------------------------------
# Tool-calling path: all channels of one request
# ---------------------------------------------------------------------------

TARGET_CONTENT = "content"      # answer text, incl. inline <think> parts
TARGET_REASONING = "reasoning"  # separate provider reasoning channel


class _InlineThinkSplitter:
    """Split a raw answer stream into answer / inline-<think> text with offsets - Claude Generated

    Same tag handling as ``agent_loop.ThinkStreamFilter`` (tags may be split
    across chunks; a held-back partial tag is released on the next chunk), but
    every emitted piece carries its offset in the raw stream so a position in
    one channel can be mapped back onto ``AgentResponse.content``.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buf = ""
        self._buf_raw = 0
        self.in_think = False
        # channel -> list of (channel_offset, raw_offset)
        self._maps: Dict[str, List[Tuple[int, int]]] = {CHANNEL_ANSWER: [], CHANNEL_THINKING: []}
        self._lengths: Dict[str, int] = {CHANNEL_ANSWER: 0, CHANNEL_THINKING: 0}

    def feed(self, token: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if not token:
            return out
        self._buf += token
        while True:
            tag = self._CLOSE if self.in_think else self._OPEN
            idx = self._buf.find(tag)
            if idx != -1:
                self._emit(self._buf[:idx], out)
                self._buf_raw += idx + len(tag)
                self._buf = self._buf[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            held = 0
            for k in range(min(len(self._buf), len(tag) - 1), 0, -1):
                if self._buf.endswith(tag[:k]):
                    held = k
                    break
            if len(self._buf) > held:
                piece = self._buf[: len(self._buf) - held]
                self._emit(piece, out)
                self._buf_raw += len(piece)
                self._buf = self._buf[len(piece):]
            return out

    def _emit(self, text: str, out: List[Tuple[str, str]]) -> None:
        if not text:
            return
        channel = CHANNEL_THINKING if self.in_think else CHANNEL_ANSWER
        self._maps[channel].append((self._lengths[channel], self._buf_raw))
        self._lengths[channel] += len(text)
        out.append((channel, text))

    def raw_offset(self, channel: str, pos: int) -> int:
        """Map a channel offset to the raw stream offset - Claude Generated"""
        entries = self._maps.get(channel) or []
        raw = 0
        for ch_start, raw_start in entries:
            if ch_start <= pos:
                raw = raw_start + (pos - ch_start)
            else:
                break
        return raw


class RepetitionGuard:
    """Repetition guard for one ``generate_with_tools`` request - Claude Generated

    Three monitors: the answer text, ``<think>`` text that arrives inline in
    the answer stream, and the separate reasoning channel (``thinking_callback``).
    The two thinking sources use the thinking thresholds. ``should_stop``
    returns True once any monitor aborted, which makes every provider loop
    close its stream and return (as ``CANCELLED``); ``apply`` then turns that
    response into ``StopReason.REPETITION`` with the loop tail cut off.
    """

    def __init__(
        self,
        config: Optional[RepetitionDetectionConfig] = None,
        on_event: Optional[Callable[[str, Optional[RepetitionResult]], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        cfg = config if isinstance(config, RepetitionDetectionConfig) else RepetitionDetectionConfig()
        self.config = cfg
        self.on_event = on_event
        grace = cfg.grace_period_seconds

        def monitor(channel: str) -> RepetitionMonitor:
            return RepetitionMonitor(
                RepetitionDetector.from_config(cfg, channel=channel),
                auto_abort=cfg.auto_abort,
                grace_seconds=grace,
                clock=clock,
            )

        self._monitors: Dict[str, RepetitionMonitor] = {
            "answer": monitor(CHANNEL_ANSWER),
            "inline_thinking": monitor(CHANNEL_THINKING),
            "reasoning": monitor(CHANNEL_THINKING),
        }
        self.reset()

    def reset(self) -> None:
        """Fresh state for a (re-)dispatch - Claude Generated"""
        for m in self._monitors.values():
            m.reset()
        self._splitter = _InlineThinkSplitter()
        self._raw_content: List[str] = []
        self._raw_reasoning: List[str] = []
        self.tripped = False
        self.result: Optional[RepetitionResult] = None
        self._trip_key: Optional[str] = None

    # -- wrapping ----------------------------------------------------------

    def wrap_stream(self, callback: Optional[Callable[[str], None]]) -> Optional[Callable[[str], None]]:
        if callback is None:
            return None

        def _stream(text: str) -> None:
            self.feed_content(text)
            callback(text)

        return _stream

    def wrap_thinking(self, callback: Optional[Callable[[str], None]]) -> Callable[[str], None]:
        def _thinking(text: str) -> None:
            self.feed_reasoning(text)
            if callback is not None:
                callback(text)

        return _thinking

    def wrap_should_stop(self, should_stop: Optional[Callable[[], bool]]) -> Callable[[], bool]:
        def _should_stop() -> bool:
            if should_stop is not None and should_stop():
                return True
            return self.tripped

        return _should_stop

    # -- feeding -----------------------------------------------------------

    def feed_content(self, text: str) -> None:
        if not text:
            return
        self._raw_content.append(text)
        if self.tripped:
            return
        for channel, piece in self._splitter.feed(text):
            key = "answer" if channel == CHANNEL_ANSWER else "inline_thinking"
            self._handle(key, self._monitors[key].feed(piece))
            if self.tripped:
                return

    def feed_reasoning(self, text: str) -> None:
        if not text:
            return
        self._raw_reasoning.append(text)
        if self.tripped:
            return
        self._handle("reasoning", self._monitors["reasoning"].feed(text))

    def _handle(self, key: str, event) -> None:
        if not event:
            return
        name, res = event
        if name == EVENT_ABORT:
            self.tripped = True
            self.result = res
            self._trip_key = key
        if self.on_event is not None:
            try:
                self.on_event(name, res)
            except Exception as e:  # a display hook must never break generation
                from .error_visibility import log_caught

                log_caught(logger, e, "RepetitionGuard.on_event")

    # -- result ------------------------------------------------------------

    def apply(self, response: Any, operator_cancelled: bool = False) -> Optional[Dict[str, Any]]:
        """Cut the loop tail off a looping response - Claude Generated

        Returns the repetition info dict when the response looped (the caller
        then sets ``StopReason.REPETITION``), ``None`` otherwise.

        * Text that did not arrive through the wrapped callbacks (non-streaming
          call) is measured here, after the fact — no early abort, but the
          loop tail is still cut.
        * Operator cancel wins: nothing is changed.
        * The cut keeps the text before the loop; tool calls of a looping turn
          are dropped.
        """
        if response is None or operator_cancelled:
            return None
        content = getattr(response, "content", "") or ""
        reasoning = getattr(response, "reasoning", "") or ""
        streamed_content = "".join(self._raw_content)
        streamed_reasoning = "".join(self._raw_reasoning)

        # A provider may return text that differs from what it streamed (the
        # text fallback strips tool JSON, say). Positions only map onto the
        # text they were measured on, so measure the returned text instead.
        if content and content != streamed_content and not self.tripped:
            self._reset_target(TARGET_CONTENT)
            self.feed_content(content)
            streamed_content = content
        if reasoning and reasoning != streamed_reasoning and not self.tripped:
            self._reset_target(TARGET_REASONING)
            self.feed_reasoning(reasoning)
            streamed_reasoning = reasoning

        if not self.tripped:
            for key in ("answer", "inline_thinking", "reasoning"):
                self._handle(key, self._monitors[key].finish())
                if self.tripped:
                    break
        if not self.tripped or self.result is None:
            return None

        res = self.result
        info = res.to_dict()
        if self._trip_key == "reasoning":
            source = streamed_reasoning
            cut = max(0, min(len(source), res.position))
            response.reasoning = source[:cut]
            info.update(target=TARGET_REASONING, kept_chars=cut, cut_chars=len(source) - cut)
        else:
            channel = CHANNEL_ANSWER if self._trip_key == "answer" else CHANNEL_THINKING
            source = streamed_content
            cut = max(0, min(len(source), self._splitter.raw_offset(channel, res.position)))
            kept = source[:cut]
            if channel == CHANNEL_THINKING and kept.count("<think>") > kept.count("</think>"):
                kept += "</think>"
            response.content = kept
            info.update(target=TARGET_CONTENT, kept_chars=cut, cut_chars=len(source) - cut)
        if hasattr(response, "tool_calls"):
            response.tool_calls = []
        return info

    def _reset_target(self, target: str) -> None:
        if target == TARGET_CONTENT:
            self._monitors["answer"].reset()
            self._monitors["inline_thinking"].reset()
            self._splitter = _InlineThinkSplitter()
            self._raw_content = []
        else:
            self._monitors["reasoning"].reset()
            self._raw_reasoning = []


def resolve_repetition_config(source: Any = None) -> RepetitionDetectionConfig:
    """The repetition settings from a config manager / AlimaConfig, else defaults - Claude Generated

    Accepts a ``RepetitionDetectionConfig``, an ``AlimaConfig``, or a config
    manager with ``load_config()``. Anything else (``None``, a test ``Mock``)
    yields the defaults, so code paths that run without a real config never
    pick up a half-built object.
    """
    if isinstance(source, RepetitionDetectionConfig):
        return source
    if source is None:
        return RepetitionDetectionConfig()
    try:
        candidate = getattr(source, "repetition_config", None)
        if isinstance(candidate, RepetitionDetectionConfig):
            return candidate
        if callable(getattr(source, "load_config", None)):
            candidate = getattr(source.load_config(), "repetition_config", None)
            if isinstance(candidate, RepetitionDetectionConfig):
                return candidate
    except Exception as e:  # config read is best-effort; defaults apply
        from .error_visibility import log_caught

        log_caught(logger, e, "resolve_repetition_config")
    return RepetitionDetectionConfig()


# ---------------------------------------------------------------------------
# Parameter variation
# ---------------------------------------------------------------------------

# Upper bound for ``RepetitionDetectionConfig.max_retries``.
MAX_RETRIES_LIMIT = 3


def retry_sampling_params(
    temperature: Optional[float], top_p: Optional[float], attempt: int
) -> Dict[str, float]:
    """Sampling parameters for automatic retry number ``attempt`` (1-based) - Claude Generated

    Only temperature and top_p: every provider accepts them (reasoning models
    that reject them have them removed in ``LlmService``). ``repetition_penalty``
    is left alone here because OpenAI-compatible endpoints differ on whether
    they accept it; an agent can still set it with ``set_generation_params``.

    attempt 1: temperature +0.3 (from at least 0.5), top_p at least 0.9
    attempt 2: temperature +0.6, top_p at least 0.95
    attempt 3: temperature +0.9, top_p 1.0       (temperature capped at 1.2)
    """
    attempt = max(1, min(MAX_RETRIES_LIMIT, int(attempt)))
    base_t = max(0.5, float(temperature if temperature is not None else 0.7))
    floor_p = {1: 0.9, 2: 0.95, 3: 1.0}[attempt]
    current_p = float(top_p if top_p is not None else 0.9)
    return {
        "temperature": round(min(1.2, base_t + 0.3 * attempt), 2),
        "top_p": round(max(current_p, floor_p), 2),
    }


def get_parameter_variation_suggestions(
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    detection_type: str = ""
) -> List[Dict[str, Any]]:
    """Parameter-variation suggestions for the warning bar - Claude Generated

    Args:
        temperature: Current temperature setting
        top_p: Current top_p setting
        repetition_penalty: Current repetition penalty
        detection_type: ``RepetitionResult.detection_type``

    Returns:
        List of suggestion dicts with 'label', 'description', and 'params' keys
    """
    suggestions = []

    # Always suggest increasing temperature if low
    if temperature < 1.0:
        new_temp = min(1.5, temperature + 0.3)
        suggestions.append({
            "label": "Temperatur erhöhen",
            "description": f"Erhöht Kreativität/Variation ({temperature:.1f} → {new_temp:.1f})",
            "params": {"temperature": new_temp},
            "priority": 1,
        })

    # Suggest repetition penalty increase
    if repetition_penalty < 1.5:
        new_penalty = min(2.0, repetition_penalty + 0.3)
        suggestions.append({
            "label": "Repetition Penalty erhöhen",
            "description": f"Bestraft wiederholte Token ({repetition_penalty:.1f} → {new_penalty:.1f})",
            "params": {"repetition_penalty": new_penalty},
            "priority": 2,
        })

    # Suggest top_p adjustment
    if top_p > 0.5:
        new_top_p = max(0.3, top_p - 0.2)
        suggestions.append({
            "label": "Top-P reduzieren",
            "description": f"Fokussiert auf wahrscheinlichere Token ({top_p:.1f} → {new_top_p:.1f})",
            "params": {"top_p": new_top_p},
            "priority": 3,
        })
    elif top_p < 0.8:
        new_top_p = min(0.95, top_p + 0.15)
        suggestions.append({
            "label": "Top-P erhöhen",
            "description": f"Erweitert Token-Auswahl ({top_p:.1f} → {new_top_p:.1f})",
            "params": {"top_p": new_top_p},
            "priority": 3,
        })

    # Combined suggestion: a loop in which nothing new is said any more.
    if detection_type == "low_information":
        suggestions.append({
            "label": "Aggressive Anpassung",
            "description": "Temperatur + Penalty kombiniert erhöhen",
            "params": {
                "temperature": min(1.5, temperature + 0.4),
                "repetition_penalty": min(2.0, repetition_penalty + 0.5),
            },
            "priority": 1,
        })

    # Sort by priority
    suggestions.sort(key=lambda x: x.get("priority", 99))

    return suggestions
