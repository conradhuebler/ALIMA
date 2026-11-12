#!/usr/bin/env python3
"""Calibration of the information-rate repetition detector - Claude Generated.

Reads saved ALIMA results (JSON exports, ``KeywordAnalysisState``, webapp
autosaves, agentic step dumps) and optional plain-text/Markdown files, plays
every text chunk-wise into ``RepetitionDetector`` and prints three tables:

  1. Smoothed information rate of loop-free texts per text type
     (min / 5th percentile / median of the per-text minimum, n texts).
  2. Synthetic loops: characters after loop onset until warn / abort
     (grace period 0), and where the cut lands relative to the onset.
  3. False alarms on loop-free texts at the configured thresholds.

The model answers are taken from these JSON keys (any depth):
``response_full_text``, ``response_text``, ``analyse_text``, ``analyse``,
``content``, ``reasoning``. ``<think>`` blocks inside an answer are split off
and counted as thinking text. Texts are deduplicated by content.

Examples:
    python scripts/calibrate_repetition.py ~/Downloads /tmp/alima_webapp_autosave
    python scripts/calibrate_repetition.py --json-only . --prose docs
    python scripts/calibrate_repetition.py results/ --warn 0.12 --abort 0.07

Only reads files; no network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_models import RepetitionDetectionConfig  # noqa: E402
from src.utils.repetition_detector import (  # noqa: E402
    CHANNEL_ANSWER,
    CHANNEL_THINKING,
    EVENT_ABORT,
    EVENT_WARN,
    EVENT_GRACE,
    RepetitionDetector,
    RepetitionMonitor,
)

RESPONSE_KEYS = {
    "response_full_text", "response_text", "analyse_text", "analyse",
    "content", "reasoning",
}
MIN_CHARS = 400
SKIP_DIRS = {".git", ".venv", "node_modules", "worktrees", "__pycache__"}
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _iter_files(paths: Iterable[str], suffixes: Tuple[str, ...]) -> Iterable[Path]:
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file() and p.suffix in suffixes:
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if any(part in SKIP_DIRS for part in f.parts):
                    continue
                if f.is_file() and f.suffix in suffixes:
                    yield f


def _walk_strings(obj, key: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v, key)
    elif isinstance(obj, str) and key in RESPONSE_KEYS:
        yield key, obj


def classify(text: str) -> str:
    """Text type of a loop-free model answer - Claude Generated"""
    s = text.strip()
    if s.startswith("```json") or s.startswith("{") or s.startswith("["):
        return "JSON"
    if re.search(r"<(final_list|keywords|class|analyse|gnd_systematik|missing_list)>", s):
        return "XML-Tags"
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        listy = sum(
            1 for ln in lines
            if re.match(r"^([-*•]|\d+[.)]|[A-ZÄÖÜ][^|]{0,60}\|)", ln) or ln.count("|") >= 2
        )
        if listy / len(lines) >= 0.5:
            return "Liste/Ketten"
    return "Fließtext"


def load_corpus(
    paths: List[str], prose_paths: List[str], log_paths: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    log_paths = log_paths or []
    corpus: Dict[str, List[str]] = {}
    seen = set()

    def add(kind: str, text: str) -> None:
        text = text.strip()
        if len(text) < MIN_CHARS:
            return
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            return
        seen.add(h)
        corpus.setdefault(kind, []).append(text)

    for f in _iter_files(paths, (".json",)):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        for key, s in _walk_strings(data):
            if key == "reasoning":
                add("Denken", s)
                continue
            for block in THINK_RE.findall(s):
                add("Denken", block)
            answer = THINK_RE.sub("", s)
            add(classify(answer), answer)
    for f in _iter_files(prose_paths, (".md", ".txt")):
        try:
            add("Doku-Prosa (kein Modell)", f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    # Log files: long lines that are (or end in) a JSON value — logged model
    # outputs such as keyword selections.
    for f in _iter_files(log_paths, (".log",)):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if len(line) < MIN_CHARS:
                continue
            start = min((i for i in (line.find("["), line.find("{")) if i >= 0), default=-1)
            if start < 0:
                continue
            try:
                json.loads(line[start:])
            except ValueError:
                continue
            add("JSON (Log)", line[start:])
    return corpus


# ---------------------------------------------------------------------------
# Synthetic loops
# ---------------------------------------------------------------------------

def synthetic_loops(normal_prefix: str, rng: random.Random) -> List[Tuple[str, str, int, str]]:
    """(name, text, loop_onset_char, channel) - Claude Generated"""
    word = "Schlagwort "
    sentence = (
        "Das Werk behandelt die Auswirkungen von Mikroplastik auf marine "
        "Ökosysteme und deren Nahrungsketten. "
    )
    subjects = ["Mikroplastik", "Cadmium", "Klimawandel", "Meeresökologie", "Toxikologie"]
    contexts = ["Nahrungsketten", "Sedimenten", "Küstenregionen", "Fischbeständen", "Plankton"]

    def varied(n: int) -> str:
        return "".join(
            f"Das Werk behandelt {rng.choice(subjects)} im Kontext von {rng.choice(contexts)}. "
            for _ in range(n)
        )

    entries = [
        "- Mikroplastik (GND: 1056637645)\n", "- Meeresökologie (GND: 4074684-1)\n",
        "- Nahrungskette (GND: 4041301-8)\n", "- Umweltgift (GND: 4061636-2)\n",
        "- Plankton (GND: 4046190-6)\n",
    ]
    json_entries = [
        '{"keyword": "Mikroplastik", "gnd_id": "1056637645"},\n',
        '{"keyword": "Meeresökologie", "gnd_id": "4074684-1"},\n',
        '{"keyword": "Nahrungskette", "gnd_id": "4041301-8"},\n',
    ]
    numbered = "".join(f"{i}. Mikroplastik in marinen Ökosystemen\n" for i in range(1, 400))
    think_loop = "".join(
        f"Wait, let me reconsider the keyword {rng.choice(subjects)}. "
        "Actually, I should check whether it is in the GND list. Let me look again. "
        for _ in range(120)
    )
    cases = [
        ("gleiches Wort", word * 600, 0, CHANNEL_ANSWER),
        ("Satz-Wiederholung", sentence * 80, 0, CHANNEL_ANSWER),
        ("variierte Sätze (5×5)", varied(150), 0, CHANNEL_ANSWER),
        ("Liste zyklisch (5)", "".join(entries) * 60, 0, CHANNEL_ANSWER),
        ("JSON zyklisch (3)", "[\n" + "".join(json_entries) * 80, 0, CHANNEL_ANSWER),
        ("Nummerierung, gleicher Text", numbered, 0, CHANNEL_ANSWER),
        ("Zeichenfolge '!'", "!" * 6000, 0, CHANNEL_ANSWER),
        ("Satz nach langem Text", normal_prefix + sentence * 80, len(normal_prefix), CHANNEL_ANSWER),
        ("Liste nach langem Text", normal_prefix + "".join(entries) * 60, len(normal_prefix), CHANNEL_ANSWER),
        ("variiert nach langem Text", normal_prefix + varied(150), len(normal_prefix), CHANNEL_ANSWER),
        ("Nummerierung nach langem Text", normal_prefix + numbered, len(normal_prefix), CHANNEL_ANSWER),
        ("Denkkanal-Schleife", think_loop, 0, CHANNEL_THINKING),
    ]
    return cases


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def chunks_of(text: str, rng: random.Random, lo: int = 1, hi: int = 40) -> Iterable[str]:
    i = 0
    while i < len(text):
        n = rng.randint(lo, hi)
        yield text[i:i + n]
        i += n


def play(text: str, cfg: RepetitionDetectionConfig, channel: str, seed: int):
    """Feed chunk-wise, grace 0; returns (events[(name, char_pos)], detector, cut) - Claude Generated"""
    rng = random.Random(seed)
    det = RepetitionDetector.from_config(cfg, channel=channel)
    mon = RepetitionMonitor(det, auto_abort=True, grace_seconds=0.0)
    events: List[Tuple[str, int]] = []
    pos = 0
    cut: Optional[int] = None
    for piece in chunks_of(text, rng):
        pos += len(piece)
        ev = mon.feed(piece)
        if ev:
            events.append((ev[0], pos))
            if ev[0] == EVENT_ABORT:
                cut = ev[1].position
                break
    if cut is None:
        ev = mon.finish()
        if ev:
            events.append((ev[0] + "@Ende", pos))
            cut = ev[1].position
    return events, det, cut


def smoothed_series(text: str, cfg: RepetitionDetectionConfig, channel: str, masked: bool = False) -> List[float]:
    """Smoothed per-segment rate (plain or digit-masked) - Claude Generated"""
    det = RepetitionDetector.from_config(cfg, channel=channel)
    det.add_chunk(text)
    rates = det.masked_segment_rates if masked else det.segment_rates
    w = det.smoothing_segments
    return [sum(rates[i - w + 1:i + 1]) / w for i in range(w - 1, len(rates))]


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _fmt(v: float) -> str:
    return "—" if v != v else f"{v:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="JSON files or directories with saved results")
    ap.add_argument("--prose", nargs="*", default=[], help=".md/.txt files or dirs (loop-free prose, not model output)")
    ap.add_argument("--log", nargs="*", default=[], help=".log files or dirs (long JSON lines = logged model output)")
    defaults = RepetitionDetectionConfig()
    ap.add_argument("--warn", type=float, default=defaults.warn_threshold)
    ap.add_argument("--abort", type=float, default=defaults.abort_threshold)
    ap.add_argument("--think-warn", type=float, default=defaults.thinking_warn_threshold)
    ap.add_argument("--think-abort", type=float, default=defaults.thinking_abort_threshold)
    ap.add_argument("--segment", type=int, default=defaults.segment_chars)
    ap.add_argument("--smooth", type=int, default=defaults.smoothing_segments)
    ap.add_argument("--seeds", type=int, default=5, help="random chunkings per text (sizes 1-40)")
    args = ap.parse_args(argv)

    cfg = RepetitionDetectionConfig(
        warn_threshold=args.warn, abort_threshold=args.abort,
        thinking_warn_threshold=args.think_warn, thinking_abort_threshold=args.think_abort,
        segment_chars=args.segment, smoothing_segments=args.smooth,
    )
    corpus = load_corpus(args.paths, args.prose, args.log)
    print(
        f"Schwellen: Antwort warn<{cfg.warn_threshold} abort<{cfg.abort_threshold}, "
        f"Denken warn<{cfg.thinking_warn_threshold} abort<{cfg.thinking_abort_threshold}; "
        f"Abschnitt {cfg.segment_chars} Zeichen, Glättung {cfg.smoothing_segments}"
    )

    # 1. rate distribution
    print("\n## 1. Geglättete Rate loop-freier Texte (Minimum je Text; bewertet = mind. "
          f"{cfg.segment_chars * cfg.smoothing_segments} Zeichen)")
    print("| Textart | n Texte | n bewertet | Zeichen median | min | P5 | Median | min ohne Ziffern |")
    print("|---|---|---|---|---|---|---|---|")
    for kind in sorted(corpus):
        texts = corpus[kind]
        channel = CHANNEL_THINKING if kind == "Denken" else CHANNEL_ANSWER
        mins, masked_mins = [], []
        for t in texts:
            series = smoothed_series(t, cfg, channel)
            if series:
                mins.append(min(series))
                masked_mins.append(min(smoothed_series(t, cfg, channel, masked=True)))
        lens = [len(t) for t in texts]
        print(
            f"| {kind} | {len(texts)} | {len(mins)} | {int(median(lens))} | "
            f"{_fmt(min(mins) if mins else float('nan'))} | {_fmt(pct(mins, 0.05))} | "
            f"{_fmt(median(mins) if mins else float('nan'))} | "
            f"{_fmt(min(masked_mins) if masked_mins else float('nan'))} |"
        )

    # 2. synthetic loops
    prose = sorted(
        (t for k, ts in corpus.items() if k in ("Fließtext", "XML-Tags") for t in ts),
        key=len, reverse=True,
    )
    prefix = prose[0][:3000] if prose else ("Normaler Text über ein Buch. " * 10)
    print(f"\n## 2. Synthetische Schleifen (Gnadenfrist 0, {args.seeds} Zerstückelungen 1-40 Zeichen; "
          "Zeichen ab Schleifenbeginn)")
    print("| Fall | Kanal | Rate in der Schleife | ohne Ziffern | bis Warnung | bis Abbruch | "
          "Schnitt − Beginn | unabhängig von Zerstückelung |")
    print("|---|---|---|---|---|---|---|---|")
    for name, text, onset, channel in synthetic_loops(prefix, random.Random(7)):
        steady = smoothed_series(text[onset:], cfg, channel)
        steady_m = smoothed_series(text[onset:], cfg, channel, masked=True)
        warn_lat, abort_lat, cut_off, signatures = [], [], [], set()
        for seed in range(args.seeds):
            events, _det, cut = play(text, cfg, channel, seed)
            # chunk independence: same event kinds at the same segment index
            signatures.add(tuple((n, p // cfg.segment_chars) for n, p in events) + (cut,))
            w = next((p for n, p in events if n in (EVENT_WARN, EVENT_GRACE)), None)
            a = next((p for n, p in events if n.startswith(EVENT_ABORT)), None)
            if w is not None:
                warn_lat.append(w - onset)
            if a is not None:
                abort_lat.append(a - onset)
            if cut is not None:
                cut_off.append(cut - onset)

        def rng_str(v):
            if not v:
                return "—"
            return f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}"

        print(
            f"| {name} | {channel} | {_fmt(steady[-1] if steady else float('nan'))} | "
            f"{_fmt(steady_m[-1] if steady_m else float('nan'))} | {rng_str(warn_lat)} | "
            f"{rng_str(abort_lat)} | {rng_str(cut_off)} | {'ja' if len(signatures) == 1 else 'NEIN'} |"
        )

    # 3. false alarms
    print("\n## 3. Fehlalarme auf loop-freien Texten")
    print("| Textart | n | Warnung | Abbruch |")
    print("|---|---|---|---|")
    for kind in sorted(corpus):
        texts = corpus[kind]
        channel = CHANNEL_THINKING if kind == "Denken" else CHANNEL_ANSWER
        warns = aborts = 0
        for i, t in enumerate(texts):
            events, _d, _c = play(t, cfg, channel, seed=i)
            names = {n.split("@")[0] for n, _p in events}
            if names & {EVENT_WARN, EVENT_GRACE, EVENT_ABORT}:
                warns += 1
            if EVENT_ABORT in names:
                aborts += 1
        print(f"| {kind} | {len(texts)} | {warns}/{len(texts)} | {aborts}/{len(texts)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
