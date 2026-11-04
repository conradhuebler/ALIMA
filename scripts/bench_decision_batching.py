#!/usr/bin/env python3
"""Misst Blockgröße und Parallelität des Entscheidungsmodells - Claude Generated.

Ein Aufruf an ``/v1/systemone`` nimmt bis zu 64 Fragen. Hier wird für ein Modell
dieselbe Kandidatenliste mit verschiedenen Blockgrößen (Fragen je Anfrage) und
verschiedener Parallelität (gleichzeitige Anfragen) bewertet und gemessen:

* Wanduhrzeit, Anzahl Anfragen, Zeit je Frage, Eingabetoken,
* ob die Scores von der Blockgröße abhängen (größte Abweichung und Überlappung der
  besten 40 gegenüber dem Lauf mit dem größten Block).

Die Kandidaten kommen aus einer Score-Datei von ``eval_decision_rank.py`` (``--scores-file``),
der Text aus ``--text-file`` (erste 1500 Zeichen, wie die Pipeline).
Ohne ``--yes`` wird nur der Plan gedruckt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import decision_client  # noqa: E402
from src.utils.decision_client import (  # noqa: E402
    MAX_QUESTIONS,
    QUESTION_VARIANTS,
    provider_base_url,
    score_items,
)
from src.utils.decision_prefilter import DEFAULT_MAX_STATE_CHARS  # noqa: E402


def compare(ref: Dict[str, float], other: Dict[str, float], top: int = 40) -> Tuple[float, int]:
    """(größte Abweichung der Scores, Überlappung der besten ``top``)."""
    common = [t for t in ref if t in other]
    worst = max((abs(ref[t] - other[t]) for t in common), default=0.0)
    best_ref = set(sorted(ref, key=lambda t: -ref[t])[:top])
    best_other = set(sorted(other, key=lambda t: -other[t])[:top])
    return worst, len(best_ref & best_other)


def configs(batch_sizes: List[int], parallel: List[int], parallel_sizes: List[int]) -> List[Tuple[int, int]]:
    """Serielle Läufe für alle Blockgrößen, parallele nur für ``parallel_sizes``."""
    out = [(b, 1) for b in batch_sizes]
    out += [(b, w) for b in parallel_sizes for w in parallel if w > 1]
    return out


def run_config(base_url: str, model: str, text: str, items: List[str], batch: int, workers: int,
               api_key: str, variant: str) -> Dict[str, Any]:
    calls = [0]
    real = decision_client.systemone

    def counting(*a: Any, **k: Any) -> Any:
        calls[0] += 1
        return real(*a, **k)

    decision_client.systemone = counting
    usage: Dict[str, int] = {}
    try:
        t0 = time.perf_counter()
        scores = score_items(base_url, model, text, items, api_key=api_key, max_parallel=workers,
                             usage_sink=usage, batch_size=batch, **QUESTION_VARIANTS[variant])
        secs = time.perf_counter() - t0
    finally:
        decision_client.systemone = real
    return {"batch": batch, "workers": workers, "requests": calls[0], "seconds": round(secs, 2),
            "per_question_ms": round(1000 * secs / max(1, len(items)), 1),
            "input_tokens": usage.get("input_tokens", 0), "scores": scores}


def _spec(spec: str) -> Tuple[Any, str]:
    from types import SimpleNamespace

    if spec.startswith(("http://", "https://")):
        url, _, model = spec.partition("#")
        return SimpleNamespace(name=url, provider_type="ollama", base_url=url, api_key=""), model
    from src.utils.config_manager import ConfigManager

    name, _, model = spec.rpartition("/")
    provider = ConfigManager().get_unified_config().get_provider_by_name(name)
    if provider is None or not model:
        raise SystemExit(f"Modell '{spec}': Anbieter/Modell oder http://host:port#modell erwartet")
    return provider, model


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-file", required=True, help="Score-Datei von eval_decision_rank.py (liefert die Kandidaten)")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variant", default="geeignet", choices=sorted(QUESTION_VARIANTS))
    ap.add_argument("--n", type=int, default=128, help="Anzahl Kandidaten je Lauf")
    ap.add_argument("--batch-sizes", default="1,4,16,64")
    ap.add_argument("--parallel", default="2,4", help="gleichzeitige Anfragen für die parallelen Läufe")
    ap.add_argument("--parallel-sizes", default="16", help="Blockgrößen, die parallel gemessen werden")
    ap.add_argument("--max-requests", type=int, default=800, help="Obergrenze je Modell (Schutz)")
    ap.add_argument("--out", default="bench_decision_batching.json")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    items = list(json.load(open(args.scores_file, encoding="utf-8"))["scores"])[: args.n]
    text = open(args.text_file, encoding="utf-8").read()[:DEFAULT_MAX_STATE_CHARS]
    bs = [min(int(b), MAX_QUESTIONS) for b in args.batch_sizes.split(",")]
    cfgs = configs(bs, [int(w) for w in args.parallel.split(",")], [int(b) for b in args.parallel_sizes.split(",")])
    per_model = sum(math.ceil(len(items) / b) for b, _ in cfgs)
    print(f"{len(items)} Kandidaten, State {len(text)} Zeichen, Variante {args.variant}")
    print(f"Plan je Modell: {len(cfgs)} Läufe {cfgs}, mindestens {per_model} Anfragen, "
          f"{len(cfgs) * len(items)} Fragen")
    if per_model > args.max_requests:
        raise SystemExit(f"{per_model} Anfragen je Modell übersteigen --max-requests {args.max_requests}")
    if not args.yes:
        print("Nichts angefragt (ohne --yes).")
        return 0

    report: Dict[str, Any] = {}
    for spec in args.models:
        provider, model = _spec(spec)
        base = provider_base_url(provider)
        key = getattr(provider, "api_key", "") or ""
        results = [run_config(base, model, text, items, b, w, key, args.variant) for b, w in cfgs]
        ref = max(results, key=lambda r: (r["batch"], -r["workers"]))["scores"]
        print(f"\n{spec}")
        print(f"{'Block':>6} {'parallel':>8} {'Anfragen':>9} {'Sek.':>7} {'ms/Frage':>9} {'Tokens':>9} {'max|Δ|':>8} {'Top40':>6}")
        for r in results:
            worst, overlap = compare(ref, r["scores"])
            r["max_diff"], r["top40_overlap"] = round(worst, 4), overlap
            print(f"{r['batch']:>6} {r['workers']:>8} {r['requests']:>9} {r['seconds']:>7} "
                  f"{r['per_question_ms']:>9} {r['input_tokens']:>9} {worst:>8.4f} {overlap:>6}")
        report[spec] = [{k: v for k, v in r.items() if k != "scores"} for r in results]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\nErgebnisse: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
