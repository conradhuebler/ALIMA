#!/usr/bin/env python3
"""Vergleicht Einstellungen des Entscheidungsmodell-Rankings an einer Soll-Liste - Claude Generated.

Ein Dokument, ein Pool, eine Soll-Liste (Schlagworte, die ein Fachreferent vergeben
würde). Je Modell und Fragetext wird jeder Kandidat einmal bewertet; Schnittgröße
(``--ks``) und ``balanced`` (reihum je Suchbegriff) werden danach aus denselben
Scores gerechnet, ohne weitere Abfragen. Die Scores landen in ``--out``; mit
``--reuse`` rechnet das Skript neu aus, was dort liegt.

Pool: entweder eine gespeicherte Datei im Format von ``aggregate_gnd_results``
(``--pool``) oder der Rohantwort-Cache zu den Suchbegriffen (``--from-cache``,
nach den ``search_*``-Aufrufen im Chat; liest nur den Cache).

Ohne ``--yes`` druckt das Skript nur den Plan (Anzahl Anfragen) und fragt nichts an.

Beispiel::

    python scripts/eval_decision_rank.py --from-cache --terms "Cadmium,Reis,Itai-Itai-Krankheit" \\
        --text-file abstract.txt --gold soll.txt --models LLMachine/d1 "Ollama Cloud/clef:27b" \\
        --variants geeignet behandelt --out eval_cadmium --yes

Soll-Liste: eine Zeile je Schlagwort, ``Cadmium``, ``Cadmium (GND-ID: 4009274-4)`` oder
nur die ID; Zeilen mit ``#`` am Anfang zählen nicht.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.decision_client import (  # noqa: E402
    MAX_QUESTIONS,
    QUESTION_VARIANTS,
    provider_base_url,
    score_items,
)
from src.utils.decision_prefilter import DEFAULT_MAX_STATE_CHARS  # noqa: E402
from src.utils.decision_tools import _balanced_pick, pool_candidates, pool_terms  # noqa: E402

_ID = re.compile(r"(\d{6,}-[\dXx]|\d{6,})")


def parse_gold(lines: List[str]) -> List[Dict[str, str]]:
    """Soll-Zeilen → ``{"label", "gnd_id"}`` (eines von beiden kann leer sein)."""
    out = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r"GND-ID:\s*([0-9Xx-]+)", line)
        if m:
            out.append({"label": line[: m.start()].strip(" ("), "gnd_id": m.group(1)})
        elif _ID.fullmatch(line):
            out.append({"label": "", "gnd_id": line})
        else:
            out.append({"label": line, "gnd_id": ""})
    return out


def find_gold(gold: List[Dict[str, str]], candidates: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Soll-Eintrag → Text des passenden Kandidaten (``None`` = nicht im Pool)."""
    def norm(text: str) -> str:
        # Pool-Titel tragen Umlaute teils zerlegt (a + U+0308): auf NFC bringen.
        return unicodedata.normalize("NFC", text).casefold()

    by_id = {c["gnd_id"]: c["keyword"] for c in candidates if c.get("gnd_id")}
    by_label = {norm(c["keyword"]): c["keyword"] for c in candidates}
    found: Dict[str, Optional[str]] = {}
    for g in gold:
        name = g["label"] or g["gnd_id"]
        found[name] = by_id.get(g["gnd_id"]) or by_label.get(norm(g["label"]))
    return found


def evaluate(
    scores: Dict[str, float], texts: List[str], groups: Dict[str, List[str]],
    found: Dict[str, Optional[str]], ks: List[int], balanced: bool,
) -> Dict[str, Any]:
    """Recall@K und Rang je Soll-Eintrag aus gegebenen Scores."""
    order = sorted(texts, key=lambda t: -scores.get(t, 0.0))
    in_pool = {n: t for n, t in found.items() if t is not None}
    result: Dict[str, Any] = {"in_pool": len(in_pool), "gold": len(found), "recall": {}, "rank": {}}
    for k in ks:
        chosen = set(_balanced_pick(order, groups, k)) if balanced else set(order[:k])
        result["recall"][k] = sum(1 for t in in_pool.values() if t in chosen)
    rank_of = {t: i + 1 for i, t in enumerate(order)}
    result["rank"] = {n: rank_of[t] for n, t in in_pool.items()}
    return result


def _provider(unified: Any, spec: str) -> Tuple[Any, str]:
    """``Anbieter/Modell`` (Anbieter aus den Einstellungen) oder ``http://host:port#modell``."""
    if spec.startswith(("http://", "https://")):
        from types import SimpleNamespace

        url, _, model = spec.partition("#")
        if not model:
            raise SystemExit(f"Modell '{spec}': nach '#' fehlt der Modellname.")
        return SimpleNamespace(name=url, provider_type="ollama", base_url=url, api_key=""), model
    name, _, model = spec.rpartition("/")
    provider = unified.get_provider_by_name(name)
    if provider is None or not model:
        raise SystemExit(f"Modell '{spec}': erwartet Anbieter/Modell, Anbieter in den Einstellungen.")
    return provider, model


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", help="Datei im Format von aggregate_gnd_results")
    src.add_argument("--from-cache", action="store_true", help="Pool aus dem Rohantwort-Cache (mit --terms)")
    ap.add_argument("--terms", help="Suchbegriffe, durch Komma getrennt (mit --from-cache)")
    ap.add_argument("--text-file", required=True, help="Abstract, gegen den bewertet wird")
    ap.add_argument("--gold", required=True, help="Soll-Liste")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Anbieter/Modell (Anbieter wie in den Einstellungen) oder http://host:port#modell")
    ap.add_argument("--variants", nargs="+", default=["geeignet"], choices=sorted(QUESTION_VARIANTS))
    ap.add_argument("--ks", default="40,80,150", help="Schnittgrößen, Komma getrennt")
    ap.add_argument("--out", default="eval_decision_rank", help="Ordner für die Scores")
    ap.add_argument("--reuse", action="store_true", help="Gespeicherte Scores in --out verwenden, nichts anfragen")
    ap.add_argument("--parallel", type=int, default=1, help="gleichzeitige Anfragen je Lauf (Standard 1)")
    ap.add_argument("--yes", action="store_true", help="Anfragen wirklich senden")
    args = ap.parse_args(argv)

    if args.from_cache:
        from src.mcp.tool_registry import ToolRegistry

        if not args.terms:
            raise SystemExit("--from-cache braucht --terms")
        reg = ToolRegistry()
        reg.register_all_tools()
        aggregate = json.loads(reg._handle_aggregate_gnd_results([t.strip() for t in args.terms.split(",") if t.strip()]))
    else:
        with open(args.pool, encoding="utf-8") as fh:
            aggregate = json.load(fh)
    candidates, skipped = pool_candidates(aggregate)
    texts = list(dict.fromkeys(c["keyword"] for c in candidates))
    groups = pool_terms(aggregate)
    text = open(args.text_file, encoding="utf-8").read()[:DEFAULT_MAX_STATE_CHARS]
    gold = parse_gold(open(args.gold, encoding="utf-8").read().splitlines())
    found = find_gold(gold, candidates)
    ks = [int(k) for k in args.ks.split(",")]
    print(f"Pool: {len(texts)} Kandidaten ({skipped} ohne Bezeichnung übersprungen); "
          f"Soll: {len(gold)}, davon im Pool: {sum(1 for v in found.values() if v)}")
    for n, t in found.items():
        if t is None:
            print(f"  nicht im Pool: {n}")

    runs = [(m, v) for m in args.models for v in args.variants]
    requests = len(runs) * -(-len(texts) // MAX_QUESTIONS)
    if not args.reuse:
        print(f"Plan: {len(runs)} Läufe × {-(-len(texts) // MAX_QUESTIONS)} Anfragen = {requests} Anfragen, "
              f"{len(texts)} Fragen je Lauf, State {len(text)} Zeichen.")
        if not args.yes:
            print("Nichts angefragt (ohne --yes). Mit --yes ausführen oder --reuse für gespeicherte Scores.")
            return 0
    os.makedirs(args.out, exist_ok=True)

    unified = None
    if not args.reuse:
        from src.utils.config_manager import ConfigManager

        unified = ConfigManager().get_unified_config()

    rows = []
    for spec, variant in runs:
        path = os.path.join(args.out, re.sub(r"[^\w.-]+", "_", f"{spec}__{variant}") + ".json")
        if args.reuse:
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
        else:
            provider, model = _provider(unified, spec)
            usage: Dict[str, int] = {}
            scores = score_items(
                provider_base_url(provider), model, text, texts, api_key=getattr(provider, "api_key", "") or "",
                max_parallel=args.parallel, usage_sink=usage, **QUESTION_VARIANTS[variant],
            )
            saved = {"model": spec, "variant": variant, "scores": scores, "input_tokens": usage.get("input_tokens", 0),
                     "n": len(texts), "state_chars": len(text)}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(saved, fh, ensure_ascii=False)
        for balanced in (False, True):
            ev = evaluate(saved["scores"], texts, groups, found, ks, balanced)
            rows.append((spec, variant, balanced, ev, saved.get("input_tokens", 0)))

    head = "Modell / Fragetext / balanced".ljust(54) + "".join(f"R@{k}".rjust(7) for k in ks) + "   Tokens"
    print("\n" + head)
    for spec, variant, balanced, ev, tokens in rows:
        label = f"{spec} / {variant} / {'ja' if balanced else 'nein'}"
        cells = "".join(f"{ev['recall'][k]}/{ev['in_pool']}".rjust(7) for k in ks)
        print(label.ljust(54) + cells + f"   {tokens}")
    print("\nRang der Soll-Einträge (global, ohne balanced):")
    for spec, variant, balanced, ev, _ in rows:
        if not balanced:
            ranks = ", ".join(f"{n} {r}" for n, r in sorted(ev["rank"].items(), key=lambda x: x[1]))
            print(f"  {spec} / {variant}: {ranks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
