#!/usr/bin/env python3
"""Chat-agent A/B evaluation harness - Claude Generated.

Runs a fixed set of queries through the real chat agent loop (real tools) for a
given model and prompt tier, and reports per-query: iterations, tool calls,
final-answer length, success, duration. Use it to compare a model across prompt
tiers / parameters before picking a chat default.

Examples:
    python scripts/chat_eval.py --model "LLMachine/north-mini-code-1.0:latest" --tier both
    python scripts/chat_eval.py --model "GWDG/..." --tier compact --queries my_queries.txt
    python scripts/chat_eval.py --model "LLMachine/..." --max-iterations 8 --max-tokens 8192

Notes:
  - Executes real tool calls (catalog/GND/DK lookups) → needs the providers and
    catalog endpoints reachable. It is a diagnostic, not a unit test.
  - "ok" = the run produced a non-empty answer that is not one of the loop's
    fallback/warning messages.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEFAULT_QUERIES = [
    "Welche Werke von Conrad Hübler gibt es im Katalog? Nutze finc.",
    "Finde Literatur zur Zuckerchemie.",
    "Welche Lehrbücher zur Biologie haben wir im Bestand?",
    "Suche den GND-Eintrag für Quantenchemie.",
    "Welche DK-Klassifikation passt zu Halbleiterphysik?",
]

# Substrings that mark the agent-loop fallback/warning answers (not a real answer).
_FALLBACK_MARKERS = (
    "keine Antwort geliefert",
    "keine finale Textantwort",
    "Token-Limit erreicht",
    "maximum iterations without conclusion",
)


def _split_model(spec: str):
    """'provider/model' or 'provider|model' → (provider, model)."""
    sep = "|" if "|" in spec else "/"
    provider, _, model = spec.partition(sep)
    return provider.strip(), model.strip()


def _is_ok(content: str) -> bool:
    if not content or not content.strip():
        return False
    return not any(m in content for m in _FALLBACK_MARKERS)


def run_one(svc, reg, provider, model, query, *, compact, max_iterations, max_tokens):
    from src.core.agent_loop import AgentLoop
    from src.core.chat_prompts import (
        build_system_prompt, get_user_prompt_template, detect_mode, apply_chat_directives,
    )

    mode = detect_mode(query)
    system_prompt = apply_chat_directives(
        build_system_prompt(mode=mode, compact=compact), language="de"
    )
    user_prompt = get_user_prompt_template(mode).format(
        context="(kein Werk geladen)", user_message=query
    )
    loop = AgentLoop(
        llm_service=svc, tool_registry=reg,
        max_iterations=max_iterations, timeout_seconds=300,
        stream_callback=None, status_callback=None,
    )
    t0 = time.time()
    res = loop.run(
        system_prompt=system_prompt, user_prompt=user_prompt, tools=[],
        provider=provider, model=model, max_tokens=max_tokens,
    )
    dt = time.time() - t0
    tools = [e.get("tool") for e in (res.tool_log or [])]
    content = res.content or ""
    return {
        "mode": mode,
        "iters": res.iterations,
        "stop": res.stop_reason,
        "tools": tools,
        "final_len": len(content),
        "ok": _is_ok(content),
        "secs": round(dt, 1),
        "preview": content[:120].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser(description="Chat-agent A/B evaluation harness")
    ap.add_argument("--model", required=True, help="provider/model, e.g. 'LLMachine/north-mini-code-1.0:latest'")
    ap.add_argument("--tier", choices=["compact", "full", "both"], default="both")
    ap.add_argument("--queries", help="Path to a file with one query per line (else built-in defaults)")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    provider, model = _split_model(args.model)
    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [l.strip() for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    tiers = [True, False] if args.tier == "both" else [args.tier == "compact"]

    from src.utils.config_manager import ConfigManager
    from src.llm.llm_service import LlmService
    from src.mcp.tool_registry import ToolRegistry

    cm = ConfigManager()
    svc = LlmService(config_manager=cm, lazy_initialization=True)
    if not svc._ensure_provider_initialized(provider):
        print(f"❌ Provider '{provider}' nicht erreichbar/initialisierbar")
        sys.exit(1)
    reg = ToolRegistry()
    reg.register_all_tools()

    print(f"Modell: {provider}/{model}  ·  max_iter={args.max_iterations}  max_tokens={args.max_tokens}")
    print(f"Queries: {len(queries)}  ·  Tiers: {'compact+full' if args.tier=='both' else args.tier}\n")

    summary = {}
    for compact in tiers:
        label = "compact" if compact else "full"
        print(f"===== PROMPT-TIER: {label} =====")
        print(f"{'#':<3}{'ok':<4}{'iters':<6}{'tools':<7}{'len':<6}{'secs':<6}{'stop':<11}query")
        oks = 0
        for i, q in enumerate(queries, 1):
            try:
                r = run_one(svc, reg, provider, model, q, compact=compact,
                            max_iterations=args.max_iterations, max_tokens=args.max_tokens)
            except Exception as e:
                print(f"{i:<3}ERR  {str(e)[:60]}")
                continue
            oks += int(r["ok"])
            print(f"{i:<3}{'✓' if r['ok'] else '✗':<4}{r['iters']:<6}{len(r['tools']):<7}"
                  f"{r['final_len']:<6}{r['secs']:<6}{r['stop']:<11}{q[:50]}")
            print(f"      tools={r['tools']}  → {r['preview']!r}")
        summary[label] = (oks, len(queries))
        print()

    print("===== SUMMARY =====")
    for label, (oks, total) in summary.items():
        print(f"  {label:<8}: {oks}/{total} ok")


if __name__ == "__main__":
    main()
