#!/usr/bin/env python3
"""Thinking-channel / token-budget probe for one provider+model - Claude Generated.

Answers three questions for a model that has a reasoning channel:

  1. Does the backend accept the thinking controls ALIMA sends
     (``reasoning_effort`` and ``extra_body.chat_template_kwargs.enable_thinking``,
     see ``LlmService._apply_openai_think``), or does it reject the request?
  2. Does ``think=False`` actually silence the reasoning channel, or is it ignored?
  3. At a given ``max_tokens``, does an answer still come back, or does the
     reasoning channel eat the whole budget (empty content, stop=max_tokens)?

It drives the same entry point as an agentic step: ``generate_with_tools``
without tools (LLMAgentStep → AgentLoop → generate_with_tools).

Examples:
    python scripts/probe_thinking.py --model "LLMachine/deepseek-v4-flash:cloud"
    python scripts/probe_thinking.py --model "LLMachine/nemotron-3-super:cloud" --max-tokens 256 4096
    python scripts/probe_thinking.py --model "GWDG/..." --abstract-file my_abstract.txt
    python scripts/probe_thinking.py --model "LLMachine/deepseek-v4-flash:cloud" \
        --workflow-step alima_v51:extraction --abstract-file abstract.txt --repeat 3

Notes:
  - Hits the real provider; it is a diagnostic, not a unit test.
  - "answer" counts characters of ``AgentResponse.content``, "reasoning"
    characters of ``AgentResponse.reasoning``. A row with answer=0 and
    reasoning>0 is the failure mode this probe exists for.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Short, self-contained stand-in for the `extraction` step of alima_v51.yaml:
# German abstract in, strict JSON out. Long enough to need a real answer,
# short enough to keep the probe cheap.
DEFAULT_ABSTRACT = (
    "Cadmium gehört zu den toxischsten Schwermetallen und reichert sich über "
    "belastete Böden in Nutzpflanzen an. Der Band beschreibt Aufnahmewege in "
    "Pflanzen, die Folgen für Bodenmikroorganismen und Verfahren zur "
    "Sanierung belasteter Standorte, darunter Phytoremediation und der "
    "Einsatz von Biokohle."
)

SYSTEM_PROMPT = (
    "Du bist ein bibliothekarischer Sacherschließungs-Assistent. "
    "Antworte ausschließlich mit gültigem JSON, ohne Vorrede."
)

USER_TEMPLATE = (
    "Extrahiere aus dem folgenden Abstract deutsche Schlagwörter und einen "
    "Arbeitstitel.\n\n"
    'Antwortformat: {{"title": "...", "keywords": ["...", "..."]}}\n\n'
    "Abstract:\n{abstract}"
)

_THINK_LABEL = {None: "default", False: "off", True: "on"}


def _split_model(spec: str):
    """'provider/model' or 'provider|model' → (provider, model)."""
    sep = "|" if "|" in spec else "/"
    provider, _, model = spec.partition(sep)
    return provider.strip(), model.strip()


def _load_workflow_prompts(spec: str):
    """'alima_v51:extraction' → (system_prompt, user_prompt) from the YAML.

    Reads the step's raw prompts so the probe measures the real workload
    instead of a stand-in. Placeholders are filled by the caller the same way
    ``LLMAgentStep._render`` does (plain ``{name}`` replacement). - Claude Generated
    """
    import yaml
    from src.core.agents.workflow_loader import find_workflow_file

    stem, _, step_id = spec.partition(":")
    if not step_id:
        raise SystemExit("--workflow-step braucht die Form WORKFLOW:STEP")
    path = find_workflow_file(stem.strip())
    if not path:
        raise SystemExit(f"Workflow '{stem}' nicht gefunden")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for entry in data.get("steps", []) or []:
        if isinstance(entry, dict) and str(entry.get("id")) == step_id.strip():
            return entry.get("system_prompt", ""), entry.get("user_prompt", "")
    raise SystemExit(f"Schritt '{step_id}' nicht in {path.name}")


def _run_cell(svc, provider, model, messages, think, max_tokens, temperature):
    """One (think × max_tokens) measurement. Returns a result dict.

    A provider-side rejection of the thinking params surfaces as an exception
    here; that is a result, not a crash, so it is caught and reported. - Claude Generated
    """
    started = time.time()
    try:
        resp = svc.generate_with_tools(
            provider=provider,
            model=model,
            messages=messages,
            tools=None,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            think=think,
        )
        return {
            "ok": True,
            "answer": len(resp.content or ""),
            "reasoning": len(getattr(resp, "reasoning", "") or ""),
            "stop": getattr(resp.stop_reason, "value", str(resp.stop_reason)),
            "secs": time.time() - started,
            "sample": (resp.content or "").strip().replace("\n", " ")[:70],
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "answer": 0,
            "reasoning": 0,
            "stop": "-",
            "secs": time.time() - started,
            "sample": "",
            "error": f"{type(e).__name__}: {e}"[:200],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="provider/model, e.g. 'LLMachine/deepseek-v4-flash:cloud'")
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[256, 4096],
                    help="budgets to probe (default: 256 4096; 4096 is the alima_v51 setting)")
    ap.add_argument("--think", choices=["default", "off", "on"], nargs="+",
                    default=["default", "off", "on"], help="which thinking settings to probe")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--abstract-file", help="file with the abstract to use instead of the built-in one")
    ap.add_argument("--workflow-step", metavar="WORKFLOW:STEP",
                    help="use a real step's prompts, e.g. 'alima_v51:extraction'")
    ap.add_argument("--repeat", type=int, default=1, help="runs per cell (models vary between calls)")
    args = ap.parse_args()

    provider, model = _split_model(args.model)
    if not provider or not model:
        print("❌ --model braucht die Form provider/model")
        return 2

    abstract = DEFAULT_ABSTRACT
    if args.abstract_file:
        abstract = Path(args.abstract_file).read_text(encoding="utf-8").strip()

    from src.llm.llm_service import LlmService
    from src.utils.config_manager import ConfigManager

    svc = LlmService(config_manager=ConfigManager(), lazy_initialization=True)
    if not svc._ensure_provider_initialized(provider):
        print(f"❌ Provider '{provider}' nicht erreichbar/initialisierbar")
        return 1

    if args.workflow_step:
        system_prompt, user_tpl = _load_workflow_prompts(args.workflow_step)
        user_prompt = user_tpl.replace("{abstract}", abstract).replace("{keywords}", "")
        workload = args.workflow_step
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_TEMPLATE.format(abstract=abstract)
        workload = "built-in"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    think_values = [{"default": None, "off": False, "on": True}[t] for t in args.think]

    print(f"Modell: {provider}/{model}  ·  temp={args.temperature}  ·  {args.repeat}× je Zelle")
    print(f"Aufgabe: {workload}  ·  Prompt {len(system_prompt) + len(user_prompt)} Zeichen "
          f"(Abstract {len(abstract)})")
    print("think=off sendet reasoning_effort='none' + enable_thinking=false "
          "(LlmService._apply_openai_think)\n")
    print(f"{'think':<9}{'budget':<9}{'req':<6}{'answer':<8}{'reason':<8}{'stop':<11}{'secs':<7}sample / error")

    rows = []
    for think in think_values:
        for budget in args.max_tokens:
            for _ in range(args.repeat):
                r = _run_cell(svc, provider, model, messages, think, budget, args.temperature)
                rows.append((_THINK_LABEL[think], budget, r))
                tail = r["error"] or r["sample"]
                print(f"{_THINK_LABEL[think]:<9}{budget:<9}{'ok' if r['ok'] else 'FAIL':<6}"
                      f"{r['answer']:<8}{r['reasoning']:<8}{r['stop']:<11}{r['secs']:<7.1f}{tail}")

    print("\nBefund:")
    for label in dict.fromkeys(r[0] for r in rows):
        cells = [r[2] for r in rows if r[0] == label]
        rejected = [c for c in cells if not c["ok"]]
        if rejected:
            print(f"  think={label}: {len(rejected)}/{len(cells)} Anfragen abgelehnt "
                  f"→ {rejected[0]['error']}")
            continue
        max_reason = max(c["reasoning"] for c in cells)
        empty = [c for c in cells if c["answer"] == 0]
        print(f"  think={label}: Reasoning max {max_reason} Zeichen, "
              f"{len(empty)}/{len(cells)} Läufe ohne Antwort")
    return 0


if __name__ == "__main__":
    sys.exit(main())
