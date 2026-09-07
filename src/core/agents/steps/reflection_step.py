"""ReflectionStep — quality-check step for MetaAgent loop - Claude Generated.

Runs an LLM-based reflection over the current SharedContext state,
producing a quality report with gaps and recommended next actions.
Called by MetaAgent between execution cycles, not directly from YAML.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.agent_loop import AgentLoop
from src.core.agents.prompt_resolver import resolve_prompts
from src.core.agents.registry import register_step
from src.core.agents.steps.base_step import BaseStep, StepConfig
from src.core.agents.steps.llm_agent_step import (
    _emit_header,
    _emit_prompt_done,
    _emit_prompts,
    _log_response,
)

logger = logging.getLogger(__name__)


# Generic, DOMAIN-NEUTRAL base. ALIMA-specific phase/GND/DK checks live in the
# workflow YAML (`meta_agent.reflection.rules`) and are injected at the
# `{workflow_rules}` slot by MetaAgent — so non-keywording pipelines don't get
# a GND/DK-flavoured quality gate. - Claude Generated
DEFAULT_REFLECTION_SYSTEM_PROMPT = (
    "Du bist der ALIMA Quality Agent. Du bewertest den aktuellen Stand einer "
    "Pipeline und empfiehlst, ob fortgefahren, ein Schritt wiederholt oder "
    "abgeschlossen wird.\n"
    "WICHTIG: Prüfe NUR Kriterien, die für den AKTUELLEN Pipeline-Schritt relevant sind.\n\n"
    "Allgemeine Logik:\n"
    "- Prüfe, ob der aktuelle Schritt sein erwartetes Ergebnis geliefert hat.\n"
    "- Wiederhole einen Schritt NUR bei einem konkreten, benannten Mangel — niemals "
    "mit der Begründung 'könnte noch optimiert/vertieft werden'.\n"
    "- status='complete', sobald alle Schritte gelaufen sind und keine offenen, noch "
    "nicht behandelten Lücken bestehen.\n"
    "{workflow_rules}\n"
    "{user_rules_gate}"
    "Ausgabe als valides JSON:\n"
    '{\n'
    '  "status": "complete" | "incomplete" | "continue",\n'
    '  "gaps": [...],\n'
    '  "action": "finish" | "continue" | "search_missing" | "rerun_search" | "rerun_selection" | "rerun_classification" | "rerun_dk_collect",\n'
    '  "reason": "..."\n'
    '}\n'
    "Keine Erläuterungen außerhalb des JSON."
)

# Filled into `{user_rules_gate}` only when the operator has rules that reach
# this gate. The reflection turn is the last LLM turn of an agentic run — the
# workflow ends in deterministic steps — so it is the only place where a rule
# that asks for something "am Ende" can still be carried out. - Claude Generated
# Always shown when rules reach this gate: they are quality criteria like any
# other. - Claude Generated
USER_RULES_INTRO = (
    "Persönliche Zusatzregeln des Betreibers:\n"
    "{user_rules}\n"
    "- Prüfe den Stand auch gegen diese Regeln.\n"
    "- Betrifft eine Regel nur die AUSGABE, ist das kein Grund, einen Schritt zu "
    "wiederholen.\n"
)

# Appended ONLY on the last reflection of a run — the one after which no step is
# pending any more. The reflection fires once per cycle; without this condition
# a model that reports 'complete' early re-generates the whole block in every
# remaining cycle. The MetaAgent decides, deterministically, via
# ``_pending_step``: it knows the step graph, the model does not. - Claude Generated
USER_RULES_FINAL_GATE = (
    "- **Dies ist die letzte Reflexion dieses Laufs**; nach dir läuft kein "
    "weiterer Schritt. Verlangt eine Regel eine Ausgabe am Ende (einen Eintrag, "
    "ein Format, ein Snippet), erzeuge sie jetzt selbst — kündige sie nicht an, "
    "sondern schreibe sie hin.\n"
    "- **Nimm die Daten aus dem Zustand oben, erfinde nichts.** Ketten,\n"
    "  Schlagworte und Notationen stehen dort ausgeschrieben; gruppiere die\n"
    "  Schlagworte nicht selbst neu und ordne keinem Schlagwort eine Kette zu,\n"
    "  in der es nicht steht.\n"
    "- Die Ausgabe gehört NICHT ins JSON. Hänge sie NACH dem JSON so an:\n"
    "  <final_output>\n"
    "  …die fertige Ausgabe, mehrzeilig, genau im geforderten Format…\n"
    "  </final_output>\n"
    "  (Das ist die einzige erlaubte Ausnahme von 'keine Erläuterungen außerhalb "
    "des JSON'. Zeilenumbrüche in einen JSON-String zu packen zerstört die "
    "Antwort.)\n"
    "- Ist nichts zu erzeugen, lass den Block ganz weg.\n"
)

#: Set by the MetaAgent on the context before the reflection that ends the run.
FINAL_GATE_FLAG = "_rules_final_gate"

DEFAULT_REFLECTION_USER_PROMPT = (
    "Aktueller Pipeline-Zustand:\n"
    "- Arbeitstitel: {working_title}\n"
    "- Extrahierte Keywords: {extracted_keywords_count}\n"
    "- GND-Einträge: {gnd_entries_count}\n"
    "- Ausgewählte Keywords: {selected_keywords_count}\n"
    "- Finale Schlagworte: {final_keywords}\n"
    "- Schlagwortketten ({keyword_chains_count}):\n{keyword_chains}\n"
    "- Kernschlagworte (RSWK-Kern): {core_keywords}\n"
    "- Formschlagworte: {form_keywords}\n"
    "- DK-Klassifikationen: {dk_classifications_count}\n"
    "- DK-Codes (Ist): {dk_codes}\n"
    "- Hat tiefe DK-Codes (≥4 Ziffern, deterministisch geprüft): {has_deep_dk}\n"
    "- Analyse-Begründung (Auszug): {analyse_excerpt}\n"
    "- Katalog-Titel gesamt: {catalog_total_titles}\n"
    "- Katalog-Notationen (unique): {catalog_unique_notations}\n"
    "- Fehlende Konzepte: {missing_concepts}\n"
    "- Bereits gesuchte Missing Concepts: {missing_concepts_searched}\n"
    "- Abgeschlossene Steps: {steps_completed}\n"
    "- Aktueller Schritt: {current_step}\n\n"
    "Anweisung: Entscheide basierend auf dem AKTUELLEN Schritt '{current_step}', "
    "nicht basierend auf dem Endzustand.\n"
    "Wenn fehlende Konzepte gemeldet wurden UND sie noch NICHT gesucht wurden, "
    "dann wähle action='search_missing' — nicht 'rerun_search'.\n"
    "Wenn fehlende Konzepte schon gesucht wurden, wähle 'continue' oder 'finish'.\n\n"
    "Welche Phase ist erreicht und was fehlt noch?"
)


def _format_chains(chains: Any) -> str:
    """The Schlagwortketten as indented ``A → B → C`` lines. - Claude Generated"""
    lines = []
    for entry in chains or []:
        if not isinstance(entry, dict):
            continue
        terms = [str(t).strip() for t in (entry.get("chain") or []) if str(t).strip()]
        if terms:
            lines.append("    " + " → ".join(terms))
    return "\n".join(lines) if lines else "    keine"


def _format_keyword_list(context: Any, field: str) -> str:
    """A KAS-style keyword bucket as a comma list, or ``keine``."""
    values = (getattr(context, "extra", None) or {}).get(field) or []
    names = [
        str(item.get("keyword", "")).strip()
        for item in values
        if isinstance(item, dict) and str(item.get("keyword", "")).strip()
    ]
    if not names:
        names = [str(v).strip() for v in values if isinstance(v, str) and v.strip()]
    return ", ".join(names) if names else "keine"


def _user_rules_values(context: Any) -> Dict[str, str]:
    """``user_rules_gate`` / ``user_rules`` for the reflection prompt.

    Both are empty strings when no rule reaches this gate, so the prompt stays
    byte-identical to the pre-feature one. - Claude Generated
    """
    from src.core.user_rules import STEP_REFLECTION, render_rule_line, rules_block_for

    workflow = str(getattr(context, "workflow_name", "") or "")
    block, rules = rules_block_for(workflow=workflow, step=STEP_REFLECTION)
    if not block:
        return {"user_rules_gate": "", "user_rules": ""}
    # The generic injection point skips this step (the gate below owns it), so
    # the run's provenance has to be recorded here. - Claude Generated
    from src.core.agents.prompt_resolver import _record_applied

    _record_applied(context, rules)
    rendered = "\n".join(render_rule_line(r) for r in rules)
    gate = USER_RULES_INTRO.replace("{user_rules}", rendered)
    if bool((getattr(context, "extra", None) or {}).get(FINAL_GATE_FLAG)):
        gate += USER_RULES_FINAL_GATE
    return {"user_rules_gate": gate + "\n", "user_rules": rendered}


@register_step("reflection")
class ReflectionStep(BaseStep):
    """Quality reflection step for MetaAgent orchestration.

    Reads SharedContext state, runs LLM reflection, outputs quality report.
    """

    def run(self, context: Any) -> Dict[str, Any]:
        raw_cfg = self.config.raw or {}

        # Build state summary for the prompt
        summary = context.get_summary() if hasattr(context, "get_summary") else {}
        missing = getattr(context, "missing_concepts", []) or []
        quality = getattr(context, "quality_report", {}) or {}
        dk_list = getattr(context, "dk_classifications", []) or []

        # Check DK depth — count only the significant DIGITS, ignoring the
        # type prefix ("DK "/"DDC "/"RVK ") and separators. The previous
        # version measured len("DK 504.064") which always passed. - Claude Generated
        import re as _re

        def _dk_digit_len(code: Any) -> int:
            return len(_re.sub(r"\D", "", str(code or "")))

        has_deep_dk = any(_dk_digit_len(cls.get("code", "")) >= 4 for cls in dk_list)
        dk_codes_str = ", ".join(
            str(cls.get("code", "")) for cls in dk_list if cls.get("code")
        ) or "keine"

        # Final keywords + analyse give the reflection agent the actual
        # content, not just counts. - Claude Generated
        extra = getattr(context, "extra", {}) or {}
        final_kws = extra.get("final_keywords", []) or []
        final_kws_str = ", ".join(
            (kw.get("keyword") if isinstance(kw, dict) else str(kw))
            for kw in final_kws[:25]
        ) or "keine"
        analyse_txt = (getattr(context, "analyse", "") or extra.get("analyse", "") or "").strip()
        analyse_excerpt = (analyse_txt[:300] + "…") if len(analyse_txt) > 300 else (analyse_txt or "—")

        # Check keyword count
        selected = getattr(context, "selected_keywords", []) or []
        has_keywords = len(selected) >= 10

        # Determine current step from execution history
        history = getattr(context, "execution_history", []) or []
        current_step = history[-1]["step"] if history else "initialisation"

        # Catalog stats for MetaAgent visibility
        catalog_stats = getattr(context, "dk_catalog_stats", {}) or {}
        catalog_total_titles = catalog_stats.get("total_titles", 0)
        catalog_unique_notations = catalog_stats.get("total_unique_notations", 0)

        # Build prompt values
        values = {
            "working_title": getattr(context, "working_title", "") or "",
            "extracted_keywords_count": len(getattr(context, "extracted_keywords", [])),
            "gnd_entries_count": len(getattr(context, "gnd_entries", [])),
            "selected_keywords_count": len(selected),
            "dk_classifications_count": len(dk_list),
            "dk_codes": dk_codes_str,
            "final_keywords": final_kws_str,
            "keyword_chains_count": len(getattr(context, "keyword_chains", []) or []),
            # The chains themselves, not just how many. A rule that asks for the
            # output to be grouped by chain cannot be followed from a count: the
            # model then partitions the flat keyword list into plausible-looking
            # groups that are not the chains. Observed on September 7 — the block
            # looked right and was wrong. - Claude Generated
            "keyword_chains": _format_chains(getattr(context, "keyword_chains", None)),
            "core_keywords": _format_keyword_list(context, "core_keywords"),
            "form_keywords": _format_keyword_list(context, "form_keywords"),
            "analyse_excerpt": analyse_excerpt,
            "catalog_total_titles": catalog_total_titles,
            "catalog_unique_notations": catalog_unique_notations,
            "has_deep_dk": "ja" if has_deep_dk else "nein",
            "has_keywords": "ja" if has_keywords else "nein",
            # Defensive: if this step ever runs without MetaAgent composing the
            # system prompt, keep the {workflow_rules} slot from leaking. - Claude Generated
            "workflow_rules": "",
            # The operator's own rules. Empty gate ⇒ the prompt is exactly what
            # it was before the feature. - Claude Generated
            **_user_rules_values(context),
            "missing_concepts": ", ".join(missing) if missing else "keine",
            "missing_concepts_searched": ", ".join(getattr(context, "missing_concepts_searched", [])) or "keine",
            "quality_report": json.dumps(quality, ensure_ascii=False) if quality else "{}",
            "steps_completed": ", ".join(getattr(context, "step_results", {}).keys()),
            "current_step": current_step,
        }

        workflow_prompts = {}
        if hasattr(context, "_workflow_prompts"):
            workflow_prompts = context._workflow_prompts or {}

        system_prompt, user_prompt, llm_override = resolve_prompts(
            raw_cfg=raw_cfg,
            resolved_inputs=values,
            context=context,
            default_system=DEFAULT_REFLECTION_SYSTEM_PROMPT,
            default_user=DEFAULT_REFLECTION_USER_PROMPT,
            workflow_prompts=workflow_prompts,
            step_id=self.step_id,
        )

        # LLM params
        llm_cfg = raw_cfg.get("llm", {}) or {}
        params = {
            "temperature": float(llm_cfg.get("temperature", 0.1)),
            "top_p": float(llm_cfg.get("top_p", 0.9)),
            # Operator budget outranks the YAML here too — the reflection turn
            # runs on the same model and hits the same wall. - Claude Generated
            "max_tokens": int(
                getattr(context, "max_tokens_override", None)
                or llm_cfg.get("max_tokens", 32768)
            ),
            "max_iterations": 1,
            "timeout_seconds": 120,
            "provider": getattr(context, "provider", "") or "",
            "model": getattr(context, "model", "") or "",
            "think": llm_cfg.get("think", getattr(context, "think", None)),
        }
        if llm_override:
            if llm_override.get("temperature") is not None:
                params["temperature"] = float(llm_override["temperature"])
            if llm_override.get("top_p") is not None:
                params["top_p"] = float(llm_override["top_p"])

        _emit_header(self.step_id, params, self.stream_callback)
        _prompt_id = uuid.uuid4().hex[:8]
        _emit_prompts(
            self.step_id, system_prompt, user_prompt, params,
            self.stream_callback, prompt_id=_prompt_id, kind="reflection",
        )

        # Reasoning of the reflection turn goes to the same 💭 block as the
        # workers' (llm_agent_step); it runs on the same model. - Claude Generated
        def _emit_thinking(text):
            try:
                from src.core.state_bus import AlimaStateBus

                AlimaStateBus().emit_event(
                    "llm.thinking", {"text": text or "", "step_id": self.step_id}
                )
            except Exception:
                logger.debug("llm.thinking bus emit failed", exc_info=True)

        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            max_iterations=1,
            timeout_seconds=120,
            stream_callback=self.stream_callback,
            on_thinking=_emit_thinking,
        )
        _t0 = time.monotonic()
        result = loop.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=None,
            provider=params["provider"],
            model=params["model"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            think=params.get("think"),
        )
        try:
            from src.core.state_bus import AlimaStateBus

            AlimaStateBus().emit_event("llm.thinking_done", {"step_id": self.step_id})
        except Exception:
            logger.debug("llm.thinking_done bus emit failed", exc_info=True)
        _emit_prompt_done(_prompt_id, self.step_id, time.monotonic() - _t0)

        _log_response(self.step_id, result.content)
        if getattr(result, "error", None):
            # No model answer — content holds the loop's diagnostic, not a
            # verdict. Failing here lets MetaAgent take its documented
            # rule-based fallback instead of reading status=None/action=None
            # out of an unparseable warning and cycling on. - Claude Generated
            raise RuntimeError(f"Reflection LLM call failed: {result.error}")
        parsed = self._extract_json(result.content)
        # A rule may ask for an output at the end of the run. The model writes it
        # after the JSON (multi-line text needs no escaping there); an older
        # answer may still carry it as a JSON field. - Claude Generated
        from src.core.agents.json_repair import extract_tagged_block

        tagged = extract_tagged_block(result.content, "final_output")
        if tagged:
            parsed["final_output"] = tagged
        logger.info(f"ReflectionStep '{self.step_id}': status={parsed.get('status')}, action={parsed.get('action')}")

        return {
            "response": parsed,
            "response_text": result.content,
            "status": parsed.get("status", "incomplete"),
            "gaps": parsed.get("gaps", []),
            "action": parsed.get("action", "finish"),
            "reason": parsed.get("reason", ""),
        }

    @staticmethod
    def _render(template: str, values: Dict[str, Any]) -> str:
        """Replace {name} markers with stringified values."""
        if not template:
            return ""
        out = template
        for name in sorted(values.keys(), key=len, reverse=True):
            v = values[name]
            if v is None:
                v = ""
            elif isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            else:
                v = str(v)
            out = out.replace("{" + name + "}", v)
        return out

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Best-effort JSON extraction."""
        if not content:
            return {}
        import re
        # Try code block first
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Fallback: find last balanced object
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and obj:
                    return obj
            except json.JSONDecodeError:
                continue
        # Last resort: a model that wrote a formatted block into a JSON string
        # left raw newlines in it. Without this the whole verdict is lost —
        # status, action and reason with it — and the run ends on the default
        # "finish" as if nothing had happened. - Claude Generated
        from src.core.agents.json_repair import repair_json_newlines

        repaired = repair_json_newlines(content)
        if repaired != content:
            for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"):
                m = re.search(pattern, repaired, re.DOTALL)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj:
                    logger.warning(
                        "JSON answer had raw newlines inside a string — salvaged"
                    )
                    return obj
        return {}
