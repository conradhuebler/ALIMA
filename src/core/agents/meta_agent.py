"""MetaAgent — orchestration layer for ALIMA Pipeline v4 - Claude Generated.

Wraps WorkflowExecutor with a planning/reflection loop:
  PLAN → EXECUTE → OBSERVE → REFLECT → (repeat or finish)

MetaAgent decides which step to run next based on current SharedContext state.
ReflectionStep produces quality reports that drive replanning.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from src.core.agents.registry import get_step_class
from src.core.agents.steps.base_step import StepResult
from src.core.agents.workflow_executor import ExecutionReport, WorkflowExecutor
from src.core.agents.workflow_loader import WorkflowDef

logger = logging.getLogger(__name__)


class MetaAgent:
    """Orchestrates workflow steps with dynamic replanning.

    Uses WorkflowExecutor internally for actual step execution.
    Maintains execution_history on SharedContext for cycle memory.
    """

    def __init__(
        self,
        llm_service: Any = None,
        tool_registry: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        context_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_cycles: int = 10,
        reflection_model: str = "",
        reflection_provider: str = "",
    ) -> None:
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.stream_callback = stream_callback
        self.context_callback = context_callback
        self.max_cycles = max_cycles
        self.reflection_model = reflection_model
        self.reflection_provider = reflection_provider
        self.executor = WorkflowExecutor(
            llm_service=llm_service,
            tool_registry=tool_registry,
            stream_callback=stream_callback,
            context_callback=context_callback,
        )

    def run(
        self,
        workflow: WorkflowDef,
        context: Any,
        meta_config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        """Run workflow with MetaAgent planning loop.

        Args:
            workflow: Parsed workflow definition.
            context: SharedContext instance (mutated in-place).
            meta_config: Optional dict with max_cycles, quality_rules, etc.

        Returns:
            ExecutionReport from final cycle.
        """
        cfg = meta_config or {}
        max_cycles = int(cfg.get("max_cycles", self.max_cycles))
        quality_rules = cfg.get("quality_rules", [])
        cycle = 0
        last_report: Optional[ExecutionReport] = None

        # Seed execution_history if not present
        if not hasattr(context, "execution_history"):
            context.execution_history = []

        while cycle < max_cycles:
            cycle += 1
            if self.stream_callback:
                self.stream_callback(
                    f"\n{'='*60}\n"
                    f"🤖 MetaAgent Cycle {cycle}/{max_cycles}\n"
                    f"{'='*60}\n"
                )
            logger.info(f"MetaAgent cycle {cycle}/{max_cycles}")

            # ── PLAN ──
            next_step = self._plan_next_step(workflow, context, quality_rules)
            if not next_step:
                if self.stream_callback:
                    self.stream_callback("✅ MetaAgent: all goals met, finishing\n")
                logger.info("MetaAgent: no more steps needed")
                break

            if self.stream_callback:
                self.stream_callback(f"📋 MetaAgent plan: run '{next_step}'\n")

            # ── EXECUTE ──
            report = self.executor.run(
                workflow=workflow,
                context=context,
                only_step=next_step,
                stop_on_error=True,
            )
            last_report = report

            # Record cycle in history
            context.execution_history.append({
                "cycle": cycle,
                "step": next_step,
                "success": report.success,
                "duration": report.duration_seconds,
                "error": report.error,
            })

            if not report.success:
                logger.error(f"MetaAgent: step '{next_step}' failed: {report.error}")
                if self.stream_callback:
                    self.stream_callback(f"❌ Step '{next_step}' failed: {report.error}\n")
                break

            # ── OBSERVE ──
            # Snapshot state for reflection
            summary = context.get_summary() if hasattr(context, "get_summary") else {}
            logger.debug(f"MetaAgent observe: {summary}")

            # ── REFLECT ──
            reflection = self._run_reflection(workflow, context)
            context.quality_report = reflection

            if self.stream_callback:
                status = reflection.get("status", "incomplete")
                action = reflection.get("action", "finish")
                gaps = reflection.get("gaps", [])
                self.stream_callback(
                    f"🔍 Reflection: status={status}, action={action}, gaps={gaps}\n"
                )

            status = reflection.get("status", "incomplete")
            action = reflection.get("action", "finish")

            if status == "complete":
                if self.stream_callback:
                    self.stream_callback("✅ MetaAgent: quality complete, finishing\n")
                logger.info("MetaAgent: quality complete")
                break

            if status == "continue" or action == "continue":
                if self.stream_callback:
                    self.stream_callback("▶️ MetaAgent: continue to next step\n")
                logger.info("MetaAgent: continue to next step")
                continue

            # ── REPLAN ──
            if action == "finish":
                if self.stream_callback:
                    self.stream_callback("✅ MetaAgent: no gaps, finishing\n")
                logger.info("MetaAgent: no gaps, finishing")
                break

            if action == "search_missing":
                missing = getattr(context, "missing_concepts", []) or []
                already_searched = set(getattr(context, "missing_concepts_searched", []) or [])
                max_reruns = getattr(context, "max_missing_reruns", 1)
                reruns_done = getattr(context, "_missing_reruns_done", 0)
                new_concepts = [c for c in missing if c not in already_searched]
                if new_concepts and reruns_done < max_reruns:
                    if self.stream_callback:
                        self.stream_callback(f"🔁 MetaAgent: adding missing concepts for search: {new_concepts}\n")
                    current = getattr(context, "extracted_keywords", []) or []
                    for concept in new_concepts:
                        if concept not in current:
                            current.append(concept)
                    context.extracted_keywords = current
                    already_searched.update(new_concepts)
                    context.missing_concepts_searched = list(already_searched)
                    context._missing_reruns_done = reruns_done + 1
                    context._force_next_step = "search"
                    logger.info(f"MetaAgent: search_missing queued #{reruns_done + 1}/{max_reruns}: {new_concepts}")
                else:
                    logger.info("MetaAgent: search_missing but no new concepts or max reruns reached")

            # Action like "rerun_search" → next cycle will plan search again
            logger.info(f"MetaAgent: will rerun due to action={action}")

        else:
            logger.warning(f"MetaAgent hit max_cycles ({max_cycles})")
            if self.stream_callback:
                self.stream_callback(f"⚠️ MetaAgent hit max_cycles ({max_cycles})\n")

        # Build final report
        return last_report or ExecutionReport(
            workflow_name=workflow.name,
            success=True,
            duration_seconds=0.0,
            step_results=[],
            error=None,
        )

    def _plan_next_step(
        self,
        workflow: WorkflowDef,
        context: Any,
        quality_rules: List[str],
    ) -> Optional[str]:
        """Decide which step to run next.

        Tries LLM-based planner first (if llm_service available), falls back
        to rule-based planner on error or missing LLM.
        """
        # Check forced step from previous reflection cycle (e.g. search_missing)
        forced = getattr(context, "_force_next_step", None)
        if forced:
            context._force_next_step = None
            enabled_ids = {s.id for s in workflow.steps if s.enabled}
            if forced in enabled_ids:
                if self.stream_callback:
                    self.stream_callback(f"📋 MetaAgent forced plan: run '{forced}'\n")
                logger.info(f"MetaAgent: forced next step='{forced}'")
                return forced
            else:
                logger.warning(f"MetaAgent: forced step '{forced}' not enabled")

        # Try LLM-based planning first
        if self.llm_service is not None:
            try:
                return self._plan_with_llm(workflow, context, quality_rules)
            except Exception as exc:
                logger.warning(f"LLM planner failed: {exc} — using rule-based fallback")

        return self._plan_rule_based(workflow, context)

    def _plan_with_llm(
        self,
        workflow: WorkflowDef,
        context: Any,
        quality_rules: List[str],
    ) -> Optional[str]:
        """Ask LLM which step to run next.

        Returns step id or None if pipeline should finish.
        """
        from src.core.agent_loop import AgentLoop

        # Build available steps list
        enabled_steps = [s for s in workflow.steps if s.enabled]
        step_list = []
        for s in enabled_steps:
            step_list.append(f"- {s.id} ({s.type}): {s.description or ''}")
        steps_text = "\n".join(step_list)

        # Build execution history
        history = getattr(context, "execution_history", []) or []
        history_text = "\n".join(
            f"  Cycle {h['cycle']}: {h['step']} {'✓' if h['success'] else '✗'}"
            for h in history[-6:]
        ) or "  Noch keine Steps ausgeführt"

        # Build current state
        catalog_stats = getattr(context, "dk_catalog_stats", {}) or {}
        catalog_summary = ""
        if catalog_stats.get("total_titles", 0) > 0:
            catalog_summary = (
                f"- Katalog-Daten: {catalog_stats.get('total_titles', 0)} Titel, "
                f"{catalog_stats.get('total_unique_notations', 0)} Notationen\n"
            )
        state_text = (
            f"- Abstract: {(getattr(context, 'abstract', '') or '')[:80]}...\n"
            f"- Extrahierte Keywords: {len(getattr(context, 'extracted_keywords', []))}\n"
            f"- GND-Einträge: {len(getattr(context, 'gnd_entries', []))}\n"
            f"- Ausgewählte Keywords: {len(getattr(context, 'selected_keywords', []))}\n"
            f"- DK-Klassifikationen: {len(getattr(context, 'dk_classifications', []))}\n"
            f"{catalog_summary}"
            f"- Fehlende Konzepte: {getattr(context, 'missing_concepts', []) or 'keine'}\n"
        )

        # Resolve planning prompts from workflow YAML (meta_agent block or prompts block)
        meta = workflow.meta_agent or {}
        wp = workflow.prompts or {}

        default_system = (
            "Du bist ALIMA MetaAgent Planer. Du entscheidest, welcher Pipeline-Schritt "
            "als nächstes ausgeführt wird, basierend auf dem aktuellen Zustand.\n\n"
            "Regeln:\n"
            "1. Wähle EXAKT eine step_id aus der Liste der verfügbaren Steps.\n"
            "2. Wenn alle notwendigen Daten vorhanden sind und keine Lücken bestehen, wähle 'finish'.\n"
            "3. Wenn fehlende Konzepte gemeldet wurden und search schon gelaufen ist, wähle 'search' (neu).\n"
            "4. Wenn noch keine Keywords extrahiert wurden, starte mit 'initialisation' oder 'extraction'.\n"
            "5. Wenn Keywords da aber keine GND-Suche, wähle 'search'.\n"
            "6. Wenn GND da aber keine Auswahl, wähle 'selection_chunks' oder 'selection'.\n"
            "7. Wenn Auswahl da aber keine DK, wähle 'dk_collect' oder 'classification'.\n"
            "8. Wenn dk_collect bereits gelaufen ist UND Katalogdaten stark vorhanden sind "
            "(z.B. >20 Notationen, >30 Titel), wähle 'classification' als nächstes.\n"
            "\nAusgabe als valides JSON:\n"
            '{\n'
            '  "next_step": "step_id oder finish",\n'
            '  "reason": "..."\n'
            '}\n'
            "Keine Erläuterungen außerhalb des JSON."
        )
        default_user = (
            "Verfügbare Steps:\n{steps_text}\n\n"
            "Ausführungsverlauf:\n{history_text}\n\n"
            "Aktueller Zustand:\n{state_text}\n\n"
            "Entscheide den nächsten Schritt."
        )

        system_prompt = meta.get("planning_system_prompt") or wp.get("planning", {}).get("system") or default_system
        user_prompt = meta.get("planning_user_prompt") or wp.get("planning", {}).get("prompt") or default_user

        # Substitute dynamic placeholders into custom user prompts
        if "{steps_text}" in user_prompt:
            user_prompt = user_prompt.replace("{steps_text}", steps_text)
        if "{history_text}" in user_prompt:
            user_prompt = user_prompt.replace("{history_text}", history_text)
        if "{state_text}" in user_prompt:
            user_prompt = user_prompt.replace("{state_text}", state_text)

        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            max_iterations=1,
            timeout_seconds=60,
            stream_callback=self.stream_callback,
        )
        result = loop.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=None,
            provider=getattr(context, "provider", "") or "",
            model=getattr(context, "model", "") or "",
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
        )

        # Parse JSON response
        parsed = self._extract_json(result.content)
        next_step = parsed.get("next_step", "")

        if next_step == "finish":
            return None

        # Validate step exists
        valid_ids = {s.id for s in enabled_steps}
        if next_step not in valid_ids:
            logger.warning(f"LLM planner returned unknown step '{next_step}' — falling back")
            return self._plan_rule_based(workflow, context)

        logger.info(f"LLM planner chose next_step='{next_step}' reason='{parsed.get('reason', '')}'")
        return next_step

    def _plan_rule_based(
        self,
        workflow: WorkflowDef,
        context: Any,
    ) -> Optional[str]:
        """Rule-based fallback planner.

        Returns step id or None if all goals met.
        """
        # Determine which steps have been run
        steps_run = set()
        if hasattr(context, "execution_history"):
            steps_run = {h["step"] for h in context.execution_history}
        else:
            steps_run = set(context.step_results.keys())

        # Find enabled steps in workflow order
        enabled_steps = [s for s in workflow.steps if s.enabled]

        # Rule: run steps in workflow order until all complete
        for step_cfg in enabled_steps:
            step_id = step_cfg.id
            if step_id == "reflection":
                continue
            if step_id not in steps_run:
                return step_id

        # All steps run but quality incomplete → rerun classification
        if "classification" in steps_run:
            return "classification"

        return None

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Best-effort JSON extraction."""
        if not content:
            return {}
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return {}


    def _run_reflection(
        self,
        workflow: WorkflowDef,
        context: Any,
    ) -> Dict[str, Any]:
        """Run reflection step to assess quality.

        Returns dict with status, gaps, action, reason.
        """
        # Build reflection step config
        step_cfg = {
            "id": "reflection",
            "type": "reflection",
            "enabled": True,
            "llm": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2048,
                "max_iterations": 1,
            },
        }

        # Use reflection model/provider if configured
        if self.reflection_model:
            step_cfg["llm"]["model"] = self.reflection_model
        if self.reflection_provider:
            step_cfg["llm"]["provider"] = self.reflection_provider

        from src.core.agents.workflow_loader import StepConfig
        cfg = StepConfig(
            id="reflection",
            type="reflection",
            raw=step_cfg,
        )

        try:
            step_cls = get_step_class("reflection")
        except KeyError:
            logger.error("ReflectionStep not registered")
            return {
                "status": "complete",
                "gaps": [],
                "action": "finish",
                "reason": "ReflectionStep not available",
            }

        step = step_cls(
            config=cfg,
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            stream_callback=self.stream_callback,
        )

        result = step.execute(context)
        if not result.success:
            logger.warning(f"Reflection step failed: {result.error}")
            return {
                "status": "complete",
                "gaps": [],
                "action": "finish",
                "reason": f"Reflection failed: {result.error}",
            }

        return result.data.get("response", {})
