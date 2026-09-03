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
                # Compact one-liner instead of a ===== banner. - Claude Generated
                self.stream_callback(f"\n🔄 Zyklus {cycle}/{max_cycles}\n")
            logger.info(f"MetaAgent cycle {cycle}/{max_cycles}")

            # ── PLAN ──
            next_step = self._plan_next_step(workflow, context, quality_rules)
            if not next_step:
                # Planner wants to finish — but don't truncate the pipeline while
                # a mandatory step is still un-run. - Claude Generated
                pending = self._pending_step(workflow, context)
                if pending is None:
                    if self.stream_callback:
                        self.stream_callback("✅ MetaAgent: all goals met, finishing\n")
                    logger.info("MetaAgent: no more steps needed")
                    break
                logger.info(f"MetaAgent: planner finish overridden — running pending '{pending}'")
                if self.stream_callback:
                    self.stream_callback(
                        f"↪️ Planner wollte beenden, aber '{pending}' fehlt noch — führe es aus\n"
                    )
                next_step = pending

            # `depends_on` is NOT enforced by the executor for a planner-chosen
            # step (`run(only_step=…)` runs whatever it is handed). Gate it here
            # so a skipped prerequisite cannot starve the step's inputs. - Claude Generated
            missing_dep = self._unmet_dependency(workflow, context, next_step)
            if missing_dep:
                logger.info(
                    f"MetaAgent: '{next_step}' requires '{missing_dep}', which has "
                    f"not run — running that first"
                )
                if self.stream_callback:
                    self.stream_callback(
                        f"↪️ '{next_step}' braucht '{missing_dep}' — führe das zuerst aus\n"
                    )
                next_step = missing_dep

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
            # Reflection is YAML-defined and opt-in: it runs ONLY if the
            # workflow declares a `meta_agent.reflection:` block. Without it,
            # the loop is a pure PLAN→EXECUTE chain and the planner alone
            # decides when to finish (returns None). - Claude Generated
            reflection_cfg = (getattr(workflow, "meta_agent", {}) or {}).get("reflection")
            if not reflection_cfg:
                continue

            reflection = self._run_reflection(workflow, context, reflection_cfg)
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

            # Reflection's "finish"/"complete" is ADVISORY. Never end the
            # pipeline while a mandatory enabled step (deps satisfied) is still
            # un-run — a weak reflection model returning "complete" mid-pipeline
            # must NOT skip dk_collect/classification. Finish only when the step
            # graph is genuinely exhausted. - Claude Generated
            if status == "complete" or action == "finish":
                pending = self._pending_step(workflow, context)
                if pending is None:
                    if self.stream_callback:
                        self.stream_callback("✅ MetaAgent: quality complete, finishing\n")
                    logger.info("MetaAgent: complete and no pending steps — finishing")
                    break
                if self.stream_callback:
                    self.stream_callback(
                        f"↪️ Reflection wollte beenden, aber Pflicht-Step '{pending}' "
                        f"fehlt noch — weiter\n"
                    )
                logger.info(f"MetaAgent: reflection finish overridden — '{pending}' still pending")
                continue

            if status == "continue" or action == "continue":
                if self.stream_callback:
                    self.stream_callback("▶️ MetaAgent: continue to next step\n")
                logger.info("MetaAgent: continue to next step")
                continue

            if action == "search_missing":
                missing = getattr(context, "missing_concepts", []) or []
                already_searched = set(getattr(context, "missing_concepts_searched", []) or [])
                max_reruns = getattr(context, "max_missing_reruns", 1)
                reruns_done = getattr(context, "_missing_reruns_done", 0)

                # Missing concepts are SPECIFIC compound phrases ("Molekulare
                # Mechanismen der Cadmium-Toleranz") that GND does not index —
                # searching them verbatim finds nothing. The selection prompt is
                # told to emit missing_concepts as ready GND SEARCH TERMS, so we
                # only normalize + skip no-GND ones and search those not already
                # in the pool. - Claude Generated
                existing_lower = {
                    str(k.get("keyword") if isinstance(k, dict) else k).lower()
                    for k in (getattr(context, "extracted_keywords", []) or [])
                }
                search_terms: List[str] = []
                seen = set(existing_lower)
                for concept in missing:
                    if concept in already_searched:
                        continue
                    already_searched.add(concept)  # mark handled either way
                    if self._concept_has_no_gnd(concept):
                        continue
                    term = self._normalize_missing_term(concept)
                    if term and term.lower() not in seen:
                        seen.add(term.lower())
                        search_terms.append(term)
                context.missing_concepts_searched = list(already_searched)

                if search_terms and reruns_done < max_reruns:
                    if self.stream_callback:
                        self.stream_callback(
                            f"🔁 MetaAgent: GND-Suche für {len(search_terms)} fehlende "
                            f"Suchbegriffe: {search_terms}\n"
                        )
                    delta = self._search_missing_concepts(workflow, context, search_terms)
                    context._missing_reruns_done = reruns_done + 1
                    # Filter ONLY the newly-found entries (not the whole pool),
                    # union into selected_keywords, then rebuild the chains via a
                    # cheap `selection` re-run (no full re-chunk). - Claude Generated
                    self._select_delta(workflow, context, delta)
                    context._force_next_step = "selection"
                    logger.info(
                        f"MetaAgent: missing-concept search #{reruns_done + 1}/{max_reruns} "
                        f"+{len(delta)} new entries: {search_terms}"
                    )
                else:
                    if self.stream_callback:
                        self.stream_callback(
                            "▶️ MetaAgent: keine neuen Suchbegriffe aus fehlenden "
                            "Konzepten (alle bereits im Pool oder ohne GND) — weiter\n"
                        )
                    logger.info("MetaAgent: search_missing — no new search terms, continuing")
                continue

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
        # Show the actual DK codes, not just the count — otherwise the planner
        # cannot tell whether classification already produced usable output and
        # keeps re-running it. - Claude Generated
        dk_list = getattr(context, "dk_classifications", []) or []
        dk_codes = ", ".join(str(c.get("code", "")) for c in dk_list if c.get("code"))
        # Only advertise missing concepts the planner can still act on. Once they
        # have been searched (missing_concepts_searched) or the re-search budget
        # (max_missing_reruns) is spent, telling the planner they are still
        # "missing" makes it loop selection→search→selection forever until
        # max_cycles. Report outstanding vs already-searched explicitly. - Claude Generated
        missing_all = getattr(context, "missing_concepts", []) or []
        searched = set(getattr(context, "missing_concepts_searched", []) or [])
        outstanding = [c for c in missing_all if c not in searched]
        budget_left = getattr(context, "_missing_reruns_done", 0) < getattr(
            context, "max_missing_reruns", 1
        )
        if outstanding and budget_left:
            missing_line = f"- Fehlende Konzepte (offen, noch nicht gesucht): {outstanding}\n"
        elif missing_all:
            missing_line = (
                "- Fehlende Konzepte: alle bereits gesucht — NICHT erneut 'search', "
                "mit dk_collect/classification fortfahren\n"
            )
        else:
            missing_line = "- Fehlende Konzepte: keine\n"
        state_text = (
            f"- Abstract: {(getattr(context, 'abstract', '') or '')[:80]}...\n"
            f"- Extrahierte Keywords: {len(getattr(context, 'extracted_keywords', []))}\n"
            f"- GND-Einträge: {len(getattr(context, 'gnd_entries', []))}\n"
            f"- Ausgewählte Keywords: {len(getattr(context, 'selected_keywords', []))}\n"
            f"- DK-Klassifikationen: {len(dk_list)}\n"
            f"- DK-Codes (Ist): {dk_codes or 'keine'}\n"
            f"{catalog_summary}"
            f"{missing_line}"
        )

        # Resolve planning prompts from workflow YAML (meta_agent block or prompts block)
        meta = workflow.meta_agent or {}
        wp = workflow.prompts or {}

        # Generic, DOMAIN-NEUTRAL base. Workflow-specific routing rules (which
        # step needs GND / DK / which phase order) come from the YAML
        # `meta_agent.planning.rules` and are injected at `{workflow_rules}`. So
        # a non-keywording pipeline isn't forced through GND/DK assumptions. - Claude Generated
        default_system = (
            "Du bist der ALIMA MetaAgent Planer. Du entscheidest, welcher Schritt "
            "als nächstes ausgeführt wird, basierend auf dem aktuellen Zustand und dem "
            "Ausführungsverlauf.\n\n"
            "Allgemeine Regeln:\n"
            "1. Wähle EXAKT eine step_id aus der Liste der verfügbaren Steps.\n"
            "2. Respektiere Reihenfolge und Abhängigkeiten: führe einen Schritt erst aus, "
            "wenn seine Vorbedingungen erfüllt sind.\n"
            "3. Wiederhole einen bereits gelaufenen Schritt NUR bei einem konkreten, "
            "benannten Mangel — niemals mit 'könnte noch optimiert werden'.\n"
            "4. Wenn alle Steps gelaufen sind und ihre erwarteten Ergebnisse vorliegen, "
            "wähle 'finish'.\n"
            "{workflow_rules}"
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

        # Nested `planning:` block (preferred) with flat / top-level fallbacks.
        planning_cfg = meta.get("planning", {}) or {}
        planning_rules = planning_cfg.get("rules") or ""
        override_system = (
            planning_cfg.get("system_prompt")
            or meta.get("planning_system_prompt")
            or wp.get("planning", {}).get("system")
        )
        override_user = (
            planning_cfg.get("user_prompt")
            or meta.get("planning_user_prompt")
            or wp.get("planning", {}).get("prompt")
        )
        system_prompt = self._inject_rules(override_system or default_system, planning_rules)
        user_prompt = override_user or default_user

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

        # All enabled steps have run → finish. Reruns are driven by reflection
        # (via `_force_next_step` / actions), NOT by the rule-based planner —
        # returning "classification" here looped forever whenever reflection
        # was absent or the LLM planner had failed. - Claude Generated
        return None

    @staticmethod
    def _steps_run(context: Any) -> set:
        """Ids of the steps already executed in this run. - Claude Generated"""
        if hasattr(context, "execution_history"):
            return {h["step"] for h in context.execution_history}
        return set(getattr(context, "step_results", {}).keys())

    @classmethod
    def _unmet_dependency(
        cls, workflow: WorkflowDef, context: Any, step_id: str
    ) -> Optional[str]:
        """Deepest un-run prerequisite of `step_id`, or None if all have run.

        The LLM planner names the next step freely and the executor does not
        check `depends_on`. Unchecked, a planner can run `classification` before
        `selection`; `${extra.final_keywords}` then resolves to empty, the
        classification prompt carries no keywords for `rvk_lookup` (no RVK at
        all), and `dk_collect` builds its catalog pool from the coarse
        `selection_chunks` output instead of the curated final keywords.

        Depth-first, so the returned step is itself runnable. Disabled or
        unknown deps can never run and count as satisfied. - Claude Generated
        """
        steps_by_id = {s.id: s for s in workflow.steps}
        steps_run = cls._steps_run(context)

        def walk(sid: str, seen: set) -> Optional[str]:
            step = steps_by_id.get(sid)
            if step is None or sid in seen:
                return None
            seen.add(sid)
            for dep in getattr(step, "depends_on", []) or []:
                dep_step = steps_by_id.get(dep)
                # 'reflection' runs outside execution_history, disabled/unknown
                # deps can never run — all three count as satisfied, otherwise
                # the redirect would never resolve. - Claude Generated
                if (
                    dep_step is None
                    or not dep_step.enabled
                    or dep == "reflection"
                    or dep in steps_run
                ):
                    continue
                return walk(dep, seen) or dep
            return None

        target = walk(step_id, set())
        # A dependency cycle resolves back to the step itself — unresolvable, so
        # leave the planner's choice alone rather than redirect to a no-op.
        return None if target == step_id else target

    @classmethod
    def _pending_step(cls, workflow: WorkflowDef, context: Any) -> Optional[str]:
        """First enabled, not-yet-run step whose dependencies are satisfied.

        Used to veto a premature finish: the planner/reflection must not end the
        pipeline while a mandatory step (e.g. dk_collect, classification) is
        still pending. Returns None only when the step graph is exhausted (or the
        remaining steps' deps can never be met). - Claude Generated
        """
        steps_run = cls._steps_run(context)
        for s in workflow.steps:
            if not s.enabled or s.id == "reflection" or s.id in steps_run:
                continue
            deps = getattr(s, "depends_on", []) or []
            if all(d in steps_run for d in deps):
                return s.id
        return None

    @staticmethod
    def _concept_has_no_gnd(concept: str) -> bool:
        """True if the LLM flagged the concept as having no GND entry."""
        low = (concept or "").lower()
        return (
            "kein gnd" in low or "keine gnd" in low
            or "no gnd" in low or "nicht in der gnd" in low
        )

    @staticmethod
    def _normalize_missing_term(concept: str) -> Optional[str]:
        """Light cleanup of a missing-concept search term.

        The selection prompt is instructed to emit missing_concepts as ready
        GND SEARCH TERMS (decomposed, analogous to extraction) — so the code no
        longer atomizes them (which produced junk like 'Molekulare'). We only
        strip parenthetical annotations and chemical charge notation, then
        collapse whitespace. - Claude Generated
        """
        import re

        if not concept or not isinstance(concept, str):
            return None
        s = re.sub(r"\([^)]*\)", " ", concept)
        s = s.replace("²⁺", "").replace("²⁻", "").replace("³⁺", "")
        s = re.sub(r"\s+", " ", s).strip(" .,:;\"'`–—\t")
        return s if len(s) >= 3 else None

    def _search_missing_concepts(
        self,
        workflow: WorkflowDef,
        context: Any,
        terms: List[str],
    ) -> List[Dict[str, Any]]:
        """Targeted GND search for clean missing-concept terms.

        Calls ``gnd_batch_search`` directly so results MERGE into the existing
        ``context.gnd_entries`` pool (the function's append path) — deduplicated
        by title against the known pool. Going through the ``search`` step would
        re-apply its ``gnd_entries: result.entries`` mapping and OVERWRITE the
        full pool. Returns the DELTA (entries new to the pool) so only those need
        re-selecting, not the whole pool. - Claude Generated
        """
        from src.core.agents.deterministic_functions import gnd_batch_search

        search_cfg: Dict[str, Any] = {}
        for s in workflow.steps:
            raw = getattr(s, "raw", {}) or {}
            if raw.get("function") == "gnd_batch_search":
                search_cfg = raw.get("config", {}) or {}
                break
        before = {(e.get("title") or "").lower() for e in (getattr(context, "gnd_entries", []) or [])}
        try:
            gnd_batch_search(
                keywords=terms,
                tool_registry=self.tool_registry,
                context=context,
                stream_callback=self.stream_callback,
                config=search_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MetaAgent: targeted missing-concept search failed: {exc}")
            if self.stream_callback:
                self.stream_callback(f"⚠️ Gezielte Missing-Concept-Suche fehlgeschlagen: {exc}\n")
            return []
        delta = [
            e for e in (getattr(context, "gnd_entries", []) or [])
            if (e.get("title") or "").lower() not in before
        ]
        return delta

    def _select_delta(
        self,
        workflow: WorkflowDef,
        context: Any,
        delta_entries: List[Dict[str, Any]],
    ) -> None:
        """Run the relevance filter over ONLY the newly-found entries.

        Re-running ``selection_chunks`` over the full enlarged pool would re-chunk
        and re-LLM everything already processed. Instead we run the selection step
        against just the delta (1 chunk) and UNION the result into the existing
        ``selected_keywords`` — the known pool is never re-chunked. - Claude Generated
        """
        sc = next(
            (s for s in workflow.steps if s.id == "selection_chunks" and s.enabled),
            None,
        )
        if sc is None or not delta_entries:
            return
        saved_pool = context.gnd_entries
        saved_sel = list(getattr(context, "selected_keywords", []) or [])
        saved_sr = (getattr(context, "step_results", {}) or {}).get("selection_chunks")
        delta_sel: List[Any] = []
        try:
            context.gnd_entries = delta_entries  # selection sees only the new entries
            step = get_step_class(sc.type)(
                config=sc,
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                stream_callback=self.stream_callback,
            )
            step.execute(context)  # replaces context.selected_keywords with the delta selection
            delta_sel = list(getattr(context, "selected_keywords", []) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MetaAgent: delta selection failed: {exc}")
        finally:
            context.gnd_entries = saved_pool
            if saved_sr is not None:
                context.step_results["selection_chunks"] = saved_sr

        # Union delta selection into the original selection (dedup by keyword).
        seen: set = set()
        merged: List[Any] = []
        for kw in saved_sel + delta_sel:
            key = str(kw.get("keyword") if isinstance(kw, dict) else kw).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(kw)
        context.selected_keywords = merged
        if self.stream_callback and len(merged) > len(saved_sel):
            self.stream_callback(
                f"   ➕ {len(merged) - len(saved_sel)} neue Schlagworte aus dem Zuwachs "
                f"selektiert (Pool nicht neu gechunkt)\n"
            )

    @staticmethod
    def _inject_rules(base: str, rules: str) -> str:
        """Augment a generic MetaAgent prompt with workflow-specific rules.

        Fills the ``{workflow_rules}`` slot if present; otherwise appends the
        rules. Keeps the generic (ALIMA-branded) base domain-neutral while each
        workflow adds its own routing/quality rules via YAML. - Claude Generated
        """
        base = base or ""
        block = (rules or "").strip()
        if "{workflow_rules}" in base:
            return base.replace("{workflow_rules}", ("\n" + block + "\n") if block else "")
        if block:
            return base.rstrip() + "\n\n" + block + "\n"
        return base

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
        reflection_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run reflection step to assess quality.

        ``reflection_cfg`` is the workflow's ``meta_agent.reflection:`` block —
        a self-contained definition (prompts + model/provider + LLM params).
        Reflection only runs when this block exists (see ``run``); omitted
        sub-fields fall back to the ReflectionStep code defaults so a bare
        ``reflection: {}`` block is enough to switch the quality gate on.

        Returns dict with status, gaps, action, reason. - Claude Generated
        """
        rc = reflection_cfg or {}

        # LLM params from the block (with sensible defaults).
        llm_block: Dict[str, Any] = {
            "temperature": float(rc.get("temperature", 0.1)),
            "top_p": float(rc.get("top_p", 0.9)),
            "max_tokens": int(rc.get("max_tokens", 32768)),
            "max_iterations": 1,
        }
        # Block model/provider win; else the constructor-level reflection_*.
        model = rc.get("model") or self.reflection_model
        provider = rc.get("provider") or self.reflection_provider
        if model:
            llm_block["model"] = model
        if provider:
            llm_block["provider"] = provider

        step_cfg: Dict[str, Any] = {
            "id": "reflection",
            "type": "reflection",
            "enabled": True,
            "llm": llm_block,
        }

        # Compose the system prompt: generic (ALIMA-branded) base + the
        # workflow's domain rules (`reflection.rules`). A full `system_prompt`
        # override replaces the base; `rules` augment whichever base is used.
        # The user prompt (state dump) stays the comprehensive code default
        # unless explicitly overridden. - Claude Generated
        from src.core.agents.steps.reflection_step import DEFAULT_REFLECTION_SYSTEM_PROMPT
        base_system = rc.get("system_prompt") or DEFAULT_REFLECTION_SYSTEM_PROMPT
        step_cfg["system_prompt"] = self._inject_rules(base_system, rc.get("rules") or "")
        if rc.get("user_prompt"):
            step_cfg["user_prompt"] = rc["user_prompt"]

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
