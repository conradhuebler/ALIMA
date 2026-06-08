"""Pipeline-Orchestration chat-tools — P-ζ. Claude Generated.

Tools that *drive* the pipeline from the chat agent:

- ``run_pipeline(input_source, mode, workflow, ...)`` — full pipeline run.
- ``rerun_step(step_id, params, ...)`` — re-execute a single step against
  the current SharedContext.

Both reuse the ``_MutationToolBase`` pattern (audit row + ProposalGateway
confirmation + ``autonomous_pipeline`` bypass) because they are
state-destructive operations.

Status streaming back to the chat panel piggybacks on the existing
``AlimaStateBus`` channel (new event ``state.pipeline_step``).

Cancel is plumbed via the running ``ChatAgentWorker``'s ``_stop_event``:
the tool reads it off ``QThread.currentThread()`` and forwards it as
``pm.set_interrupt_flag(...)`` for the duration of the call.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from src.ui.chat_tools.base import BaseChatTool
from src.ui.chat_tools.mutations import _MutationToolBase

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_should_stop() -> Callable[[], bool]:
    """Read ``_stop_event.is_set`` off the calling Qt thread, if any.

    Pipeline-tools execute inside ``ChatAgentWorker`` (a ``QThread`` that
    sets ``self._stop_event`` in ``__init__``). We grab the current thread
    and read that attribute — no new wiring needed and the code stays
    headless-friendly (returns a no-op stopper when the attribute is
    missing).
    """
    try:
        from PyQt6.QtCore import QThread
        t = QThread.currentThread()
        ev = getattr(t, "_stop_event", None)
        if ev is not None and hasattr(ev, "is_set"):
            return ev.is_set
    except Exception:
        pass
    # Headless / non-Qt fallback: look at the current threading.Thread.
    cur = threading.current_thread()
    ev = getattr(cur, "_stop_event", None)
    if ev is not None and hasattr(ev, "is_set"):
        return ev.is_set
    return lambda: False


class _StatusForwarder:
    """Context manager that pipes pipeline step callbacks to AlimaStateBus.

    Saves the current ``pm.step_started_callback`` / ``step_completed_callback``
    on entry and restores them on exit. While active, every step transition is
    forwarded both to the original callback (if any) AND to a
    ``state.pipeline_step`` event on ``AlimaStateBus`` so the chat panel can
    render a 🔄/✅ marker.
    """

    def __init__(self, pm: Any, tool_name: str) -> None:
        self.pm = pm
        self.tool_name = tool_name
        self._old_started = None
        self._old_completed = None
        self._installed = False

    def __enter__(self) -> "_StatusForwarder":
        pm = self.pm
        if pm is None:
            return self
        try:
            from src.core.state_bus import AlimaStateBus
            bus = AlimaStateBus()
        except Exception:
            return self

        self._old_started = getattr(pm, "step_started_callback", None)
        self._old_completed = getattr(pm, "step_completed_callback", None)
        tool_name = self.tool_name

        def on_started(step):
            try:
                bus.emit_event("state.pipeline_step", {
                    "tool": tool_name,
                    "step_id": getattr(step, "step_id", ""),
                    "name": getattr(step, "name", ""),
                    "status": "running",
                })
            except Exception:
                logger.debug("state.pipeline_step emit failed", exc_info=True)
            if self._old_started:
                try:
                    self._old_started(step)
                except Exception:
                    logger.exception("forwarded step_started_callback raised")

        def on_completed(step):
            try:
                bus.emit_event("state.pipeline_step", {
                    "tool": tool_name,
                    "step_id": getattr(step, "step_id", ""),
                    "name": getattr(step, "name", ""),
                    "status": "completed",
                })
            except Exception:
                logger.debug("state.pipeline_step emit failed", exc_info=True)
            if self._old_completed:
                try:
                    self._old_completed(step)
                except Exception:
                    logger.exception("forwarded step_completed_callback raised")

        pm.step_started_callback = on_started
        pm.step_completed_callback = on_completed
        self._installed = True
        return self

    def __exit__(self, *_exc) -> None:
        if not self._installed:
            return
        try:
            self.pm.step_started_callback = self._old_started
            self.pm.step_completed_callback = self._old_completed
        except Exception:
            logger.exception("_StatusForwarder restore failed")


class _InterruptInstaller:
    """Context manager that registers a ``should_stop`` with ``pm.set_interrupt_flag``.

    The pipeline's ``_check_interruption`` consults the registered callable at
    every step boundary and raises ``InterruptedError`` when it returns True.
    Caller wraps ``pm.start_pipeline(...)`` / ``execute_single_step(...)``
    in this manager so chat-side cancel propagates without permanent
    interrupt-flag leakage.
    """

    def __init__(self, pm: Any, should_stop: Callable[[], bool]) -> None:
        self.pm = pm
        self.should_stop = should_stop
        self._old_check = None
        self._installed = False

    def __enter__(self) -> "_InterruptInstaller":
        pm = self.pm
        if pm is None:
            return self
        # Snapshot under the internal lock so we don't race with another caller.
        try:
            lock = getattr(pm, "_interrupt_lock", None) or threading.Lock()
            with lock:
                self._old_check = getattr(pm, "_interrupt_check_func", None)
            pm.set_interrupt_flag(lock, self.should_stop)
            self._installed = True
        except Exception:
            logger.exception("set_interrupt_flag install failed")
        return self

    def __exit__(self, *_exc) -> None:
        if not self._installed:
            return
        try:
            lock = getattr(self.pm, "_interrupt_lock", None) or threading.Lock()
            with lock:
                self.pm._interrupt_check_func = self._old_check
                self.pm._is_interrupted = False
        except Exception:
            logger.exception("_InterruptInstaller restore failed")


def _summarise_state(pm: Any) -> Dict[str, Any]:
    """Build a tiny dict describing the post-run state for the LLM."""
    ctx = getattr(pm, "last_shared_context", None)
    state = getattr(pm, "current_analysis_state", None)
    if ctx is not None:
        return {
            "keyword_count": len(getattr(ctx, "extracted_keywords", []) or []),
            "selected_keyword_count": len(getattr(ctx, "selected_keywords", []) or []),
            "dk_count": len(getattr(ctx, "dk_classifications", []) or []),
            "working_title": getattr(ctx, "working_title", "") or "",
            "abstract_length": len(getattr(ctx, "abstract", "") or ""),
        }
    if state is not None:
        return {
            "keyword_count": len(getattr(state, "initial_keywords", []) or []),
            "dk_count": len(getattr(state, "dk_classifications", []) or []),
            "working_title": getattr(state, "working_title", "") or "",
            "abstract_length": len(getattr(state, "original_abstract", "") or ""),
        }
    return {}


# ----------------------------------------------------------------------
# run_pipeline
# ----------------------------------------------------------------------


class RunPipelineTool(_MutationToolBase):
    name = "run_pipeline"
    operation = "pipeline_run"
    description = (
        "Run a full ALIMA pipeline against the given input. DESTRUCTIVE: "
        "overwrites the current pipeline state and SharedContext. Requires "
        "user confirmation unless autonomous_pipeline is True. Use 'classic' "
        "mode for the 5-step linear pipeline, 'agentic' for a YAML workflow."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "input_source": {
                "type": "string",
                "description": "Abstract text, DOI, file path, or URL.",
            },
            "input_type": {
                "type": "string",
                "enum": ["text", "doi", "pdf", "img", "url"],
                "description": "How to interpret input_source.",
                "default": "text",
            },
            "mode": {
                "type": "string",
                "enum": ["classic", "agentic"],
                "description": "Pipeline mode.",
                "default": "classic",
            },
            "workflow": {
                "type": "string",
                "description": "Workflow YAML name (agentic mode only, e.g. 'alima_classic').",
            },
            "reason": {
                "type": "string",
                "description": "Short justification for the user.",
            },
        },
        "required": ["input_source", "reason"],
    }

    def available_for(self, session: Any) -> bool:
        return self.pipeline_manager is not None

    def execute(self, session: Any, **kwargs: Any) -> str:
        pm = self.pipeline_manager
        input_source = (kwargs.get("input_source") or "").strip()
        input_type = (kwargs.get("input_type") or "text").strip().lower()
        mode = (kwargs.get("mode") or "classic").strip().lower()
        workflow = (kwargs.get("workflow") or "").strip() or None
        reason = (kwargs.get("reason") or "").strip()

        if pm is None:
            return json.dumps({"status": "invalid", "reason": "no_pipeline_manager"})
        if not input_source:
            return json.dumps({"status": "invalid", "reason": "input_source_required"})
        if mode not in ("classic", "agentic"):
            return json.dumps({"status": "invalid", "reason": "invalid_mode", "mode": mode})

        # Collision guard: refuse if a pipeline is mid-run.
        if (
            getattr(pm, "pipeline_steps", None)
            and getattr(pm, "current_step_index", 0) > 0
            and pm.current_step_index < len(pm.pipeline_steps)
        ):
            return json.dumps({"status": "busy", "reason": "pipeline_already_running"})

        payload = {
            "input_source": input_source,
            "input_type": input_type,
            "mode": mode,
            "workflow": workflow,
            "reason": reason,
        }
        audit_id = self._record_pending(payload)
        decision = self._ask_user(audit_id, payload)
        accepted = bool(decision.get("accepted"))
        if not accepted:
            self._record_outcome(audit_id, False, decision.get("reject_reason", ""))
            return json.dumps({
                "status": "rejected",
                "audit_id": audit_id,
                "reject_reason": decision.get("reject_reason", ""),
            })

        # Snapshot config + flip mode/workflow.
        prev_config = copy.deepcopy(pm.config)
        try:
            pm.config.enable_agentic_mode = (mode == "agentic")
            if workflow:
                pm.config.workflow_name = workflow

            should_stop = _resolve_should_stop()
            cancelled = False
            try:
                with _StatusForwarder(pm, self.name), _InterruptInstaller(pm, should_stop):
                    pm.start_pipeline(input_source, input_type)
            except InterruptedError:
                cancelled = True
            except Exception as e:
                logger.exception("run_pipeline: start_pipeline raised")
                self._record_outcome(audit_id, True, f"pipeline_error: {e}")
                return json.dumps({
                    "status": "error",
                    "audit_id": audit_id,
                    "error": str(e),
                })

            summary = _summarise_state(pm)
            self._record_outcome(audit_id, True, "cancelled" if cancelled else "")
            return json.dumps({
                "status": "cancelled" if cancelled else "completed",
                "audit_id": audit_id,
                "mode": mode,
                "workflow": workflow,
                **summary,
            })
        finally:
            try:
                pm.set_config(prev_config)
            except Exception:
                logger.exception("run_pipeline: restore config failed")


# ----------------------------------------------------------------------
# rerun_step
# ----------------------------------------------------------------------


_CLASSICAL_STEP_IDS = {"initialisation", "keywords", "dk_search", "dk_classification"}


class RerunStepTool(_MutationToolBase):
    name = "rerun_step"
    operation = "pipeline_rerun_step"
    description = (
        "Re-execute a single pipeline step against the current SharedContext. "
        "Useful for trying a different model/temperature on just the keyword "
        "selection or DK classification step. Works for both classical and "
        "agentic modes (auto-detected from current pipeline config). Requires "
        "confirmation unless autonomous_pipeline is True."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "step_id": {
                "type": "string",
                "description": (
                    "Step id. Classical: 'initialisation', 'keywords', "
                    "'dk_search', 'dk_classification'. Agentic: any step id "
                    "from the currently loaded workflow YAML."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Step-config overrides. Common keys: 'provider', 'model', "
                    "'temperature', 'top_p'."
                ),
                "additionalProperties": True,
            },
            "reason": {
                "type": "string",
                "description": "Short justification for the user.",
            },
        },
        "required": ["step_id", "reason"],
    }

    def available_for(self, session: Any) -> bool:
        pm = self.pipeline_manager
        return pm is not None and getattr(pm, "last_shared_context", None) is not None

    def execute(self, session: Any, **kwargs: Any) -> str:
        pm = self.pipeline_manager
        step_id = (kwargs.get("step_id") or "").strip()
        params = kwargs.get("params") or {}
        reason = (kwargs.get("reason") or "").strip()

        if pm is None:
            return json.dumps({"status": "invalid", "reason": "no_pipeline_manager"})
        if not step_id:
            return json.dumps({"status": "invalid", "reason": "step_id_required"})
        if getattr(pm, "last_shared_context", None) is None:
            return json.dumps({"status": "invalid", "reason": "no_prior_pipeline_run"})

        agentic = bool(getattr(pm.config, "enable_agentic_mode", False))
        if not agentic and step_id not in _CLASSICAL_STEP_IDS:
            return json.dumps({
                "status": "invalid",
                "reason": "unknown_classical_step_id",
                "step_id": step_id,
                "available": sorted(_CLASSICAL_STEP_IDS),
            })

        payload = {"step_id": step_id, "params": dict(params), "reason": reason}
        audit_id = self._record_pending(payload)
        decision = self._ask_user(audit_id, payload)
        accepted = bool(decision.get("accepted"))
        if not accepted:
            self._record_outcome(audit_id, False, decision.get("reject_reason", ""))
            return json.dumps({
                "status": "rejected",
                "audit_id": audit_id,
                "reject_reason": decision.get("reject_reason", ""),
            })

        should_stop = _resolve_should_stop()
        cancelled = False
        try:
            if agentic:
                cancelled = self._rerun_agentic(step_id, params, should_stop)
            else:
                cancelled = self._rerun_classical(step_id, params, should_stop)
        except Exception as e:
            logger.exception("rerun_step: execution raised")
            self._record_outcome(audit_id, True, f"step_error: {e}")
            return json.dumps({
                "status": "error",
                "audit_id": audit_id,
                "step_id": step_id,
                "error": str(e),
            })

        summary = _summarise_state(pm)
        # Konvergenz Pipeline/Agent: notify the result tabs (MainWindow) to
        # re-render from the updated current_analysis_state. run_pipeline emits
        # state.pipeline_completed via PipelineManager; a single-step rerun has
        # no such event, so emit state.changed here. - Claude Generated
        try:
            from src.core.state_bus import AlimaStateBus
            AlimaStateBus().emit_event(
                "state.changed", {"op": "rerun_step", "step_id": step_id}
            )
        except Exception:
            logger.exception("rerun_step: state.changed emit failed")
        self._record_outcome(audit_id, True, "cancelled" if cancelled else "")
        return json.dumps({
            "status": "cancelled" if cancelled else "completed",
            "audit_id": audit_id,
            "step_id": step_id,
            "mode": "agentic" if agentic else "classic",
            **summary,
        })

    # ------------------------------------------------------------------
    # Classical: execute_single_step
    # ------------------------------------------------------------------

    def _rerun_classical(
        self, step_id: str, params: Dict[str, Any], should_stop: Callable[[], bool]
    ) -> bool:
        pm = self.pipeline_manager
        cfg = copy.deepcopy(pm.config)
        # Apply param overrides to the target step.
        step_cfg = cfg.step_configs.get(step_id) if hasattr(cfg, "step_configs") else None
        if step_cfg is not None:
            for k, v in params.items():
                if hasattr(step_cfg, k):
                    setattr(step_cfg, k, v)
                elif isinstance(step_cfg, dict):
                    step_cfg[k] = v
        # Rebuild input_data from current_analysis_state.
        state = getattr(pm, "current_analysis_state", None)
        abstract = getattr(state, "original_abstract", "") if state else ""
        input_data = abstract or ""
        if step_id == "keywords" and state is not None:
            kw_lines = []
            for sr in getattr(state, "search_results", []) or []:
                for gnd_id, info in (getattr(sr, "results", {}) or {}).items():
                    title = info.get("title") if isinstance(info, dict) else ""
                    label = title or gnd_id
                    kw_lines.append(f"{label} (GND-ID: {gnd_id})")
            if kw_lines:
                input_data = f"{abstract}\n\nExisting Keywords: {', '.join(kw_lines)}"

        cancelled = False
        try:
            with _StatusForwarder(pm, self.name), _InterruptInstaller(pm, should_stop):
                pm.execute_single_step(step_id, cfg, input_data=input_data)
        except InterruptedError:
            cancelled = True

        # Refresh shared context from updated analysis state.
        try:
            from src.core.agents.shared_context import SharedContext
            if getattr(pm, "current_analysis_state", None) is not None:
                pm.last_shared_context = SharedContext.from_keyword_analysis_state(
                    pm.current_analysis_state
                )
        except Exception:
            logger.exception("rerun_step: shared-context refresh failed")
        return cancelled

    # ------------------------------------------------------------------
    # Agentic: write SC to tmp, call _start_v4_workflow_pipeline
    # ------------------------------------------------------------------

    def _rerun_agentic(
        self, step_id: str, params: Dict[str, Any], should_stop: Callable[[], bool]
    ) -> bool:
        pm = self.pipeline_manager
        ctx = pm.last_shared_context

        prev_config = copy.deepcopy(pm.config)
        tmp_path: Optional[str] = None
        cancelled = False
        try:
            # Persist current SC for warm-start.
            fd, tmp_path = tempfile.mkstemp(prefix="alima_rerun_", suffix=".json")
            os.close(fd)
            ctx.save_to_file(tmp_path)

            pm.config.agentic_step_id = step_id
            pm.config.agentic_input_context_path = tmp_path

            # Apply param overrides to step_config[step_id] if present.
            step_cfg = pm.config.step_configs.get(step_id) if hasattr(pm.config, "step_configs") else None
            if step_cfg is not None:
                for k, v in params.items():
                    if hasattr(step_cfg, k):
                        setattr(step_cfg, k, v)
                    elif isinstance(step_cfg, dict):
                        step_cfg[k] = v

            # Resolve workflow file (same logic the manager uses on start).
            wf_path = pm._resolve_workflow_path()
            if not wf_path:
                raise RuntimeError(
                    f"Agentic workflow '{pm.config.workflow_name}' not found"
                )

            try:
                with _StatusForwarder(pm, self.name), _InterruptInstaller(pm, should_stop):
                    pm._start_v4_workflow_pipeline(
                        str(uuid.uuid4()),
                        ctx.abstract,
                        ctx.input_type or "text",
                        ctx.source_value,
                        wf_path,
                    )
            except InterruptedError:
                cancelled = True
        finally:
            try:
                pm.set_config(prev_config)
            except Exception:
                logger.exception("rerun_step: restore config failed")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return cancelled


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def pipeline_tools(
    *,
    pipeline_manager: Any,
    kb_manager: Any,
    gateway: Any,
    chat_config: Any,
    session_id: str,
) -> List[BaseChatTool]:
    """Build the pipeline orchestration tool list for a chat session."""
    common = dict(
        pipeline_manager=pipeline_manager,
        kb_manager=kb_manager,
        gateway=gateway,
        chat_config=chat_config,
        session_id=session_id,
    )
    return [RunPipelineTool(**common), RerunStepTool(**common)]
