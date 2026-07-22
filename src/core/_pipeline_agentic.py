"""Agentic v4 pipeline path for PipelineManager - Claude Generated.

Split out of ``pipeline_manager.py`` (WP cleanup D). Verbatim mixin extraction:
the methods stay on ``PipelineManager`` via MRO, so no call site changes.

The agentic side of the manager — the v4 ``WorkflowExecutor`` path
(``_start_v4_workflow_pipeline``), its entry point, and the workflow-path
resolver — plus ``_AgenticStreamFilter`` (the line-buffer that strips JSON out
of agentic LLM streaming). All of it is agentic-only: the filter is instantiated
nowhere but the v4 path, and these methods make NO cross-method calls back into
the class (they read ``self`` state and use method-local imports). Companion to
``_pipeline_classic_steps.ClassicStepExecutorMixin`` — classic step bodies in
one mixin, the agentic path in the other.
"""

import json
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils.config_models import ProviderScope


class _AgenticStreamFilter:
    """Line-buffer filter: strips JSON syntax from agentic LLM streaming output.

    Non-JSON status messages (emojis, headers) pass through unchanged.
    JSON keyword/gnd_id objects → "keyword (gnd_id)".
    Compact single-line JSON objects/arrays → parsed and extracted.
    Numeric, boolean, and structural-only lines → suppressed.
    """

    _SKIP = _re.compile(r'^[\s\{\}\[\],]*$|^```')
    _KV_STR = _re.compile(r'"([^"]+)":\s*"([^"]*)"')
    _KV_KEY_STRUCT = _re.compile(r'"[^"]+"\s*:\s*[\[\{]')
    _KV_SCALAR = _re.compile(r'"[^"]+"\s*:\s*(?:[\d.eE+-]+|true|false|null)')

    _STRUCTURAL_KEYS = frozenset({
        'keywords', 'classifications', 'search_terms',
        'missing_concepts', 'final_keywords', 'gnd_entries',
    })

    def __init__(self, raw_cb: Callable[[str], None]) -> None:
        self._cb = raw_cb
        self._buf = ""
        self._pending_keyword = ""

    def __call__(self, token: str) -> None:
        self._buf += token
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._process_line(line)

    def flush(self) -> None:
        if self._buf:
            self._process_line(self._buf)
            self._buf = ""
        if self._pending_keyword:
            self._cb(f"{self._pending_keyword}\n")
            self._pending_keyword = ""

    def _emit_parsed(self, obj: Any) -> None:
        """Recursively emit readable text from a parsed JSON value."""
        if isinstance(obj, dict):
            keyword = obj.get('keyword') or ''
            gnd_id = obj.get('gnd_id') or ''
            if keyword and gnd_id:
                self._cb(f"{keyword} ({gnd_id})\n")
                return
            if keyword:
                self._cb(f"{keyword}\n")
                return
            # No keyword field — show title/string values then recurse into lists
            for key in ('title', 'working_title'):
                val = obj.get(key) or ''
                if val:
                    self._cb(f"{val}\n")
            for key, val in obj.items():
                if key in ('title', 'working_title'):
                    continue
                if isinstance(val, list):
                    for item in val:
                        self._emit_parsed(item)
                elif isinstance(val, str) and val.strip() and key not in self._STRUCTURAL_KEYS:
                    self._cb(f"{val}\n")
        elif isinstance(obj, list):
            for item in obj:
                self._emit_parsed(item)
        elif isinstance(obj, str) and obj.strip():
            self._cb(f"{obj}\n")

    def _process_line(self, line: str) -> None:
        s = line.strip().rstrip(',')
        if not s or self._SKIP.match(s):
            return

        # Compact JSON object or array on one line — parse directly
        if s.startswith('{') or s.startswith('['):
            if self._pending_keyword:
                self._cb(f"{self._pending_keyword}\n")
                self._pending_keyword = ""
            try:
                import json as _json
                self._emit_parsed(_json.loads(s))
            except Exception:
                pass  # partial/invalid JSON — suppress
            return

        m = self._KV_STR.match(s)
        if m:
            key, val = m.groups()
            if key == 'keyword':
                self._pending_keyword = val
            elif key == 'gnd_id' and self._pending_keyword:
                self._cb(f"{self._pending_keyword} ({val})\n")
                self._pending_keyword = ""
            elif key not in self._STRUCTURAL_KEYS:
                if self._pending_keyword:
                    self._cb(f"{self._pending_keyword}\n")
                    self._pending_keyword = ""
                self._cb(f"{val}\n")
            return
        if self._KV_KEY_STRUCT.match(s) or self._KV_SCALAR.match(s):
            if self._pending_keyword:
                self._cb(f"{self._pending_keyword}\n")
                self._pending_keyword = ""
            return
        # Non-JSON line (status message with emojis, headers) → pass through
        if self._pending_keyword:
            self._cb(f"{self._pending_keyword}\n")
            self._pending_keyword = ""
        self._cb(line + '\n')


class AgenticPipelineMixin:
    """The agentic v4 pipeline path. Mixed into :class:`PipelineManager`."""

    def _resolve_workflow_path(self) -> Optional[str]:
        """Resolve the configured workflow name/custom path to a filesystem path.

        Returns the first existing match from (custom_workflow_path,
        workflow search paths).  ``None`` if nothing found.
        """
        from src.core.agents.workflow_loader import find_workflow_file

        if self.config.custom_workflow_path:
            p = Path(self.config.custom_workflow_path)
            if p.exists():
                return str(p)
        if self.config.workflow_name:
            found = find_workflow_file(self.config.workflow_name)
            if found is not None:
                return str(found)
        return None

    def _start_agentic_pipeline(self, pipeline_id: str, input_text: str,
                                 input_type: str, input_source: Optional[str]) -> str:
        """Execute pipeline via v4 WorkflowExecutor — Claude Generated.

        Resolves the configured workflow YAML and delegates to
        :meth:`_start_v4_workflow_pipeline`. The legacy v3 MetaAgent dispatch
        was removed in the Phase 5 cleanup; v3 YAMLs now live under
        ``workflows/legacy/`` and are no longer discovered.
        """
        wf_path = self._resolve_workflow_path()
        if not wf_path:
            msg = (
                f"Agentic workflow '{self.config.workflow_name}' not found. "
                "Use `alima workflows list` to see available v4 workflows."
            )
            self.logger.error(msg)
            if self.stream_callback:
                self.stream_callback(f"\n❌ {msg}", "error")
            if self.pipeline_completed_callback:
                self.pipeline_completed_callback(None)
            return pipeline_id

        return self._start_v4_workflow_pipeline(
            pipeline_id, input_text, input_type, input_source, wf_path
        )

    def _start_v4_workflow_pipeline(
        self,
        pipeline_id: str,
        input_text: str,
        input_type: str,
        input_source: Optional[str],
        workflow_path: str,
    ) -> str:
        """Execute a v4 YAML workflow through :class:`WorkflowExecutor` - Claude Generated.

        Populates ``self.current_analysis_state`` by converting the resulting
        :class:`SharedContext` via ``to_keyword_analysis_state()`` so the rest
        of the GUI/CLI stack stays untouched.
        """
        # Side-effect imports register built-in step types + tool fns.
        from src.core.agents import deterministic_functions as _fns  # noqa: F401
        from src.core.agents import steps as _steps  # noqa: F401
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.sub_agents import create_caching_registry
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import load_workflow

        self.logger.info(f"🚀 Starting v4 workflow pipeline {pipeline_id}: {workflow_path}")

        # Resolve agentic default from the unified config hierarchy first.
        # Runtime --override still wins via global_provider_override.
        provider = self.config.global_provider_override or ""
        model = self.config.global_model_override or ""
        if not provider or not model:
            unified_config = self.config_manager.get_unified_config() if self.config_manager else None
            if unified_config is not None:
                resolved_provider, resolved_model = unified_config.resolve_default_provider_model(
                    scope=ProviderScope.AGENTIC
                )
                provider = provider or resolved_provider
                model = model or resolved_model

        # Final fallback: walk step_configs for any user-supplied override.
        # This keeps legacy --step overrides and saved per-step settings working.
        if not provider or not model:
            for cfg in self.config.step_configs.values():
                if cfg is None:
                    continue
                provider = provider or (cfg.provider or "")
                model = model or (cfg.model or "")
                if provider and model:
                    break

        temperature = 0.5
        for cfg in self.config.step_configs.values():
            if cfg is None:
                continue
            if cfg.temperature is not None:
                temperature = cfg.temperature
                break

        if not provider or not model:
            self.logger.warning(
                "Agentic workflow %r starts with empty provider/model "
                "(provider=%r, model=%r). Set agentic_default_provider/model, "
                "pipeline_default_provider/model, or use --override.",
                getattr(workflow_path, "name", workflow_path),
                provider,
                model,
            )
            if self.stream_callback:
                self.stream_callback(
                    f"\n⚠️  Provider/Model leer (provider={provider!r}, "
                    f"model={model!r}) — Schritte können stillschweigend "
                    f"fehlschlagen.\n",
                    "agentic",
                )

        if self.config.agentic_input_context_path:
            ctx = SharedContext.load_from_file(self.config.agentic_input_context_path)
            self.logger.info(
                f"Loaded warm-start context from {self.config.agentic_input_context_path}"
            )
        else:
            ctx = SharedContext(
                abstract=input_text,
                initial_keywords=[],
                input_type=input_type,
                source_value=input_source,
            )
        ctx.provider = provider or ctx.provider
        ctx.model = model or ctx.model
        ctx.temperature = temperature
        ctx.verbose = self.config.agentic_verbose
        ctx.prompt_service = self.alima_manager.prompt_service

        try:
            workflow = load_workflow(workflow_path, strict=True)
        except Exception as e:
            self.logger.error(f"Failed to load v4 workflow '{workflow_path}': {e}")
            if self.pipeline_completed_callback:
                self.pipeline_completed_callback(None)
            return pipeline_id

        tool_registry = create_caching_registry(config_manager=self.config_manager)

        _json_filter: Optional[_AgenticStreamFilter] = None
        if self.stream_callback and not self.config.agentic_verbose:
            _raw_cb = self.stream_callback  # capture for closure
            _json_filter = _AgenticStreamFilter(lambda msg: _raw_cb(msg, "agentic"))

        def _stream(msg: str) -> None:
            if _json_filter is not None:
                _json_filter(msg)
            elif self.stream_callback:
                self.stream_callback(msg, "agentic")

        # MetaAgent always active in agentic mode — merge YAML meta_agent block with enabled=True
        meta_cfg = dict(workflow.raw.get("meta_agent", {}) or {})
        meta_cfg["enabled"] = True
        use_meta = True

        if use_meta:
            from src.core.agents.meta_agent import MetaAgent
            if self.stream_callback:
                # Compact one-liner (tagged "agentic") instead of a ===== banner. - Claude Generated
                self.stream_callback(
                    f"🤖 MetaAgent aktiv (max. {meta_cfg.get('max_cycles', 10)} Zyklen)\n",
                    "agentic",
                )
            self.logger.info(f"MetaAgent mode enabled for {workflow.name}")

            executor = MetaAgent(
                llm_service=self.alima_manager.llm_service,
                tool_registry=tool_registry,
                stream_callback=_stream,
                context_callback=self.agentic_context_callback,
                max_cycles=int(meta_cfg.get("max_cycles", 10)),
                reflection_model=meta_cfg.get("reflection_model", model),
                reflection_provider=meta_cfg.get("reflection_provider", provider),
            )
        else:
            executor = WorkflowExecutor(
                llm_service=self.alima_manager.llm_service,
                tool_registry=tool_registry,
                stream_callback=_stream,
                context_callback=self.agentic_context_callback,
            )

        try:
            if use_meta:
                report = executor.run(
                    workflow,
                    ctx,
                    meta_config=meta_cfg,
                )
            else:
                report = executor.run(
                    workflow,
                    ctx,
                    only_step=self.config.agentic_step_id or None,
                    stop_on_error=True,
                )
            if _json_filter is not None:
                _json_filter.flush()
            if not report.success:
                raise RuntimeError(report.error or "v4 workflow failed")

            self.current_analysis_state = ctx.to_keyword_analysis_state()
            # WP10 P-δ.1: retain the raw SharedContext for chat tools.
            self.last_shared_context = ctx
            try:
                from src.core.state_bus import AlimaStateBus
                AlimaStateBus().emit_event(
                    "state.pipeline_completed",
                    {"workflow": workflow.name},
                )
            except Exception:
                self.logger.warning("state.pipeline_completed emit failed", exc_info=True)
            if self.pipeline_completed_callback:
                self.pipeline_completed_callback(self.current_analysis_state)

        except Exception as e:  # noqa: BLE001
            if _json_filter is not None:
                _json_filter.flush()
            self.logger.error(f"v4 workflow pipeline failed: {e}")
            if self.stream_callback:
                self.stream_callback(f"\n❌ Workflow Fehler: {e}", "error")
            if self.pipeline_completed_callback:
                self.pipeline_completed_callback(None)

        return pipeline_id
