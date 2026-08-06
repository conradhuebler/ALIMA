"""AlimaStateBus event-handler mixin for PipelineChatPanel. Claude Generated.

Extracted from ``pipeline_chat_panel.py`` (F-5 god-file split): the handlers for
the agentic-pipeline / chat-agent bus events (``tool.called`` / ``tool.result``,
``state.pipeline_step`` / ``_prompt`` / ``_prompt_done`` / ``_completed``,
``state.changed``). They render collapsible tool-call blocks so bus-driven steps
look identical to native tool calls.

Method bodies are moved verbatim. ``BusEventMixin`` is mixed into
``PipelineChatPanel``; the ``bus.subscribe(...)`` wiring that binds these
handlers stays in ``PipelineChatPanel.__init__`` (so the module source keeps the
``state.pipeline_started`` / ``state.pipeline_completed`` references that
``tests/test_state_bus.py`` asserts on). Handlers reference panel state
(``self._renderer``, ``self._bus_tool_call_ids`` …) and ChatAgentMixin helpers
(``self._refresh_shared_context``) resolved through the MRO; not a standalone
widget.

Ownership note (Chat-UX 9/9): this is one of THREE StateBus→renderer
consumers — (1) this GUI mixin, (2) the webapp's ``_SessionBusSubscriber``
(``src/webapp/render_bridge.py``, Qt-free, near-identical handler bodies),
(3) ``UnifiedMessageRenderer.subscribe`` (tool events only, for embedded
mini-logs). Behavioral edits to the shared handler logic (e.g. the §9.3
error-text passthrough) must be applied to (1) and (2) in lockstep; full
extraction into one Qt-free bridge is a known follow-up.
"""
from __future__ import annotations

from datetime import datetime


class BusEventMixin:
    """Bus subscriptions for agentic-pipeline tool events."""

    def _on_state_changed(self, _diff: dict) -> None:
        self._refresh_shared_context()

    def _on_bus_notice(self, payload: dict) -> None:
        """Operational notice (e.g. an LLM rate-limit wait) as a log line.

        Lockstep counterpart of ``_SessionBusSubscriber._handle_notice``
        (webapp). - Claude Generated
        """
        try:
            text = (payload or {}).get("text") or ""
            if not text:
                return
            level = (payload or {}).get("level") or "warning"
            self._renderer.render_pipeline_log(text, level)
        except Exception:
            self.logger.exception("PipelineChatPanel: bus notice rendering failed")

    def _on_bus_tool_called(self, payload: dict) -> None:
        try:
            name = payload.get("name", "") or "tool"
            args = payload.get("arguments", {}) or {}
            tool_id = self._renderer.render_tool_call(name, args)
            bus_id = payload.get("id")
            if bus_id:
                self._bus_tool_call_ids[bus_id] = tool_id
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.called rendering failed"
            )

    def _on_bus_pipeline_prompt(self, payload: dict) -> None:
        """Render the agentic step input (SYSTEM+USER prompt) as a collapsed,
        timestamped block instead of inline streamed text. - Claude Generated"""
        try:
            prompt_id = str(payload.get("prompt_id", "") or "")
            step_id = payload.get("step_id", "") or "?"
            kind = payload.get("kind", "input") or "input"
            system = payload.get("system", "") or ""
            user = payload.get("user", "") or ""
            ts = str(payload.get("timestamp", "") or "")
            ts_hms = ts.split("T")[-1] if "T" in ts else ts
            provider = payload.get("provider", "") or ""
            model = payload.get("model", "") or ""

            meta_parts = []
            if ts_hms:
                meta_parts.append(ts_hms)
            pm = "/".join(p for p in (provider, model) if p)
            if pm:
                meta_parts.append(pm)
            if hasattr(self, "pipeline_start_time"):
                elapsed = (datetime.now() - self.pipeline_start_time).total_seconds()
                meta_parts.append(f"(+{elapsed:.1f}s)")
            meta = "  ".join(meta_parts)

            if kind == "reflection":
                icon, title = "🔍", f"Reflexion '{step_id}'"
            else:
                icon, title = "📥", f"Input '{step_id}'"

            body = f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}"
            # Close any open streaming line so the collapsible sits on its own.
            if self._renderer._is_streaming:
                self._renderer.end_streaming_line()
            tool_id = self._renderer.render_collapsible(
                title, body, collapsed=True, icon=icon, meta=meta,
            )
            if prompt_id:
                self._prompt_blocks[prompt_id] = tool_id
                self._prompt_meta[prompt_id] = meta
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: _on_bus_pipeline_prompt failed"
            )

    def _on_bus_pipeline_prompt_done(self, payload: dict) -> None:
        """Append the LLM-call duration (⏱ Ys) to a rendered input block. - Claude Generated"""
        try:
            prompt_id = str(payload.get("prompt_id", "") or "")
            dur = payload.get("duration_s")
            tool_id = self._prompt_blocks.get(prompt_id)
            if tool_id is None or dur is None:
                return
            base = self._prompt_meta.get(prompt_id, "")
            meta = f"{base}  ⏱ {float(dur):.1f}s" if base else f"⏱ {float(dur):.1f}s"
            self._renderer.update_collapsible_meta(tool_id, meta)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: _on_bus_pipeline_prompt_done failed"
            )

    def _on_bus_tool_result(self, payload: dict) -> None:
        try:
            result = payload.get("result", "") or ""
            # Register any URLs from tool results so the renderer won't
            # flag them as external (e.g. DOIs from finc records). - Claude Generated
            try:
                import json as _json
                from src.core.url_utils import extract_urls_from_json
                self._renderer.add_trusted_urls(
                    extract_urls_from_json(_json.loads(result))
                )
            except Exception:
                pass
            bus_id = payload.get("id")
            tool_id = self._bus_tool_call_ids.pop(bus_id, None) if bus_id else None
            if tool_id:
                # P-B: cache-hit branch appends a 📦 badge to the result
                # text so users can distinguish fast cache returns from
                # live calls. Renderer forwards the result verbatim.
                if payload.get("cache_hit"):
                    preview = result.strip().replace("\n", " ")
                    if len(preview) > 80:
                        preview = preview[:80] + "…"
                    result = f"📦 cache: {preview}"
                # Phase E: bus producers emit "ok" (CachingToolRegistry) but
                # the renderer's internal status enum is "success" / "error".
                # Normalize here so the ✓/✗ icon picks the right glyph.
                raw_status = payload.get("status") or "success"
                status = "success" if raw_status == "ok" else raw_status
                self._renderer.render_tool_result(
                    tool_id, result, status=status,
                )
            else:
                # Fallback: orphan result (bus id unknown — late or dropped call).
                preview = result.strip().replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:120] + "…"
                self._append_system_message(f"↳ {preview}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.result rendering failed"
            )

    def _on_bus_pipeline_completed(self, payload: dict) -> None:
        """Render a system message when ``state.pipeline_completed`` fires.

        Phase F: the legacy ``pipeline_completed_callback`` still runs
        elsewhere, but the bus event was previously dead (no
        subscriber). Now the panel reacts to it so the chat log
        acknowledges the end of a pipeline run consistently.
        """
        try:
            from ..utils.i18n import t
            label = t("render.pipeline_completed")
            workflow = (payload or {}).get("workflow")
            if workflow:
                label += f" ({workflow})"
            self._renderer.render_system_message(label)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus state.pipeline_completed render failed"
            )

    def _on_bus_pipeline_step(self, payload: dict) -> None:
        """Render step-progress as collapsible tool-call block (was: legacy marker).

        Same visual treatment as real ``tool.called``/``tool.result`` events so
        that steps triggered via ``run_pipeline`` / ``rerun_step`` look
        identical to native tool calls.
        """
        try:
            status = payload.get("status", "")
            step_id = payload.get("step_id", "") or "?"
            name = payload.get("name", "") or step_id
            tool_name = f"pipeline.{step_id}"
            args = {
                "step": step_id,
                "name": name,
                "tool": payload.get("tool", ""),
            }
            if status == "running":
                self._last_tool_call_id = self._renderer.render_tool_call(
                    tool_name, args
                )
                self._open_step_status = []
                self._pipeline_step_open = True
                return
            # Terminal: success when explicitly "completed", error otherwise.
            result_status = "success" if status == "completed" else "error"
            # Prefer the accumulated status lines (per-keyword search
            # progress, etc.) as the block's body when present. Fall
            # back to a short summary line if the step emitted no
            # status messages.
            # Post-(F) polish: if no status lines accumulated, fall back
            # to a short summary line. Use ``getattr`` so the handler
            # is robust against an older stub that doesn't define the
            # accumulator attribute yet.
            if getattr(self, "_open_step_status", None):
                result_text = "\n".join(self._open_step_status)
            else:
                result_text = f"{status or 'done'}: {name}"
            # WP12 §9.3: surface the actual error text — the payload carries
            # it since WP A, but it used to be dropped here.
            error_text = payload.get("error")
            if result_status == "error" and error_text:
                result_text = f"{error_text}\n{result_text}"
            self._open_step_status = []
            self._pipeline_step_open = False
            if self._last_tool_call_id:
                self._renderer.render_tool_result(
                    self._last_tool_call_id, result_text, status=result_status
                )
                self._last_tool_call_id = None
            else:
                # Orphan completion (no prior running) → still render a block.
                tid = self._renderer.render_tool_call(tool_name, args)
                self._renderer.render_tool_result(
                    tid, result_text, status=result_status
                )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus state.pipeline_step rendering failed"
            )
