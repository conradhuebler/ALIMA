"""Pipeline-log rendering mixin for PipelineChatPanel. Claude Generated.

Extracted from ``pipeline_chat_panel.py`` (F-5 god-file split): the former
``PipelineStreamWidget`` behavior — pipeline step events (▶/✅/❌, durations,
GND-verification + DK-catalog summaries), LLM token streaming, log save/clear,
and the per-run reset.

Method bodies are moved verbatim. ``PipelineLogMixin`` is mixed into
``PipelineChatPanel`` (which provides ``__init__``, ``self._renderer``, the Qt
base, and the chat-side helpers referenced here such as ``load_context``); it is
not a standalone widget. Slot decorators are kept for intent; new-style
connections work regardless of meta-object registration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import pyqtSlot

from ..core.pipeline_manager import PipelineStep
from ..utils.pipeline_utils import PipelineResultFormatter


class PipelineLogMixin:
    """Pipeline-side log rendering (preserved from PipelineStreamWidget)."""

    def add_pipeline_message(
        self,
        message: str,
        level: str = "info",
        step_id: Optional[str] = None,
    ):
        self._renderer.render_pipeline_log(message, level, step_id)

    def add_streaming_token(self, token: str, step_id: str):
        # The SYSTEM/USER prompt dump is rendered as a collapsible 📥 Input
        # block via the state.pipeline_prompt bus event — drop the inline
        # duplicate so it isn't shown twice. - Claude Generated
        if token.lstrip().startswith("--- SYSTEM ---"):
            return
        self._renderer.render_streaming_token(token, step_id)

    def start_streaming_line(self, step_id: str, prefix: str = ""):
        self._renderer.start_streaming_line(step_id, prefix)

    def end_streaming_line(self):
        self._renderer.end_streaming_line()

    def auto_scroll_to_bottom(self):
        self._renderer.auto_scroll_to_bottom()

    @pyqtSlot(object)
    def on_pipeline_started(self, pipeline_id: str):
        self.add_pipeline_message("🚀 Pipeline gestartet", "step")
        self.add_pipeline_message(f"Pipeline ID: {pipeline_id}", "info")
        self.pipeline_start_time = datetime.now()

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        self.current_step_id = step.step_id
        self.step_start_times[step.step_id] = datetime.now()
        args: Dict[str, Any] = {"name": step.name}
        if step.provider and step.model:
            args["provider"] = f"{step.provider}/{step.model}"
        tool_id = self._renderer.render_tool_call(f"pipeline.{step.step_id}", args)
        self._step_tool_call_ids[step.step_id] = tool_id

    def _build_step_summary(self, step: "PipelineStep", duration: str) -> str:
        """Return a multiline summary string for a completed pipeline step."""
        lines = [f"✅ Abgeschlossen in {duration}"]
        if not step.output_data:
            return "\n".join(lines)

        if step.step_id == "keywords" and (
            "keywords" in step.output_data or "final_keywords" in step.output_data
        ):
            keywords = step.output_data.get(
                "final_keywords", step.output_data.get("keywords", [])
            )
            lines.append(f"Gefunden: {len(keywords)} Keywords")
            if keywords:
                preview = ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else "")
                lines.append(f"Keywords: {preview}")
            verification = step.output_data.get("verification")
            if verification and isinstance(verification, dict):
                stats = verification.get("stats", {})
                verified_count = stats.get("verified_count", 0)
                total = stats.get("total_extracted", 0)
                rejected = verification.get("rejected", [])
                lines.append(f"✅ {verified_count}/{total} Keywords GND-verifiziert")
                if rejected:
                    rejected_names = [r.split("(")[0].strip() for r in rejected]
                    lines.append(
                        f"⚠️ {len(rejected)} Keywords ohne GND-Pool-Treffer entfernt: "
                        + ", ".join(rejected_names)
                    )

        elif step.step_id == "search" and "search_results" in step.output_data:
            count = step.output_data["search_results"]
            lines.append(f"Gefunden: {count} GND-Einträge")

        elif step.step_id == "verification" and "verified_keywords" in step.output_data:
            verified = step.output_data["verified_keywords"]
            lines.append(f"Verifiziert: {len(verified)} Keywords")

        elif step.step_id == "dk_search" and "dk_search_results" in step.output_data:
            lines.append(self._format_dk_search_results(step.output_data["dk_search_results"]))

        return "\n".join(lines)

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        duration = "unbekannt"
        if step.step_id in self.step_start_times:
            duration_seconds = (
                datetime.now() - self.step_start_times[step.step_id]
            ).total_seconds()
            duration = f"{duration_seconds:.1f}s"

        summary = self._build_step_summary(step, duration)
        tool_id = self._step_tool_call_ids.pop(step.step_id, None)
        if tool_id:
            self._renderer.render_tool_result(tool_id, summary, status="success")
        else:
            # Fallback: no tool block was opened for this step (e.g. step fired
            # before the panel was ready), emit as flat log lines.
            self.add_pipeline_message(
                f"✅ Schritt abgeschlossen in {duration}", "success", step.step_id
            )

        # Render the per-DK-code catalog-research result identically to the
        # Pipeline-Tab (shared formatter), in addition to the keyword-timing
        # summary kept in the collapsible tool block above. - Claude Generated
        if step.step_id == "dk_search" and step.output_data:
            self._render_dk_search_card(step.output_data)

    def _render_dk_search_card(self, output_data: Dict[str, Any]) -> None:
        """Render per-DK-code catalog-research results as a card (shared formatter).

        WP12: the card HTML is produced by the shared
        ``PipelineResultFormatter.format_dk_search_card_html`` so the GUI and the
        webapp emit byte-identical chrome from one source.
        """
        html, text = PipelineResultFormatter.format_dk_search_card_html(output_data)
        if html:
            self._renderer.render_html_block(html, kind="dk_search", plain_text=text)

    def _format_dk_search_results(self, dk_results: List[Dict[str, Any]]) -> str:
        """Build a summary string for DK search results (tool-block body)."""
        if not dk_results:
            return "Keine Klassifikationen (DK/RVK) gefunden"

        total_keywords = len(dk_results)
        total_classifications = sum(
            len(r.get("classifications", [])) for r in dk_results
        )
        cache_count = sum(1 for r in dk_results if r.get("source") == "cache")
        live_count = total_keywords - cache_count
        success_count = sum(1 for r in dk_results if r.get("classifications"))

        lines = [
            f"🔍 Klassifikationssuche: {total_keywords} Keywords → "
            f"{success_count} erfolgreich → {total_classifications} Klassifikationen",
        ]
        if cache_count > 0 or live_count > 0:
            lines.append(f"   📦 Cache: {cache_count} | 🔍 Live: {live_count}")

        for keyword_result in dk_results:
            keyword = keyword_result.get("keyword", "unknown")
            source = keyword_result.get("source", "unknown")
            search_time = keyword_result.get("search_time_ms", 0)
            classifications = keyword_result.get("classifications", [])
            status_icon = "✅" if classifications else "⚠️"
            status_text = f"{len(classifications)} Klassifikationen" if classifications else "Keine Klassifikationen"
            source_icon = "📦" if source == "cache" else "🔍"
            timing_text = f"({search_time:.1f}ms)" if search_time > 0 else ""
            lines.append(f"{status_icon} {source_icon} {keyword} - {status_text} {timing_text}")

        return "\n".join(lines)

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        tool_id = self._step_tool_call_ids.pop(step.step_id, None)
        if tool_id:
            self._renderer.render_tool_result(tool_id, error_message, status="error")
        else:
            self.add_pipeline_message(
                f"❌ Fehler in Schritt: {step.name}", "error", step.step_id
            )
            self.add_pipeline_message(error_message, "error", step.step_id)

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        total_duration = "unbekannt"
        if hasattr(self, "pipeline_start_time"):
            total_seconds = (
                datetime.now() - self.pipeline_start_time
            ).total_seconds()
            total_duration = f"{total_seconds:.1f}s"
        self.add_pipeline_message(
            f"\U0001f389 Pipeline vollständig abgeschlossen in {total_duration}!",
            "success",
        )

        if (
            analysis_state
            and hasattr(analysis_state, "final_llm_analysis")
            and analysis_state.final_llm_analysis
        ):
            kw_list = analysis_state.final_llm_analysis.extracted_gnd_keywords or []
            if kw_list:
                kw_display = ", ".join(kw_list)
                self.add_pipeline_message(
                    f"\U0001f4cc {len(kw_list)} GND-Schlagworte ausgewählt:\n{kw_display}",
                    "success",
                )
            response_text = (
                analysis_state.final_llm_analysis.response_full_text or ""
            )
            if (
                "Schlagwortketten" in response_text
                or "schlagwortketten" in response_text.lower()
            ):
                chain_lines = [
                    line
                    for line in response_text.split("\n")
                    if "→" in line or "->" in line
                ]
                if chain_lines:
                    self.add_pipeline_message(
                        "\U0001f517 Schlagwortketten:\n" + "\n".join(chain_lines[:10]),
                        "success",
                    )

        if analysis_state and getattr(analysis_state, "dk_classifications", None):
            # WP12: the rich colour-coded card (confidence + per-code titles) is
            # built by the shared formatter so the GUI and webapp render the
            # identical chrome from one source.
            card_html, dk_codes_text = (
                PipelineResultFormatter.format_dk_classifications_card_html(analysis_state)
            )
            if card_html:
                self.add_pipeline_message("\U0001f3f7 DK-Klassifikationen:", "success")
                self._renderer.render_html_block(
                    card_html, kind="dk_classifications", plain_text=dk_codes_text
                )

        # Reintroduced RVK-Analytik: frequency Auswertung + RVK provenance tables
        # via the shared formatter (replaces the old plain-text provenance line so
        # GUI and webapp render the identical chrome). - Claude Generated
        if analysis_state:
            ausw_html, ausw_plain = (
                PipelineResultFormatter.format_dk_auswertung_card_html(analysis_state)
            )
            if ausw_html:
                self._renderer.render_html_block(
                    ausw_html, kind="dk_statistics", plain_text=ausw_plain
                )

        # Generic convention: any workflow that populates extra.report_markdown
        # (e.g. title_list_search's render_report step) gets it rendered as an
        # actual HTML table here, instead of the unrendered pipe-table text
        # that streams into the log during execution. Keyed off the field's
        # presence, not the workflow name. - Claude Generated
        report_markdown = getattr(analysis_state, "report_markdown", "") if analysis_state else ""
        if report_markdown:
            self._renderer.render_markdown_block(report_markdown, kind="workflow_report")

        # Auto-load chat context for the just-finished pipeline.
        try:
            self.load_context(analysis_state)
        except Exception:
            self.logger.exception("PipelineChatPanel: load_context after pipeline failed")

    @pyqtSlot(str)
    def on_llm_token_received(self, token: str):
        if self.current_step_id:
            self.add_streaming_token(token, self.current_step_id)

    def start_llm_streaming(self, step_id: str):
        self.start_streaming_line(step_id, "LLM Antwort: ")

    def end_llm_streaming(self):
        self.end_streaming_line()

    def clear_stream(self):
        self._renderer.clear()
        self.add_pipeline_message("Stream geleert", "info")

    def save_stream_log(self):
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        if self.current_working_title:
            default_filename = f"{self.current_working_title}_log.txt"
        else:
            default_filename = (
                f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

        docs_dir = Path.home() / "Documents"
        if not docs_dir.exists():
            docs_dir = Path.home()
        default_path = str(docs_dir / default_filename)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Pipeline-Log speichern",
            default_path,
            "Text Files (*.txt);;All Files (*)",
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    # WebLogView has no toPlainText(); reconstruct the log from
                    # the renderer's message history (completed messages).
                    plain_text = "\n".join(
                        entry.content for entry in self._renderer.history
                    )
                    f.write(f"ALIMA Pipeline Log - {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(plain_text)
                self.add_pipeline_message(
                    f"Log gespeichert: {filename}", "success"
                )
            except Exception as e:
                self.add_pipeline_message(f"Fehler beim Speichern: {e}", "error")

    def set_working_title(self, working_title: str):
        self.current_working_title = working_title
        self.logger.info(
            f"PipelineChatPanel: working_title set to '{working_title}'"
        )

    def refresh_styles(self):
        if hasattr(self, "stream_text"):
            from .styles import get_font_size
            self.stream_text.set_font_pt(get_font_size())

    def reset_for_new_pipeline(self):
        self.current_step_id = None
        self.step_start_times.clear()
        self.current_working_title = None
        self.clear_stream()
        self.hide_repetition_warning()
        self._bus_tool_call_ids.clear()
        self._prompt_blocks.clear()
        self._prompt_meta.clear()
        self._last_tool_call_id = None
        if self.reset_toggle.isChecked():
            self.session.reset()
            self.current_context = ""
            self.working_title = ""
            self._renderer.clear()
