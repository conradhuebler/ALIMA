"""Pipeline event/lifecycle handler mixin for PipelineTab. Claude Generated.

Extracted from ``pipeline_tab.py`` (F-5 god-file split): the step/pipeline event
handlers (``on_step_*`` / ``on_pipeline_*``), abort/stop, LLM-stream + repetition
slots, the agentic step-snapshot bridge, and classical-tab state sync. Method
bodies are moved verbatim. ``PipelineTabEventsMixin`` is mixed into ``PipelineTab``
(which provides ``__init__``, the widgets/state and helper methods these
reference); it is not a standalone widget.
"""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox

from ..utils.pipeline_utils import PipelineResultFormatter


class PipelineTabEventsMixin:
    """Pipeline step/lifecycle event handlers (verbatim from PipelineTab)."""

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        """Handle step started event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Track step start time
        self.step_start_times[step.step_id] = datetime.now()
        if self.pipeline_start_time is None:
            self.pipeline_start_time = datetime.now()

        # Start live duration updates for this step
        self.current_running_step = step.step_id
        self.duration_update_timer.start()

        # Update global status bar with current provider info
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(step, "provider") and hasattr(step, "model"):
                self.main_window.global_status_bar.update_provider_info(
                    step.provider, step.model
                )
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "running"
                )
            if hasattr(self.main_window.global_status_bar, "pipeline_progress"):
                self.main_window.global_status_bar.pipeline_progress.show()

        self.pipeline_status_label.setText(f"Schritt läuft: {step.name}")

        # Auto-jump to current step tab
        if hasattr(self, "pipeline_tabs"):
            self.jump_to_step(step.step_id)

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_started(step)

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        """Handle step completed event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Stop live duration updates for this step
        if self.current_running_step == step.step_id:
            self.duration_update_timer.stop()
            self.current_running_step = None

        # Update global status bar
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "completed"
                )

        # Update result displays
        if step.step_id == "initialisation" and step.output_data:
            free_keywords = step.output_data.get("keywords", "")
            self.logger.debug(f"Initialisation step output_data: {step.output_data}")
            self.logger.debug(f"Extracted free keywords: '{free_keywords}'")
            if hasattr(self, "initialisation_result"):
                # keywords is a string, not a list
                self.initialisation_result.setPlainText(free_keywords)
                self.logger.debug(
                    f"Set initialisation_result text to: '{free_keywords}'"
                )

            # Display working title after initialisation - Claude Generated
            if (self.pipeline_manager.current_analysis_state and
                hasattr(self.pipeline_manager.current_analysis_state, 'working_title') and
                self.pipeline_manager.current_analysis_state.working_title):
                working_title = self.pipeline_manager.current_analysis_state.working_title

                # Set working title in stream widget for log filename - Claude Generated
                if hasattr(self, 'stream_widget') and self.stream_widget:
                    self.stream_widget.set_working_title(working_title)

                self.logger.info(f"Displaying working title: {working_title}")
        elif step.step_id == "search" and step.output_data:
            # Populate the GND-hit table from the full search_results (not the
            # text-reduced gnd_treffer). Selection is marked later when the
            # keywords step finishes. - Claude Generated
            state = (
                self.pipeline_manager.current_analysis_state
                if self.pipeline_manager
                else None
            )
            search_results = getattr(state, "search_results", None) if state else None
            if search_results:
                # Fresh pool → clear chunk/final marks from a previous run so a
                # later non-chunked run can't inherit a stale ☑ Chunk tier. - Claude Generated
                self._populate_gnd_hits(search_results, reset_marks=True)

        elif step.step_id == "keywords" and step.output_data:
            final_keywords = step.output_data.get("final_keywords", "")
            self.logger.debug(f"Keywords step output_data: {step.output_data}")
            self.logger.debug(f"Final keywords: '{final_keywords}'")
            # Normalise to list for cross-check below - Claude Generated
            if isinstance(final_keywords, list):
                final_keywords_list = final_keywords
                final_keywords_text = "\n".join(final_keywords)
            else:
                final_keywords_text = str(final_keywords)
                final_keywords_list = [l.strip() for l in final_keywords_text.splitlines() if l.strip()]
            if hasattr(self, "keywords_result"):
                self.keywords_result.setPlainText(final_keywords_text)
                self.logger.debug(
                    f"Set keywords_result text to: '{final_keywords_text}'"
                )

            # ── Schlagwortketten mit Verifikation anzeigen ─────────────────── Claude Generated
            if hasattr(self, "keyword_chains_result"):
                llm_analysis = step.output_data.get("llm_analysis")
                chains = llm_analysis.keyword_chains if llm_analysis else []
                self._render_keyword_chains(chains, final_keywords_list)

            # Surface the chunk-survivor pool (chunked runs only) as the ☑ Chunk
            # tier before marking the ✅ Final tier — mirrors the agentic
            # selection_chunks → selection tiering so the GND-Recherche tab
            # visibly separates "gechunkt" from "ausgewählt". - Claude Generated
            llm_analysis = step.output_data.get("llm_analysis")
            chunk_survivors = (
                getattr(llm_analysis, "chunk_keywords", None) if llm_analysis else None
            )
            if chunk_survivors:
                self._mark_gnd_selection(chunk_survivors, tier="chunk")
            # Mark which GND-Recherche hits survived the selection step - Claude Generated
            self._mark_gnd_selection(final_keywords_list)

        elif step.step_id == "dk_search" and step.output_data:
            # Display DK search results with counts and titles - Claude Generated (Enhanced with filtering)
            # Use flattened DK-centric format for display (backward compatibility fallback to original)
            dk_search_results = step.output_data.get("dk_search_results_flattened",
                                                      step.output_data.get("dk_search_results", []))
            if hasattr(self, "dk_search_results"):
                if dk_search_results:
                    # Store raw data for filtering
                    self.dk_search_raw_data = dk_search_results

                    # Display results (will respect any active filter)
                    self._display_dk_search_results(dk_search_results)

                    # Update filter count if filter is active
                    if (hasattr(self, 'dk_search_filter_input') and
                        self.dk_search_filter_input.text().strip()):
                        self._filter_dk_search_results()
                else:
                    self.dk_search_raw_data = []
                    self.dk_search_results.setPlainText("Keine DK/RVK-Klassifikationen gefunden")

        elif step.step_id == "dk_classification" and step.output_data:
            # Display final DK classification results from LLM - Claude Generated
            dk_classifications = step.output_data.get("dk_classifications", [])
            if hasattr(self, "dk_classification_results"):
                if dk_classifications:
                    # Get dk_search_results from previous step for title display
                    dk_search_results = step.output_data.get("dk_search_results_flattened", [])

                    # Generate HTML display with titles
                    html_display = self._format_dk_classifications_with_titles(
                        dk_classifications,
                        dk_search_results
                    )
                    self.dk_classification_results.setHtml(html_display)
                else:
                    self.dk_classification_results.setPlainText("Keine DK/RVK-Klassifikationen generiert")

                # Also update the input summary with search data from previous step
                if hasattr(self, "dk_input_summary"):
                    search_data = step.output_data.get("dk_search_summary", "")
                    if search_data:
                        self.dk_input_summary.setPlainText(search_data)
                    else:
                        self.dk_input_summary.setPlainText("Katalog-Suchergebnisse für LLM-Analyse")

                # Update compact stats - Claude Generated
                if hasattr(self, "dk_compact_stats"):
                    stats = step.output_data.get("statistics")
                    if stats:
                        total = stats.get("total_classifications", 0)
                        dedup = stats.get("deduplication_stats", {})
                        orig = dedup.get("original_count", 0)
                        rate = dedup.get("deduplication_rate", "0%")
                        self.dk_compact_stats.setText(
                            f"📊 <b>Klassifikations-Statistik:</b> {orig} Katalogtreffer → <b>{total}</b> unikale Klassifikationen "
                            f"(Deduplizierungsrate: {rate})"
                        )

        # End any active streaming for this step
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_completed(step)
        
        # Emit results to other tabs based on step type - Claude Generated
        self._emit_step_results_to_tabs(step)

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        """Handle step error event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Stop live duration updates for this step
        if self.current_running_step == step.step_id:
            self.duration_update_timer.stop()
            self.current_running_step = None

        # Update global status bar
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "error"
                )

        self.pipeline_status_label.setText(f"Fehler: {step.name}")

        # End any active streaming for this step
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_error(step, error_message)

        QMessageBox.critical(
            self,
            "Pipeline-Fehler",
            f"Fehler in Schritt '{step.name}':\n{error_message}",
        )

        # Re-enable start button
        self.auto_pipeline_button.setEnabled(True)

    @pyqtSlot(str)
    def on_pipeline_error(self, error_message: str):
        """Handle pipeline-level failure that escaped step handling - Claude Generated

        Without this the worker thread dies silently and the UI stays in
        "Processing…" forever (see workers.py PipelineWorker.run).
        """
        if self.current_running_step:
            self.duration_update_timer.stop()
            self.current_running_step = None

        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    "Pipeline", "error"
                )

        self.pipeline_status_label.setText("Pipeline-Fehler")

        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        QMessageBox.critical(
            self,
            "Pipeline-Fehler",
            f"Die Pipeline ist mit einem Fehler abgebrochen:\n{error_message}",
        )

        self.auto_pipeline_button.setEnabled(True)

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        """Handle pipeline completion - Claude Generated"""
        # Stop any running timer
        self.duration_update_timer.stop()
        self.current_running_step = None

        self.pipeline_status_label.setText("Pipeline abgeschlossen ✓")
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)
        self.pipeline_completed.emit()

        # Stop status bar timer and progress
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "pipeline_progress"):
                self.main_window.global_status_bar.pipeline_progress.hide()
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    "Pipeline", "completed"
                )

        # Propagate working_title to stream widget (agentic: no on_step_completed fires)
        if analysis_state and hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            working_title = analysis_state.working_title
            if hasattr(self, 'stream_widget'):
                self.stream_widget.set_working_title(working_title)

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_completed(analysis_state)

        # Emit complete analysis_state for distribution to specialized tabs - Claude Generated
        if analysis_state:
            self.pipeline_results_ready.emit(analysis_state)
            # Sync classical step tabs from agentic result (no step_completed fires in agentic mode) - Claude Generated
            if (self.pipeline_manager and self.pipeline_manager.config
                    and self.pipeline_manager.config.enable_agentic_mode):
                self._sync_classical_tabs_from_state(analysis_state)

        # Optional: Auto-save after completion - Claude Generated
        if hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            from ..utils.pipeline_utils import export_analysis_state_to_file
            from ..utils.pipeline_defaults import autosave_filename, get_autosave_dir

            auto_save_dir = get_autosave_dir(getattr(self, 'config_manager', None))
            auto_save_dir.mkdir(parents=True, exist_ok=True)

            # Timestamped so two agentic runs of the same document don't overwrite
            # each other (the agentic working_title carries no timestamp of its
            # own, unlike the classic path's). - Claude Generated
            auto_save_file = auto_save_dir / autosave_filename(analysis_state.working_title)
            try:
                export_analysis_state_to_file(analysis_state, str(auto_save_file))
                self.logger.info(f"✅ Auto-saved pipeline result to: {auto_save_file}")
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")

        QMessageBox.information(
            self,
            "Pipeline abgeschlossen",
            "Die komplette Analyse-Pipeline wurde erfolgreich abgeschlossen!",
        )

    def _render_keyword_chains(self, chains: list, final_keywords_source) -> None:
        """Render Schlagwortketten with green/red verification into keyword_chains_result - Claude Generated"""
        if not hasattr(self, "keyword_chains_result"):
            return
        if not chains:
            self.keyword_chains_result.setPlainText("Keine Schlagwortketten in LLM-Antwort gefunden.")
            return

        import re as _re

        def _norm(kw: str) -> str:
            return _re.sub(r"\s*\(GND-ID:[^)]*\)", "", kw).strip().lower()

        def _esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Normalize final keywords – accepts list[str] or list[dict{keyword}]
        final_set: set = set()
        for item in (final_keywords_source or []):
            if isinstance(item, dict):
                final_set.add(_norm(item.get("keyword") or item.get("title") or ""))
            else:
                final_set.add(_norm(str(item)))

        blocks = []
        for c in chains:
            parts = c.get("chain", [])
            reason = c.get("reason", "")
            kw_html_parts = []
            all_present = True
            for kw in parts:
                if _norm(kw) in final_set:
                    kw_html_parts.append(f'<span style="color:#4caf50;font-weight:bold">{_esc(kw)}</span>')
                else:
                    kw_html_parts.append(f'<span style="color:#f44336;font-weight:bold">{_esc(kw)} ✗</span>')
                    all_present = False
            arrow = '<span style="color:#888"> → </span>'
            status_color = "#4caf50" if all_present else "#ff9800"
            block = (
                f'<p style="margin:4px 0 0 0">'
                f'<span style="color:{status_color};font-weight:bold">{"✓" if all_present else "⚠"} </span>'
                f'{arrow.join(kw_html_parts)}</p>'
            )
            if reason:
                block += f'<p style="margin:1px 0 6px 16px;color:#aaa;font-style:italic">{_esc(reason)}</p>'
            else:
                block += '<p style="margin:0 0 6px 0"></p>'
            blocks.append(block)

        self.keyword_chains_result.setHtml(
            '<html><body style="font-family:monospace">' + "".join(blocks) + "</body></html>"
        )

    @pyqtSlot(str, dict)
    def _on_agentic_step_snapshot(self, step_id: str, snapshot: dict) -> None:
        """Update classical step-tab widgets from agentic step completion snapshots - Claude Generated.

        Mapping: extraction→initialisation, search→search, selection→keywords,
                 classification→dk_classification, dk_collect→dk_search, dk_postprocess→dk_classification
        """
        status = snapshot.get("_step_status", "")
        if status not in ("completed", "error"):
            return  # skip running/pending intermediate emissions

        # Update PipelineStepWidget status indicator (▷▶✓✗) - Claude Generated
        mapping = self._AGENTIC_STEP_MAP.get(step_id)
        if mapping:
            widget_key, _ = mapping
            step_widget = self.step_widgets.get(widget_key)
            if step_widget:
                step_widget.step.status = status
                step_widget.update_status_display()

        try:
            if step_id == "extraction":
                keywords = snapshot.get("extracted_keywords", [])
                if keywords and hasattr(self, "initialisation_result"):
                    if isinstance(keywords, list):
                        # Tolerate the {"_truncated": N} sentinel in capped lists
                        text = "\n".join(k for k in keywords if isinstance(k, str))
                    else:
                        text = str(keywords)
                    self.initialisation_result.setPlainText(text)
                working_title = snapshot.get("working_title", "")
                if working_title:
                    if hasattr(self, "stream_widget"):
                        self.stream_widget.set_working_title(working_title)

            elif step_id == "search":
                gnd_entries = snapshot.get("gnd_entries", [])
                if gnd_entries and hasattr(self, "search_results_table"):
                    # Fresh pool → clear chunk/final marks from a previous run
                    self._populate_gnd_hits(gnd_entries, reset_marks=True)

            elif step_id == "selection_chunks":
                # Tier 2 of 3: chunk-filter survivors (☑) - Claude Generated
                chunk_kws = snapshot.get("selected_keywords", [])
                if chunk_kws:
                    self._mark_gnd_selection(chunk_kws, tier="chunk")

            elif step_id == "verify_keywords":
                # Tier 3 of 3: verified final keywords (✅), GND-IDs may have
                # been corrected against the pool/DB - Claude Generated
                verified = snapshot.get("extra", {}).get("final_keywords", [])
                if verified:
                    self._mark_gnd_selection(verified, tier="final")
                    if hasattr(self, "keywords_result"):
                        lines = []
                        for kw in verified:
                            if isinstance(kw, dict):
                                lines.append(
                                    f"{kw.get('keyword') or kw.get('title') or ''} (GND-ID: {kw.get('gnd_id', '')})"
                                )
                            else:
                                lines.append(str(kw))
                        self.keywords_result.setPlainText("\n".join(lines))

            elif step_id == "selection":
                final_kws = snapshot.get("extra", {}).get("final_keywords", [])
                if final_kws and hasattr(self, "keywords_result"):
                    lines = []
                    for kw in final_kws:
                        if isinstance(kw, dict):
                            lines.append(f"{kw.get('keyword') or kw.get('title') or ''} (GND-ID: {kw.get('gnd_id', '')})")
                        else:
                            lines.append(str(kw))
                    self.keywords_result.setPlainText("\n".join(lines))
                chains = snapshot.get("keyword_chains", [])
                if chains:
                    self._render_keyword_chains(chains, final_kws)
                # Mark which GND-Recherche hits survived selection - Claude Generated
                if final_kws:
                    self._mark_gnd_selection(final_kws)

            elif step_id == "classification":
                # classification step produces dk_classifications BEFORE
                # dk_postprocess — at this point context.dk_search_results
                # still holds the keyword-centric list from dk_collect, which
                # has no top-level ``dk`` keys. ``get_titles_for_dk_code``
                # would return empty for every code, producing a "notations
                # without title assignments" view that is then immediately
                # overwritten by the dk_postprocess per-step. Skip the
                # render here and let dk_postprocess own the display. - Claude Generated
                pass

            elif step_id == "dk_collect":
                dk_results = snapshot.get("dk_search_results", [])
                if dk_results and hasattr(self, "dk_search_results"):
                    self.dk_search_raw_data = dk_results
                    self._display_dk_search_results(dk_results)

            elif step_id == "dk_postprocess":
                # dk_postprocess writes the DK-centric rich list to
                # context.dk_search_results (via build_dk_search_results)
                # BEFORE this snapshot is emitted, so the per-step sees
                # the rich source. Defensive: if for some reason
                # snapshot["dk_search_results"] is empty/missing, skip the
                # render rather than blank the widget — the end-of-pipeline
                # _sync_classical_tabs_from_state owns the final state. - Claude Generated
                dk_results = snapshot.get("dk_search_results", [])
                dk_class = snapshot.get("dk_classifications", [])
                if dk_results and hasattr(self, "dk_search_results"):
                    self.dk_search_raw_data = dk_results
                    self._display_dk_search_results(dk_results)
                if dk_class and dk_results and hasattr(self, "dk_classification_results"):
                    codes = self._dk_class_codes(dk_class)
                    html_display = self._format_dk_classifications_with_titles(
                        codes, dk_results
                    )
                    self.dk_classification_results.setHtml(html_display)

        except Exception as e:
            self.logger.warning(f"_on_agentic_step_snapshot({step_id}) failed: {e}")

    def _sync_classical_tabs_from_state(self, state) -> None:
        """Populate classical step-tab widgets from analysis_state after agentic run.

        In agentic mode step_completed never fires, so this fills the same widgets
        that on_step_completed() would normally update. Each section is guarded
        independently — a failure in the GND part must not silently skip the
        DK displays (previously one broad try aborted the whole sync). - Claude Generated
        """
        try:
            # Init tab: extracted keywords
            if state.initial_keywords and hasattr(self, "initialisation_result"):
                self.initialisation_result.setPlainText("\n".join(state.initial_keywords))

            # Search tab: show ALL GND hits (not just search terms), with the
            # final selection marked so deselected hits can be filtered. - Claude Generated
            if state.search_results and hasattr(self, "search_results_table"):
                selected = (
                    state.final_llm_analysis.extracted_gnd_keywords
                    if state.final_llm_analysis
                    else None
                )
                self._populate_gnd_hits(state.search_results, selected=selected)

            # Keywords tab: final GND keywords
            if hasattr(self, "keywords_result"):
                final_kws = []
                if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                    final_kws = state.final_llm_analysis.extracted_gnd_keywords
                elif state.initial_keywords:
                    final_kws = state.initial_keywords
                if final_kws:
                    text = "\n".join(final_kws) if isinstance(final_kws, list) else str(final_kws)
                    self.keywords_result.setPlainText(text)
        except Exception as e:
            self.logger.warning(f"_sync_classical_tabs_from_state (GND/keywords part) failed: {e}")

        try:
            # DK search + classification tabs.
            # In agentic mode the rich DK-centric catalog data (real titles +
            # counts) lives in state.dk_search_results — written by
            # build_dk_search_results / dk_postprocess and identical to what the
            # per-step snapshot shows during the run. state.dk_search_results_flattened
            # is only a thin structure derived from the final classifications
            # (titles = DK label, count = confidence×100), which clears the
            # Katalog-Recherche view and drops the title list. Prefer the rich
            # source; fall back to flattened only if it is empty. - Claude Generated
            dk_rich = PipelineResultFormatter.select_dk_title_source(
                getattr(state, "dk_search_results", None),
                getattr(state, "dk_search_results_flattened", None),
            )

            # If ``dk_rich`` has no displayable titles (e.g. dk_postprocess
            # was skipped in v5.1 because ``when: ${extra.dk_prompt_text}
            # != ''`` evaluated false, leaving only the thin flattened
            # source) fall back to the flattened list so the final DK/RVK
            # notations still get a title column rather than bare code
            # headings. ``select_dk_title_source`` already prefers the rich
            # source, so reaching this branch means the rich source was
            # either empty or had no titles. - Claude Generated
            if dk_rich and state.dk_classifications and not any(
                r.get("titles") for r in dk_rich if isinstance(r, dict)
            ):
                flattened_fallback = getattr(state, "dk_search_results_flattened", None) or []
                if any(
                    r.get("titles") for r in flattened_fallback if isinstance(r, dict)
                ):
                    self.logger.warning(
                        "_sync_classical_tabs_from_state: rich dk_search_results has "
                        "no titles, falling back to dk_search_results_flattened for "
                        "DK classification title lookup"
                    )
                    dk_rich = flattened_fallback

            if dk_rich and hasattr(self, "dk_search_results"):
                # Final sync must never CLEAR what the per-step snapshots
                # already rendered: only overwrite when the end-of-run data
                # actually formats to displayable text - Claude Generated
                if PipelineResultFormatter.format_dk_search_results_text(dk_rich):
                    self.dk_search_raw_data = dk_rich
                    self._display_dk_search_results(dk_rich)
                else:
                    self.logger.warning(
                        "_sync_classical_tabs_from_state: end-of-run DK data has no "
                        "displayable titles — keeping snapshot content in "
                        "Katalog-Recherche view"
                    )

            # DK classification tab — state.dk_classifications may be List[Dict]
            if state.dk_classifications and hasattr(self, "dk_classification_results"):
                codes = self._dk_class_codes(state.dk_classifications)
                html_display = self._format_dk_classifications_with_titles(
                    codes,
                    dk_rich,
                )
                self.dk_classification_results.setHtml(html_display)

            # DK compact stats
            if state.dk_statistics and hasattr(self, "dk_compact_stats"):
                stats = state.dk_statistics
                total = stats.get("total_classifications", 0)
                dedup = stats.get("deduplication_stats", {})
                orig = dedup.get("original_count", 0)
                rate = dedup.get("deduplication_rate", "0%")
                self.dk_compact_stats.setText(
                    f"📊 <b>Klassifikations-Statistik:</b> {orig} Katalogtreffer → "
                    f"<b>{total}</b> unikale Klassifikationen (Deduplizierungsrate: {rate})"
                )
        except Exception as e:
            self.logger.warning(f"_sync_classical_tabs_from_state (DK part) failed: {e}")

    def on_abort_current_step_requested(self):
        """Abort only the current LLM generation; pipeline continues - Claude Generated"""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.logger.info("User requested step-only abort (pipeline continues)")
            self.pipeline_worker.abort_current_step()

    def on_stop_pipeline_requested(self):
        """Handle stop button click - Claude Generated"""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.logger.info("User requested pipeline stop")
            self.stop_pipeline_button.setEnabled(False)
            self.stop_pipeline_button.setText("⏹ Stopping...")
            self.pipeline_status_label.setText("Beende Pipeline...")
            self.pipeline_worker.request_stop()

    @pyqtSlot()
    def on_pipeline_aborted(self):
        """Handle pipeline abort signal - Claude Generated"""
        self.logger.info("Pipeline aborted by user")

        # Stop any running timer
        self.duration_update_timer.stop()
        self.current_running_step = None

        # Reset button states
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setText("⏹️ Stop")
        self.stop_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)

        # Update status
        self.pipeline_status_label.setText("Pipeline abgebrochen")
        self.pipeline_status_label.setStyleSheet(
            "color: #FF9800; font-weight: bold; padding: 5px; "
            "background-color: #FFF3E0; border: 1px solid #FFB74D; border-radius: 3px;"
        )

        # End any active streaming
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Note: Removed QMessageBox - status label provides sufficient feedback - Claude Generated

    @pyqtSlot(str, str)
    def on_llm_stream_token(self, token: str, step_id: str):
        """Handle streaming LLM token - Claude Generated"""
        self.logger.debug(f"Received streaming token for {step_id}: '{token[:20]}...'")
        if hasattr(self, "stream_widget"):
            # Start streaming line if not already started
            if not self.stream_widget.is_streaming:
                self.logger.debug(f"Starting streaming for step {step_id}")
                self.stream_widget.start_llm_streaming(step_id)

            # Add the token to the streaming display
            self.stream_widget.add_streaming_token(token, step_id)

            # End streaming if we get a final token (this would need refinement based on actual LLM response patterns)
            # For now, we'll leave the line open and let the step completion handle ending

    def on_repetition_detected(self, result, suggestions: list, grace_period: bool, resolved: bool, grace_seconds: float):
        """Handle repetition detection from LLM - Claude Generated (2026-02-17)

        Args:
            result: RepetitionResult object (None if resolved)
            suggestions: List of parameter variation suggestions
            grace_period: True if grace period active
            resolved: True if repetition resolved during grace period
            grace_seconds: Grace period duration in seconds
        """
        if resolved:
            # Repetition resolved - hide warning
            self.stream_widget.hide_repetition_warning(resolved=True)
        elif result:
            # Repetition detected - show warning
            detection_type = result.detection_type
            details = result.details
            self.stream_widget.show_repetition_warning(
                detection_type=detection_type,
                details=details,
                suggestions=suggestions,
                grace_period=grace_period,
                grace_seconds=grace_seconds
            )

