"""Result-distribution + bus-handler mixin for MainWindow. Claude Generated.

Extracted from ``main_window.py`` (F-5 god-file split): pipeline-result fan-out to
the specialized tabs, the AlimaStateBus handlers (pipeline_completed/state_changed
with coalesced re-render), intermediate-analysis routing, window-title updates,
and search-selection / GND-keyword / search-field sync. Method bodies are moved
verbatim. ``MainWindowResultsMixin`` is mixed into ``MainWindow``; not a
standalone window.
"""
from __future__ import annotations

import re

from PyQt6.QtCore import pyqtSlot


class MainWindowResultsMixin:
    """Pipeline-result distribution + bus handlers (verbatim from MainWindow)."""

    @pyqtSlot(object)
    def on_pipeline_results_ready(self, analysis_state):
        """Central slot for distributing pipeline results to specialized tabs - Claude Generated"""
        self.logger.debug("Distributing pipeline results to specialized tabs")

        # 1. Abstract-Analyse Tab: Set abstract + initial keywords + LLM response
        if analysis_state.original_abstract:
            self.abstract_tab.set_abstract(analysis_state.original_abstract)

        if analysis_state.initial_keywords:
            # Robust type handling for keywords - Claude Generated
            if isinstance(analysis_state.initial_keywords, list):
                keywords_str = ", ".join(analysis_state.initial_keywords)
            else:
                # Handle case where it's already a string
                keywords_str = str(analysis_state.initial_keywords)
            self.abstract_tab.set_keywords(keywords_str)

        # Ensure history is updated during live pipeline runs - Claude Generated
        # Pass the correct step result for each tab to avoid cross-tab contamination - Claude Generated
        if analysis_state:
            init_text = (
                analysis_state.initial_llm_call_details.response_full_text
                if hasattr(analysis_state, 'initial_llm_call_details') and analysis_state.initial_llm_call_details
                else None
            )
            self.abstract_tab.add_external_analysis_to_history(analysis_state, result_text=init_text)
            self.logger.debug("Updated AbstractTab history with initialisation result")

        if hasattr(analysis_state, 'initial_llm_call_details') and analysis_state.initial_llm_call_details:
            llm_response = analysis_state.initial_llm_call_details.response_full_text
            if llm_response:
                self.abstract_tab.display_llm_response(llm_response)

        # 2. GND-Suche Tab: Set initial keywords + display search results
        if analysis_state.initial_keywords:
            # Type-safe join - Claude Generated (Fix for string parsing bug)
            if isinstance(analysis_state.initial_keywords, list):
                keywords_str = ", ".join(analysis_state.initial_keywords)
            else:
                keywords_str = str(analysis_state.initial_keywords)
            self.search_tab.update_search_field(keywords_str)

        if analysis_state.search_results:
            # Handle both Dict and List[SearchResult] formats - Claude Generated
            if isinstance(analysis_state.search_results, dict):
                # Already in correct format
                self.search_tab.display_search_results(analysis_state.search_results)
            else:
                # Convert List[SearchResult] to Dict format
                search_results_dict = {
                    sr.search_term: sr.results
                    for sr in analysis_state.search_results
                }
                self.search_tab.display_search_results(search_results_dict)

        # 3. Manuelle Analyse / Verifikation view — P-θ.2: same tab as block 1.
        # `self.analyse_keywords is self.abstract_tab` (aliased). Calls below
        # update GND-keyword view + final-LLM-history entry on the merged tab.
        if analysis_state.original_abstract:
            self.analyse_keywords.set_abstract(analysis_state.original_abstract)

        # Extract GND keywords from search_results
        if analysis_state.search_results:
            gnd_keywords = []
            # Handle both Dict and List[SearchResult] formats - Claude Generated
            if isinstance(analysis_state.search_results, dict):
                # Dict format: {search_term: {keyword: data}}
                for results in analysis_state.search_results.values():
                    for keyword, data in results.items():
                        gnd_ids = data.get("gndid", set())
                        for gnd_id in gnd_ids:
                            gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")
            else:
                # List[SearchResult] format
                for search_result in analysis_state.search_results:
                    for keyword, data in search_result.results.items():
                        gnd_ids = data.get("gndid", set())
                        for gnd_id in gnd_ids:
                            gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")
            if gnd_keywords:
                self.analyse_keywords.set_keywords("\n".join(gnd_keywords))

        # Display final LLM analysis if available
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            if hasattr(analysis_state.final_llm_analysis, 'response_full_text'):
                keywords_text = analysis_state.final_llm_analysis.response_full_text
                self.analyse_keywords.display_llm_response(keywords_text)
                # Add to history with the correct (keywords step) result text - Claude Generated
                self.analyse_keywords.add_external_analysis_to_history(analysis_state, result_text=keywords_text)

        # 4. UB-Katalog Tab: GND keywords auto-filled via pipeline_results_ready signal
        # (wired to ub_catalog_tab.update_from_pipeline — Claude Generated)

        # 5. 📚 DK-Zuordnung & DK-Statistik Tab - Claude Generated
        if analysis_state.original_abstract:
            self.dk_analysis_tab.set_abstract(analysis_state.original_abstract)

        # Distribute flattened DK results to dk_analysis_tab for display
        if hasattr(analysis_state, 'dk_search_results_flattened') and analysis_state.dk_search_results_flattened:
            # Use flattened (DK-centric) results with titles for display
            self.dk_analysis_tab.set_keywords(analysis_state.dk_search_results_flattened)
        elif hasattr(analysis_state, 'dk_search_results') and analysis_state.dk_search_results:
            # Fallback to keyword-centric if flattened not available
            self.dk_analysis_tab.set_keywords(analysis_state.dk_search_results)

        if hasattr(analysis_state, 'dk_llm_analysis') and analysis_state.dk_llm_analysis:
            dk_text = analysis_state.dk_llm_analysis.response_full_text
            self.dk_analysis_tab.display_llm_response(dk_text)
            # Pass DK-specific result text so history shows classification, not keywords - Claude Generated
            self.dk_analysis_tab.add_external_analysis_to_history(analysis_state, result_text=dk_text)

        if hasattr(analysis_state, 'classifications') and analysis_state.classifications:
            self.dk_classification_tab.update_data(analysis_state)
            self.logger.info("✅ DK statistics and analysis tabs populated with pipeline results.")

        # 6. 📊 Analyse-Review Tab - Sende finale Ergebnisse (lossless) - Claude Generated
        if analysis_state.final_llm_analysis or analysis_state.classifications:
            self.analysis_review_tab.receive_full_state(analysis_state)
            self.logger.info("✅ Analysis review tab populated with full pipeline state (lossless).")

        # 6. 📚 DK-Klassifikation (Optional) - Claude Generated
        # Note: DK search results and classifications are handled by show_loaded_state_indicator()
        # when called from populate_all_tabs_from_state(). For live pipeline results, they are
        # displayed directly in pipeline_tab via on_step_completed() callbacks.
        if analysis_state.dk_search_results:
            self.logger.info(f"DK search results available: {len(analysis_state.dk_search_results)} entries")
        if analysis_state.classifications:
            self.logger.info(f"Classifications available: {len(analysis_state.classifications)} entries")

        self.logger.info("Pipeline results successfully distributed to all tabs")

        # NB: intentionally do NOT switch tabs on completion. Results are
        # distributed to the specialized tabs above, but the user stays in the
        # Pipeline tab (chat / pipeline-logger) to read the run. Claude Generated.

        # Classical distribution finished — release the guard so subsequent
        # agent-driven bus completions are distributed by the bus handler. The
        # bus state.pipeline_completed event for this run was already delivered
        # (and skipped) before this signal handler ran. - Claude Generated
        self._classical_active = False

        # P-δ.5a: chat_dock retired. PipelineChatPanel is always visible
        # in PipelineTab — load_context is wired via pipeline_results_ready
        # signal above (and also called from on_pipeline_completed inside
        # the panel itself).

    # ------------------------------------------------------------------
    # Konvergenz Pipeline/Agent: bus-driven result-tab refresh
    # ------------------------------------------------------------------
    def _on_classical_pipeline_started(self, *args) -> None:
        """Mark that a classical GUI run owns tab distribution. - Claude Generated

        While active, ``_on_bus_pipeline_completed`` defers to the
        ``pipeline_results_ready`` signal path (which appends analysis history
        and auto-navigates). Agent-driven runs never emit this signal, so the
        flag stays False and the bus handler distributes for them.
        """
        self._classical_active = True

    def _on_bus_pipeline_completed(self, payload: dict) -> None:
        """Distribute agent-driven pipeline completions to the result tabs. - Claude Generated

        For classical GUI runs (``_classical_active``) the
        ``pipeline_results_ready`` signal already handles distribution, so we
        skip here to avoid double-rendering and duplicate history entries.
        """
        if self._classical_active:
            return
        state = getattr(self.pipeline_manager, "current_analysis_state", None)
        if not state:
            return
        try:
            # Mirror the full classical distribution (incl. analysis history)
            # so an agent `run_pipeline` looks identical to a manual run.
            self.on_pipeline_results_ready(state)
            # The direct pipeline_results_ready→slot connections only fire on
            # the signal path; replicate them here for the agent path.
            self.dk_classification_tab.update_data(state)
            self.ub_catalog_tab.update_from_pipeline(state)
            self.search_tab.update_data(state)
            self.on_pipeline_title_update(state)
            self.comparison_tab.load_from_current(state)
        except Exception:
            self.logger.exception(
                "MainWindow: bus pipeline-completed distribution failed"
            )

    def _on_bus_state_changed(self, payload: dict) -> None:
        """Coalesce incremental state.changed events into one tab re-render. - Claude Generated

        Fired by KeywordAnalysisState mutation methods (agent keyword/DK
        proposals, rerun_step). Unlike a completion these must NOT append a new
        analysis-history entry, so a separate idempotent re-render is used.
        """
        try:
            op = (payload or {}).get("op")
            if op:
                self._pending_ops.add(op)
            self._rerender_timer.start()
        except Exception:
            self.logger.exception("MainWindow: bus state-changed handling failed")

    def _flush_rerender(self) -> None:
        """Run the coalesced idempotent re-render after the debounce window. - Claude Generated"""
        ops = self._pending_ops
        self._pending_ops = set()
        state = getattr(self.pipeline_manager, "current_analysis_state", None)
        if not state:
            return
        self._rerender_result_tabs(state, ops=ops or None)

    def _rerender_result_tabs(self, state, ops=None) -> None:
        """Idempotent re-render of the result tabs WITHOUT appending history. - Claude Generated

        Used for incremental agent updates so the GND-search, DK and review tabs
        reflect the current ``KeywordAnalysisState`` regardless of whether the
        pipeline or the chat agent produced it. ``ops`` (mutation op-tags) is
        advisory; a full refresh is harmless because every call below replaces —
        never appends — the displayed data.
        """
        if not state:
            return
        try:
            # GND search-results transparency view (replaces stored state).
            self.search_tab.update_data(state)
            # UB-catalog keyword auto-fill (sets keywords; no network search).
            self.ub_catalog_tab.update_from_pipeline(state)
            # DK table + last LLM response WITHOUT add_external_analysis_to_history
            # (which dk_classification_tab.update_data would trigger).
            flat = (getattr(state, "dk_search_results_flattened", None)
                    or getattr(state, "dk_search_results", None))
            if flat:
                self.dk_analysis_tab.set_keywords(flat)
            dk_llm = getattr(state, "dk_llm_analysis", None)
            if dk_llm and getattr(dk_llm, "response_full_text", None):
                self.dk_analysis_tab.display_llm_response(dk_llm.response_full_text)
            # Lossless review view (replaces current state).
            self.analysis_review_tab.receive_full_state(state)
            # Keep comparison "current" slot and window title in sync.
            self.comparison_tab.load_from_current(state)
            self.on_pipeline_title_update(state)
        except Exception:
            self.logger.exception("MainWindow: _rerender_result_tabs failed")

    @pyqtSlot(object)
    def on_intermediate_analysis_ready(self, analysis_result):
        """Update AbstractTab during live pipeline run - Claude Generated"""
        if not analysis_result:
            return

        self.logger.info("Updating AbstractTab with intermediate pipeline result")

        # 1. Route intermediate result to the correct tab based on task_name - Claude Generated
        task_name = getattr(analysis_result, 'task_name', None)
        if task_name in ["dk_class", "dk_classification"]:
            target_tab = self.dk_analysis_tab
        elif task_name in ["keywords", "rephrase", "keywords_chunked"]:
            target_tab = self.analyse_keywords
        else:
            target_tab = self.abstract_tab

        if hasattr(analysis_result, 'full_text'):
            target_tab.display_llm_response(analysis_result.full_text)
        elif hasattr(analysis_result, 'response_full_text'):
            target_tab.display_llm_response(analysis_result.response_full_text)

        # 2. Update keywords if available
        # Don't overwrite dk_analysis_tab keywords — its keywords_edit holds
        # DK search results as INPUT for analysis, not as output display - Claude Generated
        if target_tab is not self.dk_analysis_tab:
            if hasattr(analysis_result, 'matched_keywords') and analysis_result.matched_keywords:
                keywords_str = ", ".join(analysis_result.matched_keywords.keys())
                target_tab.set_keywords(keywords_str)
            elif hasattr(analysis_result, 'extracted_gnd_keywords') and analysis_result.extracted_gnd_keywords:
                keywords_str = ", ".join(analysis_result.extracted_gnd_keywords)
                target_tab.set_keywords(keywords_str)
            elif hasattr(analysis_result, 'extracted_gnd_classes') and analysis_result.extracted_gnd_classes:
                classes_str = ", ".join(analysis_result.extracted_gnd_classes)
                target_tab.set_keywords(classes_str)

    def update_window_title(self, arbeitstitel: str = None):
        """Update window title with optional work title - Claude Generated"""
        if arbeitstitel:
            self.setWindowTitle(f"ALIMA - {arbeitstitel}")
        else:
            self.setWindowTitle("ALIMA - Automatisierte Schlagwortgenerierung")

    @pyqtSlot(object)
    def on_pipeline_title_update(self, analysis_state):
        """Update window title with working title from analysis state - Claude Generated"""
        if analysis_state and hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            self.update_window_title(analysis_state.working_title)
            self.logger.info(f"Window title updated: {analysis_state.working_title}")

    @pyqtSlot(dict)
    def on_search_selection_changed(self, changes):
        """Handle GND selection changes from SearchTab - Claude Generated

        Receives user modifications to GND keyword selection and propagates
        them to AnalysisReviewTab for final export integration.

        Args:
            changes: Dict with 'modified' (selection changes) and 'manual' (additions)
        """
        modified = changes.get('modified', {})
        manual = changes.get('manual', [])

        self.logger.info(
            f"GND selection changed: {len(modified)} modifications, {len(manual)} manual additions"
        )

        # TODO: Update analysis state with modifications
        # For now, just log the changes and show a notification
        if modified or manual:
            total_changes = len(modified) + len(manual)
            self.global_status_bar.show_notification(
                f"✅ GND-Auswahl aktualisiert: {total_changes} Änderungen",
                duration=3000
            )

        # Future enhancement: Propagate to AnalysisReviewTab
        # if hasattr(self, 'analysis_review_tab'):
        #     self.analysis_review_tab.update_gnd_selection(changes)

    @pyqtSlot(str)
    def update_gnd_keywords(self, keywords):
        self.logger.info(keywords)
        """
        Extrahiert GND-Schlagworte aus einem Text.
        
        Args:
            text (str): Der Text, der die GND-Einträge enthält
            
        Returns:
            list: Liste der GND-Schlagworte ohne IDs
        """
        self.logger.info(keywords)
        try:
            # Suche den Abschnitt mit den GND-Einträgen
            if "Schlagworte OGND Eintrage:" not in keywords:
                return []

            # Extrahiere den relevanten Teil des Textes
            gnd_section = keywords.split("Schlagworte OGND Eintrage:")[1].split(
                "FEHLENDE KONZEPTE:"
            )[0]

            # Extrahiere die Schlagworte (alles vor der URL)
            gnd_terms = []
            for line in gnd_section.split(","):
                if "(https://" in line:
                    term = line.split("(https://")[0].strip().replace('"', "")

                    self.logger.info(term)
                    if term:
                        gnd_terms.append(term)
            self.logger.info(gnd_terms)
            self.ub_catalog_tab.update_keywords(keywords)
            return gnd_terms

        except Exception as e:
            self.logger.error(f"Fehler beim Extrahieren der GND-Terme: {str(e)}")
            return []

    @pyqtSlot(str)
    def update_search_field(self, keywords):
        """
        Extrahiert Keywords aus einem String, behandelt geklammerte Terme und schützt Slashes.

        Args:
            keyword_string (str): String mit kommagetrennten Keywords in Anführungszeichen

        Returns:
            list: Liste von extrahierten Keywords
            list: Liste von extrahierten Klammer-Termen
        """

        # Entferne Leerzeichen am Anfang und Ende
        keyword_string = keywords.strip()

        # Wenn der String mit [ beginnt und mit ] endet, entferne diese
        if keyword_string.startswith("[") and keyword_string.endswith("]"):
            keyword_string = keyword_string[1:-1]

        # Regulärer Ausdruck für das Matching
        # Matches entweder:
        # 1. Terme in Anführungszeichen die Slash enthalten
        # 2. Terme in Anführungszeichen mit Klammern
        # 3. Normale Terme in Anführungszeichen
        pattern = r'"([^"]*?/[^"]*?)"|"([^"]*?\([^)]+?\)[^"]*?)"|"([^"]*?)"'

        matches = re.finditer(pattern, keyword_string)

        keywords = []
        bracketed_terms = []

        for match in matches:
            # Wenn es ein Slash-Term ist
            if match.group(1):
                term = match.group(1).replace("/", "\\/")
                keywords.append(f'"{term}"')
            # Wenn es ein Term mit Klammern ist
            elif match.group(2):
                term = match.group(2)
                # Extrahiere den Klammerinhalt
                bracket_content = re.findall(r"\(([^)]+)\)", term)
                # Füge den Hauptterm zu den Keywords hinzu
                main_term = re.sub(r"\s*\([^)]+\)", "", term).strip()
                if main_term:
                    keywords.append(main_term)
                # Füge die Klammerterme zur separaten Liste hinzu
                bracketed_terms.extend([f'"{term}"' for term in bracket_content])
            # Wenn es ein normaler Term ist
            elif match.group(3):
                keywords.append(f'"{match.group(3)}"')

        result = keywords + bracketed_terms
        self.search_tab.update_search_field(", ".join(result))

