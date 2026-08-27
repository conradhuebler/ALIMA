"""Run-control / config / workflow mixin for PipelineTab. Claude Generated.

Extracted from ``pipeline_tab.py`` (F-5 god-file split): starting the pipeline
worker, global provider/model + think overrides, the config dialog + save/load,
workflow-combo population + selection, agentic-panel rebuild, catalog config,
and result emission to other tabs. Method bodies are moved verbatim.
``PipelineTabControlMixin`` is mixed into ``PipelineTab`` (which provides
``__init__``, ``setup_ui`` and the widgets/state these drive); not a standalone
widget.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from ..core.pipeline_manager import PipelineConfig, PipelineStep
from .pipeline_config_dialog import PipelineConfigDialog
from .workers import PipelineWorker


class PipelineTabControlMixin:
    """Run control + config + workflow (verbatim from PipelineTab)."""

    def start_auto_pipeline(self):
        """Start the automatic pipeline in background thread - Claude Generated"""
        # Get input text — prefer confirmed value, fall back to whatever is in text_display.
        # This handles the case where the user pastes text directly without clicking "Text verwenden". - Claude Generated
        input_text = getattr(self, "current_input_text", "")
        if not input_text and hasattr(self, 'unified_input'):
            input_text = self.unified_input.text_display.toPlainText().strip()

        if not input_text:
            QMessageBox.warning(
                self,
                "Keine Eingabe",
                "Bitte wählen Sie eine Eingabequelle und stellen Sie Text bereit.",
            )
            return

        # Apply LLM model selection + DK config - Claude Generated
        self._apply_global_override_from_gui()
        self._update_dk_config_from_gui()
        # Ensure run mode matches the workflow picker (agentic vs. classic).
        self._apply_workflow_selection()

        # Stop any existing worker
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.quit()
            self.pipeline_worker.wait()

        # Reset streaming widget for new pipeline
        if hasattr(self, "stream_widget"):
            self.stream_widget.reset_for_new_pipeline()

        # Update status and button visibility - Claude Generated
        self.pipeline_status_label.setText("Pipeline läuft...")
        self.auto_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setVisible(True)

        # Get force_update flag from checkbox - Claude Generated
        force_update = getattr(self, 'force_update_checkbox', None)
        force_update_enabled = force_update.isChecked() if force_update else False

        # Determine source metadata for working title - Claude Generated
        # Priority: cached value from last text_ready > widget's current_source_type > doi_url_input field
        input_type = getattr(self, 'current_input_type', 'text')
        input_source = getattr(self, 'current_input_source', '')
        if input_type == 'text' and hasattr(self, 'unified_input'):
            # Read fresh from widget — set by extract_text() on last DOI/PDF/image resolution
            widget_type = getattr(self.unified_input, 'current_source_type', 'text')
            widget_data = getattr(self.unified_input, 'current_source_data', '')
            if widget_type != 'text' and widget_data:
                input_type = widget_type
                input_source = widget_data
            else:
                # Last fallback: read the DOI input field directly
                doi_val = self.unified_input.doi_url_input.text().strip()
                if doi_val:
                    input_type = 'url' if doi_val.startswith(('http://', 'https://')) else 'doi'
                    input_source = doi_val

        # Rebuild agentic-context panels from the selected workflow so the
        # widget reflects the *current* workflow instead of the hardcoded
        # 5-step pipeline view. Only relevant when agentic mode is on.
        self._rebuild_agentic_panels()

        # Create and start worker thread - Claude Generated
        self.pipeline_worker = PipelineWorker(
            self.pipeline_manager, input_text,
            input_type=input_type,
            input_source=input_source,
            force_update=force_update_enabled
        )

        # Connect worker signals
        self.pipeline_worker.step_started.connect(self.on_step_started)
        self.pipeline_worker.step_completed.connect(self.on_step_completed)
        self.pipeline_worker.step_error.connect(self.on_step_error)
        self.pipeline_worker.pipeline_error.connect(self.on_pipeline_error)  # Claude Generated
        self.pipeline_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.stream_token.connect(self.on_llm_stream_token)
        self.pipeline_worker.aborted.connect(self.on_pipeline_aborted)  # Claude Generated
        self.pipeline_worker.repetition_detected.connect(self.on_repetition_detected)  # Claude Generated (2026-02-17)
        # Forward agentic context updates to MainWindow dock via signal - Claude Generated
        self.pipeline_worker.agentic_context_updated.connect(self.agentic_context_updated)
        # Also update classical step tabs from agentic snapshots in real-time - Claude Generated
        self.pipeline_worker.agentic_context_updated.connect(self._on_agentic_step_snapshot)

        # Start the worker
        self.pipeline_worker.start()

        # Emit pipeline started signal
        self.pipeline_started.emit("pipeline_thread")

        # NOTE: the chat panel's "🚀 Pipeline gestartet" line is driven by the
        # state.pipeline_started bus event (with the real pipeline UUID). The
        # previous explicit stream_widget.on_pipeline_started("pipeline_thread")
        # call here produced a duplicate started-banner with a placeholder ID. - Claude Generated

    def _apply_global_override_from_gui(self):
        """Apply LLM model selection to pipeline config (global_provider/model_override) - Claude Generated"""
        if not hasattr(self, 'global_override_selector'):
            return
        config = self.pipeline_manager.config
        if not config:
            return
        # A complete (provider, model) pick is an override; "-- Standard --" (and
        # an incomplete provider-only pick) yields a falsy member → fall through to
        # the baseline. Requiring both avoids an invalid provider/model mix.
        provider, model = self.global_override_selector.get_selection()
        think_override = self._get_global_think_override()
        budget_override = self._get_global_max_tokens_override()
        if provider and model:
            config.global_provider_override = provider
            config.global_model_override = model
            config.global_think_override = think_override
            config.global_max_tokens_override = budget_override
            # Propagate into the per-step configs. Setting the attribute alone is
            # not enough: apply_global_override() only runs in __post_init__, so a
            # runtime selection here would be ignored and every LLM step would keep
            # its per-step pipeline default (e.g. gemma). The classic executor reads
            # step_config.provider/model directly (get_step_config). - Claude Generated
            config.apply_global_override()
            self.logger.info(f"🤖 LLM selected: {provider}/{model} think={think_override}")
        else:
            # Back to "-- Standard --": clear the override AND rebuild the per-step
            # baseline (pipeline default + task preferences). Without the rebuild a
            # previously applied override would stay mutated into step_configs and
            # stick. The think-only override is overlaid after the rebuild. - Claude Generated
            config.global_provider_override = None
            config.global_model_override = None
            self.pipeline_manager.reload_config()
            config = self.pipeline_manager.config
            if config:
                config.global_think_override = think_override
                config.global_max_tokens_override = budget_override
                if think_override is not None:
                    config.apply_global_override()

    def _get_global_think_override(self):
        """Read the global thinking override combo → None/True/False - Claude Generated"""
        if not hasattr(self, 'global_think_combo'):
            return None
        return {0: None, 1: True, 2: False}.get(self.global_think_combo.currentIndex())

    def _get_global_max_tokens_override(self):
        """Read the token-budget spinbox → None or the budget - Claude Generated.

        The spinbox shows "Standard" at 0 (``setSpecialValueText``); that means
        "leave every step the budget its workflow YAML names".
        """
        if not hasattr(self, 'global_max_tokens_spin'):
            return None
        return self.global_max_tokens_spin.value() or None

    def _update_dk_config_from_gui(self):
        """
        Update DK pipeline configuration from GUI widgets - Claude Generated
        Applies current GUI spinner values to PipelineManager configuration
        """
        if not hasattr(self.pipeline_manager, 'config') or not self.pipeline_manager.config:
            return

        config = self.pipeline_manager.config

        # Update dk_search step config
        if 'dk_search' in config.step_configs:
            dk_search_config = config.step_configs['dk_search']
            if hasattr(self, 'dk_search_max_results'):
                dk_search_config.custom_params['max_results'] = self.dk_search_max_results.value()

        # Update dk_classification step config
        if 'dk_classification' in config.step_configs:
            dk_classification_config = config.step_configs['dk_classification']
            if hasattr(self, 'dk_frequency_threshold'):
                dk_classification_config.custom_params['dk_frequency_threshold'] = self.dk_frequency_threshold.value()

        self.logger.info(
            f"✅ DK config updated from GUI: max_results={self.dk_search_max_results.value()}, "
            f"frequency_threshold={self.dk_frequency_threshold.value()}"
        )

    def show_pipeline_config(self):
        """Show pipeline configuration dialog - Claude Generated"""
        prompt_service = None
        if hasattr(self.alima_manager, "prompt_service"):
            prompt_service = self.alima_manager.prompt_service

        # Get config_manager for provider preferences integration - Claude Generated
        config_manager = getattr(self.alima_manager, 'config_manager', None) or getattr(self.llm_service, 'config_manager', None)
        
        dialog = PipelineConfigDialog(
            llm_service=self.llm_service,
            prompt_service=prompt_service,
            current_config=self.pipeline_manager.config,
            config_manager=config_manager,
            parent=self,
        )
        dialog.config_saved.connect(self.on_config_saved)
        dialog.exec()

    def on_config_saved(self, config: PipelineConfig):
        """Handle saved pipeline configuration - Claude Generated"""
        self.pipeline_manager.set_config(config)

        # Update step widgets to reflect new configuration
        self.update_step_display_from_config()

        QMessageBox.information(
            self,
            "Konfiguration gespeichert",
            "Pipeline-Konfiguration wurde erfolgreich aktualisiert!",
        )

    def load_json_state(self):
        """Load pipeline state from JSON file - Claude Generated"""
        if self.main_window and hasattr(self.main_window, 'load_analysis_state_from_file'):
            self.main_window.load_analysis_state_from_file()
        else:
            self.logger.error("Cannot load JSON: MainWindow not available")

    def update_step_display_from_config(self):
        """Update step widgets based on current configuration - Claude Generated"""
        config = self.pipeline_manager.config

        # Update provider/model display for each step
        for step_id, step_widget in self.step_widgets.items():
            if step_id in config.step_configs:
                step_config = config.step_configs[step_id]

                # Handle both dict and PipelineStepConfig objects - Claude Generated
                if isinstance(step_config, dict):
                    provider = step_config.get("provider") or ""
                    model = step_config.get("model") or ""
                    enabled = step_config.get("enabled", True)
                else:
                    provider = step_config.provider or ""
                    model = step_config.model or ""
                    enabled = step_config.enabled

                # Update step data
                step_widget.step.provider = provider
                step_widget.step.model = model

                # ENHANCED: Add task preference information - Claude Generated
                selection_reason = self._determine_selection_reason(step_id, provider, model)
                step_widget.step.selection_reason = selection_reason

                # Update display (visual styling based on enabled state)
                if not enabled:
                    step_widget.setStyleSheet("QFrame { opacity: 0.5; }")
                else:
                    step_widget.setStyleSheet("")

                step_widget.update_status_display()

    def _determine_selection_reason(self, step_id: str, provider: str, model: str) -> str:
        """Determine why this provider/model was selected for the step - Claude Generated"""
        try:
            # Get config manager from pipeline manager
            config_manager = getattr(self.pipeline_manager, 'config_manager', None)
            if not config_manager:
                return "unknown"

            # Load current config to check task preferences
            config = config_manager.load_config()
            if not config or not hasattr(config, 'task_preferences'):
                return "fallback"

            # Map step_id to task name for task_preferences lookup
            task_name_mapping = {
                "initialisation": "initialisation",
                "keywords": "keywords",
                "dk_classification": "dk_class",
                "image_text_extraction": "image_text_extraction"
            }

            task_name = task_name_mapping.get(step_id)
            if not task_name or task_name not in config.unified_config.task_preferences:
                return "provider preferences" if provider else "default"

            # Check if this provider/model matches task preferences
            task_data = config.unified_config.task_preferences[task_name]
            model_priority = task_data.model_priority if task_data else []

            for rank, priority_entry in enumerate(model_priority, 1):
                candidate_provider = priority_entry.get("provider_name")
                candidate_model = priority_entry.get("model_name")

                if candidate_provider == provider and candidate_model == model:
                    return f"task preference #{rank}"

            # Check chunked preferences
            chunked_priorities = task_data.chunked_model_priority if task_data and task_data.chunked_model_priority else []
            for rank, priority_entry in enumerate(chunked_priorities, 1):
                candidate_provider = priority_entry.get("provider_name")
                candidate_model = priority_entry.get("model_name")

                if candidate_provider == provider and candidate_model == model:
                    return f"chunked preference #{rank}"

            # If we have provider/model but it's not in task preferences
            if provider and model:
                return "provider preferences"
            else:
                return "fallback"

        except Exception as e:
            return f"error: {str(e)[:20]}"

    def reset_pipeline(self):
        """Reset pipeline to initial state - Claude Generated"""
        # Stop any running worker
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.quit()
            self.pipeline_worker.wait()

        self.pipeline_manager.reset_pipeline()

        # Reset timing tracking
        self.step_start_times.clear()
        self.pipeline_start_time = None
        self.current_running_step = None
        self.duration_update_timer.stop()

        # Reset all step widgets
        for step_widget in self.step_widgets.values():
            step_widget.step.status = "pending"
            step_widget.update_status_display()

        # Clear results
        if hasattr(self, "initialisation_result"):
            self.initialisation_result.clear()
        if hasattr(self, "search_results_table"):
            self.search_results_table.setRowCount(0)
            self.search_raw_rows = []
            self.search_chunk_ids = set()
            self.search_chunk_labels = set()
            self.search_final_ids = set()
            self.search_final_labels = set()
            if hasattr(self, "search_tier_stats"):
                self.search_tier_stats.setText("")
            if hasattr(self, "search_filter_input"):
                self.search_filter_input.clear()
            if hasattr(self, "search_tier_filter"):
                self.search_tier_filter.setCurrentIndex(0)
        if hasattr(self, "keywords_result"):
            self.keywords_result.clear()
        # DK-related widgets - Claude Generated (Fixed widget names)
        if hasattr(self, "dk_classification_results"):
            self.dk_classification_results.clear()
        if hasattr(self, "dk_search_results"):
            self.dk_search_results.clear()
        if hasattr(self, "dk_input_summary"):
            self.dk_input_summary.clear()

        # Reset DK filter controls - Claude Generated
        if hasattr(self, "dk_search_filter_input"):
            self.dk_search_filter_input.clear()
        if hasattr(self, "dk_filter_mode"):
            self.dk_filter_mode.setCurrentIndex(0)  # "Alle Felder"
        if hasattr(self, "dk_filter_count_label"):
            self.dk_filter_count_label.setText("")
        if hasattr(self, "dk_search_raw_data"):
            self.dk_search_raw_data = []

        # Reset status and button states - Claude Generated
        self.pipeline_status_label.setStyleSheet("")
        self.pipeline_status_label.setText("Bereit für Pipeline-Start")
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)

        # Reset stream widget completely - Claude Generated
        if hasattr(self, "stream_widget"):
            self.stream_widget.reset_for_new_pipeline()

    def on_config_changed(self):
        """Handle configuration changes - Claude Generated (Webcam Feature)"""
        self.logger.debug("Pipeline tab: Handling config change")
        # Refresh the embedded provider/model pickers so a provider added/removed
        # in Settings shows up immediately (no restart). Both keep the current
        # pick when the provider still exists. - Claude Generated
        if hasattr(self, "global_override_selector"):
            self._populate_global_override_combo()
        if hasattr(self, "stream_widget") and hasattr(self.stream_widget, "refresh_providers"):
            self.stream_widget.refresh_providers()

    # Preferred display order for the workflow picker. ``__classic__`` is the
    # synthetic non-agentic entry (rigid pipeline); every other key is a YAML
    # workflow stem run agentically. Unknown stems are appended alphabetically
    # before the legacy separator. Claude Generated.
    _WORKFLOW_ORDER = [
        "alima_v51",           # ⭐ default (agentic v5.1)
        "alima_v51_105",       # UB Freiberg variant: WiWi-only RVK, DK otherwise
        "__classic__",         # classic, non-agentic
        "alima",               # v5.0
        "alima_classic_v51",   # v4.1
        "alima_classic",       # v4.0
        "title_list_search",
        "catalog_search",
        "synonym_expansion",
        "batch_metadata",
    ]

    def _populate_workflow_combo(self):
        """Populate the workflow picker - Claude Generated.

        Order: ALIMA v5.1 (Voreinstellung) → „Klassische Pipeline (nicht
        agentisch)“ → übrige Workflows → Trenner → legacy-Workflows aus
        ``workflows/legacy/``. Die Auswahl steuert agentisch vs. klassisch
        (siehe ``_on_workflow_changed``).
        """
        try:
            import yaml
            from src.core.agents.workflow_loader import DEFAULT_SEARCH_PATHS, discover_workflow_files

            def _wf_version(path) -> Optional[str]:
                """Cheap top-level ``version`` read — tolerant of legacy
                (v2/v3) schemas that ``load_workflow`` would reject."""
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                    return str(data.get("version", "?"))
                except Exception:
                    return None

            # Discover stems → version, separating root and legacy workflows.
            root: dict = {}
            legacy: dict = {}
            files = discover_workflow_files()
            for path in files:
                ver = _wf_version(path)
                if ver is not None:
                    root[path.stem] = ver
            # legacy subdirs are not part of the shared top-level discovery
            seen: set = {p.resolve() for p in files}
            for base in DEFAULT_SEARCH_PATHS:
                legacy_dir = base / "legacy"
                if not legacy_dir.is_dir():
                    continue
                for path in sorted(legacy_dir.glob("*.yaml")):
                    key = path.resolve()
                    if key in seen:
                        continue
                    seen.add(key)
                    ver = _wf_version(path)
                    if ver is not None:
                        legacy[path.stem] = ver

            self.workflow_combo.blockSignals(True)
            self.workflow_combo.clear()

            def _label(stem: str) -> str:
                ver = root.get(stem)
                return f"{stem} (v{ver})" if ver else stem

            # 1. ALIMA v5.1 (default)
            if "alima_v51" in root:
                self.workflow_combo.addItem(
                    f"⭐ ALIMA v5.1 — agentisch (v{root['alima_v51']})", "alima_v51"
                )
            # 2. Classic non-agentic (synthetic entry, no YAML)
            self.workflow_combo.addItem(
                "Klassische Pipeline (nicht agentisch)", "__classic__"
            )
            # 3. Remaining workflows in preferred order, then any extras A→Z
            added = {"alima_v51", "__classic__"}
            for stem in self._WORKFLOW_ORDER:
                if stem in added or stem not in root:
                    continue
                self.workflow_combo.addItem(_label(stem), stem)
                added.add(stem)
            for stem in sorted(root):
                if stem in added:
                    continue
                self.workflow_combo.addItem(_label(stem), stem)
                added.add(stem)
            # 4. Legacy workflows after a separator
            if legacy:
                self.workflow_combo.insertSeparator(self.workflow_combo.count())
                for stem in sorted(legacy):
                    self.workflow_combo.addItem(
                        f"{stem} (legacy v{legacy[stem]})", stem
                    )

            # Respect configured default workflow (SystemConfig.default_workflow).
            default_workflow = "alima_v51"
            try:
                from ..utils.config_manager import ConfigManager

                cfg = ConfigManager().load_config()
                default_workflow = getattr(cfg.system_config, "default_workflow", "alima_v51") or "alima_v51"
            except Exception:
                pass
            default_idx = self.workflow_combo.findData(default_workflow)
            if default_idx >= 0:
                self.workflow_combo.setCurrentIndex(default_idx)
            else:
                self.workflow_combo.setCurrentIndex(0)  # ALIMA v5.1 (or classic fallback)
            self.workflow_combo.blockSignals(False)
            self.logger.debug(
                f"Workflow combo populated with {self.workflow_combo.count()} entries"
            )

            # Wire once — guard against duplicate connects on repopulate.
            try:
                self.workflow_combo.currentIndexChanged.disconnect(
                    self._on_workflow_changed
                )
            except Exception:
                pass  # not yet connected (first call) — nothing to disconnect
            self.workflow_combo.currentIndexChanged.connect(self._on_workflow_changed)
            # Push the default selection (v5.1 → agentic) into the config now.
            self._apply_workflow_selection()
        except Exception as e:
            self.logger.error(f"Error populating workflow combo: {e}")

    # Workflow-specific hints for the UnifiedInputWidget. Keys are YAML stems.
    WORKFLOW_HINTS = {
        "title_list_search": (
            "💡 Titel eingeben – eine pro Zeile oder im Fließtext "
            "(LLM extrahiert strukturierte Titel)."
        ),
        "catalog_search": (
            "💡 Suchbegriffe kommagetrennt oder als Liste – werden parallel "
            "gegen SWB / Lobid / Katalog geschickt."
        ),
    }

    def _apply_workflow_selection(self) -> bool:
        """Push the current workflow-combo choice into the pipeline config.

        The synthetic ``__classic__`` entry runs the rigid non-agentic
        pipeline (``enable_agentic_mode=False``); every YAML entry runs
        agentically with its stem as ``workflow_name``. Returns the resulting
        agentic flag. Claude Generated.
        """
        data = (
            self.workflow_combo.currentData()
            if hasattr(self, "workflow_combo")
            else None
        )
        if data is None:  # separator or empty combo
            return False
        agentic = data != "__classic__"
        if self.pipeline_manager and self.pipeline_manager.config:
            self.pipeline_manager.config.enable_agentic_mode = agentic
            if agentic:
                self.pipeline_manager.config.workflow_name = data
        return agentic

    def _on_workflow_changed(self, _index: int = 0) -> None:
        """Apply workflow selection: config (agentic/classic) + hint + panels.

        Dock visibility is NOT changed here — the agentic context dock is
        shown only on request (View ▸ 🤖 Agentic Kontext). Claude Generated.
        """
        if not hasattr(self, "workflow_combo"):
            return
        data = self.workflow_combo.currentData()
        if data is None:  # separator selected (shouldn't happen) — ignore
            return
        agentic = self._apply_workflow_selection()

        hint = self.WORKFLOW_HINTS.get(data if agentic else "", "")
        if hasattr(self, "unified_input"):
            self.unified_input.set_hint(hint)

        # Notify MainWindow so it can refresh the (possibly hidden) context
        # panels; the dock's visibility stays user-controlled.
        self.agentic_mode_changed.emit(agentic)
        # Keep the context widget panels in sync with the selected workflow.
        self._rebuild_agentic_panels()
        self.logger.info(f"Workflow '{data}' selected (agentic={agentic})")

    def _rebuild_agentic_panels(self) -> None:
        """Load active workflow YAML and emit workflow def to MainWindow dock.

        No-op when agentic mode is off or the workflow can't be found.
        Failures are logged but never abort the pipeline start.
        """
        if not self.pipeline_manager or not self.pipeline_manager.config:
            return
        if not self.pipeline_manager.config.enable_agentic_mode:
            return

        try:
            from src.core.agents.workflow_loader import (
                find_workflow_file,
                load_workflow,
            )

            wf_name = self.pipeline_manager.config.workflow_name or "alima_classic"
            wf_path = find_workflow_file(wf_name)
            if wf_path is None:
                self.logger.warning(
                    f"Agentic dock: workflow '{wf_name}' not found — panels not rebuilt"
                )
                return
            wf_def = load_workflow(wf_path, strict=False)
            self.agentic_workflow_built.emit(wf_def)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Agentic panel rebuild failed: {e}")

    def _emit_step_results_to_tabs(self, step: PipelineStep) -> None:
        """
        Emit pipeline step results to appropriate tab viewer methods - Claude Generated
        
        Args:
            step: Completed pipeline step with results
        """
        if not step.output_data:
            return
            
        try:
            # Emit search results to SearchTab
            if step.step_id == "search" and "search_results" in step.output_data:
                search_results = step.output_data["search_results"]
                self.logger.debug(f"Emitting search results to SearchTab: {len(search_results)} terms")
                self.search_results_ready.emit(search_results)
            
            # Emit keyword analysis results to AbstractTab (and DkAnalysisTab)
            elif step.step_id in ["initialisation", "keywords", "dk_classification"]:
                if "analysis_result" in step.output_data:
                    analysis_result = step.output_data["analysis_result"]
                    self.logger.debug(f"Emitting {step.step_id} analysis results to AbstractTab")
                    self.analysis_results_ready.emit(analysis_result)
                elif "llm_analysis" in step.output_data:
                    llm_analysis = step.output_data["llm_analysis"]
                    self.logger.debug(f"Emitting {step.step_id} LLM analysis results to Tabs")
                    # We reuse the same signal, as AbstractTab can handle LlmKeywordAnalysis too
                    # (Need to ensure AbstractTab's slot can handle both or we wrap it)
                    self.analysis_results_ready.emit(llm_analysis)
                
        except Exception as e:
            self.logger.error(f"Error emitting step results to tabs: {e}")

    def show_loaded_state_indicator(self, state):
        """
        Display visual indicators for loaded analysis state - Claude Generated
        Shows which pipeline steps have data from the loaded JSON
        """
        try:
            # Add visual indicator in pipeline status
            loaded_steps = []

            if state.original_abstract:
                loaded_steps.append("Input")
            if state.initial_keywords:
                loaded_steps.append("Initialisierung")
            if state.search_results:
                loaded_steps.append("Suche")
            if state.final_llm_analysis:
                loaded_steps.append("Schlagworte")
            if state.classifications:
                loaded_steps.append("Klassifikation")

            if loaded_steps:
                loaded_info = " → ".join(loaded_steps)
                self.pipeline_status_label.setText(f"📁 Geladener Zustand: {loaded_info}")
                self.pipeline_status_label.setStyleSheet(
                    "color: #2E7D32; font-weight: bold; padding: 5px; "
                    "background-color: #E8F5E8; border: 1px solid #4CAF50; border-radius: 3px;"
                )

                # Populate results displays with loaded data
                if state.initial_keywords and hasattr(self, 'initialisation_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    keywords_text = (", ".join(state.initial_keywords)
                                     if isinstance(state.initial_keywords, list)
                                     else str(state.initial_keywords))
                    self.initialisation_result.setPlainText(f"📁 Geladene Keywords:\n{keywords_text}")

                if state.search_results and hasattr(self, 'search_results_table'):
                    selected = (
                        state.final_llm_analysis.extracted_gnd_keywords
                        if state.final_llm_analysis
                        else None
                    )
                    self._populate_gnd_hits(state.search_results, selected=selected)

                if state.final_llm_analysis and hasattr(self, 'keywords_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    final_kw = state.final_llm_analysis.extracted_gnd_keywords
                    final_keywords = (", ".join(final_kw)
                                      if isinstance(final_kw, list)
                                      else str(final_kw))
                    self.keywords_result.setPlainText(f"📁 Finale Schlagwörter:\n{final_keywords}")

                # DK Classification Results Display - Claude Generated (Enhanced with titles)
                if state.classifications and hasattr(self, 'dk_classification_results'):
                    html_display = self._format_dk_classifications_with_titles(
                        state.classifications,
                        state.dk_search_results_flattened  # flattened format has {dk, titles} at top level
                    )
                    self.dk_classification_results.setHtml(
                        f"<div style='background: #E8F5E8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                        f"<strong>📁 Geladene Klassifikationen (DK/RVK)</strong>"
                        f"</div>{html_display}"
                    )

                # DK Search Results Display - Claude Generated (Enhanced for filtering)
                if state.dk_search_results and hasattr(self, 'dk_search_results'):
                    # Store raw data for filtering
                    self.dk_search_raw_data = state.dk_search_results

                    # Display results using display method
                    self._display_dk_search_results(state.dk_search_results)

                    # Add loaded indicator prefix
                    current_text = self.dk_search_results.toPlainText()
                    self.dk_search_results.setPlainText(
                        f"📁 Geladene DK-Suchergebnisse:\n\n{current_text}"
                    )

            self.logger.info(f"Pipeline tab updated with loaded state indicators: {loaded_steps}")

        except Exception as e:
            self.logger.error(f"Error showing loaded state indicator: {e}")
