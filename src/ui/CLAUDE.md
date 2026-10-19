# UI - PyQt6 User Interface Layer

## [Preserved Section - Permanent Documentation]

### UI Architecture
`src/ui/` implements the PyQt6 GUI for ALIMA.

**Main Components:**
- `MainWindow`: central window, tab + menu management
- `PipelineTab`: vertical pipeline UI (primary interface) orchestrating the full workflow via `PipelineManager`; integrated input tabs (DOI/Image/PDF/Text), per-step result + live streaming, Auto-Pipeline button
- `GlobalStatusBar`: provider info + cache stats + pipeline progress (auto-refresh)
- `AbstractTab`: text analysis with chunking workflow + real-time streaming; auto-sends results to `AnalysisReviewTab` (`analysis_completed` signal)
- `SearchTab` (`find_keywords.py`): **the** unified search tab — GND keyword search + embedded UB catalog (DK/RVK) + pipeline post-processing, ONE shared search input. Reworked Aug 2026 (WP-K5): async + cancellable (`GndSearchWorker`), source checkboxes live from the plugin system (`refresh_sources`), Häufigkeit = `display_count` (count landmine), classification column. Results are tabs: GND-Schlagwörter / UB-Katalog (injected `UBCatalogTab`, `set_embedded` shares the input) / Pipeline-Mapping (hidden until pipeline data). `SearchTabUnified` (combo switcher) was deleted. External contract: ctor signature (+`ub_catalog_tab`), `update_data`/`update_search_field`/`display_search_results`/`refresh_styles`/`refresh_sources`, signal `selection_changed`.
- `AnalysisReviewTab`: analysis review + result management (JSON import/export, auto-receive)
- `DkAnalysisUnifiedTab`: unified DK-Zuordnung + DK-Statistik + UB-Suche (inherits `AbstractTab`)
- `ImageAnalysisTab`: Vision-LLM image analysis
- DOI lookup: via the DOI input-source plugins / `src/utils/doi_resolver.py` (the standalone Crossref tab + `crossref_worker.py` were removed)

**Supporting Components:**
- `step_config_widgets.SearchStepConfigWidget`: Quellen des klassischen Suchschritts als eine Checkbox je aktivierter GND-Quelle, live aus `factory.enabled_gnd_provider_ids` (Muster von `SearchTab`, WP-K5). ⚠️ keine festen Provider-Listen — vorher waren es drei hartverdrahtete Boxen und `gnd_local` sowie jedes eigene Plugin waren unerreichbar.
- `workflow_editor_dialog.py`: form editor for agentic v4 workflow YAML (ruamel round-trip, validates `load_workflow(strict=True)`, saves to `~/.config/alima/workflows/`; engine read-only). The classic pipeline has no editor dialog of its own (`PipelineConfigDialog`, Smart/Advanced/Expert per-step override, removed Oct 2026 — every step already ran on the one pipeline default via `PipelineConfig.create_from_provider_preferences()`, so there was nothing a per-step override could do that the Sprachmodelle page's pipeline default doesn't already cover).
- `batch_processing_dialog.py`: batch file/directory processing (QThread, progress, continue-on-error)
- `ComprehensiveSettingsDialog`: **keine Tabs mehr** — `SettingsShell` (Seitenleiste + `QStackedWidget` + Suchfeld) mit fünf Seiten: Sprachmodelle, Erschließung und Workflows, Datenquellen, Daten und Speicher, Anwendung. Jede Gruppe ist ein `_build_*_group() -> QGroupBox`, jede Seite reine Komposition über `_page(...)`; jedes Bedienelement meldet sich über `self.index.add(...)` bei der Suche an. Die Bausteine liegen in `src/ui/settings/` (`shell`, `search`, `collapsible`, `resolution_view`, `provider_list`, `workflow_list`, `effective_models`). ⚠️ Laden und Speichern adressieren die Widgets über `self.<name>` — eine gebaute, aber nicht eingehängte Gruppe fällt erst dort auf; `tests/test_settings_dialog_state.py` hält das fest. ⚠️ Nichts auf einer Seite darf vertikal wachsen: eine umbrechende Hinweiszeile unter der Modelltabelle war das einzige Element, das beim Vergrößern mitwuchs. ⚠️ `resolution_view.py`s "Modell"-Spalte stand auf `ResizeToContents` — korrekt für den Modus, aber die Spalten- (und damit Tabellen-)breite wechselte sichtbar mit jeder Modellwahl (z. B. `"a1"` vs. ein langer Ollama-Tag). Jetzt fest (`Interactive`, 200px) + Tooltip für den vollen Namen; dieselbe Lektion gilt für jede feste-Zeilen-Tabelle mit variabler Modell-/Freitext-Spalte. Das Modell-Combobox in der "Ändern"-Spalte ist dort außerdem nicht mehr editierbar (`editable_model=False`) — der konfigurierte Wert steht ohnehin in der separaten "Modell"-Spalte.

### ✅ FIXED (Oct 2026): `PipelineWorker` gab Callbacks nie zurück
Die sieben `pm.set_callbacks`-Werte blieben nach dem Lauf an den fertigen Worker gebunden; der
Chat-Agent leitete sie weiter (zweiter, wirkungsloser Stop-Button). `run()` restauriert sie jetzt
im `finally` (wie `SingleStepWorker`). Tests: `tests/test_pipeline_worker_callback_cleanup.py`.

### ✅ ENTFERNT (Oct 2026): Bearbeiten → „Prompt-Konfiguration" + „Workflow-Editor"
Zwei eigenständige Menüeinträge außerhalb des Einstellungsdialogs. „Workflow-Editor" war
funktionierend, aber 100 % redundant — derselbe `WorkflowEditorDialog(initial_workflow=None)`-
Aufruf wie der „➕ Neu…"-Knopf in `settings/workflow_list.py`. „Prompt-Konfiguration"
(`PromptEditorDialog`, jetzt gelöscht) war dagegen für seinen eigentlichen Zweck kaputt: er
kannte nur `prompts.json`, nie `prompts.yaml` — und `prompts.yaml` trägt für alle zehn mit
`prompts.json` geteilten Tasks einen `"default"`-Eintrag, der beim Merge gewinnt
(`YamlPromptService.merge_into`). Eine Bearbeitung über den Dialog landete in `prompts.json`,
lief zur Laufzeit aber unverändert auf dem YAML-Text weiter — ohne Fehler, ohne Warnung. Beide
Menüeinträge sowie `show_prompt_editor`/`show_workflow_editor` entfernt; der Workflow-Editor
bleibt über Einstellungen → Erschließung und Workflows erreichbar.

### ✅ FIXED (Oct 2026): markierte Zeile in `provider_list.py` wurde weiß auf grau
`styles.py`s `QTreeWidget::item:selected { background: primary (blau); color: white; }` und das später deklarierte `QTreeWidget::item:hover { background: background_light (hellgrau); }` kollidierten: ein Klick selektiert eine Zeile UND lässt den Cursor genau darüber stehen, also sind `:selected` und `:hover` gleichzeitig wahr. Bei einer QSS-Eigenschaftskollision gewinnt die später deklarierte Regel — `:hover`s `background-color` überschrieb `:selected`s Blau, `color: white` aus `:selected` blieb aber aktiv (kein `color` in `:hover`). Per echtem Rendering + Pixel-Sampling verifiziert (nur unter `QT_QPA_PLATFORM=offscreen` reproduzierbar — der sonst automatisch gewählte Plattform-Plugin triggert den Hover-Pseudo-State in diesem Umfeld nicht verlässlich, deshalb testet `tests/test_provider_list_selection_contrast.py` nur den Stylesheet-Text, nicht Pixel). Fix: `:hover:!selected`. Dieselbe `:hover`-ohne-`:!selected`-Lücke besteht auch bei `QTabBar::tab:hover` (verdeckt beim Hovern kurz die Markierung des aktiven Tabs) — kosmetisch, kein Kontrastbruch, bewusst nicht mitgefixt.
- `settings/workflow_list.py` (`WorkflowList`): die Workflows als Liste — eigener `name:` statt Dateiname, Version, Schrittzahl, Ablage (Projekt/Benutzer/eingebaut), ⭐ = `SystemConfig.default_workflow`. Doppelklick bearbeitet eine YAML im `WorkflowEditorDialog`; die klassische Pipeline (`__classic__`) ist nicht bearbeitbar (`editable=False`) — sie läuft auf dem Pipeline-Standard von der Sprachmodelle-Seite, kein eigener Dialog. ⚠️ Der Editor schreibt **sofort** und gehört nicht zur Unit of Work des Einstellungsdialogs. Auswahl verschiebt den Stern nicht — nur „⭐ Als Standard".
- `plugin_settings_tab.py` (`PluginSettingsTab`): category-grouped per-plugin config (search + input), auto-built from each plugin's `config_fields`; enable/usage_hint + add/duplicate/remove. Single editor for provider/source config — **the Catalog tab + DOI System entries were removed** (folded into the `catalog` / DOI plugins; only the DOI `SystemConfig` fields are still derived on save — the `CatalogConfig` mirror is gone, WP P7). Replaced the checkbox-only `provider_selector.py` (deleted). Spec: [`docs/plugin_system.md`](../../docs/plugin_system.md).
- `first_start_wizard.py` (`FirstStartWizard`): Einrichtung in sieben Seiten (LLM, Modelle, Datenbank, Katalog) mit Verbindungstests und Institutions-Presets. Erreichbar über **Bearbeiten → 🧭 Einrichtungsassistent…** *und* `--wizard`/erster Start. ⚠️ `accept()` **mergt** auf die bestehende Konfiguration (`_apply_onto_existing`): es ersetzt nur die abgefragten Bereiche. Vorher baute es eine frische `AlimaConfig` und `save_config` schrieb sie ganz — ein erneuter Lauf löschte Plugin-Einstellungen, Tier-2-Freigaben, Bundles, chat/ui/repetition und alle weiteren Provider. Ein Backup gibt es nicht. Die Modell-Seite (`ModelSelectionPage`) fragt seit Oct 2026 nur noch zwei Modelle ab (Pipeline, Vision) statt sechs einzelner Pipeline-Schritte — siehe [`docs/settings_cleanup_phase1.md`](../../docs/settings_cleanup_phase1.md).
- `dialogs/rules_dialog.py` (`RulesDialog`): manage the personal indexing rules — list/edit/enable/delete plus export+import (provenance preserved). Reached from **Bearbeiten → 📌 Zusatzregeln…** (the only entry point that is found without being told), from Settings → Chat-Agent, and from 📌 in the chat panel header. Store: `src/core/user_rules.py`.
- `chat_tools/rules.py`: `list_rules` / `propose_rule` / `set_rule_enabled` / `set_rule_scope` / `delete_rule`; the writing ones ask through the existing `ProposalGateway` and ignore `autonomous_pipeline` (a rule changes every *future* run, not the current one).
- `proposal_bar.py` (`ProposalBar`): the confirmation surface for every gateway proposal — a one-shot strip between chat log and input, hidden again once answered, and withdrawn via `dismiss(audit_id)` when the gateway's `proposal_expired` fires (a click after the timeout reports "zu spät", never "akzeptiert"). The log block is a record only; ⚠️ never put accept/reject anchors back there — a custom scheme is dropped by QWebEngine, and an https one stays clickable forever.
- `chat_tools/llm_switch.py` (`switch_llm_model`): lets the chat agent change its own provider/model for the rest of the session. Registered **only** when `ChatConfig.allow_model_switch` is on *and* the host passed an `llm_service` (the headless runner's session does not survive the call). Validates against the enabled providers, never persists, and any toolbar change clears it.
- **One LLM pick per pipeline tab**: `pipeline_tab.global_override_selector` drives the run and, via `stream_widget.set_llm_override`, the chat. The chat header shows the effective model read-only (`model_status_label`) — ⚠️ do not add a second picker there; the old one silently outranked the toolbar's. On "-- Standard --" the agentic default from the settings applies: `ChatConfig` has **no** provider/model default any more (a leftover value there outranked everything and no surface showed it).
- **Abschlusskarte**: `format_final_result_card_html` closes a run with one `final_result` card (core, headings, chains, notation, shelf place, rule snippet + copy button); the detail cards follow below. Styles in `alima_render.css`, the copy handler in `alima_render.js` — both shared assets, so GUI and webapp behave alike. `data-alima-copy` names the target as a selector inside the enclosing `.alima-copy-scope` (empty = the whole scope), so copying is a generic card mechanism, not something bolted to the rule output. ⚠️ `render_markdown_block` must stay `html: False`: the commonmark preset swallowed `Toleranz <Biologie>` into a phantom tag.
- **Standard-Workflow**: the ⭐ next to `pipeline_tab.workflow_combo` writes `SystemConfig.default_workflow` (same setting as Settings ▸ System) — the star in the picker marks that setting, never a fixed stem, and the default is listed first.
- `TableWidget`, `Styles`: reusable display/theming (the unused `widgets.py` was removed June 2026)

### Design Patterns
- Signal/slot, thread-safe UI updates; QThread for non-blocking LLM/search/IO with streaming + cancellation.
- Responsive `QSplitter` layouts with dynamic resizing during streaming.

## [Variable Section - Short-term Information]

### ✅ FIXED (Oct 2026): toter Legacy-Migrationspfad in `UnifiedProviderTab`
`_migrate_from_legacy_config` rief `UnifiedProviderConfig.from_legacy_config(self.config.unified_config, self.unified_config)` — zwei Positionsargumente für eine Methode, die eines nimmt. `self.config.unified_config` ist zudem dasselbe Objekt wie `self.unified_config` (beide aus `config_to_edit.unified_config`), also gab es nie etwas zu migrieren. Der TypeError-Fallback ersetzte `self.unified_config` durch ein frisches, leeres Objekt — ein danach im Dialog angelegter Provider landete auf dem abgetrennten Objekt, „Speichern" sah ihn nie. Auslöser (`providers` leer) war vorher durch den Config-Reset aus `3247a67` verdeckt. Entfernt statt repariert — keine andere Aufrufstelle, und die eigentliche Legacy-Migration läuft längst beim Parsen in `ConfigManager`. Tests: `tests/test_unified_provider_tab_no_dead_migration.py`.

### F-5 god-file split (June 30)
- ✅ **`pipeline_chat_panel` 1923→698** — leaf widgets → `chat_input_widgets.py`; repetition bar → `repetition_warning_bar.py`; behavior → `PipelineLogMixin` / `ChatAgentMixin` / `BusEventMixin` (`_chat_panel_*.py`). Verbatim moves, reachable via MRO; `bus.subscribe` wiring stays in `__init__`.
- **Mixin pattern** (new here): for a single god-*class* with no embeddable sub-widgets, split methods into behavioral mixins — zero call-site changes, existing stub tests pass unchanged.
- Remaining F-5: `pipeline_tab` 2882, `main_window` 2650.

### Findings — `ProviderModelSelector` adoption (decide when working through cleanup)
Shared `provider_model_selector.ProviderModelSelector` is used by `unified_provider_tab`, `image_analysis_tab`, `pipeline_chat_panel`, `pipeline_tab`, and now `single_step_dialog` (migrated June 2026). The remaining builders of their own provider/model combos are **not** clean dedup — each has an *intentional* behavior difference, so adopting the selector would change UX. Decide deliberately (not as "cleanup"):
- ~~`comprehensive_settings_dialog`~~: erledigt — der dortige `ModelSelectionDialog` war eine tote Kopie hinter einem nie eingehängten Tab und ist mit ihm gelöscht (Sep 2026). Der lebende Picker ist `provider_dialogs.TaskModelSelectionDialog`; er baut seine Provider-Liste aus `ProviderDetectionService.get_available_providers()`, nicht aus den 4 üblichen plus den aktivierten. Tests: `tests/test_task_model_selection_dialog.py`.
- **`abstract_tab`**: model combo is filled from a **pushed `self.available_models` cache** (not async pull); tracks `explicit_provider/model_selection` + `user_interaction_mode`; recommended-model and history-restore use synchronous `findText`+`setCurrentIndex` that breaks under the selector's **async** loading. → adoption is a re-architecture, not a dedup.
- **`pipeline_config_dialog`**: per-step provider/model grid (~26/50 combo refs) — large, out of scope.

Detection note: the selector's default detection service wraps `LlmService.get_available_models` (TTL-cached), so the **model list source is the same** — the differences above are about provider lists / load mechanics, not the underlying models.

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
0. ✅ **`SearchTab` überarbeitet** (Aug 4, WP-K5) — async + abbrechbar, Plugin-live,
   `display_count`, Klassifikations-Spalte; Klick-Test-Feedback eingearbeitet
   (Suchmodus-Label, Mapping nur im Pipeline-Modus, Stop-Button). Offener
   Polish-Punkt: Zeilenfarben sind Light-Mode-Hexes (Dark-Theme).
1. **Pipeline Configuration UI**: graphical configuration for pipeline steps/models
2. **Pipeline Templates**: save/load workflow configurations
3. **Batch Review Table**: enhanced table view for batch results
4. **Drag & Drop to Pipeline**: files dropped directly onto pipeline steps
5. **Pipeline Step Editing**: inline editing of intermediate results
6. **Keyboard Shortcuts**: comprehensive keyboard navigation
7. **Advanced Theming**: dark mode + customizable themes
8. **Enhanced Export**: pipeline results to PDF/Excel/etc.

### Vision
- Intuitive, efficient interface for library-science workflows.
- Accessibility, multi-monitor/varied-resolution support.
- Seamless integration across all analysis and search functions.
