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
- `first_start_wizard.py` (`FirstStartWizard`): Einrichtung in sieben Seiten (LLM, Modelle, Datenbank, Katalog) mit Verbindungstests und Institutions-Presets. Erreichbar über **Bearbeiten → 🧭 Einrichtungsassistent…** *und* `--wizard`/erster Start. ⚠️ `accept()` **mergt** auf die bestehende Konfiguration (`_apply_onto_existing`): es ersetzt nur die abgefragten Bereiche. Anbieter werden **ergänzt** (`_merge_providers`: gleicher Name ersetzt genau diesen Eintrag an seiner Stelle, alle anderen bleiben); die Standards Allgemein/Pipeline/Bild setzt der Wizard auf seine Wahl. Vorher baute es eine frische `AlimaConfig` und `save_config` schrieb sie ganz — ein erneuter Lauf löschte Plugin-Einstellungen, Tier-2-Freigaben, Bundles, chat/ui/repetition und alle weiteren Provider. Ein Backup gibt es nicht. Die Modell-Seite (`ModelSelectionPage`) fragt seit Oct 2026 nur noch zwei Modelle ab (Pipeline, Vision) statt sechs einzelner Pipeline-Schritte — siehe [`docs/settings_cleanup_phase1.md`](../../docs/settings_cleanup_phase1.md).
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

### ✅ ADDED (Oct 2026): Häkchen in der Providerliste + vorkonfiguriertes lokales Ollama
`ProviderList` hat ein Häkchen je Zeile (`enabled_toggled`), das `UnifiedProvider.enabled` setzt und sofort speichert. Solange kein nativer Ollama-Eintrag auf localhost/127.0.0.1 existiert, zeigt die Liste `ollama_local` (`config_models.local_ollama_preset`, `localhost:11434`) als ausgeschalteten Vorschlag; erst das Häkchen übernimmt ihn in die Konfiguration (danach ein gewöhnlicher, löschbarer Eintrag; nach dem Löschen erscheint der Vorschlag wieder). Beim Einschalten prüft `_check_provider` die Verbindung im `ModelLoadWorker` und meldet das Ergebnis als Toast; „nicht erreichbar" und „erreichbar ohne Modelle" sind nicht unterscheidbar (`get_available_models` liefert bei Fehlern `[]`). Tests: `tests/test_provider_list.py`, `tests/test_provider_toggle_nonblocking.py`.

### ✅ ADDED (Oct 2026): Eingabe-Feedback im Anbieter-Dialog
`ProviderEditDialog` zeigt Hinweise live (`utils/ollama_endpoint.hints`, Qt-frei: fehlender Host sperrt „OK“, alles andere warnt nur), hat einen **„Verbindung testen“**-Knopf (nur Ollama und OpenAI-kompatibel; Formularwerte, im Hintergrund, nichts wird gespeichert) und bereinigt beim Übernehmen (`parse_host`: Schema/Pfad/`:port` aus dem Hostfeld, `clean_api_key`: Leerraum und „Bearer“). Ollama Cloud wird als HTTPS/443 gespeichert. Der Schlüssel des Ollama-Formulars ist jetzt maskiert; Hinweise geben ihn nie aus. `LlmService` und `OllamaConnectionValidator.test_native` bauen die URL über dieselbe Funktion (`ollama_endpoint.base_url`), Test und echter Aufruf können nicht auseinanderlaufen. Gemini/Anthropic haben keinen Testknopf. Tests: `tests/test_ollama_endpoint.py`.

### ✅ CHANGED (Oct 2026): Einrichtungsassistent speichert Ollama nativ
`ConfigurationBuilder.create_initial_config("ollama", …)` baute vorher einen `openai_compatible`-Eintrag mit `/v1` und dem Platzhalter-Schlüssel `no-key-required`, obwohl der Wizard vorher den **nativen** Client testete (und ein Cloud-Schlüssel ging verloren). Jetzt Typ `ollama` mit Host/Port/SSL/Schlüssel; Ollama Cloud heißt `ollama_cloud` und wird als HTTPS/443 gespeichert. GUI-Seite (`LLMSetupPage`) und CLI haben SSL, Schlüsselfeld und dieselben Hinweise wie der Anbieter-Dialog (`utils/ollama_endpoint`). ⚠️ Bestehende Wizard-Konfigurationen mit `openai_compatible` + `/v1` bleiben unverändert und funktionieren weiter. Der Verbindungstest im Wizard läuft weiter synchron im UI-Thread (bis 10 s Timeout). Tests: `tests/test_wizard_ollama.py`.

### ✅ ADDED (Oct 2026): `scripts/lint_qss.py`, Stylesheet-Prüfung
Qt verwirft fehlerhafte QSS-Deklarationen (oder die ganze Regel) ohne Warnung; eine Zeile `#padding: 12px 8px;` in der Tab-Regel der Pipeline hielt `min-width: 100px` aus derselben Regel monatelang unwirksam, bis das Aufräumen sie plötzlich aktivierte. Das Skript liest alle Python-Dateien unter `src/` per `ast` (Argumente von `setStyleSheet`, Regeln mit Qt-Klassen-Selektor, f-Strings und `{{ }}`-Vorlagen mit neutralisierten Platzhaltern; Docstrings und Web-CSS bleiben außen vor) und meldet Fehler (keine `property: value`-Deklaration, `#`/`//`-Kommentarzeilen, fehlendes `;`, unausgeglichene Klammern) und Warnungen (unbekannte Eigenschaft). `python scripts/lint_qss.py [Pfade]`, Exit 1 bei Fehlern; `tests/test_qss_lint.py` lässt `src/` bei jedem Testlauf prüfen. Am Stand vor dieser Sitzung fand es genau den Pipeline-Fehler (und `cursor` als unbekannte Eigenschaft), jetzt nichts. ⚠️ Nicht erfasst: Stylesheets, die zur Laufzeit aus Variablen zusammengesetzt oder aus Dateien geladen werden (es gibt keine `.qss`-Dateien), und die Webapp-CSS.

### ✅ FIXED (Oct 2026): GUI-Start 10,5 s → 3,0 s
Gemessen am echten Fensteraufbau auf einer Kopie der Nutzerdaten (Netz im Prozess gesperrt): `MainWindow()` 9,6 s → 2,1 s. Drei Posten: (1) Anbieter-SDKs wurden trotz `lazy_initialization=True` beim Start importiert (≈5 s, siehe `src/llm/CLAUDE.md`); (2) `load_settings` wandte Theme und Schriftgröße nacheinander an, je ein vollständiges Stylesheet-Polieren (≈1,2 s); jetzt ein Durchgang (`apply_theme(restyle=False)`, dann `apply_font_size`); (3) `apply_font_size` schrieb bei jedem Start die Config zurück, jetzt nur bei geänderter Größe; die Workflow-Versionen werden mit dem libyaml-Parser gelesen (0,4 s → unter 0,05 s). Nicht angefasst: Import von `alima_gui` (≈1 s, davon `ollama` 0,35 s), das Stylesheet-Polieren selbst (≈1,1 s), der Aufbau der Tabs (≈0,85 s). Messskripte nicht im Repo; Methode: `cProfile` um `MainWindow()` mit `HOME` auf eine Kopie von `~/.config/alima`.

### ✅ CHANGED (Oct 2026): Farbwelt aus dem Logo
`styles.py` (COLORS, Haupt-Stylesheet, Dark-Zweig) und die Paletten in `alima_gui.py` folgen den Tokens aus [`docs/design_refresh.md`](../../docs/design_refresh.md); in `src/ui/` stehen keine festen Farbliterale mehr (Ausnahmeliste mit Begründung in `tests/test_no_hardcoded_colors.py`, derzeit der abdunkelnde Rand um den Bildausschnitt im Webcam-Dialog). Neue Widgets holen Farben über `get_colors()`; ein Theme-Wechsel läuft über `_reapply_tab_stylesheets` durch alle Widgets (`refresh_styles`), der GUI-Log folgt dem Theme ohne Neuaufbau. ⚠️ HTML-Inhalte und Tabellenzellen (Suchdetails, Review-Zusammenfassung, Ketten-Liste, Zeilenfarben) werden beim Füllen eingefärbt und behalten das alte Theme bis zum nächsten Füllen; nur die agentischen Panels rendern neu. Die Wiederholungs-Leiste ist im Ruhezustand wirklich versteckt (Knöpfe unsichtbar, nicht klickbar).

### ✅ ADDED (Oct 2026): Vorlagen für OpenAI-kompatible Anbieter
`utils/provider_presets.py` (nur Daten): OpenAI, Mistral, NVIDIA NIM, OpenRouter, Groq, Together, DeepSeek, xAI sowie lokal LM Studio, vLLM, llama.cpp, Ollama über `/v1`; bewusst ohne GWDG/Hochschul-Dienste (dafür Institutions-Presets). Im Anbieter-Dialog (Typ openai_compatible) und im Wizard als „Vorlage“-Auswahl: setzt die Basis-URL, im Dialog Name/Beschreibung nur wenn leer; die Auswahl folgt der URL; ein Hinweis erscheint, wenn die Vorlage einen Schlüssel braucht. ⚠️ Die Adressen sind aus der Anbieter-Dokumentation übernommen und **nicht gegen die Dienste geprüft**; „Verbindung testen“ prüft sie mit dem eigenen Schlüssel. Tests: `tests/test_provider_presets.py` (Form der Daten, nicht Erreichbarkeit).

### ✅ FIXED (Oct 2026): Einfrieren nach „OK“ im Einstellungsdialog (3,8 s)
`show_settings` rief nach `dialog.exec()` das vollständige `load_settings()`: `apply_theme` + `apply_font_size` setzen `app.setStyleSheet` und alle Tab-Stylesheets neu, auch ohne Änderung (gemessen 3,8 s im UI-Thread, headless mit echtem `MainWindow`), und der Aufruf setzte nebenbei Fenstergeometrie, Tab 0 und Splitter auf den Startzustand zurück. Jetzt `load_settings(only_changes=True)`: Theme/Schrift nur bei Abweichung, Fensterzustand unberührt; der Startpfad bleibt vollständig. Gemessen nach den Änderungen (5 synthetische localhost-Provider): Häkchen ≈ 0,1 s, `_refresh_components` 0,08 s, Dialog öffnen 0,45 s. Die Zeiten echter, nicht erreichbarer Remote-Provider sind nicht gemessen. Tests: `tests/test_provider_toggle_nonblocking.py`.

### ✅ FIXED (Oct 2026): Einfrieren der Einstellungen mit mehreren Providern
`config_changed` (jedes Häkchen, jede Bearbeitung, Speichern) rief im Hauptfenster `llm_service.refresh_all_provider_status()` synchron im UI-Thread: jeder aktive Provider nacheinander, `force_check=True`, bis 3 s TCP-Timeout, also linear in der Zahl nicht erreichbarer Provider. Jetzt `_refresh_provider_status_async` (Thread, kein Überlappen); die Statusleiste fragt nach einem Einstellungswechsel keine Modellliste mehr ab (`update_provider_info(allow_network=False)`). Dazu lief „Modelle abfragen" synchron und blockierte bei einem Server, der die Verbindung annimmt und nie antwortet, unbegrenzt (der Ollama-Client hat `Timeout(None, connect=5)`); jetzt im Worker. Gemessen nur mit synthetischen localhost-Providern; die 3-s-Summe ist aus dem Code abgeleitet, nicht an echten toten Hosts gemessen.

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
