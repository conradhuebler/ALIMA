# Audit Findings — Wissensbasis für WP-Planung

**Status**: Snapshot der Code-Inspektion. Soll Annahmen in
`agentic_ui_workpackages.md` durch Fakten ersetzen.

## 1. Prompts: JSON ist Wahrheit, YAML existiert parallel
- `prompts.json` ist gewireter Standard:
  - `pipeline_utils.py:6277` baut `PromptService(prompts_path, …)` mit
    Default `prompts.json` (`config_models.py:635`).
  - `AlimaManager.prompt_service` wird damit gefüttert.
- `prompts.yaml` existiert (29 KB) + `YamlPromptService` ist
  implementiert (`src/llm/yaml_prompt_service.py`), aber **nirgends
  importiert/instanziiert** im Wiring-Pfad (`grep -r YamlPromptService`
  → 0 Treffer außerhalb der eigenen Datei).
- `prompts.json` und `prompts.yaml` haben **unterschiedlichen Inhalt**
  (`diff -q` → "verschieden"). Risiko: still drift.

**Konsequenz für WP3**: nur prompts.json ist relevant. YAML-Variante ist
totes Code-Pfad oder geplanter Migrationspfad.

## 2. Webapp ist agentic-blind
- `src/webapp/app.py:993-1006` baut `PipelineConfig` aus
  `create_from_provider_preferences(config_manager)`.
- Keine Stelle setzt `enable_agentic_mode` oder `workflow_name`.
- Webapp-Frontend hat **keinen Workflow-Picker**.
- `global_override` (provider|model) wird unterstützt, aber kein
  Workflow-Switch.

**Konsequenz für WP9**: Webapp führt heute IMMER classic-Pipeline aus.
Tier-Modell muss explizit „Workflow-Auswahl" als Webapp-Lücke listen.

## 3. AbstractTab + analyse_keywords-Tab = generischer LLM-Runner
- `main_window.py:494` → `abstract_tab = AbstractTab(...)` mit
  `set_task("initialisation")`.
- `main_window.py:507` → `analyse_keywords = AbstractTab(...)` mit
  `set_task("keywords")`.
- **Beide sind dieselbe Klasse**, nur mit unterschiedlichem Default-
  Task. `AbstractTab.set_task()` wechselt prompts.json-Task on the fly.
- AbstractTab.task ∈ {initialisation, keywords, analysis, abstract,
  rephrase, …} — definiert durch verfügbare prompts.json-Tasks.
- PDF-Button visible nur für `task in ["abstract", "initialisation"]`
  (Spezialfall hardcoded).

**Konsequenz für WP1 + WP5**: Verifikation-Tab ≠ separater Code, =
Re-Use von AbstractTab. „Single-Step pro Workflow-Step" könnte als
Erweiterung dieses Patterns realisiert werden — aber heute
**prompts.json-Task** ist der Schlüssel, nicht Workflow-Step.

## 4. ComparisonTab ist KeywordAnalysisState-Diff (nicht „Vergleich")
- `comparison_tab.py:35` → lädt zwei JSON-Files (`KeywordAnalysisState`),
  zeigt side-by-side mit color-coded chips.
- Buttons: „Aktuell → A/B" laden current pipeline state in einen Slot.
- Reine Read-Only-Vergleichsansicht.

**Konsequenz für WP1**: ComparisonTab und ReviewTab sind funktional
verschieden. Comparison = Diff-Tool, Review = single-state-viewer.
Kein Merge-Kandidat.

## 5. Worker-Klassen — kein State-Sync-Bus, alles über Signals
- `src/ui/workers.py`:
  - `StoppableWorker` (Basis-QThread mit stop)
  - `SingleStepWorker` (eine LLM-Task)
  - `PipelineWorker` (volle Pipeline)
  - `DNBSyncWorker`
- Pro Worker eigene Signal-Set. Tab-zu-Tab-Updates über direkte
  pyqtSignal-Verbindungen in `main_window.py`.
- **Kein zentraler EventBus**. Jeder Tab muss explizit subscriben.

**Konsequenz für WP6**: State-Sync-Fundament ist heute *nicht* da. Jede
neue Mutation aus Chat würde N×M Signal-Verbindungen erfordern. WP6 ist
echte Neubau-Arbeit.

## 6. Workflow-Discovery: 3 Pfade aktiv, alle live
- `workflow_loader.py:55` `DEFAULT_SEARCH_PATHS`:
  1. `Path("workflows")` (CWD)
  2. `Path.home() / ".config/alima/workflows"`
  3. `<package>/workflows` (relative zu Source-Root)
- `find_workflow_file()` durchsucht alle drei in Reihenfolge.
- `pipeline_tab.py:1845-1850` + `cli/workflow_cmd.py:119` nutzen es.
- **Custom-Workflow-Upload via UI**: existiert nicht, aber
  Filesystem-Pfad funktioniert.

**Konsequenz für WP9**: Custom-Workflow-Pfad ist „funktioniert wenn
User Datei hinkopiert", kein UI-Pfad. Tier-Modell sollte fragen ob
UI-Upload nötig.

## 7. AgenticContextWidget = pro-Step-Panels mit `output_paths`
- `agentic_context_widget.py:60` `AgenticStepPanel` rendert pro Step
  ausschließlich die Felder, die der Step laut YAML schreibt
  (`output_paths`).
- Auto-Expand bei laufendem Step, manuell toggleable.
- **Workflow-agnostisch**: arbeitet mit jedem Workflow, kennt nur
  `step_id` + `output_paths` aus der YAML-Definition.

**Konsequenz für WP4**: AgenticContextWidget ist GUTES Beispiel für
Renderer-Plug-Pattern — schon workflow-agnostisch. Renderer-Registry
könnte daran anknüpfen.

## 8. Reproduzierbarkeit: Seed in classic, nicht in agentic
- `pipeline_utils.py` (classic): mehrere Stellen `seed=kwargs.get("seed", 0)`
  (Zeile 495, 1464, 1660, 2196).
- `LLMAgentStep.run()` + `AgentLoop`: **kein** `seed`-Parameter im
  Call-Pfad (`grep` 0 Treffer).
- Classic ist also reproduzierbarer als agentic.

**Konsequenz für WP2**: Forschungspfad-Argument für classic ist
empirisch begründet — agentic fehlt seed-Support. Müsste nachgerüstet
werden ODER classic bleibt für reproduzierbare Runs.

## 9. Telemetrie: nicht vorhanden
- `grep -r "telemetry|track_event|analytics"` → 0 Treffer.
- Keine Tab-Nutzungsdaten. Keine Workflow-Run-Counts.

**Konsequenz für WP1 + WP10**: Tab-Audit muss qualitativ erfolgen
(Code-Inspektion + User-Aussage), nicht datengetrieben.

## 10. DkAnalysisUnifiedTab — keine Hardcode-Stellen sichtbar im quick-grep
- `grep "self\.dk_"` lieferte 0 Treffer auf der Datei.
- Tiefere Inspektion nötig (Methoden, nicht nur Attribute).

**Konsequenz für WP1**: DkAnalysisUnifiedTab muss vertieft auditiert
werden — quick-pass reicht nicht.

## 11. Provider-Override-Mechanik existiert
- `PipelineConfig.global_provider_override`,
  `global_model_override`, `apply_global_override()`.
- Webapp + GUI nutzen es.
- Aber: **eine** Override für **alle** Steps — keine per-Step-
  Provider-Wahl im config-Path (nur in YAML-Step `llm: { provider }`).

**Konsequenz für neuen Provider-WP**: per-Step-Provider-Mix ist
YAML-möglich aber nicht im UI-Config-Pfad (nur global-Override + YAML-Edit).

## 12. AnalysisReviewTab = Multi-Tab-Result-Viewer (10 Sub-Tabs)
- `analysis_review_tab.py:189` `details_tabs = QTabWidget()` mit:
  Original Abstract, Initial Keywords, Such-Ergebnisse, GND-Keywords,
  Finale Analyse, Chunks, Iterationsverlauf, DK/RVK, K10+ Export,
  Statistiken, plus DK-Statistik-Tabelle (Top-10).
- Plus oberhalb: Batch-Tabelle (`batch_table_widget`) für Batch-Review.
- Lädt JSON-Files (KeywordAnalysisState).

**Konsequenz für WP1 + WP4**: AnalysisReviewTab ist heute der
**de-facto-Output-Renderer** — Sub-Tabs entsprechen "Render-Slots".
Renderer-Registry kann Sub-Tabs schrittweise extrahieren.

## 13. DkAnalysisUnifiedTab erbt AbstractTab — kein Hardcode
- `dk_analysis_unified_tab.py:25` `class DkAnalysisUnifiedTab(AbstractTab)`.
- 7 zusätzliche Methoden:
  `set_keywords`, `restore_keywords_input`, `on_analysis_completed`,
  `update_data`, `receive_pipeline_results`, `receive_catalog_results`,
  `add_external_analysis_to_history`.
- Logik: nimmt `dk_search_results_flattened` (oder Fallback
  `dk_search_results`) als Keywords, ruft AbstractTab-LLM-Call.
- **DkAnalysisUnifiedTab = AbstractTab + DK-spezifische Pipeline-
  Result-Slots**. Kein eigener LLM-Code.

**Konsequenz für WP4 + WP5**: Pattern „AbstractTab subclass mit
spezialisierten Receive-Slots" ist wiederholbar für andere Steps.
Single-Step-Tabs könnten generiert werden statt manuell gepflegt.

## 14. AgentLoop ist schon multi-turn-fähig (Path B existiert!)
- `agent_loop.py:79-107`: arbeitet intern mit `messages: List[Dict[str, Any]]`.
- `messages.append(assistant_msg)` (Z.134), `messages.append({tool_result...})` (Z.146),
  `messages.append({user...})` (Z.225) — echte conversational message
  history, nicht String-Render.
- `LlmService.generate_with_tools(model, messages=messages, tools=...)`
  ist die provider-agnostische Schnittstelle.

**Konsequenz für meine S2-Selbstkritik**: **Path B ist bereits da**.
Multi-turn-Chat = nur externe Aggregation der user-turns in dasselbe
`messages`-Array. AgentLoop muss nur eine Variante bekommen die
existing messages annimmt statt sie aus system_prompt+user_prompt zu
bauen. WP8 wird kleiner als gedacht.

## 15. AbstractTab hat Prompt-Variant-Selector (multi-model-ready)
- `abstract_tab.py:725-749`: `populate_task_selector()` +
  `populate_prompt_selector()` lesen aus `prompt_manager.get_prompts_for_task(task)`.
- prompts.json kann pro Task mehrere Varianten enthalten ("Prompt Set 1",
  "Prompt Set 2"). User wählt manuell.
- AbstractTab zeigt + erlaubt Editing von system_prompt + user_prompt.

**Konsequenz für WP11**: Multi-Variant-Schema ist im UI schon präsent,
braucht nur sinnvoll gefüllte prompts.json (Varianten pro Modell-
Familie). Prompt-Auto-Auswahl basierend auf Modell wäre WP11-Kern.

## 16. LlmService — 4 Provider-Typen, OpenAI-compatible deckt GWDG
- `llm_service.py:152-176`: gemini, anthropic, ollama, openai_compatible.
- `openai_compatible` ist generisch (URL + key) → GWDG, OpenAI selbst,
  Together, Fireworks, alle drauf.
- Cancel-Mechanik provider-spezifisch (`current_request_id` für
  openai/anthropic, eigener Pfad für ollama).

**Konsequenz für WP11**: Operator-Vorgabe (OpenAI-API + GWDG + Ollama)
reduziert auf **2 effektive API-Familien**: openai_compatible + ollama.
Test-Matrix entsprechend kompakter (vs Anthropic/Gemini eigene APIs).

## 17. Worker-Inflation durch BatchProcessingDialog
- `batch_processing_dialog.py` bringt eigene 3 Worker-Klassen mit:
  `SiegelFetchWorker`, `SiegelCacheLoadWorker`, `BatchProcessingWorker`.
- Plus `BatchProcessingWorker` ruft `BatchProcessor.process_batch_file()`
  → eigener Code-Pfad für Batch.

**Konsequenz für WP1**: Batch hat 3 spezielle Worker zusätzlich zu
`workers.py`. Konsolidierung mit Pipeline-Worker ggf. möglich.

## Lücken die ich nicht aufgelöst habe (bewusst, jetzt zu tief)

- AnalysisReviewTab interne Tab-Struktur (10+ Sub-Tabs vermutlich)
- DkAnalysisUnifiedTab Methoden-Inventar
- BatchProcessingDialog vs CLI batch-Modus Differenz
- FirstStartWizard Coverage (was wird gefragt)
- Conversation-Pattern in `agent_loop.py` (echte messages oder String-Render)

→ Diese gehören in WP1 + WP4 + WP9 als reguläre Audit-Arbeit.
