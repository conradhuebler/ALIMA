# Konsolidierungs-Inventar

**Was geht schon, was wiederverwendbar?** Vermeidet Neubau wo
existierender Code Pattern liefert. Quellen: [`audit_findings.md`](../audit_findings.md).

## A. Backend-Bausteine die direkt tragen

### A1 — Plugin-Registry-Pattern (STEP_REGISTRY + TOOL_FN_REGISTRY)
- `src/core/agents/registry.py`
- Decorator-basiert (`@register_step`, `@register_tool_fn`).
- Erfolgreich für Workflow-Steps + Deterministic-Functions.

**Wiederverwendbar für**: Renderer-Registry (WP4),
Chat-Tool-Registry (WP7). Gleiche Mechanik einsetzen für UI-Konsistenz.

### A2 — workflow_loader 3-Pfad-Discovery
- Workflow-YAMLs aus `./workflows/`, `~/.config/alima/workflows/`,
  `<package>/workflows/`.
- `find_workflow_file(name)` + `_discover_workflows()`.

**Wiederverwendbar für**: Custom-Renderer-Discovery (analog),
Custom-Tool-Discovery, User-Pluggable-Erweiterungen.

### A3 — AgentLoop ist multi-turn-fähig
- Audit-Finding 14: arbeitet intern mit `messages: List[Dict]`,
  `LlmService.generate_with_tools(messages=...)` ist provider-agnostisch.
- Iteration-Cap, Repeat-Detection, Stream-Callback bereits drin.

**Wiederverwendbar für**: Chat-Multi-Turn (WP8). Braucht nur einen
neuen Einstiegspunkt der existing `messages` annimmt statt aus
system+user neu baut. Aufwand: ~1 Tag.

### A4 — ToolRegistry + Presets (`src/mcp/`)
- ToolDefinition-Schema, register/dispatch.
- Presets in YAML (`default_presets.yaml`), custom in
  `~/.config/alima/tool_presets.yaml`.

**Wiederverwendbar für**: Chat-Tool-Set (WP7). Workflow-aware Presets
(z.B. `chat_alima_classic`, `chat_title_list_search`) via gleichem
Mechanismus.

### A5 — PipelineConfig.global_*_override
- `global_provider_override`, `global_model_override`,
  `apply_global_override()`.
- Webapp + GUI nutzen es heute.

**Wiederverwendbar für**: Chat-Provider-Override (WP11). Per-Step-
Override braucht Erweiterung (heute nur global).

### A6 — model_chunking_thresholds + model_capabilities
- `src/utils/model_capabilities.py`: Pattern-Matching für Modellgrößen
  → Chunking-Threshold.
- 15+ Patterns, Default 500 Keywords.

**Wiederverwendbar für**: Provider-Capability-Profil (WP11). Schema
erweitern um json_mode, tool_use_native, vision, max_context, seed,
streaming, statt nur chunking.

### A7 — prompts.json multi-variant-Schema
- Pro Task `prompts: [[user, system, temp, p_value, models]]`.
- Models-Liste pro Variante → variante-pro-Modell-Familie technisch
  bereits möglich.
- AbstractTab zeigt Varianten als "Prompt Set 1/2/...".

**Wiederverwendbar für**: Multi-Provider-Prompts (WP11). Prompts pro
Modell-Familie definieren, Auto-Selection statt manuell.

### A8 — KeywordAnalysisState als Single-Source-of-Truth
- Zentrales DataClass das von Pipeline-Manager + Tabs gelesen wird.
- Bietet `classifications` als Property-Alias für
  `dk_classifications`.

**Wiederverwendbar für**: State-Sync-Bus (WP6). Zentrale Änderungs-API
auf diesem DataClass aufsetzen, Mutation-Events emittieren.

### A9 — PipelineJsonManager save/resume
- JSON-basierte Persistenz für KeywordAnalysisState + SharedContext
  (über `save_to_file`/`load_from_file`).
- Schon von Pipeline + ComparisonTab genutzt.

**Wiederverwendbar für**: Chat-Session-Persistenz (WP8 optional),
Single-Step-Warm-Start (WP5).

## B. UI-Bausteine die direkt tragen

### B1 — AgenticContextWidget (workflow-agnostisch)
- `agentic_context_widget.py:60` AgenticStepPanel rendert pro Step
  nur die Felder die der Step laut YAML schreibt (`output_paths`).
- Auto-Expand bei laufendem Step, manuell toggleable.

**Wiederverwendbar für**: Renderer-Plug-Vorbild (WP4). Pattern:
"YAML-deklarierte Output-Paths → Widget rendert diese Slice".

### B2 — AbstractTab als generischer LLM-Runner mit Task-Selector
- `abstract_tab.py:162` `set_task()`, `populate_task_selector()`,
  `populate_prompt_selector()`.
- Task-Combo + Prompt-Variant-Combo + System/User-Prompt-Editor +
  Stream-Output + Result-History.
- DkAnalysisUnifiedTab erbt → Pattern bewährt.

**Wiederverwendbar für**: Single-Step-Tabs (WP5). Pattern „Subclass
+ Receive-Slot" funktioniert. Auto-generierte Tabs aus Workflow-Step
sind realistisch.

### B3 — AnalysisReviewTab als Multi-Slot-Renderer
- 10 Sub-Tabs für verschiedene Result-Aspekte (Audit-Finding 12).
- Lädt JSON, populates each Sub-Tab aus KeywordAnalysisState-Feldern.

**Wiederverwendbar für**: Renderer-Registry-Migration (WP4). Sub-Tabs
sind faktische Renderer — extrahieren als Renderer-Klassen, Reuse in
Pipeline-Tab + Single-Step-Tabs.

### B4 — PipelineChatPanel für Live-Stream
- 803 Zeilen, Token-für-Token-Anzeige pro Step.
- Repetition-Warning-UI integriert.

**Wiederverwendbar für**: Universal-Stream-Komponente. Auch im Chat
für Re-Run-Stream-Output (WP8).

### B5 — ComparisonTab für Two-State-Diff
- Lädt 2 KeywordAnalysisState-JSONs side-by-side.
- "Aktuell → A/B" Buttons.

**Wiederverwendbar für**: Forschungspfad-Vergleich classic vs agentic
(WP2). Funktional schon da, ggf. um Provider-Vergleich erweitern (WP11).

### B6 — workflow_combo + workflow_hints in PipelineConfigDialog
- Heute nur in Pipeline-Konfig versteckt.
- Hint-System (`WORKFLOW_HINTS`) — Workflow-Beschreibung im UI.

**Wiederverwendbar für**: Workflow-Browser (WP1/WP10). Hint-Pattern
ausbauen, prominenter platzieren.

### B7 — BatchProcessingDialog mit eigenem Worker-Pattern
- `batch_processing_dialog.py:89` BatchProcessingWorker.
- Tab-basierte Eingabe (Datei vs Verzeichnis), Filter, Live-Log,
  Progress-Bar, Cancel.

**Wiederverwendbar für**: Single-Pipeline-Run-UI-Pattern (WP5). Modal
+ Live-Log + Cancel ist gut bewährt.

### B8 — ChatWidget als Dock + System-Prompt-Dialog
- Existiert (uncommitted im Repo).
- Reset-Toggle, Provider-Combo, Streaming-History, Modal-Dialog für
  System-Prompt.

**Wiederverwendbar für**: Chat-UI-Konzept (WP8). Skeleton vorhanden,
nur Tool-Use + Mutations-Dialoge ergänzen.

## C. CLI-Bausteine

### C1 — alima workflow / workflows list
- `src/cli/commands/workflow_cmd.py` — Workflow-Discovery + Exec.
- `--input`, `--input-file`, `--output`, `--only-step`.

**Wiederverwendbar für**: CLI-Tier-1-Backend (WP9). Single-Step
funktioniert hier schon.

### C2 — Modulare Subcommand-Struktur
- `src/cli/commands/{pipeline,workflow,search,protocol,state,provider,
  database,setup}_cmd.py`.

**Wiederverwendbar für**: Neue Subcommands (z.B. `alima chat
<workflow>`) folgen gleichem Pattern.

## D. Webapp-Bausteine

### D1 — Session-basiertes Modell + WebSocket-Streaming
- `src/webapp/app.py`: Session-Lifecycle, WS-Streaming, Auto-Save,
  Recovery (Audit-Finding 2 + AIChangelog-Eintrag).
- 30-min WS-Timeout, 5-Sek-Heartbeat.

**Wiederverwendbar für**: Webapp-Catch-Up (WP9). Streaming-Infrastruktur
trägt schon, fehlt nur agentic + Workflow-Auswahl.

### D2 — global_override-Param
- `start_analysis(...)` nimmt `global_override: provider|model`.

**Wiederverwendbar für**: Provider-Wahl in Webapp ohne neuen UI-Bau
(WP11).

## E. Was NICHT wiederverwendbar (Neubau)

| Komponente | Warum |
|---|---|
| State-Sync-Bus | Existiert nicht (Audit-Finding 5) — direkte Signal-Spider heute. |
| Renderer-Registry | Existiert nicht — sub-tabs sind hardcoded. |
| Chat-Tool-Set workflow-aware | Existiert nicht — Chat ist statischer Context-String. |
| Per-Step-Provider-UI | Heute nur global-override + YAML-edit. |
| Workflow-Browser-Tab | Heute nur Combo im Config-Dialog versteckt. |
| Provider-Capability-Profil | Heute nur Chunking-Hints. |
| YAML-Editor-UI | Heute nur File-System. |
| Telemetrie-Bus | Existiert gar nicht. |

## F. Zwei vergleichbare Code-Pfade die bewusst leben (klassisch + agentic)

| Aspekt | classic (`pipeline_utils.py`) | agentic (`WorkflowExecutor`) |
|---|---|---|
| Step-Definition | Hardcoded in Python | YAML-deklariert |
| Reproduzierbarkeit | seed-Pfad da | seed fehlt (Audit-Finding 8) |
| Chunking | Klassisch hartkodiert | YAML `chunking:`-Block |
| State | KeywordAnalysisState direkt | SharedContext → übersetzt nach KeywordAnalysisState |
| MetaAgent-Loop | Nein | Ja (optional) |
| Tools | Suggester-Plugins | MCP-Tools + Tool-Functions |
| Prompts | prompts.json | inline YAML (oder prompts.json fallback) |
| iterative Search | Da | Fehlt |
| Repetition Detection | Da | Erbt teilweise |

→ **Forschungspfad classic** profitiert von Determinismus + iterativer
Refinement. Beide bleiben (Operator-Direktive). WP2 quantifiziert
Differenz, WP10 plant wie sie nebeneinander gepflegt werden.
