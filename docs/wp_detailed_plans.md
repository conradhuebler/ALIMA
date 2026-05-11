# Detaillierte WP-Pläne (vorläufig, tief durchdacht)

**Status**: Pro WP konkretisierter Arbeitsplan: Schritte, Outputs,
Querverweise auf andere WPs + auf existierende Bausteine
(`consolidation_inventory.md`). KEINE Implementation. Soll vor
Implementation pro WP nochmal reviewt + freigegeben werden.

Querverweis-Konventionen:
- **Reuse**: existierender Code aus `consolidation_inventory.md` Sektion A/B/C/D.
- **Block**: Output dieses WP wird von WP-X gebraucht.
- **Need**: dieses WP braucht Output von WP-X.

---

## WP1 — Tab-Inventar-Audit (Tiefe)

### Ziel
Vollständige, faktenbasierte Tabelle aller UI-Einheiten (GUI-Tabs,
Webapp-Endpoints, CLI-Commands) mit Funktion, Datenfluss, Worker-
Patterns, Überlappungen.

### Schritte
1. **GUI-Tabs vertiefen** — pro Tab: Methoden-Inventar (`ast` parse),
   read-from (KeywordAnalysisState-Felder), write-to (Mutation-Stellen),
   eigene Worker, eigene Signals, geteilte Helpers. Output: Tabelle mit
   ~10 Spalten.
2. **Sub-Tabs auflisten** — AnalysisReviewTab hat 10, DkAnalysisUnifiedTab
   hat eigene Sub-Strukturen, evtl. andere auch. Komplette Sub-Tab-
   Karte.
3. **Webapp-Endpoints klassifizieren** — pro Endpoint: erfasste
   Funktion, Kanonischer Tab-Äquivalent (Mapping), agentic-Support
   ja/nein, Lücken.
4. **CLI-Subcommands** — pro Command: Funktion, GUI-Tab-Äquivalent.
5. **Überlapp-Matrix** — Tab × Funktion (Crossref, GND-Search,
   DK-Search, ...). Wo doppelte Implementierungen?
6. **Empfehlung pro Tab** — keep / merge / replace / delete / refactor.
7. **User-Stimme einholen** (außerhalb Code): welche Tabs werden
   benutzt? (Operator-Befragung statt Telemetrie da nicht vorhanden,
   Audit-Finding 9.)

### Output
- `docs/audit_tab_inventory.md` — Hauptdokument mit:
  - Tab-Tabelle (10 Tabs)
  - Sub-Tab-Karte
  - Webapp-Endpoint-Mapping
  - CLI-Mapping
  - Überlapp-Matrix
  - Empfehlung pro Tab

### Querverweise
- **Reuse**: keiner (rein Audit).
- **Block**: WP4 (Sub-Tab-Karte = Renderer-Slots), WP5 (Tab→Step-
  Mapping), WP10 (Migrationspriorität).
- **Need**: keine.

### Risiken
- Operator-Befragung kann subjektiv sein.
- 28k Zeilen UI-Code: Vollständigkeit nicht garantiert. Time-Box auf
  4-6 Stunden, dann was hat man hat.

### Dauer-Schätzung
1 Tag (mit Operator-Feedback-Pass).

---

## WP2 — Classic ↔ Agentic Pipeline-Vergleich

### Ziel
Quantifizierte Aussage: was kann classic, was kann agentic, wo
divergieren Outputs für gleichen Input. Forschungspfad-Schutz
empirisch begründen.

### Schritte
1. **Output-Feld-Tabelle** — alle Felder die jeweils erzeugt werden
   (KeywordAnalysisState + dk_search_results + classifications +
   refinement_iterations + dk_statistics + SharedContext.extra für
   agentic). Diff.
2. **Reproduzierbarkeits-Test** — gleicher Input × gleicher Provider/
   Modell × gleicher Seed × 3 Runs pro Pfad. Output-Diff messen.
   Erwartung: classic deterministisch, agentic nicht (Audit-Finding 8).
3. **Token-Cost-Vergleich** — `LlmService` loggt Tokens. Mittelwert
   pro Pfad für Standardabstract.
4. **Feature-Lücken** — was kann classic was agentic nicht (iterative
   Search, repetition detection, dk_statistics) und umgekehrt
   (MetaAgent-Loop, freie Tool-Use durch LLM).
5. **Forschungs-Use-Case-Definition** — Operator-Klärung: was ist
   "Forschung" konkret? Welche Garantien werden verlangt?
6. **Empfehlungs-Tabelle** — pro Use-Case (Routine-Erschließung,
   Forschungs-Run, Batch, exploratives Workflow): welcher Pfad?

### Output
- `docs/research_classic_vs_agentic.md`:
  - Output-Feld-Diff
  - Reproduzierbarkeitsmessung (Tabelle)
  - Token-Cost-Tabelle
  - Feature-Matrix
  - Use-Case-Empfehlung
  - Decision-Vorschlag: dual-pfad bleibt (mit klarer Verantwortungs-
    Splittung) ODER classic als Workflow-YAML konsolidieren.

### Querverweise
- **Reuse**: B5 (ComparisonTab) für sichtbaren Diff der konkreten
  Test-Runs.
- **Block**: WP10 (Migrationsplan darf classic nicht brechen),
  WP4 (Renderer für agentic-only Felder).
- **Need**: WP11 (für Reproduzierbarkeits-Test braucht es Provider-
  Test-Setup).

### Risiken
- Reproduzierbarkeits-Tests brauchen stabile Provider-Verfügbarkeit.
- Token-Vergleich ist provider-abhängig.

### Decision-Point
**T1**: classic-als-Workflow oder dual-pfad dauerhaft. Beeinflusst
WP4 + WP10.

### Dauer-Schätzung
1.5 Tage.

---

## WP3 — Output-Schema-Inventar pro Workflow

### Ziel
Für alle 6 vorhandenen Workflows + classic: vollständige Liste
welche Felder welchen Typs in `SharedContext` (typed + extra) +
KeywordAnalysisState landen. Vorbedingung WP4 + WP7.

### Schritte
1. **Pro Workflow YAML lesen + Step-Outputs extrahieren** — `outputs:`-
   Block pro Step → Feldname + Pfad. 6 Workflows × ~5 Steps = ~30
   Outputs.
2. **Datentypen ermitteln** — durch tatsächliche Test-Läufe + Code-
   Inspektion der register-fns. Output-Beispiele.
3. **Render-Slot-Vokabular ableiten** — Cluster der Output-Typen:
   `keyword_list`, `keyword_chains`, `gnd_pool`, `dk_table`,
   `classification_list`, `duplicate_table`, `metadata_record`,
   `text_blob`, `raw_json_fallback`.
4. **Tool-Inventar pro Workflow** — welche Chat-Tools machen Sinn
   pro Output (z.B. `get_dk_titles_for_code` nur wenn dk_search_results
   im Schema). Vorbereitung WP7.
5. **Schema-Versionierung** — wie umgehen mit zukünftigen
   Schema-Änderungen? Vorschlag: `schema_version:` pro Workflow.

### Output
- `docs/workflow_output_schemas.md`:
  - Schema pro Workflow (Tabelle)
  - Render-Slot-Vokabular (Liste mit Beschreibung)
  - Tool-Inventar pro Workflow
  - Versionierungs-Vorschlag.

### Querverweise
- **Reuse**: A1 (Plugin-Pattern), A7 (prompts.json multi-variant — falls
  Schema-Hinweise dort sinnvoll).
- **Block**: WP4 (Render-Slots = Input für Renderer-Registry),
  WP7 (Tool-Inventar = Input), WP11 (Provider-Schema-Robustheit).
- **Need**: keine direkte; nutzt Audit-Finding 1 (prompts.json ist
  Wahrheit).

### Decision-Point
**T1**: Render-Slot-Vokabular fixiert. Ändert sich später mit hohem
Aufwand → früh festlegen.

### Risiken
- Workflow-Outputs sind dynamisch — nicht alle Felder kommen immer
  vor (z.B. iterative_refinement nur bei Konvergenz).
- Render-Slot-Vokabular zu eng oder zu breit → schwer im Voraus.

### Dauer-Schätzung
1 Tag.

---

## WP4 — Renderer-Registry-Skizze

### Ziel
Pluggable Output-Renderer. Code-Sketch mit 2-3 Beispielen, ohne Build.
Beweisen ob Option C (aus `ui_requirements_catalog.md`) trägt.

### Schritte
1. **API-Sketch** — pseudo-Code:
   ```python
   @register_renderer("dk_table")
   class DkTableRenderer(BaseRenderer):
       output_slot = "dk_table"
       def render_qt(self, data, context) -> QWidget: ...
       def render_html(self, data, context) -> str: ...   # webapp
       def render_cli(self, data, context) -> str: ...    # ascii table
   ```
2. **Renderer-Registry** — analog A1: `RENDERER_REGISTRY: Dict[str, Type[BaseRenderer]]`
   plus `@register_renderer` decorator.
3. **YAML-Hint pro Step** — `output_schema: dk_table` als optional-
   Annotation pro Step. Default-Mapping wenn nicht gesetzt
   (z.B. `keyword_chains` aus Feldnamen ableiten).
4. **3 Beispiel-Renderer** — `dk_table` (aus DkAnalysisUnifiedTab
   extrahiert), `keyword_chains` (aus AnalysisReviewTab.Sub-Tab),
   `raw_json` (Fallback).
5. **Frontend-Mapping** — wie weit teilen GUI/Webapp den Renderer-
   Code? Vorschlag: Renderer liefert Daten + HTML-Template-String,
   GUI rendert in QTextBrowser, Webapp injiziert in Template.
6. **Migrations-Pattern** — wie bestehende Sub-Tabs (AnalysisReviewTab
   B3) zu Renderer-Klassen werden, ohne dass User-Erfahrung kippt.
7. **Backward-Compat** — Tabs die heute hardcoded rendern können
   intern Renderer aufrufen, äußerlich gleich aussehen.

### Output
- `docs/renderer_registry_design.md`:
  - API-Sketch
  - Registry-Mechanik
  - 3 Beispiel-Renderer (Pseudo-Code)
  - Frontend-Mapping (GUI/Webapp/CLI)
  - Migrations-Strategie pro Bestands-Tab.

### Querverweise
- **Reuse**: A1 (Registry-Pattern), A2 (Discovery), B1
  (AgenticContextWidget als Plug-Vorbild), B3 (AnalysisReviewTab-
  Sub-Tabs als Renderer-Quelle).
- **Block**: WP5 (Single-Step braucht Output-Renderer), WP8 (Chat
  zeigt Tool-Outputs ggf. mit Renderern), WP10 (Migration).
- **Need**: WP3 (Render-Slots).

### Decision-Point
**T2**: Renderer-Plugin-Architektur ja/nein. Wenn nein → bei Tab-pro-
Workflow-Hardcoding bleiben.

### Risiken
- Frontend-übergreifender Renderer (GUI+Webapp) ist trickreich
  (Qt vs HTML).
- Renderer-Inflation: pro Workflow neuer? Long-Tail.

### Dauer-Schätzung
1.5 Tage (Skizze + 3 Beispiel-Renderer als Pseudo-Code).

---

## WP5 — Single-Step-Execution-Modell

### Ziel
Klärt UI-Pattern + Backend-Mechanik wie ein einzelner Workflow-Step
isoliert ausführbar wird. Heute teilweise da (CLI `--only-step`,
WorkflowExecutor `only_step=`), UI fehlt strukturiert.

### Schritte
1. **Status-Quo CLI durchspielen** — wie genau funktioniert
   `--only-step`? Was muss in SharedContext schon stehen damit Step
   X läuft?
2. **Input-Schema-Auflösung** — pro Step aus YAML `inputs:`-Block
   ableiten welche Felder „User-fillable" (z.B. abstract) vs „aus
   vorigem Step" (z.B. extracted_keywords). Generator dafür.
3. **UI-Pattern** — Subclass-AbstractTab-Pattern (B2) für jeden Step?
   Auto-generiert aus Workflow-YAML beim App-Start? Oder hardcoded
   für Standard-Workflow + auto-generated für custom?
4. **Output-Übernahme-Mechanik** — Single-Step-Result soll in
   nachfolgenden Step übernehmbar sein. UI-Knopf "→ nächster Step
   übernehmen"?
5. **Warm-Start aus JSON** — A9 nutzen: lade KeywordAnalysisState
   oder SharedContext, wähle Step aus Combo, run.
6. **Chunking-Single-Step-Verhalten** — was wenn Step `chunking: enabled:
   true` ist? Alle Chunks oder einzelne Chunks?
7. **Generator vs Hardcode-Tradeoff** — Auto-generierte Tabs für 6
   Workflows × 5-7 Steps = 30-42 Tabs. Lösung: nur aktiver Workflow.
   Oder Tabs als Side-Panel nicht Top-Level.

### Output
- `docs/single_step_model.md`:
  - Status-Quo-Beschreibung
  - Input-Schema-Auflösung (Algorithmus)
  - UI-Pattern (Wireframe-Text)
  - Output-Übernahme-Flow
  - Warm-Start-Sequence
  - Tab-Generator-Konzept
  - Migration-Plan: AbstractTab + analyse_keywords ersetzen oder
    transformieren?

### Querverweise
- **Reuse**: B2 (AbstractTab + Task-Selector als Vorbild),
  B7 (Modal-Pattern für Single-Run), A9 (JSON-Persistenz),
  C1 (CLI-Pfad).
- **Block**: WP10 (Migration der Single-Step-Tabs).
- **Need**: WP3 (Input-Schema), WP4 (Output-Renderer).

### Decision-Point
**T3**: Auto-generierte Tabs vs hardcoded. Beeinflusst Tab-Inflation.

### Risiken
- Auto-Tab-Generator zu rigide (UX-Risiko).
- Pattern „pro Step ein Tab" vs Workflow mit 7 Steps → UI überladen.

### Dauer-Schätzung
2 Tage (Konzept + 2 Wireframe-Varianten).

---

## WP6 — State-Sync + Mutations-Fundament

### Ziel
Single-Source-of-Truth-Modell + Event-Bus, damit Chat-Mutationen
+ Multi-Tab-Updates konsistent bleiben. Audit-Finding 5: existiert
heute nicht.

### Schritte
1. **State-Modell-Klärung** — KeywordAnalysisState ist Master für
   classic + post-pipeline-State. SharedContext lebt nur während
   agentic-Lauf. Frage: was nach Lauf-Ende? Heute SharedContext
   wird verworfen → Audit-Finding "must keep `last_shared_context`"
   aus `agentic_chat_plan.md`.
2. **Event-Bus-Pattern** — Vorschlag: `AlimaStateBus` Singleton mit
   `emit(event_type, payload)` + `subscribe(event_type, handler)`.
   Qt-basiert (pyqtSignal-Wrapper) für GUI, FastAPI-Event für Webapp.
3. **Mutations-API auf KeywordAnalysisState** —
   `apply_keyword_replacement(old, new, gnd_id)`,
   `add_keyword(kw, gnd_id)`,
   `remove_keyword(kw)`,
   `replace_classification(old, new)`,
   `set_dk_search_results(...)`.
   Jede Mutation emittiert `state_changed(diff)`.
4. **Diff-Format** — kompakt, JSON-basiert (`{op: "replace_keyword",
   old: ..., new: ...}`).
5. **Locking/Race-Condition** — wenn Pipeline-Worker schreibt während
   Chat-Mutation: Worker hat Vorrang, Chat-Mutation queued bis
   Worker done.
6. **Undo/Redo** — auf Diff-Stack basieren. Pro Pipeline-Run reset?
7. **Persistenz** — Audit-Log der Diffs als JSON neben Pipeline-Result?

### Output
- `docs/state_sync_design.md`:
  - State-Master-Modell (post-Lauf inkl. SharedContext-Retention)
  - EventBus-API
  - Mutations-API
  - Diff-Format
  - Locking-Pattern
  - Undo/Redo-Plan
  - Persistenz-Optional.

### Querverweise
- **Reuse**: A8 (KeywordAnalysisState als SOT), A9 (JSON-Persistenz für
  Audit-Log).
- **Block**: WP7 (Schreib-Tools brauchen Mutations-API), WP8 (Modals
  emittieren Events).
- **Need**: keine direkte.

### Decision-Point
**T2**: EventBus oder Qt-Signal-Spider? Auswirkt Frontend-Übertragbarkeit.

### Risiken
- Webapp-Sessions sind isoliert — Event-Bus pro Session statt global.
- Race-Condition zwischen Pipeline-Stream und Chat-Mutation.

### Dauer-Schätzung
2 Tage.

---

## WP7 — Chat-Tools workflow-aware

### Ziel
Chat-Tool-Set generisch (über `SharedContext.extra` introspizierbar)
plus per-Workflow registrierbare Spezial-Tools. Plus Provider-
Strategie + Read-Only-Modus.

### Schritte
1. **Tool-Klassen-Hierarchie** — `BaseChatTool` mit `name`,
   `description`, `parameters_schema`, `available_for(workflow_name)`,
   `execute(session, **args)`. Subclasses pro spezifischem Tool.
2. **Generische Tools** — `list_available_data` (Schema-Dump),
   `get_extra(path)`, `get_step_result(step_id, path)`,
   `get_messages_history()`.
3. **ALIMA-spezifische Tools** — `get_keywords(kind)`,
   `get_keyword_chains`, `get_dk_classifications`,
   `get_dk_titles_for_code`, `get_chunk_response`,
   `find_chunk_for_keyword`, `search_in_gnd_pool`.
4. **Tool-Discovery pro Chat-Init** — bei aktivem Workflow Tools
   filtern via `available_for(workflow_name)`. Nutzt Render-Slot-
   Vokabular aus WP3.
5. **MCP-Tool-Integration** — existing MCP-Tools (`search_gnd`,
   `search_lobid`) auch verfügbar machen. Read-Only-Modus
   (Audit-Finding S5): Setting `chat.no_cache_writes` macht
   `search_gnd` nicht-persistent (oder direct API ohne Cache).
6. **Anti-Halluzination** — `validate_gnd_term` mandatory vor jeder
   Keyword-Empfehlung (Tool-Beschreibung sagt LLM dass es das tun
   muss).
7. **Schreib-Tools** — `propose_keyword_replacement`,
   `propose_step_rerun`. Tool-Execution emittiert Event statt direkt
   zu mutieren (verbindet WP6).
8. **Provider-Strategie** — Chat-Provider unabhängig vom Pipeline-
   Provider, eigener Default in Config (`chat.default_provider`,
   `chat.default_model`). UI-Combo bleibt für Override.
9. **Multi-Turn-Path** — AgentLoop schon multi-turn-fähig (Audit-
   Finding 14 + A3). Externe Aggregation der user-turns + assistant-
   turns als `messages`-Liste.

### Output
- `docs/chat_tools_design.md`:
  - Tool-Klassen-Hierarchie
  - Generische + ALIMA-Tools (Tabelle)
  - Discovery-Mechanik
  - Read-Only-Modus
  - Anti-Halluzination-Pattern
  - Schreib-Tools-Pattern
  - Provider-Strategie
  - Multi-Turn-Integration mit AgentLoop.

### Querverweise
- **Reuse**: A3 (AgentLoop multi-turn), A4 (ToolRegistry), A5 (Provider-
  Override), A8 (State-Mutation).
- **Block**: WP8 (Chat-UI-Konzept).
- **Need**: WP3 (Output-Schemas), WP6 (Mutations-API), WP11
  (Provider-Strategie-Bestandteil).

### Decision-Point
**T3**: Generisch vs spezialisiert + Tool-Set-Erweiterungs-Mechanik
(YAML-deklariert oder Code-registriert).

### Risiken
- Tool-Inflation: User verwirrt mit zu vielen Tools.
- Anti-Halluzination ist nur so gut wie LLM-Compliance.

### Dauer-Schätzung
2 Tage.

---

## WP8 — Chat-UI-Konzept

### Ziel
Chat als Tier-1-Feature in GUI mit Tool-Call-Visualisierung,
Mutations-Modal, Multi-Turn-History, Webapp-API-Vorbereitung.

### Schritte
1. **UI-Position-Frage** — Dock vs Tab vs beide. Vorschlag: Dock
   bleibt (B8), zusätzlich kontextueller Mini-Chat in jedem Tab
   (Frage zur sichtbaren Daten-Slice).
2. **Tool-Call-Rendering** — kollabierbarer Block in History:
   `🔧 list_available_data() → {abstract: ..., keywords: 20, ...}`.
   Default kollabiert, klickbar zum Aufklappen.
3. **Mutations-Modal** — modaler Dialog für `propose_*`-Tools:
   "Replace *Freilandökologie* with *Pflanzenökologie*
   (GND-ID 4174277-3)? [Ja / Nein / Bearbeiten]". Editier-Modus
   öffnet Form für Änderung.
4. **Re-Run-Stream** — `propose_step_rerun` ausgeführt → Pipeline-
   Worker startet → Stream geht in PipelineStreamWidget (B4) zurück,
   nicht in Chat. Chat zeigt Status-Zeile.
5. **History-Persistenz** — Chat-Sessions speicherbar als JSON
   (Optional). Reset bei neuem Pipeline-Run als Default-On.
6. **Multi-Turn-Integration** — Chat-Worker (rewrite von ChatWorker)
   nutzt AgentLoop direkt mit `messages`-Liste pro Turn (A3).
7. **Webapp-API-Vorbereitung** — Chat-Endpoints definieren
   (`POST /api/chat/{sid}`, `WS /ws/chat/{sid}`), aber nicht bauen.
   Tier-3-Notiz für WP9.
8. **Provider-Combo** — bleibt im Chat-Header (B8), Default aus
   `chat.default_*` (WP7, WP11).

### Output
- `docs/chat_ui_design.md`:
  - UI-Position-Begründung
  - Wireframe Tool-Call-Block (ASCII)
  - Wireframe Mutations-Modal (ASCII)
  - Re-Run-Flow
  - Persistenz-Konzept
  - AgentLoop-Integration-Detail
  - Webapp-API-Skizze (für WP9 Tier-Mapping).

### Querverweise
- **Reuse**: A3 (AgentLoop), B4 (PipelineStreamWidget für Re-Run),
  B8 (ChatWidget-Skelett).
- **Block**: WP10 (Chat-Migration in Roadmap).
- **Need**: WP6 (Mutations-API), WP7 (Tools), WP11 (Provider-Default).

### Risiken
- Modaler Dialog während Streaming kann irritieren.
- Re-Run aus Chat während User in anderem Tab arbeitet.

### Dauer-Schätzung
2.5 Tage (statt ursprünglich 1 — S1-Selbstkritik).

---

## WP9 — Multi-Frontend-Tier-Modell

### Ziel
Pro Feature × Frontend Matrix definieren. Webapp-Strategie expliziert
(catch-up vs lite-variant) inkl. Roadmap-Notiz.

### Schritte
1. **Feature-Inventar** — aus M1-M6 (`ui_requirements_catalog.md`)
   detailliert jede Sub-Funktion (z.B. M1 → "Workflow-Wahl",
   "Streaming", "DK-Klassifikation-Anzeige", "K10+-Export").
2. **Frontend-Matrix** — Feature × {GUI, Webapp, CLI}. Pro Zelle:
   "vorhanden", "fehlt", "tier-1/2/3 sollte".
3. **Tier-Definition** — Tier-1 (überall), Tier-2 (GUI + Webapp),
   Tier-3 (nur GUI). Begründung pro Tier.
4. **Webapp-Strategie-Optionen**:
   - **Lite**: bewusst eingeschränkt, nur Tier-1.
   - **Catch-Up**: Roadmap pro Feature.
   - **Headless-Equivalent**: Webapp = Frontend zu Workflow-Engine,
     gleiche Funktion wie GUI.
5. **CLI-Strategie** — `alima chat <workflow>` realistisch?
   Welche Tier-1-Features fehlen heute?
6. **Geteilte Schicht** — JSON-Schemas für Workflow-Inputs/Outputs
   generieren aus Workflow-YAML + register-fns. Dient als OpenAPI-
   Basis für Webapp.

### Output
- `docs/frontend_tier_model.md`:
  - Feature-Inventar
  - Frontend-Matrix (große Tabelle)
  - Tier-Definitionen
  - Webapp-Strategie-Vorschlag (Empfehlung + Alternativen)
  - CLI-Strategie
  - Geteilte-Schicht-Vorschlag.

### Querverweise
- **Reuse**: D1 (Webapp-Streaming-Infra), D2 (Webapp-Override), C1
  (CLI-Workflow-Cmd), C2 (CLI-Modul-Pattern).
- **Block**: WP4 (Renderer-Frontend-Mapping), WP8 (Chat-Webapp-API),
  WP10 (Webapp in Migration).
- **Need**: WP1 (Webapp/CLI im Inventar enthalten).

### Decision-Point
**T1**: Tier-Festlegung. Webapp-Strategie. Beeinflusst Scope WP4-8.

### Risiken
- Webapp-Catch-Up ist signifikanter Aufwand.
- Lite-Variante lässt Webapp veralten.

### Dauer-Schätzung
1 Tag.

---

## WP11 — Provider/Modell-Portabilität

### Ziel
Pipeline + Agent + Chat funktionieren reproducible auf 3 Providern
(Operator-Vorgabe: OpenAI-API + GWDG + Ollama). Effektiv 2 API-
Familien (openai_compatible + ollama).

### Schritte
1. **Provider-Capability-Schema** — pro Provider × Modell:
   ```yaml
   providers:
     openai_compatible:
       gpt-4o:
         json_mode: true
         tool_use: native
         max_context: 128000
         seed: true
         streaming: true
         vision: true
       qwen-72b-via-gwdg:
         json_mode: false
         tool_use: limited  # GWDG abhängig
         max_context: 32000
         seed: false
         streaming: true
         vision: false
     ollama:
       llama3.1:8b:
         json_mode: true  # via format=json
         tool_use: native  # neuere Modelle
         max_context: 8192
         seed: true
         streaming: true
         vision: false
       qwen2.5:14b:
         ...
   ```
   Erweiterung von `model_capabilities.py` (A6).
2. **Prompt-Varianten-Strategie** — prompts.json schon multi-
   variant (A7). Pro Workflow-Step-Task neue Varianten anlegen pro
   Modell-Familie:
   - `qwen` / `deepseek` (thinking-Markup ok)
   - `llama` / `gemma` (kein thinking)
   - `openai-gpt` (function-calling-ready)
   - `default` (Fallback)
   - GWDG hostet meist Open-Source-Modelle → matcht ollama-Varianten.
3. **Auto-Selection-Logik** — bei LLM-Call: Modellname → Familie
   ableiten → Prompt-Variante wählen → Fallback default. Pattern-
   matching analog `model_capabilities.py`.
4. **Tool-Use-Wrap-Audit** — `LlmService.generate_with_tools()` sauber
   provider-agnostisch? Audit pro Provider, nicht nur openai_compatible.
   Heute Ollama tool-use ist neuer (≥ qwen2.5, llama3.1).
5. **Seed-Pflicht für Forschungspfad** — agentic-LLMAgentStep um
   `seed`-Parameter erweitern. classic hat schon (Audit-Finding 8).
6. **Test-Matrix** — pro Workflow × {OpenAI-API, GWDG, Ollama-local}:
   smoke-test. CI-fähig wenn API-Keys gesetzt, sonst skip.
7. **Per-Step-Provider-Mix-UI** — heute global (A5) + YAML. UI-Pattern
   für „cheap-extraction + premium-classification": pro Step in
   PipelineConfigDialog ein Provider-Combo? Oder neue Workflow-
   Variante (`alima_premium.yaml`, `alima_cheap.yaml`)?
8. **Chat-Provider-Default** — `chat.default_provider` in Config,
   eigene Logik (klein + schnell, weil Multi-Turn). Empfehlung:
   Ollama-local oder OpenAI-mini-class.
9. **Output-Format-Robustheit** — Test pro Provider: kommt JSON
   sauber? Falls nicht, Tolerant-Parser anstellen
   (`json_response_parser.py` schon da).

### Output
- `docs/provider_portability_design.md`:
  - Capability-Schema (Tabelle)
  - Prompt-Varianten-Strategie (Tabelle pro Workflow)
  - Auto-Selection-Algorithmus
  - Tool-Use-Audit-Ergebnisse
  - Seed-Plan für agentic
  - Test-Matrix
  - Per-Step-UI-Vorschlag
  - Chat-Provider-Default-Vorschlag.

### Querverweise
- **Reuse**: A5 (Override-Mechanik), A6 (Capability-Pattern erweitern),
  A7 (multi-variant prompts.json), C2 (CLI für Tests).
- **Block**: WP10 (kein Hardcode auf einen Provider), WP7 (Chat-
  Provider), WP8 (Provider-Default), WP4 (Renderer ggf. provider-
  abhängig bei Tool-Output-Format).
- **Need**: WP3 (Output-Schemas — wir testen Robustheit pro Schema).

### Decision-Point
**T2**: Capability-Schema fixiert. **T3**: Test-Matrix Scope.

### Risiken
- N×M Prompt-Varianten Pflege ist Aufwand.
- Provider ändern APIs (Anthropic-Format-Migration etc. bei zukünftiger
  Erweiterung).
- GWDG-Verfügbarkeit hängt von Uni-Login ab — CI-Test schwierig.

### Dauer-Schätzung
2 Tage Konzept + 0.5 Tag Test-Matrix-Skizze.

---

## WP10 — Migrations- und Decision-Timeline

### Ziel
Synthese aller WPs zu konkretem Migrationsplan + Roll-Back-
Pfaden + Forschungspfad-Schutz.

### Schritte
1. **Phasen-Definition** — z.B.:
   - **P-α**: Renderer-Skelett + 1 Beispiel-Renderer
   - **P-β**: AnalysisReviewTab Sub-Tabs als Renderer extrahieren
   - **P-γ**: Single-Step-Generator (auto-Tabs für aktiven Workflow)
   - **P-δ**: Chat-Tools + read-only
   - **P-ε**: Chat-Mutations-Tools + Modal
   - **P-ζ**: Webapp-Catch-Up Tier-1
   - **P-η**: Provider-Variants in prompts.json
   - **P-θ**: Tab-Konsolidierung (mergen oder löschen)
2. **Pro Tab/Endpoint Entscheidung** — Tabelle aus WP1-Audit:
   keep/merge/replace/delete + in welcher Phase + abhängige WPs.
3. **Forschungspfad-Schutz** — Smoke-Tests pro Phase:
   classic-Pipeline + agentic-Pipeline mit Referenz-Input → erwartete
   Outputs. Falls Drift → Phase blockiert.
4. **Roll-Back-Punkte** — pro Phase ein Tag/Branch + Plan wie zurück.
5. **User-Kommunikation** — Release-Notes-Template, evtl. In-App-
   Banner für Workflow-Wahl-Erklärung.
6. **Risiko-Matrix** — pro Phase Risiken + Mitigation.
7. **Reihenfolge-Entscheidung** — was zuerst:
   - Tab-Konsolidierung (User-sichtbar) oder
   - Backend-Renderer-Registry (unsichtbar erstmal) oder
   - Chat (neu) oder
   - Provider-Portability (Foundation).

### Output
- `docs/migration_roadmap.md`:
  - Phasen-Tabelle
  - Tab-Entscheidungen
  - Smoke-Tests
  - Roll-Back-Plan
  - Risiko-Matrix
  - Reihenfolge-Empfehlung mit Begründung.

### Querverweise
- **Reuse**: B5 (ComparisonTab als Smoke-Test-Tool), A9 (JSON für
  Smoke-Test-Fixtures).
- **Block**: Implementation-Start.
- **Need**: ALLE anderen WPs.

### Decision-Point
**T4**: Reihenfolge der Phasen. Beeinflusst Lieferplan.

### Risiken
- Phasen-Abhängigkeiten ändern sich während Implementation.
- User-Erwartungen: Tab-Konsolidierung sichtbar = drängt zuerst.

### Dauer-Schätzung
1 Tag (Synthese, kein Bau).

---

## Globale Decision-Calendar (aktualisiert)

| Slot | Was wird entschieden | Liefernde WPs |
|------|---------------------|---------------|
| **T0** | Start aller WP1-3, WP9, WP11 | — |
| **T1** | Render-Slot-Vokabular fixiert | WP3 |
| **T1** | Tier-Festlegung Frontends + Webapp-Strategie | WP9 |
| **T1** | Provider-Capability-Schema-Sketch | WP11 |
| **T1** | classic-als-YAML oder dual-pfad | WP2 |
| **T2** | Renderer-Plugin-Architektur ja/nein | WP4 |
| **T2** | EventBus-Pattern fix | WP6 |
| **T2** | Capability-Schema fix | WP11 |
| **T3** | Auto-Tabs vs hardcoded Single-Step | WP5 |
| **T3** | Chat-Tool-Set-Design | WP7 |
| **T3** | Test-Matrix-Scope | WP11 |
| **T4** | Migrations-Reihenfolge | WP10 |

## Globale Querverweis-Matrix (Lese-Hilfe)

| WP | braucht | liefert für |
|----|---------|-------------|
| WP1 | — | WP4, WP5, WP10 |
| WP2 | WP11 (für Tests) | WP4, WP10 |
| WP3 | — | WP4, WP7, WP11 |
| WP4 | WP3 | WP5, WP8, WP10 |
| WP5 | WP3, WP4 | WP10 |
| WP6 | — | WP7, WP8 |
| WP7 | WP3, WP6, WP11 | WP8 |
| WP8 | WP6, WP7, WP11 | WP10 |
| WP9 | WP1 | WP4, WP8, WP10 |
| WP10 | ALLE | Implementation |
| WP11 | WP3 | WP10, WP7, WP8, WP4 |
